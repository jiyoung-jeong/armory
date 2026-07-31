from __future__ import annotations

import gc
import logging
import multiprocessing as mp
import signal
from multiprocessing.synchronize import Event

import zmq

from armory.scheduling.base import RequestScheduler
from armory.scheduling.baselines import (
    GreedyDeadlineScheduler,
    MaxBatchScheduler,
    RandomBatchScheduler,
    RoundRobinScheduler,
    StarvationScheduler,
)
from armory.scheduling.dynamic_action import DynamicActionScheduler
from armory.scheduling.lookahead_actions import LookaheadActionsScheduler
from armory.serving.schemas import (
    AckNotification,
    BatchProfile,
    GpuPrepared,
    PrepareAck,
    PrepareGpu,
    PrepareScheduler,
    Reconfigure,
    ResetAll,
    ResponseBatch,
    SlotRequest,
    WarmupSeed,
)
from armory.utils import logging_config
from armory_client.messages import ResetRequest

logger = logging.getLogger(__name__)

SCHEDULER_REGISTRY: dict[str, type[RequestScheduler]] = {
    "max-batch": MaxBatchScheduler,
    "greedy-deadline": GreedyDeadlineScheduler,
    "dynamic-action": DynamicActionScheduler,
    "lookahead-actions": LookaheadActionsScheduler,
    "round-robin": RoundRobinScheduler,
    "random": RandomBatchScheduler,
    "starvation": StarvationScheduler,
}


class SchedulerWorker:
    """Subprocess worker: owns all robot state and dispatches batches to GPU via mp.Queue.

    GPU sends InferResponses directly to WS (not via this process). This process only
    receives small CompletionNotifications from GPU for state bookkeeping.
    """

    def __init__(
        self,
        sched_in_ep: str,
        result_ep: str,
        batch_queue: mp.Queue,
        scheduler_metrics_queue: mp.Queue | None,
        max_batch_size: int,
        algorithm: str,
        scheduler_kwargs: dict | None,
        ready_event: Event,
        log_queue: mp.Queue | None = None,
        control_ack_queue: mp.Queue | None = None,
    ) -> None:
        self.sched_in_ep = sched_in_ep
        self.result_ep = result_ep
        self.batch_queue = batch_queue
        self.scheduler_metrics_queue = scheduler_metrics_queue
        self.max_batch_size = max_batch_size
        self.algorithm = algorithm
        self.scheduler_kwargs = scheduler_kwargs
        self.ready_event = ready_event
        self.log_queue = log_queue
        self.control_ack_queue = control_ack_queue
        self._pending_prepares: dict[str, tuple[RequestScheduler, str, dict]] = {}

    def run(self) -> None:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        signal.signal(signal.SIGTERM, signal.SIG_DFL)

        if self.log_queue is not None:
            logging_config.setup_worker_logging(self.log_queue, process_name="scheduler")

        logger.info("Scheduler starting (algorithm=%s)", self.algorithm)

        cls = SCHEDULER_REGISTRY.get(self.algorithm)
        if cls is None:
            raise ValueError(
                f"Unknown scheduling algorithm {self.algorithm!r}. "
                f"Available: {sorted(SCHEDULER_REGISTRY)}"
            )

        ctx = zmq.Context()

        req_sock = ctx.socket(zmq.SUB)
        req_sock.setsockopt(zmq.SUBSCRIBE, b"")
        req_sock.connect(self.sched_in_ep)  # WS main connects

        result_sock = ctx.socket(zmq.SUB)
        result_sock.setsockopt(zmq.SUBSCRIBE, b"")
        result_sock.connect(self.result_ep)  # GPU connects

        extra_kwargs: dict = dict(self.scheduler_kwargs or {})
        self._current_scheduler: RequestScheduler = cls(
            self.batch_queue,
            max_batch_size=self.max_batch_size,
            **extra_kwargs,
        )
        self._result_sock = result_sock
        self._current_scheduler._drain_fn = self._make_drain_fn()

        batch_profile = self._recv_batch_profile(result_sock)
        self._batch_profile: dict[int, float] = dict(batch_profile)
        for batch_size, latency in self._batch_profile.items():
            self._current_scheduler.latency_tracker.update_infer(batch_size, latency)

        poller = zmq.Poller()
        poller.register(req_sock, zmq.POLLIN)
        poller.register(result_sock, zmq.POLLIN)

        self.ready_event.set()
        logger.info("Scheduler ready")

        # Everything alive at this point (sockets, scheduler state, latency
        # tables, imported modules) is permanent for the process lifetime.
        # Move it into the frozen generation so full collections during the
        # lookahead search only traverse per-search churn instead of the whole
        # resident heap — the 844-collected-but-0.47s sweeps were almost
        # entirely scanning these never-garbage objects.
        gc.collect()
        gc.freeze()

        tick = 0
        while True:
            tick += 1
            # logger.debug("tick=%d stage=poll_wait", tick)
            poller.poll(timeout=1)

            # logger.debug("tick=%d stage=process_engine", tick)
            self._process_engine_messages(self._current_scheduler, result_sock)

            # logger.debug("tick=%d stage=process_server", tick)
            self._process_server_messages(req_sock)

            # logger.debug("tick=%d stage=schedule_begin", tick)
            # PrepareGpu is already in the work queue. Do not let the old
            # scheduler enqueue anything behind that barrier while the GPU and
            # result-stream acknowledgments are still in flight.
            decisions = [] if self._pending_prepares else self._current_scheduler.schedule()

            if self.scheduler_metrics_queue is not None:
                try:
                    self.scheduler_metrics_queue.put_nowait(decisions)
                except Exception:
                    logger.exception("tick=%d failed to enqueue scheduler decisions", tick)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    # TODO: this two functions can probably be unified
    def _recv_batch_profile(self, result_sock: zmq.Socket) -> dict[int, float]:
        """Block until the GPU worker sends its BatchProfile over result_sock."""
        logger.info("Waiting for batch profile from GPU worker...")
        while True:
            if result_sock.poll(timeout=100):
                msg = result_sock.recv_pyobj()
                assert isinstance(msg, BatchProfile), f"Unexpected message: {type(msg).__name__}"
                return msg.latencies

    def _process_engine_messages(
        self, scheduler: RequestScheduler, result_sock: zmq.Socket
    ) -> None:
        # Drain completions first so _in_flight is up-to-date before we
        # process new requests and decide whether to schedule.
        while result_sock.poll(0):
            msg = result_sock.recv_pyobj(zmq.NOBLOCK)
            if isinstance(msg, ResponseBatch):
                scheduler.on_batch_completed(msg)
            elif isinstance(msg, GpuPrepared):
                pending = self._pending_prepares.pop(msg.operation_id, None)
                if pending is None:
                    logger.warning("Ignoring unexpected GPU prepare ack: %s", msg.operation_id)
                    continue
                replacement, algorithm, scheduler_kwargs = pending
                self._install_scheduler(replacement, algorithm, scheduler_kwargs)
                self._send_prepare_ack(msg.operation_id)
                logger.info("Run prepare complete: operation_id=%s", msg.operation_id)
            else:
                raise AssertionError(f"Unexpected message: {type(msg).__name__}")

    def _process_server_messages(self, req_sock: zmq.Socket) -> None:
        while req_sock.poll(0):
            msg = req_sock.recv_pyobj(zmq.NOBLOCK)
            scheduler = self._current_scheduler
            if isinstance(msg, ResetRequest):
                scheduler.reset_robot(msg.robot_id)
                logger.debug("Received reset request: %s", msg)
            elif isinstance(msg, ResetAll):
                scheduler.reset_all()
                logger.info("Received ResetAll: cleared scheduler + mirror state")
            elif isinstance(msg, Reconfigure):
                self._handle_reconfigure(msg)
            elif isinstance(msg, PrepareScheduler):
                self._handle_prepare(msg)
            elif isinstance(msg, SlotRequest):
                scheduler.update(msg)
                logger.debug("Received slot request: %s", msg)
            elif isinstance(msg, AckNotification):
                scheduler.update_ack(msg)
                logger.debug("Received ack notification: %s", msg)
            elif isinstance(msg, WarmupSeed):
                for arrival_ts, request_ts in msg.obs_samples:
                    scheduler.latency_tracker.update_obs(msg.robot_id, arrival_ts, request_ts)
                for client_receive_time, server_send_time in msg.delivery_samples:
                    scheduler.latency_tracker.update_action_delivery(
                        msg.robot_id, client_receive_time, server_send_time
                    )
                logger.info(
                    "Seeded latency for robot %s from warmup, observation_latency: %f, action_latency: %f",
                    msg.robot_id,
                    scheduler.latency_tracker.observation_latency(msg.robot_id),
                    scheduler.latency_tracker.action_latency(msg.robot_id),
                )
            else:
                logger.warning("Unknown message type: %s", type(msg).__name__)

    def _make_drain_fn(self):
        # Bind to whichever scheduler is current at drain time so a reconfigure
        # mid-search still drains into the active instance.
        return lambda: self._process_engine_messages(self._current_scheduler, self._result_sock)

    def _handle_reconfigure(self, msg: Reconfigure) -> None:
        self._replace_scheduler(msg.algorithm, msg.scheduler_kwargs)

    def _handle_prepare(self, msg: PrepareScheduler) -> None:
        replacement, error = self._build_scheduler(msg.algorithm, msg.scheduler_kwargs)
        if error is not None:
            self._send_prepare_ack(msg.operation_id, error=error)
            return
        assert replacement is not None

        self._pending_prepares[msg.operation_id] = (
            replacement,
            msg.algorithm,
            dict(msg.scheduler_kwargs or {}),
        )
        try:
            # This worker is the sole RequestBatch producer, so the barrier is
            # ordered after every old batch even if the mp.Queue feeder thread
            # has not flushed them yet.
            self.batch_queue.put_nowait(PrepareGpu(msg.operation_id))
        except Exception as exc:
            self._pending_prepares.pop(msg.operation_id, None)
            logger.exception("Could not enqueue GPU prepare barrier")
            self._send_prepare_ack(msg.operation_id, error=repr(exc))
            return
        logger.info("Run prepare waiting for GPU: operation_id=%s", msg.operation_id)

    def _replace_scheduler(self, algorithm: str, scheduler_kwargs: dict) -> str | None:
        replacement, error = self._build_scheduler(algorithm, scheduler_kwargs)
        if error is not None:
            return error
        assert replacement is not None
        self._install_scheduler(replacement, algorithm, scheduler_kwargs)
        return None

    def _build_scheduler(
        self, algorithm: str, scheduler_kwargs: dict
    ) -> tuple[RequestScheduler | None, str | None]:
        cls = SCHEDULER_REGISTRY.get(algorithm)
        if cls is None:
            # WS main validates before publishing; this branch is defence-in-depth.
            error = f"unknown algorithm {algorithm!r} (available: {sorted(SCHEDULER_REGISTRY)})"
            logger.error("Scheduler replacement ignored: %s", error)
            return None, error
        try:
            new_scheduler = cls(
                self.batch_queue,
                max_batch_size=self.max_batch_size,
                **dict(scheduler_kwargs or {}),
            )
        except Exception as exc:
            logger.exception(
                "Reconfigure failed to construct %s with kwargs=%s; keeping current scheduler",
                algorithm,
                scheduler_kwargs,
            )
            return None, repr(exc)
        new_scheduler._drain_fn = self._make_drain_fn()
        for batch_size, latency in self._batch_profile.items():
            new_scheduler.latency_tracker.update_infer(batch_size, latency)
        return new_scheduler, None

    def _install_scheduler(
        self,
        scheduler: RequestScheduler,
        algorithm: str,
        scheduler_kwargs: dict,
    ) -> None:
        self._current_scheduler = scheduler
        self.algorithm = algorithm
        self.scheduler_kwargs = dict(scheduler_kwargs or {})
        logger.info(
            "Reconfigured scheduler: algorithm=%s kwargs=%s",
            algorithm,
            scheduler_kwargs,
        )

    def _send_prepare_ack(self, operation_id: str, *, error: str | None = None) -> None:
        if self.control_ack_queue is None:
            logger.error("No control ack queue for prepare operation %s", operation_id)
            return
        self.control_ack_queue.put_nowait(
            PrepareAck(operation_id=operation_id, worker="scheduler", error=error)
        )
