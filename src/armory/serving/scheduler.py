from __future__ import annotations

import gc
import json
import logging
import multiprocessing as mp
import pathlib
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
from armory.serving.config import ServerConfig
from armory.serving.schemas import (
    AckNotification,
    BatchProfile,
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
        metrics_dir: pathlib.Path,
        config: ServerConfig,
        ready_event: Event,
        log_queue: mp.Queue | None = None,
    ) -> None:
        self.sched_in_ep = sched_in_ep
        self.result_ep = result_ep
        self.batch_queue = batch_queue
        self.metrics_dir = metrics_dir
        self.config = config
        self.ready_event = ready_event
        self.log_queue = log_queue

    def run(self) -> None:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        signal.signal(signal.SIGTERM, signal.SIG_DFL)

        if self.log_queue is not None:
            logging_config.setup_worker_logging(self.log_queue, process_name="scheduler")

        logger.info("Scheduler starting (%s)", self.config)

        if self.config.scheduler.scheduling_algorithm not in SCHEDULER_REGISTRY:
            raise ValueError(
                f"Unknown scheduling algorithm {self.config.scheduler.scheduling_algorithm!r}. "
                f"Available: {sorted(SCHEDULER_REGISTRY)}"
            )

        decisions_log = open(self.metrics_dir / "scheduler_decisions.jsonl", "w")

        ctx = zmq.Context()

        req_sock = ctx.socket(zmq.SUB)
        req_sock.setsockopt(zmq.SUBSCRIBE, b"")
        req_sock.connect(self.sched_in_ep)  # WS main connects

        result_sock = ctx.socket(zmq.SUB)
        result_sock.setsockopt(zmq.SUBSCRIBE, b"")
        result_sock.connect(self.result_ep)  # GPU connects

        self._result_sock = result_sock
        self._current_scheduler: RequestScheduler = self._build_scheduler(self.config)

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
            decisions = self._current_scheduler.schedule()

            if decisions:
                try:
                    for decision in decisions:
                        # vars instead of asdict to avoid copy
                        decisions_log.write(json.dumps(vars(decision), default=float) + "\n")
                    decisions_log.flush()
                except Exception:
                    logger.exception("tick=%d failed to record scheduler decisions", tick)

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
            assert isinstance(msg, ResponseBatch), f"Unexpected message: {type(msg).__name__}"
            scheduler.on_batch_completed(msg)

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

    def _build_scheduler(self, config: ServerConfig) -> RequestScheduler:
        cls = SCHEDULER_REGISTRY[config.scheduler.scheduling_algorithm]
        scheduler = cls(config.scheduler, self.batch_queue, max_batch_size=config.max_batch_size)
        scheduler._drain_fn = self._make_drain_fn()
        return scheduler

    def _handle_reconfigure(self, msg: Reconfigure) -> None:
        if msg.config.scheduler.scheduling_algorithm not in SCHEDULER_REGISTRY:
            # WS main validates before publishing; this branch is defence-in-depth.
            logger.error(
                "Reconfigure ignored: unknown algorithm %r (available: %s)",
                msg.config.scheduler.scheduling_algorithm,
                sorted(SCHEDULER_REGISTRY),
            )
            return
        try:
            new_scheduler = self._build_scheduler(msg.config)
        except Exception:
            logger.exception(
                "Reconfigure failed to construct %s; keeping current scheduler",
                msg.config,
            )
            return
        for batch_size, latency in self._batch_profile.items():
            new_scheduler.latency_tracker.update_infer(batch_size, latency)
        self._current_scheduler = new_scheduler
        self.config = msg.config
        logger.info("Reconfigured scheduler: %s", msg.config)
