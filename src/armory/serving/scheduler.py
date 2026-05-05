from __future__ import annotations

import logging
import multiprocessing as mp
import signal
from multiprocessing.synchronize import Event

import zmq

from armory.scheduling.base import RequestScheduler
from armory.scheduling.baselines import (
    FixedMaxBatchScheduler,
    GreedyActionScheduler,
    GreedyDeadlineScheduler,
    MaxBatchScheduler,
    RandomBatchScheduler,
    RoundRobinScheduler,
)
from armory.scheduling.dynamic_action import DynamicActionScheduler
from armory.scheduling.lookahead import LookaheadScheduler
from armory.scheduling.lookahead_actions import LookaheadActionsScheduler
from armory.serving.schemas import (
    AckNotification,
    BatchProfile,
    CompletionNotification,
    SlotRequest,
    WarmupSeed,
)
from armory.utils import logging_config
from armory_client.messages import ResetRequest

logger = logging.getLogger(__name__)


def _recv_batch_profile(result_sock: zmq.Socket) -> dict[int, float]:
    """Block until the GPU worker sends its BatchProfile over result_sock."""
    logger.info("Waiting for batch profile from GPU worker...")
    while True:
        if result_sock.poll(timeout=100):
            msg = result_sock.recv_pyobj()
            if isinstance(msg, BatchProfile):
                return msg.latencies
            logger.warning("Unexpected message before batch profile: %s", type(msg).__name__)


SCHEDULER_REGISTRY: dict[str, type[RequestScheduler]] = {
    "max-batch": MaxBatchScheduler,
    "fixed-max-batch": FixedMaxBatchScheduler,
    "greedy-action": GreedyActionScheduler,
    "greedy-deadline": GreedyDeadlineScheduler,
    "dynamic-action": DynamicActionScheduler,
    "lookahead": LookaheadScheduler,
    "lookahead-actions": LookaheadActionsScheduler,
    "round-robin": RoundRobinScheduler,
    "random": RandomBatchScheduler,
}


# FIXME: underscore method is a weird naming convention
def _run_scheduler(
    sched_in_ep: str,
    result_ep: str,
    batch_queue: mp.Queue,
    scheduler_metrics_queue: mp.Queue | None,
    max_batch_size: int,
    algorithm: str,
    scheduler_kwargs: dict | None,
    ready_event: Event,
    log_queue: mp.Queue | None = None,
) -> None:
    """Owns all robot state; dispatches batches to GPU via mp.Queue.

    GPU sends InferResponses directly to WS (not via this process). This process only receives small CompletionNotifications
    from GPU for state bookkeeping.
    """

    # NOTE: uncomment this to attach a debugger to the scheduler process
    # import debugpy

    # debugpy.listen(("0.0.0.0", 5679))  # different port from main process
    # debugpy.wait_for_client()

    # might need to run `ssh -NL 5679:localhost:5679 <server node>`
    # also might need to add this to vscode launch.json:
    # "configurations": [
    #     {
    #         "name": "Attach scheduler",
    #         "type": "debugpy",
    #         "request": "attach",
    #         "connect": {"host": "localhost", "port": 5679}
    #     }
    # ]

    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, signal.SIG_DFL)

    if log_queue is not None:
        logging_config.setup_worker_logging(log_queue, process_name="scheduler")

    logger.info("Scheduler starting (algorithm=%s)", algorithm)

    cls = SCHEDULER_REGISTRY.get(algorithm)
    if cls is None:
        raise ValueError(
            f"Unknown scheduling algorithm {algorithm!r}. Available: {sorted(SCHEDULER_REGISTRY)}"
        )
    ctx = zmq.Context()

    req_sock = ctx.socket(zmq.PULL)
    req_sock.bind(sched_in_ep)  # WS main connects

    result_sock = ctx.socket(zmq.PULL)
    result_sock.bind(result_ep)  # GPU connects

    extra_kwargs: dict = dict(scheduler_kwargs or {})
    scheduler = cls(batch_queue, max_batch_size=max_batch_size, **extra_kwargs)

    batch_profile = _recv_batch_profile(result_sock)
    for batch_size, latency in batch_profile.items():
        scheduler.latency_tracker.update_infer(batch_size, latency)

    poller = zmq.Poller()
    poller.register(req_sock, zmq.POLLIN)
    poller.register(result_sock, zmq.POLLIN)

    ready_event.set()
    logger.info("Scheduler ready")

    while True:
        poller.poll(timeout=1)

        # Drain completions first so _in_flight is up-to-date before we
        # process new requests and decide whether to schedule.
        while result_sock.poll(0):
            msg = result_sock.recv_pyobj(zmq.NOBLOCK)
            if isinstance(msg, list):
                for item in msg:
                    if isinstance(item, CompletionNotification):
                        scheduler.update_completion(item)
                # Any list from the GPU (including an empty [] sent when the
                # batch was skipped) signals that the batch slot is free.
                scheduler.notify_batch_complete()

        while req_sock.poll(0):
            msg = req_sock.recv_pyobj(zmq.NOBLOCK)
            if isinstance(msg, ResetRequest):
                scheduler.reset_robot(msg.robot_id)
                logger.debug("Received reset request: %s", msg)
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

        scheduler.advance()
        if scheduler.in_flight == 0:
            scheduler.schedule()
            if scheduler_metrics_queue is not None:
                samples = scheduler.flush_decisions()
                if samples:
                    scheduler_metrics_queue.put_nowait(samples)
            scheduler.advance()
