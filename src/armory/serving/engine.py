from __future__ import annotations

import logging
import multiprocessing as mp
import signal
import time
from collections.abc import Callable
from multiprocessing.synchronize import Event

import numpy as np
import zmq

from armory.scheduling.latency import LatencyTracker
from armory.serving.schemas import (
    BatchProfile,
    CompletionNotification,
    RequestBatch,
    ResponseBatch,
    RobotID,
    SlotRequest,
)
from armory.serving.slots import RobotSlots, SlotData
from armory.utils import logging_config
from armory_client.messages import (
    AckNotification,
    InferResponse,
    InferType,
    InternalRequest,
    ResetRequest,
    RTCParams,
    TrainTimeRTCParams,
    VlashParams,
    WarmupSeed,
)

logger = logging.getLogger(__name__)

PROFILE_ITERATIONS = 5


def _profile_and_send(policy, max_batch_size: int, notify_sock: zmq.Socket) -> None:
    """Profile inference latency for each batch size and send a BatchProfile to the scheduler."""
    logger.info("Profiling batch latency for sizes 1..%d", max_batch_size)
    profile: dict[int, float] = {}

    request = policy.make_infer_request()
    for batch_size in range(1, max_batch_size + 1):
        latencies = []
        for _ in range(PROFILE_ITERATIONS):
            start = time.perf_counter()
            policy.infer_batch([request] * batch_size)
            latencies.append(time.perf_counter() - start)
        profile[batch_size] = sum(latencies) / len(latencies)
        logger.info("  batch_size=%d: %.1f ms", batch_size, profile[batch_size] * 1000)
    notify_sock.send_pyobj(BatchProfile(latencies=profile))
    logger.info("Sent batch profile to scheduler")


def process_server_messages(latency_tracker: LatencyTracker, req_sock: zmq.Socket) -> None:
    while req_sock.poll(0):
        msg = req_sock.recv_pyobj(zmq.NOBLOCK)
        if isinstance(msg, ResetRequest):
            latency_tracker.reset_robot(msg.robot_id)
            logger.debug("Received reset request: %s", msg)
        elif isinstance(msg, SlotRequest):
            latency_tracker.update_obs(msg.robot_id, msg.arrival_timestamp, msg.request_timestamp)
            logger.debug("Received slot request: %s", msg)
        elif isinstance(msg, AckNotification):
            latency_tracker.update_action_delivery(
                msg.robot_id, msg.receive_time, msg.server_send_time
            )
            logger.debug("Received ack notification: %s", msg)
        elif isinstance(msg, WarmupSeed):
            for arrival_ts, request_ts in msg.obs_samples:
                latency_tracker.update_obs(msg.robot_id, arrival_ts, request_ts)
            for client_receive_time, server_send_time in msg.delivery_samples:
                latency_tracker.update_action_delivery(
                    msg.robot_id, client_receive_time, server_send_time
                )
            logger.info(
                "Seeded latency for robot %s from warmup, observation_latency: %f, action_latency: %f",
                msg.robot_id,
                latency_tracker.observation_latency(msg.robot_id),
                latency_tracker.action_latency(msg.robot_id),
            )
        else:
            logger.warning("Unknown message type: %s", type(msg).__name__)


def make_params(
    slot_data: SlotData,
    last_infer_step: dict[RobotID, int],
    latency_tracker: LatencyTracker,
    prev_actions: dict[RobotID, np.ndarray],
) -> RTCParams | VlashParams | TrainTimeRTCParams | None:
    if (
        slot_data.infer_type == InferType.INFERENCE_TIME_RTC
        and slot_data.robot_id in last_infer_step
    ):
        s = slot_data.action_index_start - last_infer_step[slot_data.robot_id]
        d = latency_tracker.total_latency(slot_data.robot_id, len(slot_data)) / slot_data.control_hz
        return RTCParams(prev_action=prev_actions[slot_data.robot_id], s_param=s, d_param=d)
    return None


# TODO: at this point, the class is so bloated this should be a class
def _run_gpu_worker(
    policy_factory: Callable,
    max_batch_size: int,
    slots: RobotSlots,
    batch_queue: mp.Queue,
    server_out_ep: str,
    gpu_out_ep: str,
    result_ep: str,
    ready_event: Event,
    log_queue: mp.Queue | None = None,
) -> None:
    """Loads model, then loops: recv batch → read obs from shared memory → infer → send results.

    Sends InferResponse objects directly to WS (gpu_out_ep) and small CompletionNotifications
    to the scheduler (result_ep) for state updates.
    """
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, signal.SIG_DFL)

    if log_queue is not None:
        logging_config.setup_worker_logging(log_queue, process_name="gpu-worker")

    logger.info("GPU worker starting")

    policy = policy_factory()
    policy.warmup(max_batch_size)

    ctx = zmq.Context()

    # Server messages (SlotRequest, ResetRequest, AckNotification, WarmupSeed)
    req_sock = ctx.socket(zmq.SUB)
    req_sock.connect(server_out_ep)

    # Direct path to WS _router_task (WS process binds)
    response_sock = ctx.socket(zmq.PUSH)
    response_sock.connect(gpu_out_ep)

    # State-update path to scheduler (scheduler binds)
    notify_sock = ctx.socket(zmq.PUSH)
    notify_sock.connect(result_ep)

    _profile_and_send(policy, max_batch_size, notify_sock)

    ready_event.set()
    logger.info("GPU worker ready")

    latency_tracker = LatencyTracker()
    _last_served_action_index: dict[RobotID, int] = {}
    _prev_actions: dict[RobotID, np.ndarray] = {}
    _action_shape: tuple[int, int] | None = None

    def should_serve_request(sd: SlotData) -> bool:
        return sd.is_padding or sd.action_index_start > _last_served_action_index.get(
            sd.robot_id, 0
        )

    def update_state() -> None:
        # Update per-robot RTC state
        # for sr, sd, action_dict in zip(slot_reqs, slot_datas, actions, strict=True):
        #     if not sr.is_padding and sd.infer_type == InferType.INFERENCE_TIME_RTC:
        #         prev_action = action_dict.get(
        #             "rtc_prev_actions", action_dict["actions"]
        #         )  # shape (ah, ad)
        #         if _action_shape is None:
        #             _action_shape = prev_action.shape
        #         _last_infer_step[sr.robot_id] = sd.observation_step
        #         _prev_actions[sr.robot_id] = prev_action
        pass

    while True:
        process_server_messages(latency_tracker, req_sock)

        batch: RequestBatch = batch_queue.get()  # blocking
        slot_reqs: list[SlotRequest] = batch.requests

        slot_datas = [slots.read(sr.slot_index) for sr in slot_reqs]
        slot_datas = [sd for sd in slot_datas if should_serve_request(sd)]

        if len(slot_datas) == 0:
            notify_sock.send_pyobj([])
            continue

        # slot_reqs, slot_datas = zip(*fresh, strict=True)
        # TODO: padding requests should be handled minimally
        # TODO: how to estimate d_params well

        infer_requests = [InternalRequest.from_slot_data(sd, make_params(sd)) for sd in slot_datas]

        logger.debug("Inferring batch of %d", len(infer_requests))
        t0 = time.time()
        actions = policy.infer_batch(infer_requests)
        t1 = time.time()
        inference_duration = t1 - t0

        # TODO: eventually make this a class method on InferResponse or whatever
        responses = [
            InferResponse(
                robot_id=sd.robot_id,
                request_id=sd.request_id,
                observation_step=sd.observation_step,
                action_index_start=sd.action_index_start,
                request_timestamp=sd.request_timestamp,
                execution_horizon=sd.execution_horizon,
                actions=action_dict["actions"],
                noise=action_dict["noise"],
                server_arrival_time=sd.arrival_timestamp,
                inference_start_time=t0,
                inference_end_time=t1,
            )
            for sd, action_dict in zip(slot_datas, actions, strict=True)
        ]

        # TODO: implement it this way
        update_state()

        # TODO: handle padding, also don't duplicate so much code
        # Send responses directly to WS — not via scheduler
        # TODO: maybe these two things can be unified under a single pub sub interface
        response_sock.send_pyobj(
            ResponseBatch(responses=responses, batch_id=batch.batch_id, batch_size=len(slot_datas))
        )

        # Notify scheduler of completion so it can update latency estimates
        notify_sock.send_pyobj(
            [
                CompletionNotification.from_slot_data(sd, len(slot_datas), inference_duration)
                for sd in slot_datas
            ],
        )
