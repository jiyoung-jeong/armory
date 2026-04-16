from __future__ import annotations

from collections.abc import Callable
import logging
import multiprocessing as mp
from multiprocessing.synchronize import Event
import signal
import time

import numpy as np
from openpi_client.messages import InferRequest
from openpi_client.messages import InferResponse
from openpi_client.messages import InferType
from openpi_client.messages import RTCParams
import zmq

from openpi.serving.schemas import BatchProfile
from openpi.serving.schemas import CompletionNotification
from openpi.serving.schemas import SlotRequest
from openpi.serving.slots import RobotSlots
from openpi.shared import logging_config

logger = logging.getLogger(__name__)


def _profile_and_send(policy, max_batch_size: int, notify_sock: zmq.Socket) -> None:
    """Profile inference latency for each batch size and send a BatchProfile to the scheduler."""
    logger.info("Profiling batch latency for sizes 1..%d", max_batch_size)
    profile: dict[int, float] = {}

    request = policy.make_infer_request()
    for batch_size in range(1, max_batch_size + 1):
        latencies = []
        for _ in range(5):
            t0 = time.perf_counter()
            policy.infer_batch([request] * batch_size)
            t1 = time.perf_counter()
            latency = (t1 - t0) * 1e3
            latencies.append(latency)
        profile[batch_size] = sum(latencies) / len(latencies)
        logger.info("  batch_size=%d → %.1f ms", batch_size, profile[batch_size])
    notify_sock.send_pyobj(BatchProfile(latency_ms=profile))
    logger.info("Sent batch profile to scheduler")


def _run_gpu_worker(
    policy_factory: Callable,
    max_batch_size: int,
    slots: RobotSlots,
    batch_queue: mp.Queue,
    gpu_out_ep: str,
    result_ep: str,
    ready_event: Event,
    log_queue: mp.Queue | None = None,
) -> None:
    """Loads model, then loops: recv batch → read obs from shared memory → infer → send results.

    Sends InferResponse objects directly to WS (gpu_out_ep) and small CompletionNotifications
    to the scheduler (result_ep) for state updates. These are decoupled so ILP solving in the
    scheduler cannot delay response delivery to clients.
    """
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, signal.SIG_DFL)

    if log_queue is not None:
        logging_config.setup_worker_logging(log_queue, process_name="gpu-worker")

    logger.info("GPU worker starting")

    policy = policy_factory()
    policy.warmup(max_batch_size)

    ctx = zmq.Context()

    # Direct path to WS _router_task (WS process binds)
    response_sock = ctx.socket(zmq.PUSH)
    response_sock.connect(gpu_out_ep)

    # State-update path to scheduler (scheduler binds)
    notify_sock = ctx.socket(zmq.PUSH)
    notify_sock.connect(result_ep)

    _profile_and_send(policy, max_batch_size, notify_sock)

    ready_event.set()
    logger.info("GPU worker ready")

    _last_infer_step: dict[str, int] = {}
    _prev_actions: dict[str, np.ndarray] = {}
    _action_shape: tuple[int, int] | None = None
    _last_served_request_id: dict[str, int] = {}  # robot_id -> last sd.request_id sent as a response

    def _make_rtc_params(robot_id: str, start_step: int, d_param: int) -> RTCParams | None:
        nonlocal _action_shape
        last = _last_infer_step.get(robot_id)
        if last is not None and start_step < last:
            _last_infer_step.pop(robot_id, None)
            _prev_actions.pop(robot_id, None)
            last = None
        s = start_step - last if last is not None else 0
        prev = _prev_actions.get(robot_id)
        if prev is None and _action_shape is not None:
            prev = np.zeros(_action_shape, dtype=np.float32)
        if prev is None:
            return None
        logger.debug(
            "Built RTC params for robot=%s start_step=%d s=%d d=%d prev_action_shape=%s",
            robot_id,
            start_step,
            s,
            d_param,
            tuple(prev.shape),
        )
        return RTCParams(prev_action=prev, s_param=s, d_param=d_param)

    while True:
        slot_reqs: list[SlotRequest] = batch_queue.get()  # blocking

        # Read obs and metadata together — guarantees they correspond to the same request,
        # even if the slot was overwritten after the SlotRequest was enqueued.
        slot_datas = [slots.read(sr.slot_index) for sr in slot_reqs]

        # Drop any slot whose request_id has already been served.  This happens when the
        # scheduler dispatches multiple SlotRequests for the same robot before the GPU
        # finishes the first one: both read the same (overwritten) slot and would produce
        # two InferResponses with identical request_ids.  request_ids are monotonically
        # increasing, so a strict > check also handles episode resets correctly.
        fresh = [
            (sr, sd)
            for sr, sd in zip(slot_reqs, slot_datas, strict=True)
            if sd.request_id > _last_served_request_id.get(sr.robot_id, 0)
        ]
        if not fresh:
            continue
        slot_reqs, slot_datas = zip(*fresh, strict=True)

        infer_requests = [
            InferRequest(
                robot_id=sr.robot_id,
                observation=sd.obs,
                observation_step=sd.observation_step,
                action_start_step=sd.action_start_step,
                execution_horizon=sd.execution_horizon,
                request_timestamp=sd.request_timestamp,
                deadline=sd.deadline,
                infer_type=sd.infer_type,
                params=_make_rtc_params(sr.robot_id, sd.observation_step, sr.estimated_d_param)
                if sd.infer_type == InferType.INFERENCE_TIME_RTC
                else sd.params,
                noise=sd.noise,
            )
            for sr, sd in zip(slot_reqs, slot_datas, strict=True)
        ]

        logger.debug("Inferring batch of %d", len(infer_requests))
        t0 = time.time()
        actions = policy.infer_batch(infer_requests)
        t1 = time.time()

        responses = [
            InferResponse(
                robot_id=sr.robot_id,
                request_id=sd.request_id,
                observation_step=sd.observation_step,
                action_start_step=sd.action_start_step,
                request_timestamp=sd.request_timestamp,
                execution_horizon=sd.execution_horizon,
                actions=action_dict["actions"],
                noise=action_dict["noise"],
                server_arrival_time=sd.arrival_timestamp,
                inference_start_time=t0,
                inference_end_time=t1,
            )
            for sr, sd, action_dict in zip(slot_reqs, slot_datas, actions, strict=True)
        ]

        # Update per-robot RTC state
        for sr, sd, action_dict in zip(slot_reqs, slot_datas, actions, strict=True):
            if sd.infer_type == InferType.INFERENCE_TIME_RTC:
                prev_action = action_dict.get("rtc_prev_actions", action_dict["actions"])  # shape (ah, ad)
                if _action_shape is None:
                    _action_shape = prev_action.shape
                _last_infer_step[sr.robot_id] = sd.observation_step
                _prev_actions[sr.robot_id] = prev_action

        # Record served request_ids before sending so the duplicate check stays consistent.
        for sr, sd in zip(slot_reqs, slot_datas, strict=True):
            _last_served_request_id[sr.robot_id] = sd.request_id

        # Send responses directly to WS — not via scheduler, so ILP latency doesn't affect clients
        response_sock.send_pyobj(responses)

        # Notify scheduler of completion so it can update latency estimates
        inference_duration_ms = (t1 - t0) * 1e3
        notify_sock.send_pyobj(
            [
                CompletionNotification(
                    robot_id=sr.robot_id,
                    action_start_step=sd.action_start_step,
                    request_id=sd.request_id,
                    batch_size=len(slot_reqs),
                    inference_duration_ms=inference_duration_ms,
                )
                for sr, sd in zip(slot_reqs, slot_datas, strict=True)
            ],
        )
