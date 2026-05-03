from __future__ import annotations

import logging
import multiprocessing as mp
import signal
import time
from collections.abc import Callable
from multiprocessing.synchronize import Event

import numpy as np
import zmq

from armory.serving.schemas import (
    BatchProfile,
    CompletionNotification,
    RequestBatch,
    ResponseBatch,
    SlotRequest,
)
from armory.serving.slots import RobotSlots
from armory.utils import logging_config
from armory_client.messages import InferRequest, InferResponse, InferType, RTCParams

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
            latency = t1 - t0
            latencies.append(latency)
        profile[batch_size] = sum(latencies) / len(latencies)
        logger.info("  batch_size=%d: %.1f ms", batch_size, profile[batch_size] * 1000)
    notify_sock.send_pyobj(BatchProfile(latencies=profile))
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
    _last_served_request_id: dict[
        str, int
    ] = {}  # robot_id -> last sd.request_id sent as a response

    def _make_rtc_params(
        robot_id: str,
        request_id: int,
        observation_step: int,
        action_start_step: int,
        execution_horizon: int,
        d_param: float,
    ) -> RTCParams | None:
        nonlocal _action_shape
        last = _last_infer_step.get(robot_id)
        if last is not None and observation_step < last:
            _last_infer_step.pop(robot_id, None)
            _prev_actions.pop(robot_id, None)
            last = None
        s = observation_step - last if last is not None else 0
        prev = _prev_actions.get(robot_id)
        if prev is None and _action_shape is not None:
            prev = np.zeros(_action_shape, dtype=np.float32)
        if prev is None:
            logger.info(
                "RTC params unavailable for first chunk: robot=%s request_id=%d "
                "obs_step=%d action_start_step=%d d=%.2f",
                robot_id,
                request_id,
                observation_step,
                action_start_step,
                d_param,
            )
            return None

        action_horizon = int(prev.shape[0])
        action_lag = observation_step - action_start_step
        rtc_window = s + float(d_param)
        log_message = (
            "RTC timing check: robot=%s request_id=%d obs_step=%d "
            "action_start_step=%d obs_minus_action_start=%d last_rtc_obs_step=%s "
            "s=%d d=%.2f s_plus_d=%.2f action_horizon=%d execution_horizon=%d "
            "prev_action_shape=%s"
        )
        log_args = (
            robot_id,
            request_id,
            observation_step,
            action_start_step,
            action_lag,
            str(last),
            s,
            d_param,
            rtc_window,
            action_horizon,
            execution_horizon,
            tuple(prev.shape),
        )
        if action_lag > 1 or rtc_window > action_horizon or float(d_param) > action_horizon:
            logger.warning(log_message, *log_args)
        else:
            logger.debug(log_message, *log_args)
        return RTCParams(prev_action=prev, s_param=s, d_param=d_param)

    while True:
        batch: RequestBatch = batch_queue.get()  # blocking
        slot_reqs: list[SlotRequest] = batch.requests

        # Read obs and metadata together — guarantees they correspond to the same request,
        # even if the slot was overwritten after the SlotRequest was enqueued.
        slot_datas = [slots.read(sr.slot_index) for sr in slot_reqs]

        # Drop any real slot whose request_id has already been served.  This happens when the
        # scheduler dispatches multiple SlotRequests for the same robot before the GPU
        # finishes the first one: both read the same (overwritten) slot and would produce
        # two InferResponses with identical request_ids.  request_ids are monotonically
        # increasing, so a strict > check also handles episode resets correctly. Padding
        # slots are still sent through inference to preserve the requested GPU batch size,
        # but they never produce responses or scheduler state updates.
        fresh = [
            (sr, sd)
            for sr, sd in zip(slot_reqs, slot_datas, strict=True)
            if sr.is_padding or sd.request_id > _last_served_request_id.get(sr.robot_id, 0)
        ]
        if not fresh or not any(not sr.is_padding for sr, _ in fresh):
            # Notify the scheduler so it can decrement _in_flight, even though
            # no real inference happened.
            notify_sock.send_pyobj([])
            continue
        slot_reqs, slot_datas = zip(*fresh, strict=True)
        actual_batch_size = len(slot_reqs)

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
                params=_make_rtc_params(
                    sr.robot_id,
                    sd.request_id,
                    sd.observation_step,
                    sd.action_start_step,
                    sd.execution_horizon,
                    sr.estimated_d_param,
                )
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
            if not sr.is_padding
        ]

        # Update per-robot RTC state
        for sr, sd, action_dict in zip(slot_reqs, slot_datas, actions, strict=True):
            if not sr.is_padding and sd.infer_type == InferType.INFERENCE_TIME_RTC:
                prev_action = action_dict.get(
                    "rtc_prev_actions", action_dict["actions"]
                )  # shape (ah, ad)
                if _action_shape is None:
                    _action_shape = prev_action.shape
                _last_infer_step[sr.robot_id] = sd.observation_step
                _prev_actions[sr.robot_id] = prev_action

        # Record served request_ids before sending so the duplicate check stays consistent.
        for sr, sd in zip(slot_reqs, slot_datas, strict=True):
            if not sr.is_padding:
                _last_served_request_id[sr.robot_id] = sd.request_id

        # Send responses directly to WS — not via scheduler
        response_sock.send_pyobj(
            ResponseBatch(
                responses=responses, batch_id=batch.batch_id, batch_size=actual_batch_size
            )
        )

        # Notify scheduler of completion so it can update latency estimates
        inference_duration = t1 - t0
        notify_sock.send_pyobj(
            [
                CompletionNotification(
                    robot_id=sr.robot_id,
                    action_start_step=sd.action_start_step,
                    request_id=sd.request_id,
                    batch_size=actual_batch_size,
                    inference_duration=inference_duration,
                    observation_step=sd.observation_step,
                    execution_horizon=sd.execution_horizon,
                    server_arrival_time=sd.arrival_timestamp,
                )
                for sr, sd in zip(slot_reqs, slot_datas, strict=True)
                if not sr.is_padding
            ],
        )
