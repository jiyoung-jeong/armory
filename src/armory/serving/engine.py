from __future__ import annotations

import logging
import multiprocessing as mp
import signal
import time
from multiprocessing.synchronize import Event

import numpy as np
import zmq

from armory.backends.types import PolicyFactory, PolicyResult, ServingPolicy
from armory.scheduling.latency import EMALatencyTracker
from armory.serving.rtc import InferType, RTCParams
from armory.serving.schemas import (
    AckNotification,
    BatchProfile,
    InternalRequest,
    RequestBatch,
    ResetAll,
    ResponseBatch,
    RobotID,
    SlotRequest,
    WarmupSeed,
)
from armory.serving.slots import RobotSlots, SlotData
from armory.utils import logging_config
from armory_client.messages import (
    InferResponse,
    ResetRequest,
)

logger = logging.getLogger(__name__)

PROFILE_ITERATIONS = 5


class GpuWorker:
    """Subprocess worker: loads model, loops recv batch -> infer -> send results.

    Sends InferResponse objects directly to WS (gpu_out_ep) and small
    CompletionNotifications to the scheduler (result_ep) for state updates.
    """

    def __init__(
        self,
        policy_factory: PolicyFactory,
        max_batch_size: int,
        slots: RobotSlots,
        batch_queue: mp.Queue,
        server_out_ep: str,
        gpu_out_ep: str,
        ready_event: Event,
        log_queue: mp.Queue | None = None,
    ) -> None:
        self.policy_factory = policy_factory
        self.max_batch_size = max_batch_size
        self.slots = slots
        self.batch_queue = batch_queue
        self.server_out_ep = server_out_ep
        self.gpu_out_ep = gpu_out_ep
        self.ready_event = ready_event
        self.log_queue = log_queue

    def run(self) -> None:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        signal.signal(signal.SIGTERM, signal.SIG_DFL)

        if self.log_queue is not None:
            logging_config.setup_worker_logging(self.log_queue, process_name="gpu-worker")

        logger.info("GPU worker starting")

        policy = self.policy_factory()
        policy.warmup(self.max_batch_size)

        ctx = zmq.Context()

        # Server messages (SlotRequest, ResetRequest, AckNotification, WarmupSeed)
        req_sock = ctx.socket(zmq.SUB)
        req_sock.setsockopt(zmq.SUBSCRIBE, b"")
        req_sock.connect(self.server_out_ep)

        # Direct path to WS _router_task (WS process binds)
        result_sock = ctx.socket(zmq.PUB)
        result_sock.bind(self.gpu_out_ep)

        # Per-robot inference state — initialised here (post-fork, not in __init__)
        self._latency_tracker = EMALatencyTracker()
        self._last_served_action_index: dict[RobotID, int] = {}
        self._prev_actions: dict[RobotID, np.ndarray] = {}

        self._profile_and_send(policy, result_sock)

        self.ready_event.set()
        logger.info("GPU worker ready")

        while True:
            self._process_server_messages(req_sock)

            batch: RequestBatch = self.batch_queue.get()  # blocking

            # Synthetic idle batch: occupy the GPU for the requested duration
            # (no inference), then report an empty completion so the scheduler's
            # timing model stays in sync with the real GPU clock.
            if batch.idle_duration > 0:
                t0 = time.time()
                end_time = time.perf_counter() + batch.idle_duration
                while time.perf_counter() < end_time:
                    pass
                result_sock.send_pyobj(
                    ResponseBatch(
                        responses=[],
                        batch_id=batch.batch_id,
                        batch_size=0,
                        inference_start_time=t0,
                        inference_duration=time.time() - t0,
                    )
                )
                continue

            slot_reqs: list[SlotRequest] = batch.requests

            # FIXME: can be much more concise
            slot_datas = []
            chunk_ids = []
            slot_requests = []
            for sr, chunk_id in zip(slot_reqs, batch.chunk_ids, strict=True):
                sd = self.slots.read(sr.slot_index)
                if sr.robot_id not in self._last_served_action_index or sr.can_serve(
                    self._last_served_action_index[sr.robot_id], sd.action_index_start
                ):
                    slot_datas.append(sd)
                    chunk_ids.append(chunk_id)
                    slot_requests.append(sr)
                else:
                    pass
                    # logger.info("Dropping request %s because it's not schedulable", sr.robot_id)

            if len(slot_datas) == 0:
                result_sock.send_pyobj(
                    ResponseBatch(
                        responses=[],
                        batch_id=batch.batch_id,
                        batch_size=len(slot_datas),
                        inference_start_time=time.time(),
                        inference_duration=0.0,
                    )
                )
                # logger.warning("Sent empty response batch")
                continue

            batch_size = len(slot_datas)
            infer_requests = [
                InternalRequest.from_slot_data(sd, self._make_params(sd, batch_size))
                for sd in slot_datas
            ]

            logger.info("Inferring batch of %d", len(infer_requests))
            t0 = time.time()
            actions = policy.infer_batch(infer_requests)
            t1 = time.time()
            inference_duration = t1 - t0

            responses = [
                self._make_infer_response(slot_data, action, chunk_id)
                for slot_data, action, chunk_id in zip(slot_datas, actions, chunk_ids, strict=True)
            ]

            self._update_state(
                slot_requests, slot_datas, actions
            )  # NOTE from Rohan: this was originally slot_reqs

            # Send responses directly to WS — not via scheduler
            result_sock.send_pyobj(
                ResponseBatch(
                    responses=responses,
                    batch_id=batch.batch_id,
                    batch_size=len(slot_requests),
                    inference_start_time=t0,
                    inference_duration=inference_duration,
                )
            )
            logger.debug("Sent response batch: %s", responses)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_infer_response(
        slot_data: SlotData,
        result: PolicyResult,
        chunk_id: int,
    ) -> InferResponse:
        """Translate one internal policy result into the client wire response."""
        return InferResponse(
            robot_id=slot_data.robot_id,
            request_id=slot_data.request_id,
            chunk_id=chunk_id,
            observation_step=slot_data.observation_step,
            action_index_start=slot_data.action_index_start,
            request_timestamp=slot_data.request_timestamp,
            min_execution_horizon=slot_data.min_execution_horizon,
            max_execution_horizon=slot_data.max_execution_horizon,
            actions=result["actions"],
            noise=result["noise"],
        )

    def _profile_and_send(self, policy: ServingPolicy, notify_sock: zmq.Socket) -> None:
        logger.info("Profiling batch latency for sizes 1..%d", self.max_batch_size)
        profile: dict[int, float] = {}
        request = policy.make_infer_request()
        for batch_size in range(1, self.max_batch_size + 1):
            latencies = []
            for _ in range(PROFILE_ITERATIONS):
                start = time.perf_counter()
                policy.infer_batch([request] * batch_size)
                latencies.append(time.perf_counter() - start)
            profile[batch_size] = sum(latencies) / len(latencies)
            self._latency_tracker.update_infer(batch_size, profile[batch_size])
            logger.info("  batch_size=%d: %.1f ms", batch_size, profile[batch_size] * 1000)
        notify_sock.send_pyobj(BatchProfile(latencies=profile))
        logger.info("Sent batch profile to scheduler")

    def _process_server_messages(self, req_sock: zmq.Socket) -> None:
        while req_sock.poll(0):
            msg = req_sock.recv_pyobj(zmq.NOBLOCK)
            if isinstance(msg, ResetRequest):
                # Latency intentionally preserved across resets — matches scheduler
                # behavior (see scheduling/base.py:reset_robot).
                self._last_served_action_index.pop(msg.robot_id, None)
                self._prev_actions.pop(msg.robot_id, None)
                logger.debug("Received reset request: %s", msg)
            elif isinstance(msg, ResetAll):
                self._last_served_action_index.clear()
                self._prev_actions.clear()
                logger.info("Received ResetAll: cleared engine RTC state (latency preserved)")
            elif isinstance(msg, SlotRequest):
                self._latency_tracker.update_obs(
                    msg.robot_id, msg.arrival_timestamp, msg.request_timestamp
                )
                logger.debug("Received slot request: %s", msg)
            elif isinstance(msg, AckNotification):
                self._latency_tracker.update_action_delivery(
                    msg.robot_id, msg.receive_time, msg.server_send_time
                )
                logger.debug("Received ack notification: %s", msg)
            elif isinstance(msg, WarmupSeed):
                for arrival_ts, request_ts in msg.obs_samples:
                    self._latency_tracker.update_obs(msg.robot_id, arrival_ts, request_ts)
                for client_receive_time, server_send_time in msg.delivery_samples:
                    self._latency_tracker.update_action_delivery(
                        msg.robot_id, client_receive_time, server_send_time
                    )
                logger.info(
                    "Seeded latency for robot %s from warmup, observation_latency: %f, action_latency: %f",
                    msg.robot_id,
                    self._latency_tracker.observation_latency(msg.robot_id),
                    self._latency_tracker.action_latency(msg.robot_id),
                )
            else:
                logger.warning("Unknown message type: %s", type(msg).__name__)

    def _make_params(self, slot_data: SlotData, batch_size: int) -> RTCParams | None:
        if (
            slot_data.infer_type == InferType.INFERENCE_TIME_RTC
            and slot_data.robot_id in self._last_served_action_index
        ):
            s = slot_data.action_index_start - self._last_served_action_index[slot_data.robot_id]
            d = (
                self._latency_tracker.total_latency(slot_data.robot_id, batch_size)
                * slot_data.control_hz
            )
            return RTCParams(
                prev_action=self._prev_actions[slot_data.robot_id], s_param=s, d_param=d
            )
        return None

    def _update_state(
        self,
        slot_reqs: list[SlotRequest],
        slot_datas: list[SlotData],
        actions: list[PolicyResult],
    ) -> None:
        for sr, sd, action_dict in zip(slot_reqs, slot_datas, actions, strict=True):
            self._last_served_action_index[sr.robot_id] = sd.action_index_start
            self._prev_actions[sr.robot_id] = action_dict["rtc_prev_actions"]
