import dataclasses
import itertools
import logging
import multiprocessing as mp
import time
from abc import ABC, abstractmethod
from collections.abc import Callable, Generator
from contextlib import contextmanager

from armory.scheduling.latency import EMALatencyTracker
from armory.scheduling.mirror import ActionChunk, Mirror
from armory.serving.schemas import (
    AckNotification,
    CompletionNotification,
    RequestBatch,
    SchedulerDecision,
    SlotRequest,
)
from armory_client.messages import InferType

logger = logging.getLogger(__name__)


class RequestScheduler(ABC):
    def __init__(
        self,
        batch_queue: mp.Queue,
        max_batch_size: int = 1,
    ):
        self._batch_queue = batch_queue
        self._max_batch_size = max_batch_size

        self.mirror = Mirror()

        # TODO: eventually a bunch of this can be moved to the mirror
        self._latest_requests: dict[str, SlotRequest] = {}
        self._latest_scheduled_requests: dict[str, SlotRequest] = {}
        self._deadlines: dict[
            str, float
        ] = {}  # includes chunks that have been sent to the GPU but not yet completed
        self._decisions: list[SchedulerDecision] = []
        self.latency_tracker = EMALatencyTracker()  # TODO: allow different latency trackers
        self.next_batch_id = itertools.count(1)
        self._in_flight = 0

    def update(self, request: SlotRequest) -> None:
        self._latest_requests[request.robot_id] = request
        if request.deadline is not None and request.deadline > self._deadlines.get(
            request.robot_id, 0
        ):
            self._deadlines[request.robot_id] = request.deadline
        self.latency_tracker.update_obs(
            request.robot_id, request.arrival_timestamp, request.request_timestamp
        )
        self.mirror.receive_request(request, request.control_hz)

    def update_completion(self, notification: CompletionNotification) -> None:
        self.latency_tracker.update_infer(notification.batch_size, notification.inference_duration)

    def update_ack(self, notification: AckNotification) -> None:
        self.latency_tracker.update_action_delivery(
            notification.robot_id,
            notification.receive_time,
            notification.server_send_time,
        )
        self.mirror.receive_response(notification)

    def schedule(self) -> None:
        """Return a list of batches of requests to be sent to the GPU."""
        all_requests = list(self._latest_requests.values())
        candidates = list(self.schedulable_requests)
        with self.record_timing() as duration:
            batches = self.get_next_batches()

        now = time.time()
        request_entries = self._request_entries(all_requests, now)
        candidate_entries = self._candidate_entries(candidates, now)
        cumulative_infer_latency = 0.0
        for batch in batches:
            dispatch_time = now + cumulative_infer_latency
            batch = self._filter_batch(batch)
            real_batch = [request for request in batch if not request.is_padding]
            if not real_batch:
                continue

            batch_size = len(batch)
            annotated = self._annotate_batch(batch, batch_size)
            self._apply_batch(batch, batch_size, dispatch_time)
            batch_id = next(self.next_batch_id)
            self._decisions.append(
                SchedulerDecision(
                    scheduler_name=self.__class__.__name__,
                    metric_name="batch_scheduled",
                    duration=duration(),
                    recorded_at=now,
                    requests=request_entries,
                    candidates=candidate_entries,
                    scheduled=self._batch_entries(real_batch, now),
                    batch_id=batch_id,
                )
            )
            self._batch_queue.put_nowait(RequestBatch(requests=annotated, batch_id=batch_id))
            self._in_flight += 1
            cumulative_infer_latency += self.latency_tracker.infer_latency(batch_size)

    def _filter_batch(self, batch: list[SlotRequest]) -> list[SlotRequest]:
        batch = [
            request for request in batch if request.is_padding or self._is_new_request(request)
        ]
        real_robot_ids = {request.robot_id for request in batch if not request.is_padding}
        return [
            request
            for request in batch
            if not request.is_padding or request.robot_id in real_robot_ids
        ]

    def _request_entries(self, requests: list[SlotRequest], now: float) -> list[dict]:
        return sorted(
            (
                {
                    "robot_id": request.robot_id,
                    "observation_step": request.observation_step,
                    "action_start_step": request.action_start_step,
                    "deadline": self._deadlines.get(request.robot_id, request.deadline) - now,
                }
                for request in requests
            ),
            key=lambda x: x["deadline"],
        )

    def _candidate_entries(self, candidates: list[SlotRequest], now: float) -> list[dict]:
        return sorted(
            (
                {
                    "robot_id": request.robot_id,
                    "deadline": self._deadlines.get(request.robot_id, request.deadline) - now,
                }
                for request in candidates
            ),
            key=lambda x: x["deadline"],
        )

    def _batch_entries(self, batch: list[SlotRequest], now: float) -> list[dict]:
        return sorted(
            (
                {
                    "robot_id": request.robot_id,
                    "deadline": self._deadlines.get(request.robot_id, request.deadline) - now,
                }
                for request in batch
            ),
            key=lambda x: x["deadline"],
        )

    def _annotate_batch(self, batch: list[SlotRequest], batch_size: int) -> list[SlotRequest]:
        annotated = []
        inference_latency = self.latency_tracker.infer_latency(batch_size)
        for request in batch:
            observation_latency = self.latency_tracker.observation_latency(request.robot_id)
            action_latency = self.latency_tracker.action_latency(request.robot_id)
            total_latency_steps = (
                observation_latency + inference_latency + action_latency
            ) * request.control_hz
            if request.infer_type == InferType.INFERENCE_TIME_RTC and not request.is_padding:
                logger.info(
                    "RTC d estimate: robot=%s request_id=%d batch_size=%d "
                    "obs_step=%d action_start_step=%d control_hz=%.2f "
                    "obs_latency_ms=%.1f infer_latency_ms=%.1f "
                    "action_latency_ms=%.1f d_steps=%.2f execution_horizon=%d",
                    request.robot_id,
                    request.request_id,
                    batch_size,
                    request.observation_step,
                    request.action_start_step,
                    request.control_hz,
                    observation_latency * 1000,
                    inference_latency * 1000,
                    action_latency * 1000,
                    total_latency_steps,
                    request.execution_horizon,
                )
            # FIXME: only pass inference + action latency, can determine observation latency when processing
            annotated.append(dataclasses.replace(request, estimated_d_param=total_latency_steps))
        return annotated

    def _apply_batch(self, batch: list[SlotRequest], batch_size: int, dispatch_time: float) -> None:
        for request in batch:
            if request.is_padding:
                continue
            self._deadlines[request.robot_id] = request.deadline
            self._latest_scheduled_requests[request.robot_id] = request
            self.mirror.schedule_pending_chunk(
                request.robot_id,
                self._action_chunk_for_request(request, batch_size, dispatch_time),
            )

    def _action_chunk_for_request(
        self, request: SlotRequest, batch_size: int, dispatch_time: float
    ) -> ActionChunk:
        inference_latency = self.latency_tracker.infer_latency(batch_size)
        action_latency = self.latency_tracker.action_latency(request.robot_id)
        return ActionChunk(
            observation_step=request.observation_step,
            arrival_time=dispatch_time + inference_latency + action_latency,
            action_start_step=request.action_start_step,
            execution_horizon=request.execution_horizon,
            arrived=False,
        )

    def notify_batch_complete(self) -> None:
        self._in_flight = max(0, self._in_flight - 1)

    def advance(self) -> None:
        """Advance any background/incremental scheduler work."""

    @property
    def in_flight(self) -> int:
        return self._in_flight

    @abstractmethod
    def get_next_batches(self) -> list[list[SlotRequest]]:
        pass

    def reset_robot(self, robot_id: str) -> None:
        self._deadlines.pop(robot_id, None)
        self._latest_requests.pop(robot_id, None)
        self._latest_scheduled_requests.pop(robot_id, None)
        self.mirror.reset_robot(robot_id)

    def clear(self, robot_id: str) -> None:
        self.reset_robot(robot_id)
        self.latency_tracker.clear(robot_id)

    @contextmanager
    def record_timing(self) -> Generator[Callable[[], float], None, None]:
        start = time.perf_counter()
        yield lambda: time.perf_counter() - start

    def flush_decisions(self) -> list[SchedulerDecision]:
        samples = self._decisions
        self._decisions = []
        return samples

    @property
    def schedulable_requests(self) -> list[SlotRequest]:
        """Get all requests that have a greater action start step."""
        result = []
        for req in self._latest_requests.values():
            if not self._is_new_request(req):
                continue
            result.append(req)
        return result

    def _is_new_request(self, request: SlotRequest) -> bool:
        last = self._latest_scheduled_requests.get(request.robot_id)
        if last is None:
            return True
        return (
            request.action_start_step > last.action_start_step
            and request.observation_step > last.observation_step
        )
