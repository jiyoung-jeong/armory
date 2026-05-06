import dataclasses
import itertools
import logging
import multiprocessing as mp
import time
from abc import ABC, abstractmethod
from collections.abc import Callable, Generator
from contextlib import contextmanager

from armory.scheduling.latency import EMALatencyTracker
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

    def update_completion(self, notification: CompletionNotification) -> None:
        self.latency_tracker.update_infer(notification.batch_size, notification.inference_duration)

    def update_ack(self, notification: AckNotification) -> None:
        self.latency_tracker.update_action_delivery(
            notification.robot_id,
            notification.receive_time,
            notification.server_send_time,
        )

    def schedule(self) -> None:
        """Return a list of batches of requests to be sent to the GPU."""
        all_requests = self._latest_requests.values()
        candidates = self.schedulable_requests
        with self.record_timing() as duration:
            batches = self.get_next_batches()

        now = time.time()
        for batch in batches:
            batch_size = len(batch)
            real_batch = [request for request in batch if not request.is_padding]
            # Capture deadlines before the loop overwrites them, sort earliest first.
            requests = sorted(
                (
                    {
                        "robot_id": r.robot_id,
                        "observation_step": r.observation_step,
                        "action_start_step": r.action_start_step,
                        "deadline": self._deadlines.get(r.robot_id, r.deadline) - now,
                    }
                    for r in all_requests
                ),
                key=lambda x: x["deadline"],
            )
            candidate_entries = sorted(
                (
                    {
                        "robot_id": r.robot_id,
                        "deadline": self._deadlines.get(r.robot_id, r.deadline) - now,
                    }
                    for r in candidates
                ),
                key=lambda x: x["deadline"],
            )
            batch_entries = sorted(
                (
                    {
                        "robot_id": r.robot_id,
                        "deadline": self._deadlines.get(r.robot_id, r.deadline) - now,
                    }
                    for r in real_batch
                ),
                key=lambda x: x["deadline"],
            )
            annotated = []
            for request in batch:
                # FIXME: this might monotonically increase if we end up serving a newer observation?
                if not request.is_padding:
                    self._deadlines[request.robot_id] = (
                        request.request_timestamp + request.execution_horizon / request.control_hz
                    )
                    self._latest_scheduled_requests[request.robot_id] = request
                observation_latency = self.latency_tracker.observation_latency(request.robot_id)
                inference_latency = self.latency_tracker.infer_latency(batch_size)
                action_latency = self.latency_tracker.action_latency(request.robot_id)
                total_latency_steps = (
                    observation_latency + inference_latency + action_latency
                ) * request.control_hz
                # FIXME: only pass inference + action latency, can determine observation latency when processing
                annotated.append(
                    dataclasses.replace(request, estimated_d_param=total_latency_steps)
                )

            batch_id = next(self.next_batch_id)

            # FIXME: this branch only has single batch decisions for now, will need to refactor timing for multi batch decisions
            self._decisions.append(
                SchedulerDecision(
                    scheduler_name=self.__class__.__name__,
                    metric_name="batch_scheduled",
                    duration=duration(),
                    recorded_at=now,
                    requests=requests,
                    candidates=candidate_entries,
                    scheduled=batch_entries,
                    batch_id=batch_id,
                )
            )
            self._batch_queue.put_nowait(RequestBatch(requests=annotated, batch_id=batch_id))
            self._in_flight += 1

    def notify_batch_complete(self) -> None:
        self._in_flight = max(0, self._in_flight - 1)

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
            last = self._latest_scheduled_requests.get(req.robot_id)
            if last is not None and req.action_start_step <= last.action_start_step:
                continue
            result.append(req)
        return result
