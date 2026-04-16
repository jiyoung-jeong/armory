from abc import ABC
from abc import abstractmethod
from contextlib import contextmanager
import dataclasses
import multiprocessing as mp
import time

from openpi.scheduling.latency import LatencyTracker
from openpi.serving.schemas import AckNotification
from openpi.serving.schemas import CompletionNotification
from openpi.serving.schemas import SchedulerDecision
from openpi.serving.schemas import SlotRequest


class RequestScheduler(ABC):
    def __init__(
        self,
        batch_queue: mp.Queue,
        max_batch_size: int = 1,
        batch_profile: dict[int, float] | None = None,
    ):
        self._batch_queue = batch_queue
        self._max_batch_size = max_batch_size
        self._batch_profile_ms: dict[int, float] = batch_profile or {}

        self._latest_requests: dict[str, SlotRequest] = {}
        self._latest_scheduled_requests: dict[str, SlotRequest] = {}
        self._deadlines: dict[str, float] = {}  # includes chunks that have been sent to the GPU but not yet completed
        self._decisions: list[SchedulerDecision] = []
        self.latency = LatencyTracker()

    def update(self, request: SlotRequest) -> None:
        self._latest_requests[request.robot_id] = request
        if request.deadline is not None and request.deadline > self._deadlines.get(request.robot_id, 0):
            self._deadlines[request.robot_id] = request.deadline
        self.latency.update_obs(request.robot_id, request.arrival_timestamp, request.request_timestamp)

    def update_completion(self, notification: CompletionNotification) -> None:
        self.latency.update_infer(notification.batch_size, notification.inference_duration_ms)

    def update_ack(self, notification: AckNotification) -> None:
        self.latency.update_action_delivery(
            notification.robot_id,
            notification.receive_time,
            notification.server_send_time,
        )

    def schedule(self) -> None:
        """Return a list of batches of requests to be sent to the GPU."""
        candidates = self._get_schedulable_requests()
        with self.record_timing() as duration:
            batches = self.get_next_batches()

        for batch in batches:
            batch_size = len(batch)
            # Capture deadlines before the loop overwrites them, sort earliest first.
            candidate_entries = sorted(
                ({"robot_id": r.robot_id, "deadline": self._deadlines.get(r.robot_id, r.deadline)} for r in candidates),
                key=lambda x: x["deadline"],
            )
            batch_entries = sorted(
                ({"robot_id": r.robot_id, "deadline": self._deadlines.get(r.robot_id, r.deadline)} for r in batch),
                key=lambda x: x["deadline"],
            )
            annotated = []
            for request in batch:
                self._deadlines[request.robot_id] = request.deadline + request.execution_horizon / request.control_hz
                self._latest_scheduled_requests[request.robot_id] = request
                d_ms = self.latency.total_delivery_ms(request.robot_id, batch_size)
                step_ms = 1000.0 / request.control_hz
                d_steps = round(d_ms / step_ms) if d_ms is not None else 0
                annotated.append(dataclasses.replace(request, estimated_d_param=d_steps))

            # FIXME: this branch only has single batch decisions for now, will need to refactor timing for multi batch decisions
            self._decisions.append(
                SchedulerDecision(
                    scheduler_name=self.__class__.__name__,
                    metric_name="batch_scheduled",
                    duration_ms=duration() * 1e3,
                    recorded_at=time.time(),
                    candidates=candidate_entries,
                    scheduled=batch_entries,
                )
            )
            self._batch_queue.put_nowait(annotated)

    @abstractmethod
    def get_next_batches(self) -> list[list[SlotRequest]]:
        pass

    def reset_robot(self, robot_id: str) -> None:
        self._deadlines.pop(robot_id, None)
        self._latest_requests.pop(robot_id, None)
        self._latest_scheduled_requests.pop(robot_id, None)
        self.latency.reset_robot(robot_id)

    @contextmanager
    def record_timing(self) -> float:
        start = time.perf_counter()
        yield lambda: time.perf_counter() - start

    def flush_decisions(self) -> list[SchedulerDecision]:
        samples = self._decisions
        self._decisions = []
        return samples

    def _get_schedulable_requests(self) -> list[SlotRequest]:
        """Get all requests that are not yet scheduled and past the minimum execution horizon."""
        result = []
        for req in self._latest_requests.values():
            last = self._latest_scheduled_requests.get(req.robot_id)
            if last is req:
                continue
            if last is not None and req.action_start_step <= last.action_start_step:
                continue
            result.append(req)
        return result
