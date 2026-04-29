import dataclasses
import itertools
import multiprocessing as mp
import random
import time

from armory.scheduling import RequestScheduler
from armory.scheduling.latency import LatencyTracker
from armory.serving.schemas import SlotRequest


def calculate_usable_time(latency_tracker: LatencyTracker, slot_request: SlotRequest, batch_size: int) -> float:
    total_latency = latency_tracker.total_latency(slot_request.robot_id, batch_size)
    total_chunk_time = slot_request.execution_horizon / slot_request.control_hz
    return total_chunk_time - total_latency


class MaxBatchScheduler(RequestScheduler):
    """Greedy scheduler that always fills to max_batch_size, prioritizing requests with earliest deadlines."""

    def get_next_batches(self) -> list[list[SlotRequest]]:
        if self._batch_queue.qsize() > 0 or (candidates := self.schedulable_requests) == []:
            return []

        candidates = sorted(candidates, key=lambda r: self._deadlines.get(r.robot_id, r.deadline))
        return [candidates[: self._max_batch_size]]


class FixedMaxBatchScheduler(RequestScheduler):
    """Always dispatch max_batch_size rows, padding with artificial duplicate requests if needed."""

    def get_next_batches(self) -> list[list[SlotRequest]]:
        if self._batch_queue.qsize() > 0 or (candidates := self.schedulable_requests) == []:
            return []

        candidates = sorted(candidates, key=lambda r: self._deadlines.get(r.robot_id, r.deadline))
        batch = candidates[: self._max_batch_size]
        if len(batch) == self._max_batch_size:
            return [batch]

        pad_sources = list(batch)
        pad_index = 0
        while len(batch) < self._max_batch_size:
            source = pad_sources[pad_index % len(pad_sources)]
            batch.append(dataclasses.replace(source, is_padding=True))
            pad_index += 1
        return [batch]


class GreedyActionScheduler(RequestScheduler):
    """Earliest-deadline-first: sort all pending requests by deadline."""

    def get_next_batches(self) -> list[list[SlotRequest]]:
        if self._batch_queue.qsize() > 0 or (candidates := self.schedulable_requests) == []:
            return []

        potential_batches = itertools.chain.from_iterable(
            itertools.combinations(candidates, i) for i in range(1, self._max_batch_size + 1)
        )
        return [list(max(potential_batches, key=lambda batch: self.calculate_actions_per_second(batch)))]

    def calculate_actions_per_second(self, batch: tuple[SlotRequest, ...]) -> float:
        """Return the number of usable actions created per second spent on inference."""
        return sum(
            calculate_usable_time(self.latency_tracker, request, len(batch)) for request in batch
        ) / self.latency_tracker.infer_latency(len(batch))

class GreedyDeadlineScheduler(RequestScheduler):
    """Earliest-deadline-first: sort all pending requests by deadline."""

    def get_next_batches(self) -> list[list[SlotRequest]]:
        if self._batch_queue.qsize() > 0 or (candidates := self.schedulable_requests) == []:
            return []

        candidates_and_infer_deadlines = sorted(
            [
                (
                    slot_request,
                    self._deadlines.get(slot_request.robot_id, slot_request.deadline)
                    - self.latency_tracker.action_latency(slot_request.robot_id),
                )
                for slot_request in candidates
            ],
            key=lambda x: x[1],
        )
        _, earliest_infer_deadline = candidates_and_infer_deadlines[0]
        batch_size = self.get_largest_batch_size(earliest_infer_deadline)
        return [[x[0] for x in candidates_and_infer_deadlines[:batch_size]]]

    def get_largest_batch_size(self, infer_deadline: float) -> int:
        """Return the largest batch size whose profiled latency fits within the time remaining until deadline."""
        # we can assume inference starts right away because queue is empty
        time_remaining = infer_deadline - time.time()
        for batch_size in range(self._max_batch_size, 0, -1):
            if self.latency_tracker.infer_latency(batch_size) <= time_remaining:
                return batch_size
        return self.most_efficient_batch_size

    @property
    def most_efficient_batch_size(self) -> int:
        """Batch size with the best throughput (requests / ms)."""
        return max(range(1, self._max_batch_size + 1), key=lambda bs: bs / self.latency_tracker.infer_latency(bs))

class RoundRobinScheduler(RequestScheduler):
    """Cycle through robots starting from the current pointer, fill to max_batch_size."""

    def __init__(
        self,
        batch_queue: mp.Queue,
        max_batch_size: int = 1,
    ):
        super().__init__(batch_queue, max_batch_size)
        self._rr_index: int = 0
        self._rr_robot_order: list[str] = []

    def update(self, request: SlotRequest) -> None:
        super().update(request)
        if request.robot_id not in self._rr_robot_order:
            self._rr_robot_order.append(request.robot_id)

    def get_next_batches(self) -> list[list[SlotRequest]]:
        if self._batch_queue.qsize() > 0:
            return []

        candidate_by_robot = {req.robot_id: req for req in self.schedulable_requests}
        n_robots = len(self._rr_robot_order)
        if not candidate_by_robot or n_robots == 0:
            return []

        batch: list[SlotRequest] = []
        idx = self._rr_index % n_robots
        for _ in range(n_robots):
            robot_id = self._rr_robot_order[idx]
            if robot_id in candidate_by_robot:
                batch.append(candidate_by_robot[robot_id])
            idx = (idx + 1) % n_robots
            if len(batch) == self._max_batch_size:
                break

        self._rr_index = idx
        return [batch] if batch else []

    def reset_robot(self, robot_id: str) -> None:
        super().reset_robot(robot_id)
        # FIXME: temporary hack to remove robot on reset
        if robot_id in self._rr_robot_order:
            removed_index = self._rr_robot_order.index(robot_id)
            self._rr_robot_order.remove(robot_id)
            if removed_index < self._rr_index:
                self._rr_index = max(0, self._rr_index - 1)


class RandomBatchScheduler(RequestScheduler):
    """Randomly select up to max_batch_size from pending requests."""

    def get_next_batches(self) -> list[list[SlotRequest]]:
        if self._batch_queue.qsize() > 0:
            return []

        candidates = list(self.schedulable_requests)
        if not candidates:
            return []

        k = min(self._max_batch_size, len(candidates))
        return [random.sample(candidates, random.randint(1, k))]
