import dataclasses
import logging
import multiprocessing as mp
import random
import time
from typing import Any

from armory.scheduling.base import RequestScheduler
from armory.serving.schemas import RobotID, SlotRequest

logger = logging.getLogger(__name__)


class MaxBatchScheduler(RequestScheduler):
    """Greedy scheduler that always fills to max_batch_size, prioritizing requests with earliest deadlines."""

    def get_next_batches(
        self, candidates: list[SlotRequest]
    ) -> tuple[list[list[SlotRequest]], dict[str, Any]]:
        if self.mirror.in_flight_batches_count > 0:
            return [], {"reason": "server_busy"}
        if not candidates:
            return [], {"reason": "no_candidates"}

        deadlines = self.mirror.deadlines()
        ordered = sorted(candidates, key=lambda r: deadlines[r.robot_id])
        batch = ordered[: self._max_batch_size]
        notes = {
            "rule": "edf_prefix",
            "max_batch_size": self._max_batch_size,
            "ordered": [r.robot_id for r in ordered],
        }
        return [batch], notes


class FixedMaxBatchScheduler(RequestScheduler):
    """Always dispatch max_batch_size rows, padding with artificial duplicate requests if needed."""

    def get_next_batches(
        self, candidates: list[SlotRequest]
    ) -> tuple[list[list[SlotRequest]], dict[str, Any]]:
        if self.mirror.in_flight_batches_count > 0:
            return [], {"reason": "server_busy"}
        if not candidates:
            return [], {"reason": "no_candidates"}

        deadlines = self.mirror.deadlines()
        ordered = sorted(candidates, key=lambda r: deadlines[r.robot_id])
        batch = list(ordered[: self._max_batch_size])
        real_size = len(batch)
        if real_size == self._max_batch_size:
            return [batch], {
                "rule": "edf_prefix_fixed",
                "max_batch_size": self._max_batch_size,
                "padded": 0,
            }

        pad_sources = list(batch)
        pad_index = 0
        while len(batch) < self._max_batch_size:
            source = pad_sources[pad_index % len(pad_sources)]
            batch.append(dataclasses.replace(source, is_padding=True))
            pad_index += 1
        notes = {
            "rule": "edf_prefix_fixed",
            "max_batch_size": self._max_batch_size,
            "padded": self._max_batch_size - real_size,
        }
        return [batch], notes


class GreedyDeadlineScheduler(RequestScheduler):
    """Earliest-deadline-first: sort all pending requests by deadline."""

    def get_next_batches(
        self, candidates: list[SlotRequest]
    ) -> tuple[list[list[SlotRequest]], dict[str, Any]]:
        if self.mirror.in_flight_batches_count > 0:
            return [], {"reason": "server_busy"}
        if not candidates:
            return [], {"reason": "no_candidates"}

        deadlines = self.mirror.deadlines()
        candidates_and_infer_deadlines = sorted(
            [
                (
                    slot_request,
                    deadlines[slot_request.robot_id]
                    - self.latency_tracker.action_latency(slot_request.robot_id),
                )
                for slot_request in candidates
            ],
            key=lambda x: x[1],
        )
        _, earliest_infer_deadline = candidates_and_infer_deadlines[0]
        batch_size = self.get_largest_batch_size(earliest_infer_deadline)
        batch = [x[0] for x in candidates_and_infer_deadlines[:batch_size]]
        notes = {
            "rule": "edf_with_latency_fit",
            "max_batch_size": self._max_batch_size,
            "chosen_batch_size": batch_size,
            "earliest_infer_deadline": earliest_infer_deadline,
            "infer_deadlines": {slot.robot_id: d for slot, d in candidates_and_infer_deadlines},
        }
        return [batch], notes

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
        return max(
            range(1, self._max_batch_size + 1),
            key=lambda bs: bs / self.latency_tracker.infer_latency(bs),
        )


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

    def get_next_batches(
        self, candidates: list[SlotRequest]
    ) -> tuple[list[list[SlotRequest]], dict[str, Any]]:
        if self.mirror.in_flight_batches_count > 0:
            return [], {"reason": "server_busy"}

        candidate_by_robot = {req.robot_id: req for req in candidates}
        n_robots = len(self._rr_robot_order)
        if not candidate_by_robot or n_robots == 0:
            return [], {"reason": "no_candidates" if not candidate_by_robot else "no_robots_known"}

        batch: list[SlotRequest] = []
        idx = self._rr_index % n_robots
        starting_index = idx
        for _ in range(n_robots):
            robot_id = self._rr_robot_order[idx]
            if robot_id in candidate_by_robot:
                batch.append(candidate_by_robot[robot_id])
            idx = (idx + 1) % n_robots
            if len(batch) == self._max_batch_size:
                break

        self._rr_index = idx
        notes = {
            "rule": "round_robin",
            "max_batch_size": self._max_batch_size,
            "rr_index_before": starting_index,
            "rr_index_after": idx,
            "robot_order": list(self._rr_robot_order),
        }
        return ([batch], notes) if batch else ([], notes)

    def reset_robot(self, robot_id: RobotID) -> None:
        super().reset_robot(robot_id)
        # FIXME: temporary hack to remove robot on reset
        if robot_id in self._rr_robot_order:
            removed_index = self._rr_robot_order.index(robot_id)
            self._rr_robot_order.remove(robot_id)
            if removed_index < self._rr_index:
                self._rr_index = max(0, self._rr_index - 1)


class RandomBatchScheduler(RequestScheduler):
    """Randomly select up to max_batch_size from pending requests."""

    def get_next_batches(
        self, candidates: list[SlotRequest]
    ) -> tuple[list[list[SlotRequest]], dict[str, Any]]:
        if self.mirror.in_flight_batches_count > 0:
            return [], {"reason": "server_busy"}
        if not candidates:
            return [], {"reason": "no_candidates"}

        k = min(self._max_batch_size, len(candidates))
        chosen_size = random.randint(1, k)
        batch = random.sample(candidates, chosen_size)
        notes = {
            "rule": "random",
            "max_batch_size": self._max_batch_size,
            "candidate_pool_size": len(candidates),
            "chosen_batch_size": chosen_size,
        }
        return [batch], notes
