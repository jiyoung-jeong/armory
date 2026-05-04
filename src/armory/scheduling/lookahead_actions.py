import copy
import logging
import multiprocessing as mp
import time

from armory.scheduling.base import RequestScheduler
from armory.scheduling.latency import LatencyTracker
from armory.scheduling.mirror import Mirror, robot_id
from armory.serving.schemas import SlotRequest

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


# TODO: can be modified to support different search strategies/pruning/stopping criteria
class Search:
    """Search through all schedules that keep the GPU busy until start_time + horizon."""

    def __init__(
        self,
        mirror: Mirror,
        latency_tracker: LatencyTracker,
        start_time: float,
        horizon: float,
        max_depth: int = 1,
    ) -> None:
        self.mirror = mirror
        self.latency_tracker = latency_tracker
        self.start_time = start_time
        self.end_time = start_time + horizon
        self.max_depth = max_depth
        self.initial_deadlines = mirror.deadlines()
        # FIXME: don't access private
        self.max_batch_size = max(latency_tracker._infer_latency.keys())

        self.best_objective = -float("inf")
        self.best_schedule: list[tuple[robot_id, ...]] = []
        self.nodes_visited = 0
        self.branches_pruned_time = 0
        self.leaves = 0

    def run(self) -> list[tuple[robot_id, ...]]:
        logger.debug(
            "search start: t=%.4f end_time=%.4f robots=%d max_batch=%d",
            self.start_time,
            self.end_time,
            len(self.mirror.robots),
            self.max_batch_size,
        )
        self._dfs(self.mirror, self.start_time, [])
        logger.debug(
            "search done: nodes=%d leaves=%d pruned_time=%d best_objective=%.4f best_len=%d",
            self.nodes_visited,
            self.leaves,
            self.branches_pruned_time,
            self.best_objective,
            len(self.best_schedule),
        )
        return self.best_schedule

    def _objective(self, mirror: Mirror, time: float) -> float:
        gpu_time = time - self.start_time
        if gpu_time <= 0:
            return -float("inf")
        new_deadlines = mirror.deadlines()
        gained_time = sum(new_deadlines[rid] - self.initial_deadlines[rid] for rid in new_deadlines)
        return gained_time / gpu_time

    def _generate_candidates(self, mirror: Mirror):
        # FIXME: reducing search space for now
        # Sort robots by their deadlines (earliest first)
        deadlines = mirror.deadlines()
        sorted_robot_ids = sorted(mirror.robots.keys(), key=lambda rid: deadlines[rid])

        # Yield batch choices of increasing size up to max_batch_size, always prefixing the sorted list
        return (
            tuple(sorted_robot_ids[:i])
            for i in range(1, min(self.max_batch_size, len(sorted_robot_ids)) + 1)
        )

    def _dfs(
        self,
        mirror: Mirror,
        time: float,
        schedule: list[tuple[robot_id, ...]],
    ) -> None:
        self.nodes_visited += 1
        objective_value = self._objective(mirror, time)
        if objective_value > self.best_objective:
            self.best_objective = objective_value
            self.best_schedule = schedule
            logger.debug(
                "new best: depth=%d objective=%.4f schedule=%s",
                len(schedule),
                objective_value,
                schedule,
            )

        if len(schedule) == self.max_depth:
            return

        candidates = list(self._generate_candidates(mirror))
        if not candidates:
            self.leaves += 1
            logger.debug(
                "leaf (no candidates): depth=%d t=%.4f robots=%d",
                len(schedule),
                time,
                len(mirror.robots),
            )
            return

        expanded = 0
        for batch in candidates:
            next_time = time + self.latency_tracker.infer_latency(len(batch))
            if next_time > self.end_time:
                self.branches_pruned_time += 1
                continue
            expanded += 1

            next_state = copy.deepcopy(mirror)
            chunks = next_state.get_chunks(list(batch), self.latency_tracker, time)
            next_state.fast_forward(next_time, list(batch), chunks)
            self._dfs(next_state, next_time, schedule + [batch])

        if expanded == 0:
            self.leaves += 1
            logger.debug(
                "leaf (all branches past end_time): depth=%d t=%.4f candidates=%d",
                len(schedule),
                time,
                len(candidates),
            )


class LookaheadActionsScheduler(RequestScheduler):
    def __init__(
        self, batch_queue: mp.Queue, max_batch_size: int = 1, *, horizon: float = 0.5
    ) -> None:
        super().__init__(batch_queue, max_batch_size)
        self.horizon = horizon

    def get_next_batches(self) -> list[list[SlotRequest]]:
        if not self._batch_queue.empty() or self.schedulable_requests == []:
            return []

        schedule = Search(self.mirror, self.latency_tracker, time.time(), self.horizon).run()
        return [[self._latest_requests[robot_id] for robot_id in batch] for batch in schedule]
