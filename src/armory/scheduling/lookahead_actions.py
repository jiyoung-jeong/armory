import copy
import itertools
import logging
import multiprocessing as mp
import time
from collections.abc import Iterator
from dataclasses import dataclass

from armory.scheduling.base import RequestScheduler
from armory.scheduling.latency import LatencyTracker
from armory.scheduling.mirror import ActionChunk, Mirror, Robot, robot_id
from armory.serving.schemas import SlotRequest

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def _action_time(robot: Robot) -> float:
    """Wall-clock duration of all actions ever queued for a robot."""
    if not robot.chunks:
        return 0.0
    return (robot.max_overall_action_step + 1) / robot.control_hz


def _action_times(mirror: Mirror) -> dict[robot_id, float]:
    return {rid: _action_time(robot) for rid, robot in mirror.robots.items()}


@dataclass(frozen=True)
class ScheduledBatch:
    robot_ids: tuple[robot_id, ...]
    chunks: tuple[ActionChunk, ...]


class _Frame:
    """A DFS frame: state at ``time`` after committing ``schedule``."""

    __slots__ = ("mirror", "time", "schedule", "candidate_iter")

    def __init__(
        self,
        mirror: Mirror,
        time: float,
        schedule: list[ScheduledBatch],
        candidate_iter: Iterator[tuple[robot_id, ...]] | None,
    ) -> None:
        self.mirror = mirror
        self.time = time
        self.schedule = schedule
        self.candidate_iter = candidate_iter


class IncrementalSearch:
    """Stack-based DFS that yields control after a bounded budget of nodes.

    Searches schedules that keep the GPU busy until ``start_time + horizon``
    and tracks the best objective seen so far. Drive it by calling ``step``
    repeatedly until ``is_done`` returns True; ``best`` is safe to read at any
    point.
    """

    def __init__(
        self,
        mirror: Mirror,
        latency_tracker: LatencyTracker,
        start_time: float,
        horizon: float,
        max_depth: int = 3,
    ) -> None:
        self.latency_tracker = latency_tracker
        self.start_time = start_time
        self.end_time = start_time + horizon
        self.max_depth = max_depth
        self.initial_action_times = _action_times(mirror)
        # FIXME: don't access private
        self.max_batch_size = max(latency_tracker._infer_latency.keys())

        self.best_objective = -float("inf")
        self.best_schedule: list[ScheduledBatch] = []
        self.nodes_visited = 1
        self.branches_pruned_time = 0

        root_mirror = copy.deepcopy(mirror)
        root = _Frame(root_mirror, start_time, [], None)
        if max_depth > 0:
            root.candidate_iter = self._candidate_iter(root_mirror)
        self._stack: list[_Frame] = [root]

    def is_done(self) -> bool:
        return not self._stack

    def step(self, budget_nodes: int = 32) -> None:
        for _ in range(budget_nodes):
            if not self._stack:
                return
            self._step_one()

    def best(self) -> list[ScheduledBatch]:
        return list(self.best_schedule)

    def _step_one(self) -> None:
        frame = self._stack[-1]
        if frame.candidate_iter is None:
            self._stack.pop()
            return

        try:
            batch = next(frame.candidate_iter)
        except StopIteration:
            self._stack.pop()
            return

        next_time = frame.time + self.latency_tracker.infer_latency(len(batch))
        if next_time > self.end_time:
            self.branches_pruned_time += 1
            return

        next_state = copy.deepcopy(frame.mirror)
        chunks = tuple(next_state.get_chunks(list(batch), self.latency_tracker, frame.time))
        next_state.fast_forward(next_time, list(batch), list(chunks))

        new_schedule = frame.schedule + [ScheduledBatch(batch, chunks)]
        new_frame = _Frame(next_state, next_time, new_schedule, None)
        self.nodes_visited += 1
        self._evaluate(new_frame)

        if len(new_schedule) < self.max_depth:
            new_frame.candidate_iter = self._candidate_iter(next_state)
            self._stack.append(new_frame)

    def _evaluate(self, frame: _Frame) -> None:
        gpu_time = frame.time - self.start_time
        if gpu_time <= 0:
            return
        new_times = _action_times(frame.mirror)
        gained = sum(new_times[rid] - self.initial_action_times[rid] for rid in new_times)
        objective = gained / gpu_time
        if objective > self.best_objective:
            self.best_objective = objective
            self.best_schedule = list(frame.schedule)
            logger.debug(
                "new best: depth=%d objective=%.4f schedule=%s",
                len(frame.schedule),
                objective,
                [batch.robot_ids for batch in frame.schedule],
            )

    def _candidate_iter(self, mirror: Mirror) -> Iterator[tuple[robot_id, ...]]:
        # FIXME: reducing search space for now
        deadlines = mirror.deadlines()
        sorted_robot_ids = sorted(mirror.robots.keys(), key=lambda rid: deadlines[rid])
        return iter(
            tuple(sorted_robot_ids[:i])
            for i in range(1, min(self.max_batch_size, len(sorted_robot_ids)) + 1)
        )


class LookaheadActionsScheduler(RequestScheduler):
    """Lookahead scheduler that searches incrementally between dispatch points.

    On each ``advance`` tick we step the in-progress search; on commit
    (``in_flight == 0``) we publish the best schedule found so far and start
    fresh on the next tick.
    """

    def __init__(
        self,
        batch_queue: mp.Queue,
        max_batch_size: int = 1,
        *,
        horizon: float = 0.5,
        max_depth: int = 3,
        step_budget_nodes: int = 32,
    ) -> None:
        super().__init__(batch_queue, max_batch_size)
        self.horizon = horizon
        self.max_depth = max_depth
        self.step_budget_nodes = step_budget_nodes
        self._search: IncrementalSearch | None = None
        self._anticipated_id_counter = itertools.count(start=-1, step=-1)
        self._planned_chunks_by_request_id: dict[int, ActionChunk] = {}

    def advance(self) -> None:
        if not self.schedulable_requests:
            self._search = None
            return
        if self._search is None:
            self._search = IncrementalSearch(
                self.mirror,
                self.latency_tracker,
                time.time(),
                self.horizon,
                self.max_depth,
            )
        if not self._search.is_done():
            self._search.step(self.step_budget_nodes)

    def get_next_batches(self) -> list[list[SlotRequest]]:
        if not self._batch_queue.empty() or not self.schedulable_requests:
            return []
        if self._search is None:
            self.advance()
        if self._search is None:
            return []
        if not self._search.best() and not self._search.is_done():
            self._search.step(self.step_budget_nodes)

        schedule = self._search.best()
        logger.debug(
            "search committed: nodes=%d pruned_time=%d objective=%.4f len=%d",
            self._search.nodes_visited,
            self._search.branches_pruned_time,
            self._search.best_objective,
            len(schedule),
        )
        self._search = None
        if not schedule:
            return []
        return self._build_batches(schedule, time.time())

    def _build_batches(self, schedule: list[ScheduledBatch], now: float) -> list[list[SlotRequest]]:
        seen: set[robot_id] = set()
        result: list[list[SlotRequest]] = []
        cumulative_infer = 0.0
        for scheduled_batch in schedule:
            dispatch_time = now + cumulative_infer
            requests: list[SlotRequest] = []
            for rid, chunk in zip(scheduled_batch.robot_ids, scheduled_batch.chunks, strict=True):
                if rid not in self._latest_requests:
                    continue
                base = self._latest_requests[rid]
                if rid not in seen:
                    request = base
                    seen.add(rid)
                else:
                    request = self.mirror.anticipate_request(
                        base,
                        chunk,
                        dispatch_time,
                        next(self._anticipated_id_counter),
                    )
                self._planned_chunks_by_request_id[request.request_id] = chunk
                requests.append(request)
            if requests:
                result.append(requests)
                cumulative_infer += self.latency_tracker.infer_latency(len(requests))
        return result

    def _action_chunk_for_request(
        self, request: SlotRequest, batch_size: int, dispatch_time: float
    ) -> ActionChunk:
        planned = self._planned_chunks_by_request_id.pop(request.request_id, None)
        if planned is not None:
            return planned
        return super()._action_chunk_for_request(request, batch_size, dispatch_time)
