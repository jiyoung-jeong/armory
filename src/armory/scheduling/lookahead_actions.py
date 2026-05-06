import copy
import itertools
import logging
import multiprocessing as mp
import time
from collections.abc import Iterator
from dataclasses import dataclass, replace

from armory.scheduling.base import RequestScheduler
from armory.scheduling.latency import LatencyTracker
from armory.scheduling.mirror import ActionChunk, Checkpoint, Mirror, Robot
from armory.serving.schemas import RobotID, SlotRequest

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def _action_time(robot: Robot) -> float:
    """Wall-clock duration of all actions ever queued for a robot."""
    if not robot.chunks:
        return 0.0
    return (robot.max_overall_action_step + 1) / robot.control_hz


def _action_times(mirror: Mirror) -> dict[RobotID, float]:
    return {rid: _action_time(robot) for rid, robot in mirror.robots.items()}


def _mirror_summary(mirror: Mirror, now: float) -> str:
    """One-line summary of per-robot buffer health: deadline gap and chunk count."""
    if not mirror.robots:
        return "no robots"
    parts = []
    deadlines = mirror.deadlines()
    for rid, robot in sorted(mirror.robots.items()):
        deadline_in = deadlines[rid] - now
        n_chunks = len(robot.chunks)
        buffer_steps = robot.max_overall_action_step + 1 if robot.chunks else 0
        parts.append(
            f"{rid}: deadline_in={deadline_in:+.3f}s chunks={n_chunks} buffer_steps={buffer_steps}"
        )
    return " | ".join(parts)


def _anticipate_request(base: SlotRequest, chunk: ActionChunk, dispatch_time: float) -> SlotRequest:
    """Fabricate a SlotRequest representing a future (depth>1) dispatch for a robot."""
    return replace(
        base,
        request_id=chunk.request_id,
        observation_step=chunk.observation_step,
        action_index_start=chunk.action_index_start,
        request_timestamp=dispatch_time,
        arrival_timestamp=dispatch_time,
    )


@dataclass(frozen=True)
class ScheduledBatch:
    robot_ids: tuple[RobotID, ...]
    chunks: tuple[ActionChunk, ...]


class _Frame:
    """A DFS frame: snapshot state at ``time`` after committing ``schedule``.

    Holds a Checkpoint, not a Mirror copy. ``IncrementalSearch.snapshot``
    reflects the top-of-stack frame's state; restore is called on backtrack
    or on max-depth siblings.
    """

    __slots__ = ("checkpoint", "time", "schedule", "candidate_iter")

    def __init__(
        self,
        checkpoint: Checkpoint,
        time: float,
        schedule: list[ScheduledBatch],
        candidate_iter: Iterator[tuple[RobotID, ...]] | None,
    ) -> None:
        self.checkpoint = checkpoint
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
        anticipated_id_counter: itertools.count,
        start_time: float,
        horizon: float,
        max_depth: int = 3,
    ) -> None:
        self.latency_tracker = latency_tracker
        self.start_time = start_time
        self.end_time = start_time + horizon
        self.max_depth = max_depth
        self._anticipated_id_counter = anticipated_id_counter
        # FIXME: don't access private
        self.max_batch_size = max(latency_tracker._infer_latency.keys())

        self.best_objective = -float("inf")
        self.best_schedule: list[ScheduledBatch] = []
        self.nodes_visited = 1
        self.branches_pruned_time = 0

        # Single snapshot mutated in place; checkpoints in frames revert it.
        self.snapshot = copy.deepcopy(mirror)
        # Advance to search-start so get_chunks finds an observation cutoff
        # for the first dispatch.
        self.snapshot.fast_forward(start_time, [], [])
        self.initial_action_times = _action_times(self.snapshot)

        root = _Frame(self.snapshot.checkpoint(), start_time, [], None)
        if max_depth > 0:
            root.candidate_iter = self._candidate_iter()
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
            if self._stack:
                self.snapshot.restore(self._stack[-1].checkpoint)
            return

        try:
            batch = next(frame.candidate_iter)
        except StopIteration:
            self._stack.pop()
            if self._stack:
                self.snapshot.restore(self._stack[-1].checkpoint)
            return

        next_time = frame.time + self.latency_tracker.infer_latency(len(batch))
        if next_time > self.end_time:
            self.branches_pruned_time += 1
            return  # snapshot still in frame state, no restore needed

        request_ids = [next(self._anticipated_id_counter) for _ in batch]
        chunks = tuple(self.snapshot.get_chunks(list(batch), request_ids, frame.time))
        self.snapshot.fast_forward(next_time, list(batch), list(chunks))

        new_schedule = frame.schedule + [ScheduledBatch(batch, chunks)]
        self.nodes_visited += 1
        self._evaluate(new_schedule, next_time)

        if len(new_schedule) < self.max_depth:
            self._stack.append(
                _Frame(self.snapshot.checkpoint(), next_time, new_schedule, self._candidate_iter())
            )
        else:
            # Don't push; restore so the next sibling starts from frame state.
            self.snapshot.restore(frame.checkpoint)

    def _evaluate(self, schedule: list[ScheduledBatch], frame_time: float) -> None:
        gpu_time = frame_time - self.start_time
        if gpu_time <= 0:
            return
        new_times = _action_times(self.snapshot)
        gained = sum(new_times[rid] - self.initial_action_times.get(rid, 0.0) for rid in new_times)
        objective = gained / gpu_time
        if objective > self.best_objective:
            self.best_objective = objective
            self.best_schedule = list(schedule)
            logger.debug(
                "new best: depth=%d objective=%.4f schedule=%s",
                len(schedule),
                objective,
                [batch.robot_ids for batch in schedule],
            )

    def _candidate_iter(self) -> Iterator[tuple[RobotID, ...]]:
        # FIXME: reducing search space for now
        deadlines = self.snapshot.deadlines()
        sorted_robot_ids = sorted(self.snapshot.robots.keys(), key=lambda rid: deadlines[rid])
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
        self._batches_dispatched: int = 0
        self._planned_chunk_hits: int = 0
        self._planned_chunk_misses: int = 0

    def advance(self) -> None:
        if not self.schedulable_requests:
            self._search = None
            return
        if self._search is None:
            now = time.time()
            logger.debug(
                "search started: robots=%d in_flight=%d batches_dispatched=%d | %s",
                len(self.mirror.robots),
                self._in_flight,
                self._batches_dispatched,
                _mirror_summary(self.mirror, now),
            )
            self._search = IncrementalSearch(
                self.mirror,
                self.latency_tracker,
                self._anticipated_id_counter,
                now,
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
        now = time.time()
        logger.debug(
            "search committed: nodes=%d pruned_time=%d objective=%.4f"
            " schedule_len=%d in_flight=%d batches_dispatched=%d"
            " chunk_hits=%d chunk_misses=%d | %s",
            self._search.nodes_visited,
            self._search.branches_pruned_time,
            self._search.best_objective,
            len(schedule),
            self._in_flight,
            self._batches_dispatched,
            self._planned_chunk_hits,
            self._planned_chunk_misses,
            _mirror_summary(self.mirror, now),
        )
        self._search = None
        if not schedule:
            return []
        return self._build_batches(schedule, now)

    def _build_batches(self, schedule: list[ScheduledBatch], now: float) -> list[list[SlotRequest]]:
        seen: set[RobotID] = set()
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
                    logger.debug(
                        "batch slot: rid=%s real obs_step=%d", rid, request.observation_step
                    )
                else:
                    request = _anticipate_request(base, chunk, dispatch_time)
                    logger.debug(
                        "batch slot: rid=%s anticipated obs_step=%d action_start=%d dispatch_in=%.3fs",
                        rid,
                        request.observation_step,
                        request.action_index_start,
                        dispatch_time - now,
                    )
                requests.append(request)
            if requests:
                result.append(requests)
                self._batches_dispatched += 1
                cumulative_infer += self.latency_tracker.infer_latency(len(requests))
        return result

    def _action_chunk_for_request(
        self, request: SlotRequest, batch_size: int, dispatch_time: float
    ) -> ActionChunk:
        robot = self.mirror.robots.get(request.robot_id)
        if robot is not None:
            for chunk in robot.chunks:
                if chunk.request_id == request.request_id:
                    self._planned_chunk_hits += 1
                    return chunk
        self._planned_chunk_misses += 1
        logger.debug(
            "planned chunk miss: rid=%s request_id=%d falling back",
            request.robot_id,
            request.request_id,
        )
        return ActionChunk(
            request_id=request.request_id,
            observation_step=request.observation_step,
            arrival_time=dispatch_time
            + self.latency_tracker.infer_latency(batch_size)
            + self.latency_tracker.action_latency(request.robot_id),
            action_index_start=request.action_index_start,
            execution_horizon=request.execution_horizon,
            arrived=False,
        )
