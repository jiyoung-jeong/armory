import copy
import itertools
import logging
import multiprocessing as mp
from collections.abc import Iterator
from dataclasses import dataclass

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

    def get_next_batches(self) -> list[list[SlotRequest]]:
        SCHEDULING_BUFFER = 0.01
        logger.debug(
            "search started: robots=%d in_flight=%d batches_dispatched=%d | %s",
            len(self.mirror.robots),
            self._in_flight,
            self._batches_dispatched,
            _mirror_summary(self.mirror, self.mirror.next_time_server_available),
        )
        self._search = IncrementalSearch(
            self.mirror,
            self.latency_tracker,
            self._anticipated_id_counter,
            self.mirror.next_time_server_available,
            self.horizon,
            self.max_depth,
        )
        if self.mirror.time_until_server_available < SCHEDULING_BUFFER:
            # TODO: just do greedy
            pass

        while (
            self.mirror.time_until_server_available > SCHEDULING_BUFFER
            and not self._search.is_done()
        ):
            logger.debug("Advancing search")  # TODO:
            self.advance()
            logger.debug("Search advanced, updated best to")  # TODO:

        return self._search.best()
