"""Lookahead scheduler that searches batches during GPU slack time."""

import itertools
import logging
import multiprocessing as mp
import time
from collections import deque, namedtuple
from collections.abc import Mapping
from typing import Any, TypeAlias

from armory.scheduling.base import RequestScheduler
from armory.scheduling.latency import LatencyTracker
from armory.scheduling.mirror import Mirror, Robot
from armory.serving.protocol import SchedulerConfig
from armory.serving.schemas import Idle, RobotID, SlotRequest

logger = logging.getLogger(__name__)

# Synthetic idle durations (seconds) the search may insert to defer the next
# dispatch — letting robots progress until a better batch becomes schedulable.
DEFAULT_IDLE_DURATIONS: tuple[float, ...] = tuple()
DEBUG_MODE = False


def _serialize_action(action: "Batch | Idle") -> Any:
    """JSON-friendly form of a schedule entry for debug notes."""
    if isinstance(action, Idle):
        return {"idle": action.duration}
    return list(action)


def _action_time(robot: Robot) -> float:
    """Wall-clock duration of all actions ever queued for a robot."""
    if not robot.chunks:
        return 0.0
    return (robot.max_overall_action_step + 1) / robot.control_hz


def _action_times(mirror: Mirror) -> dict[RobotID, float]:
    return {rid: _action_time(robot) for rid, robot in mirror.robots.items()}


def _execution_times(mirror: Mirror) -> dict[RobotID, float]:
    return {rid: robot.executed_steps / robot.control_hz for rid, robot in mirror.robots.items()}


def _starvation_time(robot: Robot) -> float:
    return robot.starved_steps() / robot.control_hz


def _starvation_times(mirror: Mirror) -> dict[RobotID, float]:
    return {rid: _starvation_time(robot) for rid, robot in mirror.robots.items()}


def _coerce_horizon_multipliers(
    multipliers: Mapping[int | str, float] | None,
) -> dict[int, float]:
    if multipliers is None:
        return {}
    return {int(horizon): float(multiplier) for horizon, multiplier in multipliers.items()}


def _mirror_summary(mirror: Mirror, now: float) -> str:
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


HORIZON = 1.0
GAMMA = 1

Batch: TypeAlias = tuple[RobotID, ...]
SearchNode: namedtuple = namedtuple(
    "SearchNode", ["schedule", "batch_to_queue", "next_time_server_available", "node"]
)


class IncrementalSearch:
    """Frontier-based BFS over batches in ``[start_time, end_time]``.

    Each frontier entry carries a Mirror so we can resume from any node. ``step``
    pops the front node, restores its node, expands every candidate child, evaluates
    each, and pushes the child nodes onto the back of the frontier. BFS gives anytime
    "best at fully-explored depth" — the caller can stop whenever slack runs out.
    ``best`` is safe to read at any point.
    """

    def __init__(
        self,
        mirror: Mirror,
        latency_tracker: LatencyTracker,
        max_depth: int = 5,
        action_horizon_multipliers: Mapping[int | str, float] | None = None,
        max_batch_size: int = 1,
        idle_durations: tuple[float, ...] = DEFAULT_IDLE_DURATIONS,
    ) -> None:
        self.latency_tracker = latency_tracker
        self.start_time = mirror.next_time_server_available()
        self.max_depth = max_depth
        self.action_horizon_multipliers = _coerce_horizon_multipliers(action_horizon_multipliers)
        self.max_batch_size = max_batch_size
        self.idle_durations = tuple(idle_durations)
        logger.debug(
            "incremental search, action_horizon_multipliers=%s", self.action_horizon_multipliers
        )

        self.root_node = mirror.get_twin()
        self.root_node.chunk_id_counter = itertools.count(1)
        self.root_node.fast_forward(self.start_time)
        self.root_node.reset_scores(GAMMA)
        self.initial_action_times = _action_times(self.root_node)
        self.initial_starvation_times = _starvation_times(self.root_node)

        eval_node = self.root_node.get_twin()
        eval_node.fast_forward(self.start_time + HORIZON)
        self.initial_execution_times = _execution_times(eval_node)
        self._search_batch_id = itertools.count(1)

        self.best_objective: float = -float("inf")
        self.best_schedule: list[tuple[RobotID, ...]] = []
        self.nodes_visited = 0
        self.all_evaluated: list[dict] = []
        # Per-op cumulative timings inside _expand (seconds): twin clone,
        # queue_batch, fast_forward, evaluate. Read by caller for diagnostics.
        self.op_time: dict[str, float] = {
            "twin": 0.0,
            "queue": 0.0,
            "fastforward": 0.0,
            "evaluate": 0.0,
        }
        self.max_node_time: float = 0.0

        self.search_level = 0
        self.frontier: deque[tuple[tuple[Batch, ...], Batch, float, Mirror]] = deque()

        for batch in self._candidate_actions(self.root_node, allow_idle=True):
            self.frontier.append(SearchNode((), batch, self.start_time, self.root_node))

    def is_done(self) -> bool:
        return not self.frontier

    def step(self, budget_nodes: int = 32) -> None:
        """Searches until it finishes budget nodes or it finishes one depth, whichever comes first."""
        for _ in range(budget_nodes):
            if self.is_done():
                return
            self._expand(*self.frontier.popleft())
            if not self.is_done() and len(self.frontier[0].schedule) > self.search_level:
                assert len(self.frontier[0].schedule) == self.search_level + 1
                self.search_level += 1
                return

    def best(self) -> list[Batch]:
        return list(self.best_schedule)

    def _candidate_batches(self, mirror: Mirror) -> tuple[tuple[RobotID, ...], ...]:
        schedulable_robot_ids = mirror.schedulable_robot_ids(fast_forward=False)
        # NOTE: commenting out for now to see why [0, 0, 0] is not available
        # schedulable_robot_ids = mirror.robots.keys()

        deadlines = mirror.deadlines()
        sorted_robot_ids = sorted(schedulable_robot_ids, key=lambda rid: deadlines[rid])

        if not sorted_robot_ids:
            return ()

        # Group into tiers by priority, preserving EDF order within each tier
        tiers: dict[float, list[RobotID]] = {}
        for rid in sorted_robot_ids:  # already EDF-sorted
            p = self.action_horizon_multipliers[mirror.robots[rid].max_execution_horizon]
            tiers.setdefault(p, []).append(rid)
        tier_list = [tiers[p] for p in sorted(tiers.keys(), reverse=True)]

        pool_size = len(sorted_robot_ids)
        tier_sizes = [len(t) for t in tier_list]
        n_tiers = len(tier_list)

        def compositions(n: int, k: int, caps: list[int]):
            """Yield all ways to write n as ordered sum of k non-negative ints, each <= caps[i]."""
            if k == 1:
                if n <= caps[0]:
                    yield (n,)
                return
            for i in range(min(n, caps[0]) + 1):
                for rest in compositions(n - i, k - 1, caps[1:]):
                    yield (i,) + rest

        seen = set()
        batches = []
        for size in range(1, min(pool_size, self.max_batch_size) + 1):
            for counts in compositions(size, n_tiers, tier_sizes):
                batch = tuple(rid for tier, count in zip(tier_list, counts) for rid in tier[:count])
                if batch not in seen:
                    seen.add(batch)
                    batches.append(batch)

        return tuple(batches)

    def _candidate_actions(self, mirror: Mirror, *, allow_idle: bool) -> list["Batch | Idle"]:
        """Real dispatch batches plus, when ``allow_idle``, one synthetic idle
        per configured duration. ``allow_idle`` is False right after an idle so
        we never chain idles back-to-back (idle 10ms + idle 10ms is already
        covered by the single idle 20ms candidate)."""
        actions: list[Batch | Idle] = list(self._candidate_batches(mirror))
        if allow_idle:
            actions.extend(Idle(d) for d in self.idle_durations)
        return actions

    def _expand(
        self,
        parent_schedule: tuple[Batch, ...],
        queued_batch: Batch,
        gpu_end_time: float,
        parent_node: Mirror,
    ):
        node = parent_node.get_twin()
        if isinstance(queued_batch, Idle):
            next_time = gpu_end_time + queued_batch.duration
            node.queue_idle(
                queued_batch.duration, next(self._search_batch_id), dispatch_time=gpu_end_time
            )
        else:
            next_time = gpu_end_time + self.latency_tracker.infer_latency(len(queued_batch))
            # don't need to fast forward since we've already fast_forwarded to time before batch.
            # Pass dispatch_time explicitly: the node was fast_forwarded to exactly gpu_end_time, so
            # reusing it keeps queue_batch from re-reading the wall clock (next_time_server_available
            # falls through to time.time() when the GPU is idle) and drifting past the simulated steps.
            node.queue_batch(
                list(queued_batch),
                next(self._search_batch_id),
                origin="searched",
                fast_forward=False,
                dispatch_time=gpu_end_time,
            )
        schedule = parent_schedule + (queued_batch,)
        node.fast_forward(next_time)

        self._evaluate(schedule, next_time, node)

        self.nodes_visited += 1
        if len(schedule) == self.max_depth:
            return

        for batch in self._candidate_actions(node, allow_idle=not isinstance(queued_batch, Idle)):
            self.frontier.append(SearchNode(schedule, batch, next_time, node))

    def _evaluate(self, schedule: tuple[Batch, ...], gpu_end_time: float, node: Mirror) -> None:
        gpu_time = gpu_end_time - self.start_time
        if gpu_time <= 0:
            return

        end_time = self.start_time + HORIZON
        eval_node = node.get_twin()
        eval_node.fast_forward(end_time)

        scores = {
            rid: eval_node.robots[rid].score / eval_node.robots[rid].control_hz
            for rid in eval_node.robots
        }
        weighted_scores = {
            rid: scores[rid]
            * self.action_horizon_multipliers[eval_node.robots[rid].max_execution_horizon]
            for rid in scores
        }
        score_sum = sum(weighted_scores.values())

        objective = score_sum / gpu_time

        if objective > self.best_objective:
            self.best_objective = objective
            self.best_schedule = list(schedule)


class LookaheadActionsScheduler(RequestScheduler):
    """Plan one batch per tick, using GPU slack to search.

    Flow per call:
    - If slack before next dispatch slot is tiny, dispatch greedily by EDF.
    - Otherwise, run a fresh search until either it exhausts or slack runs
      out. Return the first batch of the best schedule.
    """

    def __init__(
        self,
        config: SchedulerConfig,
        batch_queue: mp.Queue,
        max_batch_size: int = 1,
        *,
        max_depth: int = 1,
        max_in_flight: int = 1,
        step_budget_nodes: int = 8,
        scheduling_buffer: float = 0.05,
        idle_durations: tuple[float, ...] = DEFAULT_IDLE_DURATIONS,
    ) -> None:
        super().__init__(config, batch_queue, max_batch_size)
        self.max_depth = max_depth
        self.max_in_flight = max_in_flight
        self.step_budget_nodes = step_budget_nodes
        self.scheduling_buffer = scheduling_buffer
        self.action_horizon_multipliers = _coerce_horizon_multipliers(
            self._config.action_horizon_multipliers
        )
        self.idle_durations = tuple(idle_durations)
        logger.debug(
            "lookahead actions scheduler, action_horizon_multipliers=%s",
            self.action_horizon_multipliers,
        )

    def get_next_batches(
        self, candidates: list[SlotRequest]
    ) -> tuple[list[list[SlotRequest]], dict[str, Any]]:
        # We plan up to ``max_depth`` batches ahead but only commit the first
        # ``max_in_flight - in_flight`` of them — the rest are model-predictive
        # context that informs the choice of the immediate dispatches and will
        # be re-planned on the next tick. Robot IDs in the plan map back through
        # the most-recent SlotRequest the scheduler has on file.
        entry_time = time.time()
        self._last_entry_time = entry_time

        # Phase records — absolute timestamps, used by the gantt plot.
        phases: list[dict[str, Any]] = []

        def _phase(name: str, start: float, end: float) -> None:
            phases.append({"name": name, "start": start, "end": end})

        if not self._latest_requests:
            # logger.debug("lookahead stage=exit reason=no_requests")
            return [], {"reason": "no_requests", "phases": phases}

        next_avail = self.mirror.next_time_server_available()
        slack = next_avail - time.time()
        in_flight = self.mirror.in_flight_batches_count
        dispatch_budget = max(0, self.max_in_flight - in_flight)

        logger.debug(
            "search start: robots=%d slack=%+.3fs in_flight=%d budget=%d | %s",
            len(self.mirror.robots),
            slack,
            in_flight,
            dispatch_budget,
            _mirror_summary(self.mirror, next_avail),
        )

        notes: dict[str, Any] = {
            "rule": "lookahead_actions",
            "max_depth": self.max_depth,
            "max_in_flight": self.max_in_flight,
            "step_budget_nodes": self.step_budget_nodes,
            "scheduling_buffer": self.scheduling_buffer,
            "action_horizon_multipliers": dict(self.action_horizon_multipliers),
            "slack_s": slack,
            "next_server_available": next_avail,
            "in_flight": in_flight,
            "dispatch_budget": dispatch_budget,
            "mirror_state": self.mirror.to_dict(),
            "phases": phases,
        }

        logger.debug(
            "lookahead stage=search_init max_batch_size=%d max_depth=%d",
            self._max_batch_size,
            self.max_depth,
        )

        search_init_start = time.time()
        search = IncrementalSearch(
            self.mirror,
            self.latency_tracker,
            self.max_depth,
            self.action_horizon_multipliers,
            self._max_batch_size,
            self.idle_durations,
        )
        search_started_at = time.time()
        _phase("search_init", search_init_start, search_started_at)
        search_iters = 0
        step_durations: list[float] = []
        step_end = search_started_at
        while not search.is_done() and (
            search_iters < 1
            or (self.mirror.next_time_server_available() - time.time()) > self.scheduling_buffer
        ):
            if self._drain_fn is not None:
                self._drain_fn()
            step_start = step_end
            search.step(self.step_budget_nodes)
            step_end = time.time()
            _phase("search_step", step_start, step_end)
            step_durations.append(step_end - step_start)
            search_iters += 1
        search_end = step_end
        search_duration = search_end - search_started_at

        notes.update(
            {
                "search_duration_s": search_duration,
                "search_done": search.is_done(),
                "search_nodes_visited": search.nodes_visited,
                "best_objective": (
                    None if search.best_objective == -float("inf") else search.best_objective
                ),
                "best_schedule_depth": len(search.best_schedule),
                "best_schedule": [_serialize_action(b) for b in search.best_schedule],
                "all_evaluated": search.all_evaluated,
            }
        )

        best = search.best()

        notes["mode"] = "search"
        # Commit the plan prefix: real batches become SlotRequest lists; idle
        # actions pass through as-is for base.schedule to dispatch as GPU sleeps.
        batches: list[list[SlotRequest] | Idle] = [
            action if isinstance(action, Idle) else [self._latest_requests[rid] for rid in action]
            for action in best[:dispatch_budget]
        ]
        _phase("postprocess", search_end, time.time())
        logger.debug(
            "lookahead stage=return mode=search batches=%d plan_depth=%d objective=%.4f",
            len(batches),
            len(best),
            search.best_objective,
        )
        return batches, notes
