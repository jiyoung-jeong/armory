"""Lookahead scheduler that searches batches during GPU slack time."""

import gc
import itertools
import logging
import multiprocessing as mp
import time
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from armory.scheduling.base import RequestScheduler
from armory.scheduling.latency import LatencyTracker
from armory.scheduling.mirror import ActionChunk, Mirror, Robot
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


@dataclass(frozen=True)
class ScheduledBatch:
    robot_ids: tuple[RobotID, ...]
    chunks: tuple[ActionChunk, ...]


class IncrementalSearch:
    """Frontier-based BFS over batches in ``[start_time, start_time + horizon]``.

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
        start_time: float,
        horizon: float,
        candidate_robot_ids: tuple[RobotID, ...],
        action_multipliers: Mapping[RobotID, float] | None = None,
        max_depth: int = 5,
    ) -> None:
        self.latency_tracker = latency_tracker
        self.start_time = start_time
        self.end_time = start_time + horizon
        self.candidate_robot_ids = tuple(rid for rid in candidate_robot_ids if rid in mirror.robots)
        self.action_multipliers = dict(action_multipliers or {})
        self.max_depth = max_depth
        # FIXME: don't access private
        self.max_batch_size = max(latency_tracker._infer_latency.keys())

        self.snapshot = mirror.get_twin()
        self.snapshot.chunk_id_counter = itertools.count(1)
        self.snapshot.fast_forward(start_time)
        self.initial_action_times = _action_times(self.snapshot)
        self._search_batch_id = itertools.count(1)
        self.candidate_batches = self._candidate_batches()

        self.best_objective = -float("inf")
        self.best_schedule: list[ScheduledBatch] = []
        self.nodes_visited = 0
        # Per-op cumulative timings inside _expand (seconds): twin clone,
        # queue_batch, fast_forward, evaluate. Read by caller for diagnostics.
        self.op_time: dict[str, float] = {
            "twin": 0.0,
            "queue": 0.0,
            "fastforward": 0.0,
            "evaluate": 0.0,
        }
        self.max_node_time: float = 0.0

        root_node = self.snapshot.get_twin()
        self.frontier: deque[tuple[tuple[ScheduledBatch, ...], float, Mirror, int]] = deque(
            [((), start_time, root_node, 0)]
        )

    def is_done(self) -> bool:
        return not self.frontier

    def step(self, budget_nodes: int = 32) -> None:
        remaining = max(1, budget_nodes)
        while remaining > 0 and self.frontier:
            schedule, end_time, node, next_candidate = self.frontier.popleft()
            remaining -= self._expand(schedule, end_time, node, next_candidate, remaining)

    def best(self) -> list[ScheduledBatch]:
        return list(self.best_schedule)

    def _candidate_batches(self) -> tuple[tuple[RobotID, ...], ...]:
        sorted_ids = sorted(self.candidate_robot_ids)
        max_size = min(self.max_batch_size, len(sorted_ids))
        return tuple(
            itertools.chain.from_iterable(
                itertools.combinations(sorted_ids, size) for size in range(max_size, 0, -1)
            )
        )

    def _expand(
        self,
        schedule: tuple[ScheduledBatch, ...],
        end_time: float,
        parent_node: Mirror,
        start_candidate: int,
        budget_nodes: int,
    ) -> int:
        count = 0
        candidate_index = start_candidate
        while candidate_index < len(self.candidate_batches) and count < budget_nodes:
            batch = self.candidate_batches[candidate_index]
            candidate_index += 1
            next_time = end_time + self.latency_tracker.infer_latency(len(batch))
            if next_time > self.end_time:
                continue

            t0 = time.perf_counter()
            node = parent_node.get_twin()
            t1 = time.perf_counter()
            chunks = tuple(
                node.queue_batch(list(batch), next(self._search_batch_id), origin="searched")
            )
            t2 = time.perf_counter()
            node.fast_forward(next_time)
            t3 = time.perf_counter()

            new_schedule = schedule + (ScheduledBatch(batch, chunks),)
            self.nodes_visited += 1
            count += 1
            self._evaluate(new_schedule, next_time, node)
            t4 = time.perf_counter()

            self.op_time["twin"] += t1 - t0
            self.op_time["queue"] += t2 - t1
            self.op_time["fastforward"] += t3 - t2
            self.op_time["evaluate"] += t4 - t3
            node_total = t4 - t0
            if node_total > self.max_node_time:
                self.max_node_time = node_total

            if len(new_schedule) < self.max_depth:
                self.frontier.append((new_schedule, next_time, node, 0))

        if candidate_index < len(self.candidate_batches):
            self.frontier.appendleft((schedule, end_time, parent_node, candidate_index))

        return count

    def _evaluate(self, schedule: tuple[ScheduledBatch, ...], now: float, node: Mirror) -> None:
        gpu_time = now - self.start_time
        if gpu_time <= 0:
            return
        new_times = _action_times(node)
        gained = sum(
            self.action_multipliers.get(rid, 1.0)
            * (new_times[rid] - self.initial_action_times.get(rid, 0.0))
            for rid in new_times
        )
        objective = gained / gpu_time
        if objective > self.best_objective:
            self.best_objective = objective
            self.best_schedule = list(schedule)
            logger.debug(
                "new best: depth=%d objective=%.4f schedule=%s",
                len(schedule),
                objective,
                [b.robot_ids for b in schedule],
            )


class LookaheadActionsScheduler(RequestScheduler):
    """Plan one batch per tick, using GPU slack to search.

    Flow per call:
    - If slack before next dispatch slot is tiny, dispatch greedily by EDF.
    - Otherwise, run a fresh search until either it exhausts or slack runs
      out. Return the first batch of the best schedule.
    """

    def __init__(
        self,
        batch_queue: mp.Queue,
        max_batch_size: int = 1,
        *,
        horizon: float = 1.0,
        max_depth: int = 6,
        max_in_flight: int = 5,
        step_budget_nodes: int = 8,
        scheduling_buffer: float = 0.05,
        action_horizon_multipliers: Mapping[int | str, float] | None = None,
    ) -> None:
        super().__init__(batch_queue, max_batch_size)
        self.horizon = horizon
        self.max_depth = max_depth
        self.max_in_flight = max_in_flight
        self.step_budget_nodes = step_budget_nodes
        self.scheduling_buffer = scheduling_buffer
        self.action_horizon_multipliers = _coerce_horizon_multipliers(action_horizon_multipliers)

    def get_next_batches(
        self, candidates: list[SlotRequest]
    ) -> tuple[list[list[SlotRequest]], dict[str, Any]]:
        # We plan up to ``max_depth`` batches ahead but only commit the first
        # ``max_in_flight - in_flight`` of them — the rest are model-predictive
        # context that informs the choice of the immediate dispatches and will
        # be re-planned on the next tick. Robot IDs in the plan map back through
        # the most-recent SlotRequest the scheduler has on file.
        entry_time = time.time()
        last_entry = getattr(self, "_last_entry_time", entry_time)
        inter_call_gap = entry_time - last_entry
        self._last_entry_time = entry_time

        # Phase records — absolute timestamps, used by the gantt plot.
        phases: list[dict[str, Any]] = []

        def _phase(name: str, start: float, end: float) -> None:
            phases.append({"name": name, "start": start, "end": end})

        if not self._latest_requests:
            logger.debug("lookahead stage=exit reason=no_requests")
            return [], {"reason": "no_requests", "phases": phases}
        if not candidates:
            logger.debug("lookahead stage=exit reason=no_candidates")
            return [], {"reason": "no_candidates", "phases": phases}

        next_avail = self.mirror.next_time_server_available()
        slack = next_avail - time.time()
        in_flight = self.mirror.in_flight_batches_count
        dispatch_budget = max(0, self.max_in_flight - in_flight)
        candidate_robot_ids = tuple(sorted(r.robot_id for r in candidates))
        action_multipliers = {
            r.robot_id: self.action_horizon_multipliers.get(r.max_execution_horizon, 1.0)
            for r in candidates
        }

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
            "horizon": self.horizon,
            "max_depth": self.max_depth,
            "max_in_flight": self.max_in_flight,
            "step_budget_nodes": self.step_budget_nodes,
            "scheduling_buffer": self.scheduling_buffer,
            "action_horizon_multipliers": dict(self.action_horizon_multipliers),
            "action_multipliers": action_multipliers,
            "slack_s": slack,
            "next_server_available": next_avail,
            "in_flight": in_flight,
            "dispatch_budget": dispatch_budget,
            "mirror_state": self.mirror.to_dict(),
            "phases": phases,
        }

        if dispatch_budget == 0:
            logger.debug(
                "lookahead exit=at_in_flight_cap inter_call=%+.3fs slack=%+.3fs in_flight=%d max=%d",
                inter_call_gap,
                slack,
                in_flight,
                self.max_in_flight,
            )
            _phase("setup", entry_time, time.time())
            notes["mode"] = "at_in_flight_cap"
            return [], notes

        if slack < self.scheduling_buffer:
            logger.debug(
                "lookahead exit=greedy_no_slack inter_call=%+.3fs slack=%+.3fs buffer=%.3fs in_flight=%d candidates=%d",
                inter_call_gap,
                slack,
                self.scheduling_buffer,
                in_flight,
                len(candidates),
            )
            setup_end = time.time()
            _phase("setup", entry_time, setup_end)
            greedy_batch = self._greedy(candidates)
            _phase("greedy", setup_end, time.time())
            notes["mode"] = "greedy_no_slack"
            return [greedy_batch], notes

        logger.debug(
            "lookahead stage=search_init horizon=%.3fs max_depth=%d", self.horizon, self.max_depth
        )
        # Run GC now, before the latency-critical search. GC stays disabled
        # afterward (across postprocess and the return path) so an automatic
        # collection cannot fire in the post-search dispatch window. The next
        # scheduler call will collect again at its head — where we have slack
        # to absorb the cost.
        gc_start = time.time()
        _phase("setup", entry_time, gc_start)
        gc_counts_before = gc.get_count()
        gc.collect()
        if gc.isenabled():
            gc.disable()
        gc_end = time.time()
        gc_counts_after = gc.get_count()
        _phase("gc", gc_start, gc_end)

        search = IncrementalSearch(
            self.mirror,
            self.latency_tracker,
            next_avail,
            self.horizon,
            candidate_robot_ids,
            action_multipliers,
            self.max_depth,
        )
        search_started_at = time.time()
        _phase("search_init", gc_end, search_started_at)
        search_iters = 0
        step_durations: list[float] = []
        step_end = search_started_at
        while not search.is_done() and (next_avail - time.time()) > self.scheduling_buffer:
            step_start = step_end
            search.step(self.step_budget_nodes)
            step_end = time.time()
            _phase("search_step", step_start, step_end)
            step_durations.append(step_end - step_start)
            search_iters += 1
        search_end = step_end
        search_duration = search_end - search_started_at
        remaining_slack = next_avail - search_end
        max_step = max(step_durations, default=0.0)
        avg_step = (sum(step_durations) / len(step_durations)) if step_durations else 0.0
        nodes = search.nodes_visited
        per_node = (search_duration / nodes) if nodes else 0.0
        logger.debug(
            "lookahead search inter_call=%+.3fs slack_in=%+.3fs slack_out=%+.3fs buffer=%.3fs "
            "iters=%d nodes=%d budget=%d total=%.4fs max_step=%.4fs avg_step=%.4fs "
            "per_node=%.4fs max_node=%.4fs ops twin=%.4f queue=%.4f ff=%.4f eval=%.4f "
            "candidate_batches=%d in_flight=%d candidates=%d done=%s gc_before=%s gc_after=%s",
            inter_call_gap,
            slack,
            remaining_slack,
            self.scheduling_buffer,
            search_iters,
            nodes,
            self.step_budget_nodes,
            search_duration,
            max_step,
            avg_step,
            per_node,
            search.max_node_time,
            search.op_time["twin"],
            search.op_time["queue"],
            search.op_time["fastforward"],
            search.op_time["evaluate"],
            len(search.candidate_batches),
            in_flight,
            len(candidates),
            search.is_done(),
            gc_counts_before,
            gc_counts_after,
        )

        notes.update(
            {
                "search_duration_s": search_duration,
                "search_done": search.is_done(),
                "search_nodes_visited": search.nodes_visited,
                "best_objective": (
                    None if search.best_objective == -float("inf") else search.best_objective
                ),
                "best_schedule_depth": len(search.best_schedule),
                "best_schedule": [list(b.robot_ids) for b in search.best_schedule],
            }
        )

        best = search.best()
        if not best:
            logger.debug("lookahead stage=exit reason=greedy_search_empty")
            greedy_batch = self._greedy(candidates)
            _phase("postprocess", search_end, time.time())
            notes["mode"] = "greedy_search_empty"
            return [greedy_batch], notes

        notes["mode"] = "search"
        batches: list[list[SlotRequest]] = []
        for sb in best[:dispatch_budget]:
            batch = [
                self._latest_requests[rid] for rid in sb.robot_ids if rid in self._latest_requests
            ]
            if batch:
                batches.append(batch)
        _phase("postprocess", search_end, time.time())
        logger.debug(
            "lookahead stage=return mode=search batches=%d plan_depth=%d objective=%.4f",
            len(batches),
            len(best),
            search.best_objective,
        )
        return batches, notes

    def _greedy(self, candidates: list[SlotRequest]) -> list[SlotRequest]:
        deadlines = self.mirror.deadlines()
        return sorted(candidates, key=lambda r: deadlines.get(r.robot_id, r.deadline))[
            : self._max_batch_size
        ]
