"""Lookahead scheduler that searches batches during GPU slack time."""

import copy
import itertools
import logging
import multiprocessing as mp
import time
from collections import deque
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

from armory.scheduling.base import RequestScheduler
from armory.scheduling.latency import LatencyTracker
from armory.scheduling.mirror import ActionChunk, Checkpoint, Mirror, Robot
from armory.serving.schemas import RobotID, SlotRequest

logger = logging.getLogger(__name__)


def _action_time(robot: Robot) -> float:
    """Wall-clock duration of all actions ever queued for a robot."""
    if not robot.chunks:
        return 0.0
    return (robot.max_overall_action_step + 1) / robot.control_hz


def _action_times(mirror: Mirror) -> dict[RobotID, float]:
    return {rid: _action_time(robot) for rid, robot in mirror.robots.items()}


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

    Each frontier entry carries a Mirror ``Checkpoint`` so we can resume from
    any node. ``step`` pops the front node, restores its snapshot, expands
    every candidate child, evaluates each, and pushes the child checkpoints
    onto the back of the frontier. BFS gives anytime "best at fully-explored
    depth" — the caller can stop whenever slack runs out. ``best`` is safe
    to read at any point.
    """

    def __init__(
        self,
        mirror: Mirror,
        latency_tracker: LatencyTracker,
        start_time: float,
        horizon: float,
        max_depth: int = 5,
    ) -> None:
        self.latency_tracker = latency_tracker
        self.start_time = start_time
        self.end_time = start_time + horizon
        self.max_depth = max_depth
        # FIXME: don't access private
        self.max_batch_size = max(latency_tracker._infer_latency.keys())

        self.snapshot = copy.deepcopy(mirror)
        self.snapshot.fast_forward(start_time)
        self.initial_action_times = _action_times(self.snapshot)
        self._search_batch_id = itertools.count(1)

        self.best_objective = -float("inf")
        self.best_schedule: list[ScheduledBatch] = []
        self.nodes_visited = 0

        root_ckpt = self.snapshot.checkpoint()
        self.frontier: deque[tuple[tuple[ScheduledBatch, ...], float, Checkpoint]] = deque(
            [((), start_time, root_ckpt)]
        )

    def is_done(self) -> bool:
        return not self.frontier

    def step(self, budget_nodes: int = 32) -> None:
        visited = 0
        while visited < budget_nodes and self.frontier:
            schedule, end_time, ckpt = self.frontier.popleft()
            visited += self._expand(schedule, end_time, ckpt)

    def best(self) -> list[ScheduledBatch]:
        return list(self.best_schedule)

    def _candidates(self) -> Iterator[tuple[RobotID, ...]]:
        # FIXME: search space is just prefix-of-EDF batches
        deadlines = self.snapshot.deadlines()
        sorted_ids = sorted(self.snapshot.robots.keys(), key=lambda rid: deadlines[rid])
        return iter(
            tuple(sorted_ids[:i]) for i in range(1, min(self.max_batch_size, len(sorted_ids)) + 1)
        )

    def _expand(
        self,
        schedule: tuple[ScheduledBatch, ...],
        end_time: float,
        parent_ckpt: Checkpoint,
    ) -> int:
        self.snapshot.restore(parent_ckpt)
        count = 0
        for batch in self._candidates():
            next_time = end_time + self.latency_tracker.infer_latency(len(batch))
            if next_time > self.end_time:
                continue

            chunks = tuple(
                self.snapshot.queue_batch(
                    list(batch), next(self._search_batch_id), origin="searched"
                )
            )
            self.snapshot.fast_forward(next_time)

            new_schedule = schedule + (ScheduledBatch(batch, chunks),)
            self.nodes_visited += 1
            count += 1
            self._evaluate(new_schedule, next_time)

            if len(new_schedule) < self.max_depth:
                self.frontier.append((new_schedule, next_time, self.snapshot.checkpoint()))

            self.snapshot.restore(parent_ckpt)
        return count

    def _evaluate(self, schedule: tuple[ScheduledBatch, ...], now: float) -> None:
        gpu_time = now - self.start_time
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
        min_execution_horizon: int = 0,
        *,
        horizon: float = 1.0,
        max_depth: int = 5,
        max_in_flight: int = 5,
        step_budget_nodes: int = 32,
        scheduling_buffer: float = 0.01,
    ) -> None:
        super().__init__(batch_queue, max_batch_size, min_execution_horizon=min_execution_horizon)
        self.horizon = horizon
        self.max_depth = max_depth
        self.max_in_flight = max_in_flight
        self.step_budget_nodes = step_budget_nodes
        self.scheduling_buffer = scheduling_buffer

    def get_next_batches(
        self, candidates: list[SlotRequest]
    ) -> tuple[list[list[SlotRequest]], dict[str, Any]]:
        # We plan up to ``max_depth`` batches ahead but only commit the first
        # ``max_in_flight - in_flight`` of them — the rest are model-predictive
        # context that informs the choice of the immediate dispatches and will
        # be re-planned on the next tick. Robot IDs in the plan map back through
        # the most-recent SlotRequest the scheduler has on file.
        if not self._latest_requests:
            logger.debug("lookahead stage=exit reason=no_requests")
            return [], {"reason": "no_requests"}

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
            "horizon": self.horizon,
            "max_depth": self.max_depth,
            "max_in_flight": self.max_in_flight,
            "step_budget_nodes": self.step_budget_nodes,
            "scheduling_buffer": self.scheduling_buffer,
            "slack_s": slack,
            "next_server_available": next_avail,
            "in_flight": in_flight,
            "dispatch_budget": dispatch_budget,
            "mirror_state": self.mirror.to_dict(),
        }

        if dispatch_budget == 0:
            logger.debug(
                "lookahead stage=exit reason=at_in_flight_cap in_flight=%d max=%d",
                in_flight,
                self.max_in_flight,
            )
            notes["mode"] = "at_in_flight_cap"
            return [], notes

        if slack < self.scheduling_buffer:
            logger.debug(
                "lookahead stage=exit reason=greedy_no_slack slack=%+.3fs buffer=%.3fs",
                slack,
                self.scheduling_buffer,
            )
            notes["mode"] = "greedy_no_slack"
            return [self._greedy()], notes

        logger.debug(
            "lookahead stage=search_init horizon=%.3fs max_depth=%d", self.horizon, self.max_depth
        )
        search = IncrementalSearch(
            self.mirror,
            self.latency_tracker,
            next_avail,
            self.horizon,
            self.max_depth,
        )
        search_started_at = time.time()
        search_iters = 0
        while not search.is_done() and (next_avail - time.time()) > self.scheduling_buffer:
            search.step(self.step_budget_nodes)
            search_iters += 1
        search_duration = time.time() - search_started_at
        logger.debug(
            "lookahead stage=search_done iters=%d nodes=%d duration=%.3fs done=%s remaining_slack=%+.3fs",
            search_iters,
            search.nodes_visited,
            search_duration,
            search.is_done(),
            next_avail - time.time(),
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
            notes["mode"] = "greedy_search_empty"
            return [self._greedy()], notes

        notes["mode"] = "search"
        batches: list[list[SlotRequest]] = []
        for sb in best[:dispatch_budget]:
            batch = [
                self._latest_requests[rid] for rid in sb.robot_ids if rid in self._latest_requests
            ]
            if batch:
                batches.append(batch)
        logger.debug(
            "lookahead stage=return mode=search batches=%d plan_depth=%d objective=%.4f",
            len(batches),
            len(best),
            search.best_objective,
        )
        return batches, notes

    def _greedy(self) -> list[SlotRequest]:
        deadlines = self.mirror.deadlines()
        requests = list(self._latest_requests.values())
        return sorted(requests, key=lambda r: deadlines.get(r.robot_id, r.deadline))[
            : self._max_batch_size
        ]
