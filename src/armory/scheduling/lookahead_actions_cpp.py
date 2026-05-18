"""Lookahead scheduler that delegates BFS search to a C++ extension.

Parallel to ``lookahead_actions.py``; same dispatch decisions but the inner
loop runs in C++ (no GC, faster Mirror clone). For benchmarking.
"""

from __future__ import annotations

import itertools
import logging
import multiprocessing as mp
import time
from collections.abc import Mapping
from typing import Any

try:
    from armory_lookahead_cpp import InFlight, RobotState, find_best_schedule

    _HAS_CPP = True
except ImportError:  # pragma: no cover
    _HAS_CPP = False

from armory.scheduling.base import RequestScheduler
from armory.serving.schemas import SlotRequest

logger = logging.getLogger(__name__)


def _coerce_horizon_multipliers(
    multipliers: Mapping[int | str, float] | None,
) -> dict[int, float]:
    if multipliers is None:
        return {}
    return {int(horizon): float(multiplier) for horizon, multiplier in multipliers.items()}


class LookaheadActionsCppScheduler(RequestScheduler):
    """Same scheduling shape as ``LookaheadActionsScheduler`` but with C++ search."""

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
        if not _HAS_CPP:
            raise RuntimeError(
                "armory_lookahead_cpp not installed. Build the package first: "
                "`uv pip install -e packages/armory-lookahead-cpp`."
            )
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
        if not self._latest_requests:
            return [], {"reason": "no_requests"}
        if not candidates:
            return [], {"reason": "no_candidates"}

        next_avail = self.mirror.next_time_server_available()
        slack = next_avail - time.time()
        in_flight = self.mirror.in_flight_batches_count
        dispatch_budget = max(0, self.max_in_flight - in_flight)

        action_multipliers = {
            r.robot_id: self.action_horizon_multipliers.get(r.max_execution_horizon, 1.0)
            for r in candidates
        }

        notes: dict[str, Any] = {
            "rule": "lookahead_actions_cpp",
            "horizon": self.horizon,
            "max_depth": self.max_depth,
            "max_in_flight": self.max_in_flight,
            "step_budget_nodes": self.step_budget_nodes,
            "scheduling_buffer": self.scheduling_buffer,
            "slack_s": slack,
            "next_server_available": next_avail,
            "in_flight": in_flight,
            "dispatch_budget": dispatch_budget,
        }

        if dispatch_budget == 0:
            notes["mode"] = "at_in_flight_cap"
            return [], notes

        if slack < self.scheduling_buffer:
            notes["mode"] = "greedy_no_slack"
            return [self._greedy(candidates)], notes

        candidate_ids = sorted({r.robot_id for r in candidates if r.robot_id in self.mirror.robots})
        if not candidate_ids:
            notes["mode"] = "greedy_no_slack"
            return [self._greedy(candidates)], notes

        # Map robot_id -> index for the C++ side.
        robots = []
        for rid in candidate_ids:
            r = self._latest_requests[rid]
            rs = RobotState()
            rs.control_hz = float(r.control_hz)
            rs.max_execution_horizon = int(r.max_execution_horizon)
            rs.action_multiplier = float(action_multipliers.get(rid, 1.0))
            robots.append(rs)

        max_batch_size_known = max(self.latency_tracker._infer_latency.keys())
        max_size = min(max_batch_size_known, len(candidate_ids))
        candidate_batches: list[list[int]] = []
        ids_indices = list(range(len(candidate_ids)))
        for size in range(max_size, 0, -1):
            for combo in itertools.combinations(ids_indices, size):
                candidate_batches.append(list(combo))

        # infer_latency table: index 0 = 0.0, index k = latency for batch size k.
        infer_latency = [0.0] * (max_size + 1)
        for k in range(1, max_size + 1):
            infer_latency[k] = float(self.latency_tracker.infer_latency(k))

        # Seed C++ with current in-flight batch completion times (sorted).
        in_flight_records = []
        for b in sorted(self.mirror.in_flight_batches, key=lambda x: x.completion_time):
            inf = InFlight()
            inf.batch_id = int(b.batch_id)
            inf.completion_time = float(b.completion_time)
            in_flight_records.append(inf)

        wall_deadline = next_avail - self.scheduling_buffer
        t0 = time.time()
        result = find_best_schedule(
            robots=robots,
            candidate_batches=candidate_batches,
            infer_latency=infer_latency,
            in_flight=in_flight_records,
            start_time=next_avail,
            horizon=self.horizon,
            max_depth=self.max_depth,
            step_budget_nodes=self.step_budget_nodes,
            wall_deadline=wall_deadline,
        )
        search_duration = time.time() - t0

        best_indices: list[list[int]] = result["best_schedule"]
        notes.update(
            {
                "search_duration_s": search_duration,
                "search_done": result["done"],
                "search_nodes_visited": result["nodes_visited"],
                "best_objective": (
                    None if result["best_objective"] == -float("inf") else result["best_objective"]
                ),
                "best_schedule_depth": len(best_indices),
                "best_schedule": [[candidate_ids[i] for i in batch] for batch in best_indices],
            }
        )

        if not best_indices:
            notes["mode"] = "greedy_search_empty"
            return [self._greedy(candidates)], notes

        notes["mode"] = "search"
        batches: list[list[SlotRequest]] = []
        for indices in best_indices[:dispatch_budget]:
            batch = [
                self._latest_requests[candidate_ids[i]]
                for i in indices
                if candidate_ids[i] in self._latest_requests
            ]
            if batch:
                batches.append(batch)
        return batches, notes

    def _greedy(self, candidates: list[SlotRequest]) -> list[SlotRequest]:
        deadlines = self.mirror.deadlines()
        return sorted(candidates, key=lambda r: deadlines.get(r.robot_id, r.deadline))[
            : self._max_batch_size
        ]
