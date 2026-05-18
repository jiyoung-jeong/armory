"""Lookahead scheduler that delegates BFS search to a C++ extension.

Parallel to ``lookahead_actions.py``; dispatch decisions are the same and
the inner loop runs in C++ against a faithful port of the Mirror search
path. See ``packages/armory-lookahead-cpp`` for the C++ source.
"""

from __future__ import annotations

import gc
import logging
import multiprocessing as mp
import time
from collections.abc import Mapping
from typing import Any

try:
    import armory_lookahead_cpp as _alc

    _HAS_CPP = True
except ImportError:  # pragma: no cover
    _HAS_CPP = False

from armory.scheduling.base import RequestScheduler
from armory.scheduling.latency import LatencyTracker
from armory.scheduling.mirror import Mirror as PyMirror
from armory.serving.schemas import RobotID, SlotRequest

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def _coerce_horizon_multipliers(
    multipliers: Mapping[int | str, float] | None,
) -> dict[int, float]:
    if multipliers is None:
        return {}
    return {int(horizon): float(multiplier) for horizon, multiplier in multipliers.items()}


def _mirror_summary(mirror: PyMirror, now: float) -> str:
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


def snapshot_mirror_to_cpp(
    mirror: PyMirror,
    latency_tracker: LatencyTracker,
    action_multipliers: Mapping[RobotID, float] | None = None,
) -> tuple[Any, list[RobotID]]:
    """Copy a live Python Mirror into the C++ Mirror representation.

    Returns the C++ Mirror plus the ordered list of robot_ids whose index in
    the list matches the integer ``robot_idx`` used by the C++ side. Order
    matches ``mirror.robots`` insertion order so candidate-batch stability
    matches Python.

    Raises KeyError if any robot in the mirror lacks measured
    observation/action latencies — callers should fall back to the greedy
    path in that case (matches what the Python search would do, where
    ``calculate_chunk_context`` raises on missing latency).
    """
    if not _HAS_CPP:
        raise RuntimeError("armory_lookahead_cpp not installed")

    multipliers = dict(action_multipliers or {})

    cpp_mirror = _alc.Mirror()
    cpp_mirror.last_batch_completed_time = mirror.last_batch_completed_time

    # infer_latency table: index 0 = 0.0, index k = latency for batch size k.
    # Match Python's LatencyTracker._infer_latency dict shape — caller must
    # ensure every batch size in [1, max] has a measurement.
    max_batch = max(latency_tracker._infer_latency.keys())
    table = [0.0] * (max_batch + 1)
    for k in range(1, max_batch + 1):
        table[k] = float(latency_tracker.infer_latency(k))
    cpp_mirror.infer_latency = table

    robot_ids: list[RobotID] = list(mirror.robots.keys())
    rid_to_idx = {rid: i for i, rid in enumerate(robot_ids)}

    cpp_robots = []
    for idx, rid in enumerate(robot_ids):
        py_robot = mirror.robots[rid]
        r = _alc.Robot()
        r.robot_idx = idx
        r.control_hz = float(py_robot.control_hz)
        r.min_execution_horizon = int(py_robot.min_execution_horizon)
        r.max_execution_horizon = int(py_robot.max_execution_horizon)
        r.observation_latency = float(latency_tracker.observation_latency(rid))
        r.action_latency = float(latency_tracker.action_latency(rid))
        r.action_multiplier = float(multipliers.get(rid, 1.0))

        r.has_last_request = py_robot.last_request is not None
        lr = _alc.LastRequest()
        if py_robot.last_request is not None:
            lr.min_execution_horizon = int(py_robot.last_request.min_execution_horizon)
            lr.max_execution_horizon = int(py_robot.last_request.max_execution_horizon)
            lr.action_index_start = int(py_robot.last_request.action_index_start)
        r.last_request = lr

        cpp_steps = []
        for s in py_robot.steps:
            cs = _alc.ControlStep()
            cs.time = float(s.time)
            cs.observation_step = int(s.observation_step)
            cs.action_step = -1 if s.action_step is None else int(s.action_step)
            cs.next_action_step = int(s.next_action_step)
            cpp_steps.append(cs)
        r.steps = cpp_steps

        cpp_chunks = []
        for c in py_robot.chunks:
            cc = _alc.ActionChunk()
            cc.chunk_id = int(c.chunk_id)
            cc.observation_step = int(c.observation_step)
            cc.action_index_start = int(c.action_index_start)
            cc.min_execution_horizon = int(c.min_execution_horizon)
            cc.max_execution_horizon = int(c.max_execution_horizon)
            cc.arrival_time = float(c.arrival_time)
            cc.execution_start_step = int(c.execution_start_step)
            cc.first_executed_index = int(c.first_executed_index)
            cpp_chunks.append(cc)
        r.chunks = cpp_chunks

        cpp_robots.append(r)
    cpp_mirror.robots = cpp_robots

    cpp_in_flight = []
    for b in mirror.in_flight_batches:
        cb = _alc.Batch()
        cb.batch_id = int(b.batch_id)
        cb.robot_indices = [rid_to_idx[rid] for rid in b.robot_ids if rid in rid_to_idx]
        cb.chunk_ids = list(b.chunk_ids)
        cb.completion_time = float(b.completion_time)
        cpp_in_flight.append(cb)
    cpp_mirror.in_flight_batches = cpp_in_flight

    return cpp_mirror, robot_ids


def run_cpp_search(
    mirror: PyMirror,
    latency_tracker: LatencyTracker,
    start_time: float,
    horizon: float,
    action_multipliers: Mapping[RobotID, float],
    max_depth: int,
    max_batch_size: int,
    step_budget_nodes: int,
    next_avail_fn: Any,
    scheduling_buffer: float,
    drain_fn: Any = None,
) -> tuple[Any, dict[str, Any]]:
    """Drive the C++ IncrementalSearch with the same stop-condition that the
    Python scheduler uses. Returns (search, diagnostics).

    ``next_avail_fn`` is a zero-arg callable returning the current
    ``mirror.next_time_server_available()`` value. Caller-supplied so the
    inner loop checks the real (mutating) mirror's slack.
    """
    cpp_mirror, robot_ids = snapshot_mirror_to_cpp(mirror, latency_tracker, action_multipliers)
    multipliers_vec = [float(action_multipliers.get(rid, 1.0)) for rid in robot_ids]
    search = _alc.IncrementalSearch(
        cpp_mirror,
        start_time,
        horizon,
        multipliers_vec,
        max_depth,
        max_batch_size,
    )

    search_iters = 0
    step_durations: list[float] = []
    step_end = time.time()
    while not search.is_done() and (next_avail_fn() - time.time()) > scheduling_buffer:
        if drain_fn is not None:
            drain_fn()
        step_start = step_end
        search.step(step_budget_nodes)
        step_end = time.time()
        step_durations.append(step_end - step_start)
        search_iters += 1

    diagnostics = {
        "robot_ids": robot_ids,
        "search_iters": search_iters,
        "step_durations": step_durations,
        "max_step": max(step_durations, default=0.0),
        "avg_step": (sum(step_durations) / len(step_durations)) if step_durations else 0.0,
    }
    return search, diagnostics


class LookaheadActionsCppScheduler(RequestScheduler):
    """Same scheduling shape as ``LookaheadActionsScheduler`` but with C++ search.

    Decision flow per call mirrors ``lookahead_actions.LookaheadActionsScheduler``:
      - exit early if no requests/candidates/at-cap,
      - greedy if slack < scheduling_buffer,
      - otherwise run the C++ BFS and return ``best[:dispatch_budget]``.
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
        entry_time = time.time()
        last_entry = getattr(self, "_last_entry_time", entry_time)
        inter_call_gap = entry_time - last_entry
        self._last_entry_time = entry_time

        phases: list[dict[str, Any]] = []

        def _phase(name: str, start: float, end: float) -> None:
            phases.append({"name": name, "start": start, "end": end})

        if not self._latest_requests:
            logger.debug("lookahead_cpp stage=exit reason=no_requests")
            return [], {"reason": "no_requests", "phases": phases}
        if not candidates:
            logger.debug("lookahead_cpp stage=exit reason=no_candidates")
            return [], {"reason": "no_candidates", "phases": phases}

        next_avail = self.mirror.next_time_server_available()
        slack = next_avail - time.time()
        in_flight = self.mirror.in_flight_batches_count
        dispatch_budget = max(0, self.max_in_flight - in_flight)
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
            "rule": "lookahead_actions_cpp",
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
                "lookahead_cpp exit=at_in_flight_cap inter_call=%+.3fs slack=%+.3fs in_flight=%d max=%d",
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
                "lookahead_cpp exit=greedy_no_slack inter_call=%+.3fs slack=%+.3fs buffer=%.3fs in_flight=%d candidates=%d",
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

        max_batch_size_known = max(self.latency_tracker._infer_latency.keys())
        if max_batch_size_known < 1:
            # Latency table not populated yet — match Python's behavior by
            # falling back to greedy rather than crashing in C++.
            _phase("setup", entry_time, time.time())
            notes["mode"] = "greedy_no_slack"
            return [self._greedy(candidates)], notes

        logger.debug(
            "lookahead_cpp stage=search_init horizon=%.3fs max_depth=%d",
            self.horizon,
            self.max_depth,
        )
        gc_start = time.time()
        _phase("setup", entry_time, gc_start)
        gc_counts_before = gc.get_count()
        gc.collect()
        if gc.isenabled():
            gc.disable()
        gc_end = time.time()
        gc_counts_after = gc.get_count()
        _phase("gc", gc_start, gc_end)

        # Snapshot + search in C++. Defensive try/except: a missing latency
        # measurement raises KeyError; fall back to greedy (matches Python's
        # behavior, which would crash inside calculate_chunk_context).
        try:
            search, diag = run_cpp_search(
                self.mirror,
                self.latency_tracker,
                next_avail,
                self.horizon,
                action_multipliers,
                self.max_depth,
                max_batch_size_known,
                self.step_budget_nodes,
                next_avail_fn=self.mirror.next_time_server_available,
                scheduling_buffer=self.scheduling_buffer,
                drain_fn=self._drain_fn,
            )
        except KeyError as e:
            logger.debug("lookahead_cpp stage=exit reason=missing_latency key=%s", e)
            _phase("search", gc_end, time.time())
            notes["mode"] = "greedy_no_slack"
            return [self._greedy(candidates)], notes

        search_end = time.time()
        _phase("search", gc_end, search_end)

        robot_ids = diag["robot_ids"]
        best_objective = search.best_objective()
        best_schedule = search.best_schedule()
        notes.update(
            {
                "search_duration_s": search_end - gc_end,
                "search_done": search.is_done(),
                "search_nodes_visited": search.nodes_visited(),
                "best_objective": None if best_objective == -float("inf") else best_objective,
                "best_schedule_depth": len(best_schedule),
                "best_schedule": [[robot_ids[i] for i in sb.robot_indices] for sb in best_schedule],
                "search_iters": diag["search_iters"],
                "max_step": diag["max_step"],
                "avg_step": diag["avg_step"],
                "op_time_twin": search.op_time_twin(),
                "op_time_queue": search.op_time_queue(),
                "op_time_fastforward": search.op_time_fastforward(),
                "op_time_evaluate": search.op_time_evaluate(),
                "max_node_time": search.max_node_time(),
                "gc_before": gc_counts_before,
                "gc_after": gc_counts_after,
            }
        )

        logger.debug(
            "lookahead_cpp search inter_call=%+.3fs slack_in=%+.3fs buffer=%.3fs "
            "iters=%d nodes=%d budget=%d total=%.4fs max_step=%.4fs avg_step=%.4fs "
            "in_flight=%d candidates=%d done=%s",
            inter_call_gap,
            slack,
            self.scheduling_buffer,
            diag["search_iters"],
            search.nodes_visited(),
            self.step_budget_nodes,
            search_end - gc_end,
            diag["max_step"],
            diag["avg_step"],
            in_flight,
            len(candidates),
            search.is_done(),
        )

        if not best_schedule:
            logger.debug("lookahead_cpp stage=exit reason=greedy_search_empty")
            greedy_batch = self._greedy(candidates)
            _phase("postprocess", search_end, time.time())
            notes["mode"] = "greedy_search_empty"
            return [greedy_batch], notes

        notes["mode"] = "search"
        batches: list[list[SlotRequest]] = []
        for sb in best_schedule[:dispatch_budget]:
            batch_rids = [robot_ids[i] for i in sb.robot_indices]
            batch = [
                self._latest_requests[rid] for rid in batch_rids if rid in self._latest_requests
            ]
            if batch:
                batches.append(batch)
        _phase("postprocess", search_end, time.time())
        logger.debug(
            "lookahead_cpp stage=return mode=search batches=%d plan_depth=%d objective=%.4f",
            len(batches),
            len(best_schedule),
            best_objective,
        )
        return batches, notes

    def _greedy(self, candidates: list[SlotRequest]) -> list[SlotRequest]:
        deadlines = self.mirror.deadlines()
        return sorted(candidates, key=lambda r: deadlines.get(r.robot_id, r.deadline))[
            : self._max_batch_size
        ]
