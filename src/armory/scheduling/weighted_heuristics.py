import math
import multiprocessing as mp
import time
from typing import Any

from armory.scheduling.base import RequestScheduler
from armory.scheduling.baselines import RoundRobinScheduler
from armory.serving.protocol import SchedulerConfig
from armory.serving.schemas import RobotID, SlotRequest


def _validate_weight(request: SlotRequest) -> float:
    weight = float(request.weight)
    if not math.isfinite(weight) or weight <= 0.0:
        raise ValueError(
            f"Robot {request.robot_id!r} must have a positive finite weight; got {weight!r}"
        )
    return weight


def _action_coverage_cost(request: SlotRequest) -> float:
    control_hz = float(request.control_hz)
    horizon = float(request.max_execution_horizon)
    if not math.isfinite(control_hz) or control_hz <= 0.0:
        raise ValueError(
            f"Robot {request.robot_id!r} must have a positive finite control_hz; got {control_hz!r}"
        )
    if not math.isfinite(horizon) or horizon <= 0.0:
        raise ValueError(
            f"Robot {request.robot_id!r} must have a positive finite "
            f"max_execution_horizon; got {horizon!r}"
        )
    cost = horizon / control_hz
    if not math.isfinite(cost) or cost <= 0.0:
        raise ValueError(
            f"Robot {request.robot_id!r} must have a positive finite "
            f"action-coverage cost; got {cost!r}"
        )
    return cost


def _candidate_map(candidates: list[SlotRequest]) -> dict[RobotID, SlotRequest]:
    candidate_by_robot = {request.robot_id: request for request in candidates}
    if len(candidate_by_robot) != len(candidates):
        raise ValueError("Round-robin candidates must contain at most one request per robot")
    return candidate_by_robot


def _validate_registered(
    candidate_by_robot: dict[RobotID, SlotRequest], robot_order: list[RobotID]
) -> None:
    unknown = sorted(set(candidate_by_robot) - set(robot_order))
    if unknown:
        raise RuntimeError(f"Candidate robots were not registered through update(): {unknown}")


class WeightedEDFScheduler(RequestScheduler):
    """Earliest-deadline-first batching with weighted service debt."""

    def __init__(
        self,
        config: SchedulerConfig,
        batch_queue: mp.Queue,
        max_batch_size: int = 1,
    ) -> None:
        super().__init__(config, batch_queue, max_batch_size)
        self._service_debt: dict[RobotID, float] = {}
        self._demand_rate: dict[RobotID, float] = {}
        self._last_advance = time.time()

    def update(self, request: SlotRequest) -> None:
        _validate_weight(request)
        self._advance_debts(time.time())
        super().update(request)
        self._demand_rate[request.robot_id] = float(request.control_hz) / max(
            1.0, float(request.max_execution_horizon)
        )
        self._service_debt.setdefault(request.robot_id, 0.0)

    def get_next_batches(
        self, candidates: list[SlotRequest]
    ) -> tuple[list[list[SlotRequest]], dict[str, Any]]:
        if self.mirror.in_flight_batches_count > 0:
            return [], {"reason": "server_busy"}
        if not candidates:
            return [], {"reason": "no_candidates"}

        for request in candidates:
            _validate_weight(request)

        now = time.time()
        self._advance_debts(now)
        deadlines = self.mirror.deadlines()
        ordered = sorted(
            candidates,
            key=lambda request: (
                self._infer_deadline(request, deadlines),
                request.robot_id,
            ),
        )
        max_size = min(self._max_batch_size, len(ordered))
        best_batch = max(
            (tuple(ordered[:size]) for size in range(1, max_size + 1)),
            key=lambda batch: self._score(batch, now, deadlines),
        )

        for request in best_batch:
            self._service_debt[request.robot_id] = max(
                0.0,
                self._service_debt.get(request.robot_id, 0.0) - 1.0,
            )

        chosen = list(best_batch)
        notes = {
            "rule": "weighted_edf",
            "max_batch_size": self._max_batch_size,
            "chosen_batch_size": len(chosen),
            "ordered": [request.robot_id for request in ordered],
            "infer_deadlines": {
                request.robot_id: self._infer_deadline(request, deadlines) for request in ordered
            },
            "weights": {request.robot_id: request.weight for request in ordered},
            "service_debt": dict(self._service_debt),
            "demand_rate": dict(self._demand_rate),
        }
        return [chosen], notes

    def reset_robot(self, robot_id: RobotID) -> None:
        self._advance_debts(time.time())
        super().reset_robot(robot_id)
        self._service_debt.pop(robot_id, None)
        self._demand_rate.pop(robot_id, None)

    def reset_all(self) -> None:
        super().reset_all()
        self._service_debt.clear()
        self._demand_rate.clear()
        self._last_advance = time.time()

    def _advance_debts(self, now: float) -> None:
        elapsed = now - self._last_advance
        if elapsed <= 0.0:
            return
        for robot_id, rate in self._demand_rate.items():
            self._service_debt[robot_id] = self._service_debt.get(robot_id, 0.0) + elapsed * rate
        self._last_advance = now

    def _infer_deadline(
        self,
        request: SlotRequest,
        deadlines: dict[RobotID, float],
    ) -> float:
        return deadlines.get(request.robot_id, request.deadline) - (
            self.latency_tracker.action_latency(request.robot_id)
        )

    def _score(
        self,
        batch: tuple[SlotRequest, ...],
        now: float,
        deadlines: dict[RobotID, float],
    ) -> tuple[int, float, float, float]:
        infer_latency = max(self.latency_tracker.infer_latency(len(batch)), 1e-6)
        earliest = min(self._infer_deadline(request, deadlines) for request in batch)
        feasible = int(infer_latency <= earliest - now)
        priority = sum(
            _validate_weight(request) * (1.0 + self._service_debt.get(request.robot_id, 0.0))
            for request in batch
        )
        throughput = len(batch) / infer_latency
        return feasible, priority / infer_latency, throughput, -earliest


class WeightedRoundRobinScheduler(RoundRobinScheduler):
    """Smooth weighted round robin over pending robots."""

    def __init__(
        self,
        config: SchedulerConfig,
        batch_queue: mp.Queue,
        max_batch_size: int = 1,
    ) -> None:
        super().__init__(config, batch_queue, max_batch_size)
        self._current_weight: dict[RobotID, float] = {}

    def update(self, request: SlotRequest) -> None:
        _validate_weight(request)
        super().update(request)
        self._current_weight.setdefault(request.robot_id, 0.0)

    def get_next_batches(
        self, candidates: list[SlotRequest]
    ) -> tuple[list[list[SlotRequest]], dict[str, Any]]:
        if self.mirror.in_flight_batches_count > 0:
            return [], {"reason": "server_busy"}
        if not candidates:
            return [], {"reason": "no_candidates"}

        candidate_by_robot = _candidate_map(candidates)
        weights = {request.robot_id: _validate_weight(request) for request in candidates}
        _validate_registered(candidate_by_robot, self._rr_robot_order)
        n_robots = len(self._rr_robot_order)
        cursor_before = self._rr_index % n_robots
        current_weight_before = {
            robot_id: self._current_weight[robot_id] for robot_id in candidate_by_robot
        }
        robot_positions = {
            robot_id: index for index, robot_id in enumerate(self._rr_robot_order)
        }
        total_weight = sum(weights.values())
        remaining = set(candidate_by_robot)
        chosen: list[SlotRequest] = []
        while remaining and len(chosen) < self._max_batch_size:
            for robot_id, weight in weights.items():
                self._current_weight[robot_id] += weight

            cursor = self._rr_index % n_robots
            selected_id = min(
                remaining,
                key=lambda robot_id: (
                    -self._current_weight[robot_id],
                    (robot_positions[robot_id] - cursor) % n_robots,
                ),
            )
            self._current_weight[selected_id] -= total_weight
            chosen.append(candidate_by_robot[selected_id])
            remaining.remove(selected_id)
            self._rr_index = (robot_positions[selected_id] + 1) % n_robots

        notes = {
            "rule": "weighted_round_robin",
            "max_batch_size": self._max_batch_size,
            "chosen_batch_size": len(chosen),
            "weights": weights,
            "current_weight_before": current_weight_before,
            "current_weight": dict(self._current_weight),
            "rr_index_before": cursor_before,
            "rr_index_after": self._rr_index,
            "robot_order": list(self._rr_robot_order),
        }
        return [chosen], notes

    def reset_robot(self, robot_id: RobotID) -> None:
        super().reset_robot(robot_id)
        self._current_weight.pop(robot_id, None)

    def reset_all(self) -> None:
        super().reset_all()
        self._current_weight.clear()


class DeficitRoundRobinScheduler(RoundRobinScheduler):
    """Deficit round robin with action-coverage seconds as request cost.

    Each pending robot receives the same quantum when visited. The quantum is the
    minimum registered request cost. Explicit robot weights are ignored.
    """

    def __init__(
        self,
        config: SchedulerConfig,
        batch_queue: mp.Queue,
        max_batch_size: int = 1,
    ) -> None:
        super().__init__(config, batch_queue, max_batch_size)
        self._deficit: dict[RobotID, float] = {}
        self._request_cost: dict[RobotID, float] = {}

    def update(self, request: SlotRequest) -> None:
        request_cost = _action_coverage_cost(request)
        super().update(request)
        self._deficit.setdefault(request.robot_id, 0.0)
        self._request_cost[request.robot_id] = request_cost

    def get_next_batches(
        self, candidates: list[SlotRequest]
    ) -> tuple[list[list[SlotRequest]], dict[str, Any]]:
        if self.mirror.in_flight_batches_count > 0:
            return [], {"reason": "server_busy"}
        if not candidates:
            return [], {"reason": "no_candidates"}

        candidate_by_robot = _candidate_map(candidates)
        for request in candidates:
            _action_coverage_cost(request)
        _validate_registered(candidate_by_robot, self._rr_robot_order)
        request_costs = {
            robot_id: self._request_cost[robot_id] for robot_id in candidate_by_robot
        }
        n_robots = len(self._rr_robot_order)
        cursor_before = self._rr_index % n_robots
        quantum = min(self._request_cost.values())
        deficit_before = {robot_id: self._deficit[robot_id] for robot_id in candidate_by_robot}
        remaining = set(candidate_by_robot)
        chosen: list[SlotRequest] = []
        while remaining and len(chosen) < self._max_batch_size:
            robot_id = self._rr_robot_order[self._rr_index % n_robots]
            self._rr_index = (self._rr_index + 1) % n_robots
            if robot_id not in remaining:
                continue

            self._deficit[robot_id] += quantum
            request_cost = self._request_cost[robot_id]
            if self._deficit[robot_id] + 1e-12 < request_cost:
                continue

            self._deficit[robot_id] = max(0.0, self._deficit[robot_id] - request_cost)
            chosen.append(candidate_by_robot[robot_id])
            remaining.remove(robot_id)

        notes = {
            "rule": "deficit_round_robin",
            "max_batch_size": self._max_batch_size,
            "chosen_batch_size": len(chosen),
            "action_coverage_cost": request_costs,
            "quantum": quantum,
            "deficit_before": deficit_before,
            "deficit": dict(self._deficit),
            "rr_index_before": cursor_before,
            "rr_index_after": self._rr_index,
            "robot_order": list(self._rr_robot_order),
        }
        return [chosen], notes

    def reset_robot(self, robot_id: RobotID) -> None:
        super().reset_robot(robot_id)
        self._deficit.pop(robot_id, None)
        self._request_cost.pop(robot_id, None)

    def reset_all(self) -> None:
        super().reset_all()
        self._deficit.clear()
        self._request_cost.clear()
