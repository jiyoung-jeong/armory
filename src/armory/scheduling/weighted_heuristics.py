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


def _validate_integer_weight(request: SlotRequest) -> int:
    weight = _validate_weight(request)
    if not weight.is_integer():
        raise ValueError(
            f"Robot {request.robot_id!r} must have an integer weight for weighted "
            f"round robin; got {weight!r}"
        )
    return int(weight)


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
        mean_priority = sum(
            _validate_weight(request) * (1.0 + self._service_debt.get(request.robot_id, 0.0))
            for request in batch
        ) / len(batch)
        throughput = len(batch) / infer_latency
        return feasible, mean_priority, throughput, -earliest


class WeightedRoundRobinScheduler(RoundRobinScheduler):
    """Conventional counter-based weighted round robin.

    Each robot receives ``weight`` visits per cycle, with at most one visit per
    circular scan. GPU batches preserve that serial WRR order. If the next visit
    would repeat a robot already in the current batch, the partial batch is
    dispatched and that visit is deferred until the next scheduling pass.
    """

    def __init__(
        self,
        config: SchedulerConfig,
        batch_queue: mp.Queue,
        max_batch_size: int = 1,
    ) -> None:
        super().__init__(config, batch_queue, max_batch_size)
        self._weights: dict[RobotID, int] = {}
        self._remaining_quota: dict[RobotID, int] = {}

    def update(self, request: SlotRequest) -> None:
        weight = _validate_integer_weight(request)
        previous_weight = self._weights.get(request.robot_id)
        if previous_weight is not None and previous_weight != weight:
            raise ValueError(
                f"Robot {request.robot_id!r} changed its weighted-round-robin weight "
                f"from {previous_weight} to {weight}"
            )
        super().update(request)
        self._weights.setdefault(request.robot_id, weight)
        self._remaining_quota.setdefault(request.robot_id, weight)

    def get_next_batches(
        self, candidates: list[SlotRequest]
    ) -> tuple[list[list[SlotRequest]], dict[str, Any]]:
        if self.mirror.in_flight_batches_count > 0:
            return [], {"reason": "server_busy"}
        if not candidates:
            # All queues are empty, so the next request starts a fresh WRR cycle.
            self._remaining_quota = dict(self._weights)
            return [], {"reason": "no_candidates"}

        candidate_by_robot = _candidate_map(candidates)
        candidate_weights = {
            request.robot_id: _validate_integer_weight(request) for request in candidates
        }
        _validate_registered(candidate_by_robot, self._rr_robot_order)
        for robot_id, weight in candidate_weights.items():
            registered_weight = self._weights[robot_id]
            if weight != registered_weight:
                raise ValueError(
                    f"Robot {robot_id!r} changed its weighted-round-robin weight "
                    f"from {registered_weight} to {weight}"
                )

        n_robots = len(self._rr_robot_order)
        cursor_before = self._rr_index % n_robots
        remaining_quota_before = dict(self._remaining_quota)
        chosen: list[SlotRequest] = []
        chosen_ids: set[RobotID] = set()
        cycle_resets = 0
        stopped_before_repeat: RobotID | None = None

        while len(chosen) < self._max_batch_size:
            if not any(
                self._remaining_quota[robot_id] > 0 for robot_id in candidate_by_robot
            ):
                # Every nonempty queue has exhausted its quota. Empty queues do
                # not hold the cycle open, matching the paper's reset rule.
                self._remaining_quota = dict(self._weights)
                cycle_resets += 1

            selected_id: RobotID | None = None
            for _ in range(n_robots):
                index = self._rr_index % n_robots
                robot_id = self._rr_robot_order[index]
                if (
                    robot_id in candidate_by_robot
                    and self._remaining_quota[robot_id] > 0
                ):
                    if robot_id in chosen_ids:
                        stopped_before_repeat = robot_id
                        break
                    selected_id = robot_id
                    self._rr_index = (index + 1) % n_robots
                    break
                self._rr_index = (index + 1) % n_robots

            if stopped_before_repeat is not None:
                break
            if selected_id is not None:
                chosen.append(candidate_by_robot[selected_id])
                chosen_ids.add(selected_id)
                self._remaining_quota[selected_id] -= 1
                if not any(
                    self._remaining_quota[robot_id] > 0
                    for robot_id in candidate_by_robot
                ):
                    # Refill immediately, even when this request fills the GPU
                    # batch, so newly arriving queues cannot observe a stale cycle.
                    self._remaining_quota = dict(self._weights)
                    cycle_resets += 1
                continue

            raise RuntimeError("Weighted round robin found no eligible candidate")

        notes = {
            "rule": "weighted_round_robin",
            "max_batch_size": self._max_batch_size,
            "chosen_batch_size": len(chosen),
            "weights": dict(self._weights),
            "remaining_quota_before": remaining_quota_before,
            "remaining_quota": dict(self._remaining_quota),
            "cycle_resets": cycle_resets,
            "stopped_before_repeat": stopped_before_repeat,
            "rr_index_before": cursor_before,
            "rr_index_after": self._rr_index,
            "robot_order": list(self._rr_robot_order),
        }
        return [chosen], notes

    def reset_robot(self, robot_id: RobotID) -> None:
        super().reset_robot(robot_id)
        self._weights.pop(robot_id, None)
        self._remaining_quota.pop(robot_id, None)

    def reset_all(self) -> None:
        super().reset_all()
        self._weights.clear()
        self._remaining_quota.clear()


class DeficitRoundRobinScheduler(RoundRobinScheduler):
    """Deficit round robin with action-coverage seconds as request cost.

    Each pending robot receives the same quantum when visited. The quantum is the
    minimum registered request cost. GPU batches preserve the serial DRR service
    order and stop before selecting the same robot twice. Explicit robot weights
    are ignored.
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
        chosen: list[SlotRequest] = []
        chosen_ids: set[RobotID] = set()
        stopped_before_repeat: RobotID | None = None
        while len(chosen) < self._max_batch_size:
            robot_id = self._rr_robot_order[self._rr_index % n_robots]
            if robot_id not in candidate_by_robot:
                self._rr_index = (self._rr_index + 1) % n_robots
                continue

            request_cost = self._request_cost[robot_id]
            next_deficit = self._deficit[robot_id] + quantum
            if next_deficit + 1e-12 < request_cost:
                self._deficit[robot_id] = next_deficit
                self._rr_index = (self._rr_index + 1) % n_robots
                continue
            if robot_id in chosen_ids:
                stopped_before_repeat = robot_id
                break

            self._deficit[robot_id] = max(0.0, next_deficit - request_cost)
            chosen.append(candidate_by_robot[robot_id])
            chosen_ids.add(robot_id)
            self._rr_index = (self._rr_index + 1) % n_robots

        notes = {
            "rule": "deficit_round_robin",
            "max_batch_size": self._max_batch_size,
            "chosen_batch_size": len(chosen),
            "action_coverage_cost": request_costs,
            "quantum": quantum,
            "deficit_before": deficit_before,
            "deficit": dict(self._deficit),
            "stopped_before_repeat": stopped_before_repeat,
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
