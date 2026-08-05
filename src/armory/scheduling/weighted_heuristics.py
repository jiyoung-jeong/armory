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


class WeightedDeficitRoundRobinScheduler(RoundRobinScheduler):
    """Round robin with persistent, weight-proportional service deficits."""

    def __init__(
        self,
        config: SchedulerConfig,
        batch_queue: mp.Queue,
        max_batch_size: int = 1,
    ) -> None:
        super().__init__(config, batch_queue, max_batch_size)
        self._deficit: dict[RobotID, float] = {}

    def update(self, request: SlotRequest) -> None:
        _validate_weight(request)
        super().update(request)
        self._deficit.setdefault(request.robot_id, 0.0)

    def get_next_batches(
        self, candidates: list[SlotRequest]
    ) -> tuple[list[list[SlotRequest]], dict[str, Any]]:
        if self.mirror.in_flight_batches_count > 0:
            return [], {"reason": "server_busy"}
        if not candidates:
            return [], {"reason": "no_candidates"}

        candidate_by_robot = {request.robot_id: request for request in candidates}
        weights = {request.robot_id: _validate_weight(request) for request in candidates}
        robot_positions = {robot_id: index for index, robot_id in enumerate(self._rr_robot_order)}
        unknown = sorted(set(candidate_by_robot) - set(robot_positions))
        if unknown:
            raise RuntimeError(f"Candidate robots were not registered through update(): {unknown}")

        n_robots = len(self._rr_robot_order)
        cursor_before = self._rr_index % n_robots
        max_weight = max(weights.values())
        quantum = {robot_id: weight / max_weight for robot_id, weight in weights.items()}
        deficit_before = {
            robot_id: self._deficit.setdefault(robot_id, 0.0) for robot_id in candidate_by_robot
        }

        remaining = set(candidate_by_robot)
        chosen: list[SlotRequest] = []
        while remaining and len(chosen) < self._max_batch_size:
            cursor = self._rr_index % n_robots
            distance = {
                robot_id: (robot_positions[robot_id] - cursor) % n_robots for robot_id in remaining
            }
            visits_needed = {
                robot_id: max(
                    1,
                    math.ceil((1.0 - self._deficit[robot_id]) / quantum[robot_id] - 1e-12),
                )
                for robot_id in remaining
            }
            service_offset = {
                robot_id: distance[robot_id] + (visits_needed[robot_id] - 1) * n_robots
                for robot_id in remaining
            }
            selected_id = min(remaining, key=service_offset.__getitem__)
            selected_offset = service_offset[selected_id]

            for robot_id in remaining:
                if distance[robot_id] <= selected_offset:
                    visits = (selected_offset - distance[robot_id]) // n_robots + 1
                    self._deficit[robot_id] += visits * quantum[robot_id]

            self._deficit[selected_id] = max(0.0, self._deficit[selected_id] - 1.0)
            chosen.append(candidate_by_robot[selected_id])
            remaining.remove(selected_id)
            self._rr_index = (robot_positions[selected_id] + 1) % n_robots

        notes = {
            "rule": "weighted_deficit_round_robin",
            "max_batch_size": self._max_batch_size,
            "chosen_batch_size": len(chosen),
            "weights": weights,
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

    def reset_all(self) -> None:
        super().reset_all()
        self._deficit.clear()
