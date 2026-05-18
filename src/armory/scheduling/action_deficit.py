import multiprocessing as mp
from collections.abc import Iterable
from typing import Any

from armory.scheduling.base import RequestScheduler
from armory.serving.schemas import RobotID, SlotRequest


class ActionDeficitScheduler(RequestScheduler):
    """Deficit scheduler with alpha-shaped action demand weights.

    Each scheduling pass grants every known robot a bounded amount of credit.
    Scheduled robots spend one credit. With ``alpha=0`` all robots receive the
    same quantum, so the round-robin tie-breaker determines service order. With
    ``alpha=1`` credit accrues proportional to estimated action demand.
    """

    def __init__(
        self,
        batch_queue: mp.Queue,
        max_batch_size: int = 1,
        *,
        alpha: float = 0.0,
        credit_cap: float | None = None,
    ):
        super().__init__(batch_queue, max_batch_size)
        self._alpha = min(1.0, max(0.0, alpha))
        self._credit_cap = credit_cap if credit_cap is not None else float(2 * max_batch_size)
        self._credit: dict[RobotID, float] = {}
        self._demand: dict[RobotID, float] = {}
        self._rr_index: int = 0
        self._robot_order: list[RobotID] = []

    def update(self, request: SlotRequest) -> None:
        super().update(request)
        if request.robot_id not in self._robot_order:
            self._robot_order.append(request.robot_id)
        self._demand[request.robot_id] = self._action_demand(request)
        self._credit.setdefault(request.robot_id, 0.0)

    def reset_robot(self, robot_id: RobotID) -> None:
        super().reset_robot(robot_id)
        self._credit.pop(robot_id, None)
        self._demand.pop(robot_id, None)
        if robot_id in self._robot_order:
            removed_index = self._robot_order.index(robot_id)
            self._robot_order.remove(robot_id)
            if removed_index < self._rr_index:
                self._rr_index = max(0, self._rr_index - 1)
            if self._robot_order:
                self._rr_index %= len(self._robot_order)
            else:
                self._rr_index = 0

    def reset_all(self) -> None:
        super().reset_all()
        self._credit.clear()
        self._demand.clear()
        self._robot_order.clear()
        self._rr_index = 0

    def get_next_batches(
        self, candidates: list[SlotRequest]
    ) -> tuple[list[list[SlotRequest]], dict[str, Any]]:
        if self.mirror.in_flight_batches_count > 0:
            return [], {"reason": "server_busy"}

        candidate_by_robot = {request.robot_id: request for request in candidates}
        n_robots = len(self._robot_order)
        if not candidate_by_robot or n_robots == 0:
            reason = "no_candidates" if not candidate_by_robot else "no_robots_known"
            return [], {"reason": reason}

        self._accrue_credit(candidate_by_robot.keys())

        rr_distance = self._rr_distances()
        ordered_candidates = sorted(
            candidate_by_robot.values(),
            key=lambda request: (
                -self._credit.get(request.robot_id, 0.0),
                rr_distance.get(request.robot_id, n_robots),
                request.robot_id,
            ),
        )
        batch = ordered_candidates[: self._max_batch_size]

        for request in batch:
            if not request.is_padding:
                self._credit[request.robot_id] = self._credit.get(request.robot_id, 0.0) - 1.0

        rr_index_before = self._rr_index % n_robots
        self._advance_rr_index(batch)
        notes = {
            "rule": "action_deficit",
            "alpha": self._alpha,
            "max_batch_size": self._max_batch_size,
            "chosen_batch_size": len(batch),
            "rr_index_before": rr_index_before,
            "rr_index_after": self._rr_index,
            "robot_order": list(self._robot_order),
            "credit": dict(self._credit),
            "demand": dict(self._demand),
            "candidate_order": [request.robot_id for request in ordered_candidates],
        }
        return ([batch], notes) if batch else ([], notes)

    def _accrue_credit(self, active_robot_ids: Iterable[RobotID]) -> None:
        active = set(active_robot_ids)
        weights = self._weights()
        for robot_id in self._robot_order:
            if robot_id not in active:
                continue
            next_credit = self._credit.get(robot_id, 0.0) + weights.get(robot_id, 1.0)
            self._credit[robot_id] = min(self._credit_cap, next_credit)

    def _weights(self) -> dict[RobotID, float]:
        if not self._demand:
            return {}

        max_demand = max(self._demand.values(), default=1.0)
        if max_demand <= 0.0:
            return {robot_id: 1.0 for robot_id in self._robot_order}

        return {
            robot_id: (max(self._demand.get(robot_id, max_demand), 0.0) / max_demand) ** self._alpha
            for robot_id in self._robot_order
        }

    def _rr_distances(self) -> dict[RobotID, int]:
        n_robots = len(self._robot_order)
        if n_robots == 0:
            return {}
        start = self._rr_index % n_robots
        return {
            self._robot_order[(start + offset) % n_robots]: offset for offset in range(n_robots)
        }

    def _advance_rr_index(self, batch: list[SlotRequest]) -> None:
        if not batch or not self._robot_order:
            return

        chosen_positions = [
            self._robot_order.index(request.robot_id)
            for request in batch
            if request.robot_id in self._robot_order
        ]
        if not chosen_positions:
            return

        n_robots = len(self._robot_order)
        start = self._rr_index % n_robots
        last_position = max(chosen_positions, key=lambda idx: (idx - start) % n_robots)
        self._rr_index = (last_position + 1) % n_robots

    @staticmethod
    def _action_demand(request: SlotRequest) -> float:
        return float(request.control_hz) / max(1.0, float(request.max_execution_horizon))
