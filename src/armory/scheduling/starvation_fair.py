import multiprocessing as mp
import time
from typing import Any

from armory.scheduling.base import RequestScheduler
from armory.serving.schemas import RobotID, SlotRequest


class StarvationFairScheduler(RequestScheduler):
    """Batch scheduler with one alpha knob for total-vs-worst starvation.

    Starvation is tracked as action-chunk debt. A robot's debt accrues at its
    estimated action demand rate and drops by one when the scheduler queues a
    chunk for that robot.

    For each possible batch size, the scheduler picks the robots with the
    largest marginal reduction in ``sum(starvation ** (1 + alpha))`` and chooses
    the batch with the most reduction per second of inference latency.

    ``alpha=0`` optimizes linear total starvation. Larger alpha values make the
    largest starvation debts dominate, approaching max-min fairness.
    """

    def __init__(
        self,
        batch_queue: mp.Queue,
        max_batch_size: int = 1,
        *,
        alpha: float = 0.0,
    ):
        super().__init__(batch_queue, max_batch_size)
        self._alpha = max(0.0, alpha)
        self._objective_power = 1.0 + self._alpha
        self._starvation: dict[RobotID, float] = {}
        self._demand: dict[RobotID, float] = {}
        self._last_advance = time.time()

    def update(self, request: SlotRequest) -> None:
        self._advance_starvation(time.time())
        super().update(request)
        self._demand[request.robot_id] = self._action_demand(request)
        self._starvation.setdefault(request.robot_id, 0.0)

    def reset_robot(self, robot_id: RobotID) -> None:
        super().reset_robot(robot_id)
        self._starvation.pop(robot_id, None)
        self._demand.pop(robot_id, None)

    def reset_all(self) -> None:
        super().reset_all()
        self._starvation.clear()
        self._demand.clear()
        self._last_advance = time.time()

    def get_next_batches(
        self, candidates: list[SlotRequest]
    ) -> tuple[list[list[SlotRequest]], dict[str, Any]]:
        if self.mirror.in_flight_batches_count > 0:
            return [], {"reason": "server_busy"}
        if not candidates:
            return [], {"reason": "no_candidates"}

        self._advance_starvation(time.time())
        candidate_by_robot = {request.robot_id: request for request in candidates}
        ranked = sorted(
            candidate_by_robot.values(),
            key=lambda request: self._candidate_sort_key(request),
        )

        best_batch: list[SlotRequest] = []
        best_key: tuple[float, int, float, float] | None = None
        batch_scores: list[dict[str, Any]] = []
        max_size = min(self._max_batch_size, len(ranked))
        for batch_size in range(1, max_size + 1):
            batch = ranked[:batch_size]
            reduction = sum(self._marginal_reduction(request.robot_id) for request in batch)
            infer_latency = max(self.latency_tracker.infer_latency(batch_size), 1e-6)
            value_per_second = reduction / infer_latency
            key = (value_per_second, len(batch), reduction, -infer_latency)
            batch_scores.append(
                {
                    "batch_size": batch_size,
                    "robots": [request.robot_id for request in batch],
                    "reduction": reduction,
                    "infer_latency": infer_latency,
                    "value_per_second": value_per_second,
                }
            )
            if best_key is None or key > best_key:
                best_key = key
                best_batch = batch

        for request in best_batch:
            if not request.is_padding:
                self._starvation[request.robot_id] = max(
                    0.0, self._starvation.get(request.robot_id, 0.0) - 1.0
                )

        notes = {
            "rule": "starvation_fair",
            "alpha": self._alpha,
            "objective_power": self._objective_power,
            "max_batch_size": self._max_batch_size,
            "chosen_batch_size": len(best_batch),
            "starvation": dict(self._starvation),
            "demand": dict(self._demand),
            "candidate_order": [request.robot_id for request in ranked],
            "candidate_reductions": {
                request.robot_id: self._marginal_reduction(request.robot_id) for request in ranked
            },
            "batch_scores": batch_scores,
        }
        return ([best_batch], notes) if best_batch else ([], notes)

    def _advance_starvation(self, now: float) -> None:
        elapsed = now - self._last_advance
        if elapsed <= 0.0:
            return
        self._last_advance = now
        for robot_id, demand in self._demand.items():
            self._starvation[robot_id] = self._starvation.get(robot_id, 0.0) + elapsed * demand

    def _candidate_sort_key(self, request: SlotRequest) -> tuple[float, float, RobotID]:
        robot_id = request.robot_id
        return (
            -self._marginal_reduction(robot_id),
            -self._starvation.get(robot_id, 0.0),
            robot_id,
        )

    def _marginal_reduction(self, robot_id: RobotID) -> float:
        starvation = max(0.0, self._starvation.get(robot_id, 0.0))
        after_service = max(0.0, starvation - 1.0)
        return starvation**self._objective_power - after_service**self._objective_power

    @staticmethod
    def _action_demand(request: SlotRequest) -> float:
        return float(request.control_hz) / max(1.0, float(request.max_execution_horizon))
