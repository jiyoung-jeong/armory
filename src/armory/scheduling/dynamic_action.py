import multiprocessing as mp
import time
from typing import Any

from armory.scheduling.base import RequestScheduler
from armory.serving.schemas import SlotRequest


class DynamicActionScheduler(RequestScheduler):
    def __init__(
        self,
        batch_queue: mp.Queue,
        max_batch_size: int = 1,
        *,
        alpha: float = 0.0,
    ):
        super().__init__(batch_queue, max_batch_size)
        self._alpha = max(0.0, alpha)
        self._service_debt: dict[str, float] = {}
        self._demand_rate: dict[str, float] = {}
        self._last_advance = time.time()

    def update(self, request: SlotRequest) -> None:
        # advance debts to reflect the demand rate
        self._advance_debts(time.time())
        super().update(request)
        # update the demand rate for the robot
        self._demand_rate[request.robot_id] = float(request.control_hz) / max(
            1.0, float(request.max_execution_horizon)
        )

    def reset_robot(self, robot_id: str) -> None:
        # reset the service debt and demand rate for the robot
        super().reset_robot(robot_id)
        self._service_debt.pop(robot_id, None)
        self._demand_rate.pop(robot_id, None)

    def get_next_batches(
        self, candidates: list[SlotRequest]
    ) -> tuple[list[list[SlotRequest]], dict[str, Any]]:
        if self.mirror.in_flight_batches_count > 0:
            return [], {"reason": "server_busy"}
        if not candidates:
            return [], {"reason": "no_candidates"}

        # advance debts to reflect the demand rate
        now = time.time()
        self._advance_debts(now)
        deadlines = self.mirror.deadlines()
        # sort the candidates by deadline and robot id
        ordered = sorted(
            candidates,
            key=lambda r: (self._infer_deadline(r, deadlines), r.robot_id),
        )
        # get the maximum batch size
        max_size = min(self._max_batch_size, len(ordered))
        # get the best batch by scoring the batches
        best_batch = max(
            (tuple(ordered[:k]) for k in range(1, max_size + 1)),
            key=lambda b: self._score(b, now, deadlines),
        )

        # update the service debt for the best batch
        for r in best_batch:
            if not r.is_padding:
                self._service_debt[r.robot_id] = max(
                    0.0, self._service_debt.get(r.robot_id, 0.0) - 1.0
                )
        chosen = list(best_batch)
        notes = {
            "rule": "alpha_fair_dynamic_action",
            "alpha": self._alpha,
            "max_batch_size": self._max_batch_size,
            "chosen_batch_size": len(chosen),
            "infer_deadlines": {r.robot_id: self._infer_deadline(r, deadlines) for r in ordered},
            "service_debt": dict(self._service_debt),
            "demand_rate": dict(self._demand_rate),
        }
        return [chosen], notes

    def _advance_debts(self, now: float) -> None:
        # return if the time elapsed is less than or equal to 0
        elapsed = now - self._last_advance
        if elapsed <= 0:
            return
        # advance the debts for each robot based on the demand rate
        for robot_id, rate in self._demand_rate.items():
            self._service_debt[robot_id] = self._service_debt.get(robot_id, 0.0) + elapsed * rate
        self._last_advance = now

    def _infer_deadline(self, request: SlotRequest, deadlines: dict[str, float]) -> float:
        # return the deadline for the request (from greedy deadline scheduler)
        return deadlines.get(
            request.robot_id, request.deadline
        ) - self.latency_tracker.action_latency(request.robot_id)

    def _score(
        self,
        batch: tuple[SlotRequest, ...],
        now: float,
        deadlines: dict[str, float],
    ) -> tuple:
        # infer latency for the batch
        infer_latency = max(self.latency_tracker.infer_latency(len(batch)), 1e-6)
        # get the earliest deadline for the batch
        earliest = min(self._infer_deadline(r, deadlines) for r in batch)
        # check if the batch fits within the earliest deadline
        fits = int(infer_latency <= earliest - now)
        # get the base score for the batch
        base = len(batch) if fits else len(batch) / infer_latency
        # alpha-fair priority: priority_i = (d_i * (1 + debt_i))^alpha
        # alpha=0 -> priority=1 (action-throughput greedy: max batch size)
        # alpha=1 -> demand-weighted debt (proportional fair)
        # alpha large -> dominated by argmax(d_i*(1+debt_i)) (max-min)
        weighted_priority = sum(
            (
                self._demand_rate.get(r.robot_id, 0.0)
                * (1.0 + self._service_debt.get(r.robot_id, 0.0))
            )
            ** self._alpha
            for r in batch
        )
        return (weighted_priority / infer_latency, base, -earliest)
