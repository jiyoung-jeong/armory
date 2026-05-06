import multiprocessing as mp
import time

from armory.scheduling import RequestScheduler
from armory.serving.schemas import SlotRequest


class DynamicActionScheduler(RequestScheduler):
    def __init__(
        self,
        batch_queue: mp.Queue,
        max_batch_size: int = 1,
        *,
        deficit_weight: float = 0.0,
    ):
        super().__init__(batch_queue, max_batch_size)
        self._deficit_weight = max(0.0, deficit_weight)
        self._service_debt: dict[str, float] = {}
        self._demand_rate: dict[str, float] = {}
        self._last_advance = time.time()

    def update(self, request: SlotRequest) -> None:
        # advance debts to reflect the demand rate
        self._advance_debts(time.time())
        super().update(request)
        # update the demand rate for the robot
        self._demand_rate[request.robot_id] = float(request.control_hz) / max(
            1.0, float(request.execution_horizon)
        )

    def reset_robot(self, robot_id: str) -> None:
        # reset the service debt and demand rate for the robot
        super().reset_robot(robot_id)
        self._service_debt.pop(robot_id, None)
        self._demand_rate.pop(robot_id, None)

    def get_next_batches(self) -> list[list[SlotRequest]]:
        # return if there are any batches in the queue or no schedulable requests
        if not self._batch_queue.empty() or not (candidates := self.schedulable_requests):
            return []

        # advance debts to reflect the demand rate
        now = time.time()
        self._advance_debts(now)
        # sort the candidates by deadline and robot id
        ordered = sorted(candidates, key=lambda r: (self._infer_deadline(r), r.robot_id))
        # get the maximum batch size
        max_size = min(self._max_batch_size, len(ordered))
        # get the best batch by scoring the batches
        best_batch = max(
            (tuple(ordered[:k]) for k in range(1, max_size + 1)),
            key=lambda b: self._score(b, now),
        )

        # update the service debt for the best batch
        for r in best_batch:
            if not r.is_padding:
                self._service_debt[r.robot_id] = max(
                    0.0, self._service_debt.get(r.robot_id, 0.0) - 1.0
                )
        return [list(best_batch)]

    def _advance_debts(self, now: float) -> None:
        # return if the time elapsed is less than or equal to 0
        elapsed = now - self._last_advance
        if elapsed <= 0:
            return
        # advance the debts for each robot based on the demand rate
        for robot_id, rate in self._demand_rate.items():
            self._service_debt[robot_id] = (
                self._service_debt.get(robot_id, 0.0) + elapsed * rate
            )
        self._last_advance = now

    def _infer_deadline(self, request: SlotRequest) -> float:
        # return the deadline for the request (from greedy deadline scheduler)
        return self._deadlines.get(
            request.robot_id, request.deadline
        ) - self.latency_tracker.action_latency(request.robot_id)

    def _score(self, batch: tuple[SlotRequest, ...], now: float) -> tuple:
        # infer latency for the batch
        infer_latency = max(self.latency_tracker.infer_latency(len(batch)), 1e-6)
        # get the earliest deadline for the batch
        earliest = min(self._infer_deadline(r) for r in batch)
        # check if the batch fits within the earliest deadline
        fits = int(infer_latency <= earliest - now)
        # get the base score for the batch
        base = len(batch) if fits else len(batch) / infer_latency
        # get the weighted debt for the batch
        weighted_debt = sum(
            self._demand_rate.get(r.robot_id, 0.0) * self._service_debt.get(r.robot_id, 0.0)
            for r in batch
        )
        # return the score for the batch
        return (fits, self._deficit_weight * weighted_debt / infer_latency, base, -earliest)
