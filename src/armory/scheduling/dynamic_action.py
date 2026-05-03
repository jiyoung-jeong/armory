import multiprocessing as mp
import time

from armory.scheduling import RequestScheduler
from armory.serving.schemas import SlotRequest


class DynamicActionScheduler(RequestScheduler):
    """Greedy-deadline baseline with weighted useful-action adjustments.

    Candidate batches are prefixes of the earliest-deadline ordering, so the
    scheduler inherits a greedy EDF structure instead of arbitrarily skipping an
    urgent robot in favor of a later one. The configurable weights then score how
    far to deviate in batch size from that baseline:

    - useful_action_weight: reward chunks that are expected to contain more still-
      executable action steps at delivery time;
    - tardiness_weight: penalize chunks whose latency makes some returned actions
      unusable on arrival;
    - slack_weight: reward serving robots with low remaining infer slack;
    - deficit_weight: reward serving robots that have accumulated service debt.

    When all weights are zero, the score reduces to the underlying greedy-deadline
    batch-size rule without any explicit special-case branch.
    """

    def __init__(
        self,
        batch_queue: mp.Queue,
        max_batch_size: int = 1,
        *,
        useful_action_weight: float = 1.0,
        tardiness_weight: float = 0.0,
        slack_weight: float = 0.0,
        deficit_weight: float = 0.0,
    ):
        super().__init__(batch_queue, max_batch_size)
        self._useful_action_weight = max(0.0, useful_action_weight)
        self._tardiness_weight = max(0.0, tardiness_weight)
        self._slack_weight = max(0.0, slack_weight)
        self._deficit_weight = max(0.0, deficit_weight)

        self._service_debt: dict[str, float] = {}
        self._demand_rate: dict[str, float] = {}
        self._last_debt_update_time = time.time()

    def update(self, request: SlotRequest) -> None:
        self._advance_debts(time.time())
        super().update(request)
        self._demand_rate[request.robot_id] = self._compute_demand_rate(request)
        self._service_debt.setdefault(request.robot_id, 0.0)

    def get_next_batches(self) -> list[list[SlotRequest]]:
        if not self._batch_queue.empty() or (candidates := self.schedulable_requests) == []:
            return []

        now = time.time()
        self._advance_debts(now)
        ordered_candidates = sorted(candidates, key=lambda r: (self._infer_deadline(r), r.robot_id))

        best_batch: tuple[SlotRequest, ...] | None = None
        best_score: tuple | None = None
        max_size = min(self._max_batch_size, len(ordered_candidates))
        for batch_size in range(1, max_size + 1):
            batch = tuple(ordered_candidates[:batch_size])
            score = self._score_batch(batch, now)
            if best_score is None or score > best_score:
                best_batch = batch
                best_score = score

        if best_batch is None:
            return []

        self._charge_service(best_batch)
        return [list(best_batch)]

    def reset_robot(self, robot_id: str) -> None:
        self._advance_debts(time.time())
        super().reset_robot(robot_id)
        self._service_debt.pop(robot_id, None)
        self._demand_rate.pop(robot_id, None)

    def _compute_demand_rate(self, request: SlotRequest) -> float:
        execution_horizon = max(1.0, float(request.execution_horizon))
        control_hz = max(1.0, float(request.control_hz))
        return control_hz / execution_horizon

    def _advance_debts(self, now: float) -> None:
        elapsed = max(0.0, now - self._last_debt_update_time)
        if elapsed <= 0:
            return

        for robot_id, demand_rate in self._demand_rate.items():
            self._service_debt[robot_id] = (
                self._service_debt.get(robot_id, 0.0) + elapsed * demand_rate
            )
        self._last_debt_update_time = now

    def _charge_service(self, batch: tuple[SlotRequest, ...]) -> None:
        for request in batch:
            if request.is_padding:
                continue
            self._service_debt[request.robot_id] = max(
                0.0, self._service_debt.get(request.robot_id, 0.0) - 1.0
            )

    def _infer_deadline(self, request: SlotRequest) -> float:
        return self._deadlines.get(
            request.robot_id, request.deadline
        ) - self.latency_tracker.action_latency(request.robot_id)

    def _score_batch(self, batch: tuple[SlotRequest, ...], now: float) -> tuple:
        infer_latency = self.latency_tracker.infer_latency(len(batch))
        earliest_infer_deadline = min(self._infer_deadline(request) for request in batch)
        time_remaining = earliest_infer_deadline - now
        fits_earliest_deadline = int(infer_latency <= time_remaining)
        base_greedy_value = (
            len(batch) if fits_earliest_deadline else len(batch) / max(infer_latency, 1e-6)
        )

        weighted_useful_actions = 0.0
        feasible_count = 0
        weighted_tardiness = 0.0
        weighted_slack_bonus = 0.0
        weighted_deficit_bonus = 0.0
        total_demand = 0.0

        for request in batch:
            robot_id = request.robot_id
            demand_rate = self._demand_rate.get(robot_id, self._compute_demand_rate(request))
            predicted_latency_steps = (
                self.latency_tracker.total_latency(robot_id, len(batch)) * request.control_hz
            )
            useful_actions = max(0.0, request.execution_horizon - predicted_latency_steps)
            tardiness_steps = max(0.0, predicted_latency_steps - request.execution_horizon)

            infer_deadline = self._infer_deadline(request)
            slack_steps = (infer_deadline - now - infer_latency) * request.control_hz
            chunk_steps = max(1.0, float(request.execution_horizon))
            slack_bonus = max(0.0, 1.0 - (slack_steps / chunk_steps))
            deficit_bonus = self._service_debt.get(robot_id, 0.0)

            weighted_useful_actions += demand_rate * useful_actions
            weighted_tardiness += demand_rate * tardiness_steps
            weighted_slack_bonus += demand_rate * slack_bonus
            weighted_deficit_bonus += demand_rate * deficit_bonus
            total_demand += demand_rate
            if useful_actions > 0:
                feasible_count += 1

        utility_adjustment = (
            self._useful_action_weight * weighted_useful_actions
            - self._tardiness_weight * weighted_tardiness
            + self._slack_weight * weighted_slack_bonus
            + self._deficit_weight * weighted_deficit_bonus
        ) / max(infer_latency, 1e-6)
        robot_ids = tuple(request.robot_id for request in batch)

        return (
            fits_earliest_deadline,
            utility_adjustment,
            base_greedy_value,
            weighted_useful_actions,
            feasible_count,
            -weighted_tardiness,
            weighted_slack_bonus,
            weighted_deficit_bonus,
            total_demand,
            -earliest_infer_deadline,
            robot_ids,
        )
