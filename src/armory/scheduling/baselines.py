import dataclasses
import itertools
import multiprocessing as mp
import random
import time

from armory.scheduling import RequestScheduler
from armory.scheduling.latency import LatencyTracker
from armory.serving.schemas import SlotRequest


def calculate_usable_time(latency_tracker: LatencyTracker, slot_request: SlotRequest, batch_size: int) -> float:
    total_latency = latency_tracker.total_latency(slot_request.robot_id, batch_size)
    total_chunk_time = slot_request.execution_horizon / slot_request.control_hz
    return total_chunk_time - total_latency


class MaxBatchScheduler(RequestScheduler):
    """Greedy scheduler that always fills to max_batch_size, prioritizing requests with earliest deadlines."""

    def get_next_batches(self) -> list[list[SlotRequest]]:
        if self._batch_queue.qsize() > 0 or (candidates := self.schedulable_requests) == []:
            return []

        candidates = sorted(candidates, key=lambda r: self._deadlines.get(r.robot_id, r.deadline))
        return [candidates[: self._max_batch_size]]


class FixedMaxBatchScheduler(RequestScheduler):
    """Always dispatch max_batch_size rows, padding with artificial duplicate requests if needed."""

    def get_next_batches(self) -> list[list[SlotRequest]]:
        if self._batch_queue.qsize() > 0 or (candidates := self.schedulable_requests) == []:
            return []

        candidates = sorted(candidates, key=lambda r: self._deadlines.get(r.robot_id, r.deadline))
        batch = candidates[: self._max_batch_size]
        if len(batch) == self._max_batch_size:
            return [batch]

        pad_sources = list(batch)
        pad_index = 0
        while len(batch) < self._max_batch_size:
            source = pad_sources[pad_index % len(pad_sources)]
            batch.append(dataclasses.replace(source, is_padding=True))
            pad_index += 1
        return [batch]


class GreedyActionScheduler(RequestScheduler):
    """Earliest-deadline-first: sort all pending requests by deadline."""

    def get_next_batches(self) -> list[list[SlotRequest]]:
        if self._batch_queue.qsize() > 0 or (candidates := self.schedulable_requests) == []:
            return []

        potential_batches = itertools.chain.from_iterable(
            itertools.combinations(candidates, i) for i in range(1, self._max_batch_size + 1)
        )
        return [list(max(potential_batches, key=lambda batch: self.calculate_actions_per_second(batch)))]

    def calculate_actions_per_second(self, batch: tuple[SlotRequest, ...]) -> float:
        """Return the number of usable actions created per second spent on inference."""
        return sum(
            calculate_usable_time(self.latency_tracker, request, len(batch)) for request in batch
        ) / self.latency_tracker.infer_latency(len(batch))


class UsefulActionScheduler(RequestScheduler):
    """Score batches by expected useful actions after latency, with urgency and service-debt terms.

    The primary signal is the number of actions from a returned chunk that are still
    expected to be executable when the robot receives them. This is measured in
    action steps, not seconds, so short-horizon robots naturally become more
    sensitive to latency. Two secondary terms help in the regimes where a purely
    throughput-oriented policy under-serves urgent robots:

    - tardiness penalty: discourages batches whose latency pushes a robot beyond the
      end of its returned chunk;
    - slack bonus: boosts robots whose remaining infer slack is below one chunk;
    - deficit bonus: gives a mild lift to robots that have accumulated service debt.
    """

    def __init__(
        self,
        batch_queue: mp.Queue,
        max_batch_size: int = 1,
        *,
        tardiness_weight: float = 0.0,
        slack_weight: float = 0.0,
        deficit_weight: float = 0.0,
    ):
        super().__init__(batch_queue, max_batch_size)
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
        if self._batch_queue.qsize() > 0 or (candidates := self.schedulable_requests) == []:
            return []

        now = time.time()
        self._advance_debts(now)
        candidates = sorted(candidates, key=lambda r: r.robot_id)

        best_batch: tuple[SlotRequest, ...] | None = None
        best_score: tuple | None = None
        max_size = min(self._max_batch_size, len(candidates))
        for batch_size in range(1, max_size + 1):
            for batch in itertools.combinations(candidates, batch_size):
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
            self._service_debt[robot_id] = self._service_debt.get(robot_id, 0.0) + elapsed * demand_rate
        self._last_debt_update_time = now

    def _charge_service(self, batch: tuple[SlotRequest, ...]) -> None:
        for request in batch:
            if request.is_padding:
                continue
            self._service_debt[request.robot_id] = max(0.0, self._service_debt.get(request.robot_id, 0.0) - 1.0)

    def _infer_deadline(self, request: SlotRequest) -> float:
        return self._deadlines.get(request.robot_id, request.deadline) - self.latency_tracker.action_latency(
            request.robot_id
        )

    def _score_batch(self, batch: tuple[SlotRequest, ...], now: float) -> tuple:
        infer_latency = self.latency_tracker.infer_latency(len(batch))
        total_latency_value = 0.0
        total_useful_actions = 0.0
        feasible_count = 0
        total_tardiness = 0.0
        total_slack_bonus = 0.0
        total_deficit_bonus = 0.0
        total_demand = 0.0

        for request in batch:
            robot_id = request.robot_id
            demand_rate = self._demand_rate.get(robot_id, self._compute_demand_rate(request))
            predicted_latency_steps = self.latency_tracker.total_latency(robot_id, len(batch)) * request.control_hz
            useful_actions = max(0.0, request.execution_horizon - predicted_latency_steps)
            tardiness_steps = max(0.0, predicted_latency_steps - request.execution_horizon)

            infer_deadline = self._infer_deadline(request)
            slack_steps = (infer_deadline - now - infer_latency) * request.control_hz
            chunk_steps = max(1.0, float(request.execution_horizon))
            slack_bonus = max(0.0, 1.0 - (slack_steps / chunk_steps))
            deficit_bonus = self._service_debt.get(robot_id, 0.0)

            per_robot_value = (
                useful_actions
                - self._tardiness_weight * tardiness_steps
                + self._slack_weight * slack_bonus
                + self._deficit_weight * deficit_bonus
            )

            total_latency_value += demand_rate * per_robot_value
            total_useful_actions += useful_actions
            total_tardiness += tardiness_steps
            total_slack_bonus += slack_bonus
            total_deficit_bonus += deficit_bonus
            total_demand += demand_rate
            if useful_actions > 0:
                feasible_count += 1

        useful_action_rate = total_latency_value / max(infer_latency, 1e-6)
        robot_ids = tuple(request.robot_id for request in batch)

        return (
            useful_action_rate,
            total_useful_actions,
            feasible_count,
            -total_tardiness,
            total_slack_bonus,
            total_deficit_bonus,
            total_demand,
            len(batch),
            robot_ids,
        )


class GreedyDeadlineScheduler(RequestScheduler):
    """Earliest-deadline-first: sort all pending requests by deadline."""

    def get_next_batches(self) -> list[list[SlotRequest]]:
        if self._batch_queue.qsize() > 0 or (candidates := self.schedulable_requests) == []:
            return []

        candidates_and_infer_deadlines = sorted(
            [
                (
                    slot_request,
                    self._deadlines.get(slot_request.robot_id, slot_request.deadline)
                    - self.latency_tracker.action_latency(slot_request.robot_id),
                )
                for slot_request in candidates
            ],
            key=lambda x: x[1],
        )
        _, earliest_infer_deadline = candidates_and_infer_deadlines[0]
        batch_size = self.get_largest_batch_size(earliest_infer_deadline)
        return [[x[0] for x in candidates_and_infer_deadlines[:batch_size]]]

    def get_largest_batch_size(self, infer_deadline: float) -> int:
        """Return the largest batch size whose profiled latency fits within the time remaining until deadline."""
        # we can assume inference starts right away because queue is empty
        time_remaining = infer_deadline - time.time()
        for batch_size in range(self._max_batch_size, 0, -1):
            if self.latency_tracker.infer_latency(batch_size) <= time_remaining:
                return batch_size
        return self.most_efficient_batch_size

    @property
    def most_efficient_batch_size(self) -> int:
        """Batch size with the best throughput (requests / ms)."""
        return max(range(1, self._max_batch_size + 1), key=lambda bs: bs / self.latency_tracker.infer_latency(bs))


class RelativeSlackEDFScheduler(GreedyDeadlineScheduler):
    """EDF variant that prioritizes the smallest slack relative to chunk duration.

    This preserves the basic greedy-deadline batching logic, but normalizes urgency
    by each robot's chunk time so short-horizon robots receive a bounded priority
    boost instead of being treated the same as long-horizon robots with equal
    absolute slack.
    """

    def get_next_batches(self) -> list[list[SlotRequest]]:
        if self._batch_queue.qsize() > 0 or (candidates := self.schedulable_requests) == []:
            return []

        now = time.time()
        most_urgent_request = min(
            candidates,
            key=lambda request: (
                self._relative_slack(request, batch_size=1, now=now),
                self._infer_deadline(request),
                request.robot_id,
            ),
        )
        batch_size = self._get_largest_batch_size_at(self._infer_deadline(most_urgent_request), now)
        candidates = sorted(
            candidates,
            key=lambda request: (
                self._relative_slack(request, batch_size=batch_size, now=now),
                self._infer_deadline(request),
                request.robot_id,
            ),
        )
        return [candidates[:batch_size]]

    def _infer_deadline(self, request: SlotRequest) -> float:
        return self._deadlines.get(request.robot_id, request.deadline) - self.latency_tracker.action_latency(
            request.robot_id
        )

    def _relative_slack(self, request: SlotRequest, batch_size: int, now: float) -> float:
        chunk_duration = max(request.execution_horizon / request.control_hz, 1e-6)
        slack = self._infer_deadline(request) - now - self.latency_tracker.infer_latency(batch_size)
        return slack / chunk_duration

    def _get_largest_batch_size_at(self, infer_deadline: float, now: float) -> int:
        time_remaining = infer_deadline - now
        for batch_size in range(self._max_batch_size, 0, -1):
            if self.latency_tracker.infer_latency(batch_size) <= time_remaining:
                return batch_size
        return self.most_efficient_batch_size


class DemandWeightedDebtEDFScheduler(RequestScheduler):
    """Prioritize robots with the most accumulated demand-weighted service debt.

    Debt grows at each robot's chunk demand rate (control_hz / execution_horizon) and
    decreases when the scheduler serves that robot. Candidate batches are scored by
    debt repaid minus demand-weighted tardiness, then tie-broken toward deadline-feasible
    and higher-demand batches. This keeps overdue robots recoverable instead of
    permanently excluding them once they miss an early deadline.
    """

    def __init__(
        self,
        batch_queue: mp.Queue,
        max_batch_size: int = 1,
    ):
        super().__init__(batch_queue, max_batch_size)
        self._service_debt: dict[str, float] = {}
        self._demand_rate: dict[str, float] = {}
        self._last_debt_update_time = time.time()

    def update(self, request: SlotRequest) -> None:
        self._advance_debts(time.time())
        super().update(request)
        self._demand_rate[request.robot_id] = self._compute_demand_rate(request)
        self._service_debt.setdefault(request.robot_id, 0.0)

    def get_next_batches(self) -> list[list[SlotRequest]]:
        if self._batch_queue.qsize() > 0 or (candidates := self.schedulable_requests) == []:
            return []

        now = time.time()
        self._advance_debts(now)
        candidates = sorted(candidates, key=lambda r: r.robot_id)

        best_batch: tuple[SlotRequest, ...] | None = None
        best_score: tuple | None = None
        for batch_size in range(1, min(self._max_batch_size, len(candidates)) + 1):
            for batch in itertools.combinations(candidates, batch_size):
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
        execution_horizon = max(1, request.execution_horizon)
        control_hz = max(float(request.control_hz), 1.0)
        return control_hz / execution_horizon

    def _advance_debts(self, now: float) -> None:
        elapsed = max(0.0, now - self._last_debt_update_time)
        if elapsed <= 0:
            return

        for robot_id, demand_rate in self._demand_rate.items():
            self._service_debt[robot_id] = self._service_debt.get(robot_id, 0.0) + elapsed * demand_rate
        self._last_debt_update_time = now

    def _charge_service(self, batch: tuple[SlotRequest, ...]) -> None:
        for request in batch:
            if request.is_padding:
                continue
            self._service_debt[request.robot_id] = max(0.0, self._service_debt.get(request.robot_id, 0.0) - 1.0)

    def _score_batch(self, batch: tuple[SlotRequest, ...], now: float) -> tuple:
        infer_latency = self.latency_tracker.infer_latency(len(batch))
        slacks = []
        debt_benefit = 0.0
        demand_weighted_tardiness = 0.0
        total_demand = 0.0

        for request in batch:
            robot_id = request.robot_id
            demand_rate = self._demand_rate.get(robot_id, self._compute_demand_rate(request))
            infer_deadline = self._deadlines.get(robot_id, request.deadline) - self.latency_tracker.action_latency(
                robot_id
            )
            slack = infer_deadline - now - infer_latency
            slacks.append(slack)
            debt_benefit += self._service_debt.get(robot_id, 0.0)
            total_demand += demand_rate
            if slack < 0:
                demand_weighted_tardiness += demand_rate * (-slack)

        feasible_count = sum(slack >= 0 for slack in slacks)
        min_slack = min(slacks)
        robot_ids = tuple(request.robot_id for request in batch)
        utility = debt_benefit - demand_weighted_tardiness

        return (
            utility,
            feasible_count,
            -min_slack,
            total_demand,
            len(batch),
            robot_ids,
        )


class RoundRobinScheduler(RequestScheduler):
    """Cycle through robots starting from the current pointer, fill to max_batch_size."""

    def __init__(
        self,
        batch_queue: mp.Queue,
        max_batch_size: int = 1,
    ):
        super().__init__(batch_queue, max_batch_size)
        self._rr_index: int = 0
        self._rr_robot_order: list[str] = []

    def update(self, request: SlotRequest) -> None:
        super().update(request)
        if request.robot_id not in self._rr_robot_order:
            self._rr_robot_order.append(request.robot_id)

    def get_next_batches(self) -> list[list[SlotRequest]]:
        if self._batch_queue.qsize() > 0:
            return []

        candidate_by_robot = {req.robot_id: req for req in self.schedulable_requests}
        n_robots = len(self._rr_robot_order)
        if not candidate_by_robot or n_robots == 0:
            return []

        batch: list[SlotRequest] = []
        idx = self._rr_index % n_robots
        for _ in range(n_robots):
            robot_id = self._rr_robot_order[idx]
            if robot_id in candidate_by_robot:
                batch.append(candidate_by_robot[robot_id])
            idx = (idx + 1) % n_robots
            if len(batch) == self._max_batch_size:
                break

        self._rr_index = idx
        return [batch] if batch else []

    def reset_robot(self, robot_id: str) -> None:
        super().reset_robot(robot_id)
        # FIXME: temporary hack to remove robot on reset
        if robot_id in self._rr_robot_order:
            removed_index = self._rr_robot_order.index(robot_id)
            self._rr_robot_order.remove(robot_id)
            if removed_index < self._rr_index:
                self._rr_index = max(0, self._rr_index - 1)


class RandomBatchScheduler(RequestScheduler):
    """Randomly select up to max_batch_size from pending requests."""

    def get_next_batches(self) -> list[list[SlotRequest]]:
        if self._batch_queue.qsize() > 0:
            return []

        candidates = list(self.schedulable_requests)
        if not candidates:
            return []

        k = min(self._max_batch_size, len(candidates))
        return [random.sample(candidates, random.randint(1, k))]
