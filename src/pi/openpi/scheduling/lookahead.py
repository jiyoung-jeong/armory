from __future__ import annotations

from functools import cache
import math
import multiprocessing as mp
import time

from openpi.scheduling import RequestScheduler
from openpi.serving.schemas import SlotRequest


class LookaheadScheduler(RequestScheduler):
    def __init__(
        self,
        batch_queue: mp.Queue,
        max_batch_size: int = 1,
        batch_profile: dict[int, float] | None = None,
        *,
        horizon_ms: int = 1000,
        timestep_ms: int = 50,
        action_horizon_steps: int = 10,
        control_hz: int = 20,
    ) -> None:
        super().__init__(batch_queue, max_batch_size, batch_profile)
        assert timestep_ms > 0, "timestep_ms must be positive"
        assert horizon_ms > 0, "horizon_ms must be positive"
        assert action_horizon_steps > 0, "action_horizon_steps must be positive"
        assert control_hz > 0, "control_hz must be positive"

        self._timestep_ms = float(timestep_ms)
        self._horizon_ticks = self._to_ticks(horizon_ms)
        self._chunk_duration_s = action_horizon_steps / control_hz
        self._chunk_ticks = self._to_ticks(self._chunk_duration_s * 1000.0)
        self._latency_ticks = {
            batch_size: self._to_ticks(self._latency_ms(batch_size))
            for batch_size in range(1, self._max_batch_size + 1)
        }
        self._latency_s = {
            batch_size: latency_ticks * self._timestep_ms / 1000.0
            for batch_size, latency_ticks in self._latency_ticks.items()
        }

        self._server_available_at: float = 0.0
        self._predicted_valid_until: dict[str, float] = {}

    def schedule(self) -> None:
        """Dispatch the best batch and update predicted in-flight timing state."""
        batches = self.get_next_batches()
        now = time.time()
        for batch in batches:
            batch_size = len(batch)
            start_time = max(now, self._server_available_at)
            finish_time = start_time + self._latency_s[batch_size]
            self._server_available_at = finish_time

            for request in batch:
                self._deadlines[request.robot_id] = request.deadline
                self._latest_scheduled_requests[request.robot_id] = request
                self._predicted_valid_until[request.robot_id] = finish_time + self._chunk_duration_s

            self._batch_queue.put_nowait(batch)
            now = finish_time

    def get_next_batches(self) -> list[list[SlotRequest]]:
        now = time.time()
        self._prune_predictions(now)

        if self._batch_queue.qsize() > 0 or now < self._server_available_at:
            return []

        schedulable = self._get_schedulable_requests()
        if not schedulable:
            return []

        request_by_robot = {request.robot_id: request for request in schedulable}
        active_robot_ids = sorted(set(self._latest_requests) | set(self._deadlines) | set(self._predicted_valid_until))
        if not active_robot_ids:
            active_robot_ids = sorted(request_by_robot)

        initial_state = tuple(self._remaining_ticks(robot_id, now) for robot_id in active_robot_ids)
        initial_candidates = self._candidate_prefixes(
            robot_ids=active_robot_ids,
            valid_until=initial_state,
            eligible_robot_ids=set(request_by_robot),
        )
        if not initial_candidates:
            return []

        @cache
        def dfs(current_tick: int, valid_until: tuple[int, ...]) -> int:
            if current_tick >= self._horizon_ticks:
                return 0

            best_cost = math.inf
            for candidate in self._candidate_prefixes(active_robot_ids, valid_until):
                arrival_tick = min(self._horizon_ticks, current_tick + self._latency_ticks[len(candidate)])
                interval_cost = self._interval_starvation_cost(valid_until, current_tick, arrival_tick)
                next_state = self._apply_batch(valid_until, candidate, arrival_tick)
                total_cost = interval_cost + dfs(arrival_tick, next_state)
                best_cost = min(best_cost, total_cost)

            if best_cost is math.inf:
                return self._interval_starvation_cost(valid_until, current_tick, self._horizon_ticks)

            return best_cost

        best_candidate: tuple[int, ...] | None = None
        best_cost = math.inf
        for candidate in initial_candidates:
            arrival_tick = min(self._horizon_ticks, self._latency_ticks[len(candidate)])
            interval_cost = self._interval_starvation_cost(initial_state, 0, arrival_tick)
            next_state = self._apply_batch(initial_state, candidate, arrival_tick)
            total_cost = interval_cost + dfs(arrival_tick, next_state)
            if total_cost < best_cost or (
                total_cost == best_cost and self._prefer_candidate(candidate, best_candidate, active_robot_ids)
            ):
                best_cost = total_cost
                best_candidate = candidate

        if best_candidate is None:
            return []

        return [[request_by_robot[active_robot_ids[index]] for index in best_candidate]]

    def reset_robot(self, robot_id: str) -> None:
        super().reset_robot(robot_id)
        self._predicted_valid_until.pop(robot_id, None)

    def _candidate_prefixes(
        self,
        robot_ids: list[str],
        valid_until: tuple[int, ...],
        eligible_robot_ids: set[str] | None = None,
    ) -> list[tuple[int, ...]]:
        eligible_indices = [
            index
            for index, robot_id in enumerate(robot_ids)
            if eligible_robot_ids is None or robot_id in eligible_robot_ids
        ]
        ordered = sorted(eligible_indices, key=lambda index: (valid_until[index], robot_ids[index]))
        max_size = min(self._max_batch_size, len(ordered))
        return [tuple(ordered[:batch_size]) for batch_size in range(1, max_size + 1)]

    def _apply_batch(
        self,
        valid_until: tuple[int, ...],
        candidate: tuple[int, ...],
        arrival_tick: int,
    ) -> tuple[int, ...]:
        updated = list(valid_until)
        refreshed_until = min(self._horizon_ticks + self._chunk_ticks, arrival_tick + self._chunk_ticks)
        for index in candidate:
            updated[index] = refreshed_until
        return tuple(updated)

    def _interval_starvation_cost(self, valid_until: tuple[int, ...], start_tick: int, end_tick: int) -> int:
        if end_tick <= start_tick:
            return 0
        return sum(max(0, end_tick - max(start_tick, expiry_tick)) for expiry_tick in valid_until)

    def _remaining_ticks(self, robot_id: str, now: float) -> int:
        valid_until = max(
            self._deadlines.get(robot_id, 0.0),
            self._predicted_valid_until.get(robot_id, 0.0),
            self._latest_requests.get(robot_id, None).deadline if robot_id in self._latest_requests else 0.0,
        )
        if valid_until <= now:
            return 0
        return min(self._horizon_ticks + self._chunk_ticks, self._to_ticks((valid_until - now) * 1000.0))

    def _latency_ms(self, batch_size: int) -> float:
        latency_ms = self._batch_profile_ms.get(batch_size)
        if latency_ms is None:
            raise ValueError(f"Missing batch profile entry for batch_size={batch_size}")
        return latency_ms

    def _to_ticks(self, duration_ms: float) -> int:
        return max(1, math.ceil(duration_ms / self._timestep_ms))

    def _prune_predictions(self, now: float) -> None:
        self._predicted_valid_until = {
            robot_id: valid_until for robot_id, valid_until in self._predicted_valid_until.items() if valid_until > now
        }
        self._server_available_at = max(now, self._server_available_at)

    def _prefer_candidate(
        self,
        candidate: tuple[int, ...],
        best_candidate: tuple[int, ...] | None,
        robot_ids: list[str],
    ) -> bool:
        if best_candidate is None:
            return True
        if len(candidate) != len(best_candidate):
            return len(candidate) > len(best_candidate)
        candidate_robot_ids = tuple(robot_ids[index] for index in candidate)
        best_robot_ids = tuple(robot_ids[index] for index in best_candidate)
        return candidate_robot_ids < best_robot_ids
