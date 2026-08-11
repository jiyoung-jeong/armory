from __future__ import annotations

import math
from collections import Counter
from dataclasses import replace
from typing import Any

import pytest

from armory.scheduling import weighted_heuristics
from armory.scheduling.baselines import RoundRobinScheduler
from armory.scheduling.weighted_heuristics import (
    DeficitRoundRobinScheduler,
    WeightedEDFScheduler,
    WeightedRoundRobinScheduler,
)
from armory.serving.protocol import SchedulerConfig
from armory.serving.rtc import InferType
from armory.serving.schemas import SlotRequest


class _FakeLatencyTracker:
    def __init__(self, infer_latency: dict[int, float]) -> None:
        self._infer_latency = infer_latency
        self.cleared = False

    def update_obs(
        self, robot_id: str, server_arrival_time: float, client_request_time: float
    ) -> None:
        del robot_id, server_arrival_time, client_request_time

    def infer_latency(self, batch_size: int) -> float:
        return self._infer_latency[batch_size]

    def action_latency(self, robot_id: str) -> float:
        del robot_id
        return 0.0

    def clear_all(self) -> None:
        self.cleared = True


class _FakeMirror:
    def __init__(self) -> None:
        self.in_flight_batches_count = 0
        self._deadlines: dict[str, float] = {}
        self.reset_robots: list[str] = []
        self.clear_count = 0

    def receive_request(self, request: SlotRequest) -> bool:
        self._deadlines[request.robot_id] = request.deadline
        return True

    def deadlines(self) -> dict[str, float]:
        return dict(self._deadlines)

    def reset_robot(self, robot_id: str) -> None:
        self.reset_robots.append(robot_id)
        self._deadlines.pop(robot_id, None)

    def clear_all(self) -> None:
        self.clear_count += 1
        self._deadlines.clear()


def _request(
    robot_id: str,
    *,
    deadline: float = 1_000.0,
    weight: float = 1.0,
    control_hz: float = 1.0,
    max_execution_horizon: int = 1,
) -> SlotRequest:
    return SlotRequest(
        slot_index=0,
        robot_id=robot_id,
        request_id=0,
        arrival_timestamp=0.0,
        observation_step=0,
        action_index_start=0,
        request_timestamp=0.0,
        deadline=deadline,
        min_execution_horizon=1,
        max_execution_horizon=max_execution_horizon,
        infer_type=InferType.SYNC,
        params=None,
        noise=None,
        control_hz=control_hz,
        weight=weight,
    )


def _scheduler(
    scheduler_type: (
        type[WeightedEDFScheduler]
        | type[WeightedRoundRobinScheduler]
        | type[DeficitRoundRobinScheduler]
    ),
    *,
    max_batch_size: int,
    infer_latency: dict[int, float] | None = None,
) -> WeightedEDFScheduler | WeightedRoundRobinScheduler | DeficitRoundRobinScheduler:
    scheduler = scheduler_type(SchedulerConfig(), object(), max_batch_size=max_batch_size)
    scheduler.mirror = _FakeMirror()  # type: ignore[assignment]
    scheduler.latency_tracker = _FakeLatencyTracker(  # type: ignore[assignment]
        infer_latency or {size: 1.0 for size in range(1, max_batch_size + 1)}
    )
    return scheduler


def _choose(
    scheduler: WeightedEDFScheduler | WeightedRoundRobinScheduler | DeficitRoundRobinScheduler,
    candidates: list[SlotRequest],
) -> tuple[list[str], dict[str, Any]]:
    mirror = scheduler.mirror
    assert isinstance(mirror, _FakeMirror)
    mirror._deadlines.update({request.robot_id: request.deadline for request in candidates})
    batches, notes = scheduler.get_next_batches(candidates)
    assert len(batches) == 1
    return [request.robot_id for request in batches[0]], notes


def _register(
    scheduler: WeightedRoundRobinScheduler | DeficitRoundRobinScheduler,
    candidates: list[SlotRequest],
) -> None:
    for request in candidates:
        scheduler.update(request)


def test_weighted_edf_selects_only_edf_prefixes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(weighted_heuristics.time, "time", lambda: 0.0)
    scheduler = _scheduler(WeightedEDFScheduler, max_batch_size=1)
    urgent = _request("urgent", deadline=10.0, weight=0.01)
    valuable = _request("valuable", deadline=20.0, weight=1_000.0)

    chosen, _ = _choose(scheduler, [valuable, urgent])

    assert chosen == ["urgent"]


def test_weighted_edf_explicit_weights_change_the_chosen_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(weighted_heuristics.time, "time", lambda: 0.0)
    urgent = _request("a", deadline=100.0, weight=3.0)
    low_weight = _request("b", deadline=101.0, weight=1.0)
    high_weight = replace(low_weight, weight=10.0)

    low_scheduler = _scheduler(
        WeightedEDFScheduler,
        max_batch_size=2,
        infer_latency={1: 1.0, 2: 1.0},
    )
    high_scheduler = _scheduler(
        WeightedEDFScheduler,
        max_batch_size=2,
        infer_latency={1: 1.0, 2: 1.0},
    )

    assert _choose(low_scheduler, [low_weight, urgent])[0] == ["a"]
    assert _choose(high_scheduler, [high_weight, urgent])[0] == ["a", "b"]


def test_weighted_edf_prefers_a_feasible_prefix_over_any_infeasible_priority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(weighted_heuristics.time, "time", lambda: 100.0)
    scheduler = _scheduler(
        WeightedEDFScheduler,
        max_batch_size=2,
        infer_latency={1: 1.0, 2: 3.0},
    )
    urgent = _request("a", deadline=102.0)
    enormous_weight = _request("b", deadline=200.0, weight=1_000_000.0)

    chosen, _ = _choose(scheduler, [enormous_weight, urgent])

    assert chosen == ["a"]


def test_weighted_edf_accrues_unweighted_demand_debt_and_charges_service(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = [100.0]
    monkeypatch.setattr(weighted_heuristics.time, "time", lambda: now[0])
    scheduler = _scheduler(
        WeightedEDFScheduler,
        max_batch_size=1,
        infer_latency={1: 0.1},
    )
    served = _request(
        "a",
        deadline=200.0,
        weight=9.0,
        control_hz=4.0,
        max_execution_horizon=2,
    )
    waiting = _request(
        "b",
        deadline=201.0,
        weight=1.0,
        control_hz=4.0,
        max_execution_horizon=2,
    )
    scheduler.update(served)
    scheduler.update(waiting)

    now[0] = 102.0
    chosen, notes = _choose(scheduler, [waiting, served])

    assert chosen == ["a"]
    assert notes["demand_rate"] == {"a": 2.0, "b": 2.0}
    assert notes["service_debt"] == {"a": pytest.approx(3.0), "b": pytest.approx(4.0)}


def test_weighted_edf_resets_debt_state(monkeypatch: pytest.MonkeyPatch) -> None:
    now = [10.0]
    monkeypatch.setattr(weighted_heuristics.time, "time", lambda: now[0])
    scheduler = _scheduler(WeightedEDFScheduler, max_batch_size=1)
    scheduler.update(_request("a"))
    scheduler.update(_request("b"))
    now[0] = 12.0
    scheduler._advance_debts(now[0])

    scheduler.reset_robot("a")

    assert "a" not in scheduler._service_debt
    assert "a" not in scheduler._demand_rate
    assert scheduler.mirror.reset_robots == ["a"]  # type: ignore[attr-defined]

    now[0] = 15.0
    scheduler.reset_all()

    assert scheduler._service_debt == {}
    assert scheduler._demand_rate == {}
    assert scheduler._last_advance == 15.0
    assert scheduler.mirror.clear_count == 1  # type: ignore[attr-defined]
    assert scheduler.latency_tracker.cleared  # type: ignore[attr-defined]


def test_weighted_round_robin_equal_weights_match_round_robin_batches() -> None:
    scheduler = _scheduler(WeightedRoundRobinScheduler, max_batch_size=2)
    candidates = [_request("c"), _request("b"), _request("a")]
    _register(scheduler, candidates)

    batches = [_choose(scheduler, candidates)[0] for _ in range(3)]

    assert isinstance(scheduler, RoundRobinScheduler)
    assert batches == [["c", "b"], ["a", "c"], ["b", "a"]]


def test_weighted_round_robin_tracks_long_run_weight_ratio() -> None:
    scheduler = _scheduler(WeightedRoundRobinScheduler, max_batch_size=1)
    candidates = [_request("a", weight=3.0), _request("b", weight=2.0), _request("c")]
    _register(scheduler, candidates)

    counts = Counter(_choose(scheduler, candidates)[0][0] for _ in range(60))

    assert counts == {"a": 30, "b": 20, "c": 10}


def test_weighted_round_robin_follows_conventional_counter_cycle() -> None:
    scheduler = _scheduler(WeightedRoundRobinScheduler, max_batch_size=1)
    candidates = [
        _request("a", weight=2.0),
        _request("b"),
        _request("c", weight=3.0),
    ]
    _register(scheduler, candidates)

    chosen = [_choose(scheduler, candidates)[0][0] for _ in range(12)]

    assert chosen == ["a", "b", "c", "a", "c", "c"] * 2


def test_weighted_round_robin_stops_before_repeating_a_robot() -> None:
    scheduler = _scheduler(WeightedRoundRobinScheduler, max_batch_size=2)
    candidates = [
        _request("a", weight=2.0),
        _request("b"),
        _request("c", weight=3.0),
    ]
    _register(scheduler, candidates)

    batches = [_choose(scheduler, candidates)[0] for _ in range(6)]

    assert batches == [
        ["a", "b"],
        ["c", "a"],
        ["c"],
        ["c", "a"],
        ["b", "c"],
        ["a", "c"],
    ]
    assert all(len(batch) == len(set(batch)) for batch in batches)


def test_weighted_round_robin_fills_from_available_unique_candidates() -> None:
    scheduler = _scheduler(WeightedRoundRobinScheduler, max_batch_size=4)
    candidates = [_request("a", weight=3.0), _request("c"), _request("d")]
    _register(scheduler, candidates)

    chosen, notes = _choose(scheduler, candidates)

    assert len(chosen) == len(set(chosen)) == 3
    assert set(chosen) == {"a", "c", "d"}
    assert notes["rule"] == "weighted_round_robin"


def test_weighted_round_robin_preserves_skipped_quota_during_active_cycle() -> None:
    scheduler = _scheduler(WeightedRoundRobinScheduler, max_batch_size=2)
    a = _request("a", weight=2.0)
    b = _request("b")
    c = _request("c", weight=3.0)
    _register(scheduler, [a, b, c])

    assert _choose(scheduler, [a, c])[0] == ["a", "c"]
    assert _choose(scheduler, [a, b, c])[0] == ["a", "b"]


def test_weighted_round_robin_empty_queues_do_not_hold_cycle_open() -> None:
    scheduler = _scheduler(WeightedRoundRobinScheduler, max_batch_size=1)
    a = _request("a", weight=2.0)
    b = _request("b")
    _register(scheduler, [a, b])

    assert _choose(scheduler, [b])[0] == ["b"]
    assert _choose(scheduler, [a, b])[0] == ["a"]
    assert _choose(scheduler, [a, b])[0] == ["b"]


@pytest.mark.parametrize("weight", [0.0, -1.0, 1.5, math.nan, math.inf, -math.inf])
def test_weighted_round_robin_rejects_invalid_weights_without_mutating_state(
    weight: float,
) -> None:
    scheduler = _scheduler(WeightedRoundRobinScheduler, max_batch_size=2)
    valid = _request("valid")
    _register(scheduler, [valid])
    state_before = (
        dict(scheduler._weights),
        dict(scheduler._remaining_quota),
        scheduler._rr_index,
    )

    with pytest.raises(ValueError, match="weight"):
        scheduler.get_next_batches([valid, _request("invalid", weight=weight)])

    assert (scheduler._weights, scheduler._remaining_quota, scheduler._rr_index) == state_before


def test_weighted_round_robin_rejects_weight_changes() -> None:
    scheduler = _scheduler(WeightedRoundRobinScheduler, max_batch_size=1)
    request = _request("a", weight=1.0)
    _register(scheduler, [request])

    with pytest.raises(ValueError, match="changed"):
        scheduler.update(replace(request, weight=2.0))


def test_weighted_round_robin_reset_robot_and_reset_all_clear_state() -> None:
    scheduler = _scheduler(WeightedRoundRobinScheduler, max_batch_size=1)
    a = _request("a")
    b = _request("b")
    _register(scheduler, [a, b])
    _choose(scheduler, [a, b])
    _choose(scheduler, [a, b])

    scheduler.reset_robot("a")

    assert "a" not in scheduler._weights
    assert "a" not in scheduler._remaining_quota
    assert scheduler._rr_robot_order == ["b"]
    assert scheduler.mirror.reset_robots == ["a"]  # type: ignore[attr-defined]

    scheduler.reset_all()

    assert scheduler._weights == {}
    assert scheduler._remaining_quota == {}
    assert scheduler._rr_robot_order == []
    assert scheduler._rr_index == 0
    assert scheduler.mirror.clear_count == 1  # type: ignore[attr-defined]
    assert scheduler.latency_tracker.cleared  # type: ignore[attr-defined]
    _register(scheduler, [b, a])
    assert _choose(scheduler, [b, a])[0] == ["b"]


def test_weighted_round_robin_busy_pass_does_not_change_state_and_empty_resets_quota() -> None:
    scheduler = _scheduler(WeightedRoundRobinScheduler, max_batch_size=1)
    a = _request("a", weight=2.0)
    _register(scheduler, [a])
    _choose(scheduler, [a])
    state_before = (dict(scheduler._remaining_quota), scheduler._rr_index)

    scheduler.mirror.in_flight_batches_count = 1
    assert scheduler.get_next_batches([a]) == (
        [],
        {"reason": "server_busy"},
    )
    assert (scheduler._remaining_quota, scheduler._rr_index) == state_before

    scheduler.mirror.in_flight_batches_count = 0
    assert scheduler.get_next_batches([]) == ([], {"reason": "no_candidates"})

    assert scheduler._remaining_quota == {"a": 2}
    assert scheduler._rr_index == state_before[1]


def test_deficit_round_robin_equal_coverage_matches_round_robin_batches() -> None:
    scheduler = _scheduler(DeficitRoundRobinScheduler, max_batch_size=2)
    candidates = [
        _request("c", control_hz=10.0, max_execution_horizon=2),
        _request("b", control_hz=20.0, max_execution_horizon=4),
        _request("a", control_hz=5.0, max_execution_horizon=1, weight=1_000.0),
    ]
    _register(scheduler, candidates)

    batches = [_choose(scheduler, candidates)[0] for _ in range(3)]

    assert isinstance(scheduler, RoundRobinScheduler)
    assert batches == [["c", "b"], ["a", "c"], ["b", "a"]]


def test_deficit_round_robin_balances_action_coverage_and_ignores_weights() -> None:
    scheduler = _scheduler(DeficitRoundRobinScheduler, max_batch_size=1)
    candidates = [
        _request("a", control_hz=5.0, max_execution_horizon=1, weight=0.01),
        _request("b", control_hz=10.0, max_execution_horizon=3, weight=1_000.0),
        _request("c", control_hz=10.0, max_execution_horizon=6, weight=7.0),
    ]
    _register(scheduler, candidates)

    counts = Counter(_choose(scheduler, candidates)[0][0] for _ in range(60))

    assert counts == {"a": 30, "b": 20, "c": 10}


def test_deficit_round_robin_preserves_serial_order_with_variable_batches() -> None:
    scheduler = _scheduler(DeficitRoundRobinScheduler, max_batch_size=2)
    candidates = [
        _request("a"),
        _request("b", max_execution_horizon=3),
    ]
    _register(scheduler, candidates)

    batches = [_choose(scheduler, candidates)[0] for _ in range(6)]

    assert batches == [["a"], ["a"], ["a", "b"]] * 2
    assert all(len(batch) == len(set(batch)) for batch in batches)


def test_deficit_round_robin_uses_action_coverage_cost() -> None:
    scheduler = _scheduler(DeficitRoundRobinScheduler, max_batch_size=4)
    candidates = [
        _request("a", control_hz=20.0, max_execution_horizon=6),
        _request("b", control_hz=20.0, max_execution_horizon=10),
        _request("c", control_hz=10.0, max_execution_horizon=10),
    ]
    _register(scheduler, candidates)

    chosen, notes = _choose(scheduler, candidates)

    assert chosen == ["a"]
    assert notes["rule"] == "deficit_round_robin"
    assert notes["quantum"] == pytest.approx(0.3)
    assert notes["stopped_before_repeat"] == "a"
    assert notes["action_coverage_cost"] == {
        "a": pytest.approx(0.3),
        "b": pytest.approx(0.5),
        "c": pytest.approx(1.0),
    }


def test_deficit_round_robin_does_not_accrue_for_unavailable_robots() -> None:
    scheduler = _scheduler(DeficitRoundRobinScheduler, max_batch_size=1)
    a = _request("a", control_hz=20.0, max_execution_horizon=6)
    b = _request("b", control_hz=20.0, max_execution_horizon=10)
    _register(scheduler, [a, b])

    _choose(scheduler, [a, b])
    deficit_before = scheduler._deficit["b"]
    for _ in range(4):
        assert _choose(scheduler, [a])[0] == ["a"]

    assert scheduler._deficit["b"] == deficit_before


@pytest.mark.parametrize(
    ("control_hz", "horizon", "match"),
    [
        (0.0, 1, "control_hz"),
        (-1.0, 1, "control_hz"),
        (math.inf, 1, "control_hz"),
        (1.0, 0, "max_execution_horizon"),
        (1.0, -1, "max_execution_horizon"),
        (1e-308, 10**308, "action-coverage cost"),
    ],
)
def test_deficit_round_robin_rejects_invalid_coverage_without_mutating_state(
    control_hz: float, horizon: int, match: str
) -> None:
    scheduler = _scheduler(DeficitRoundRobinScheduler, max_batch_size=1)
    valid = _request("valid")
    _register(scheduler, [valid])
    state_before = (dict(scheduler._deficit), scheduler._rr_index)

    with pytest.raises(ValueError, match=match):
        scheduler.get_next_batches(
            [_request("invalid", control_hz=control_hz, max_execution_horizon=horizon)]
        )

    assert (scheduler._deficit, scheduler._rr_index) == state_before


def test_deficit_round_robin_reset_and_idle_paths_preserve_state() -> None:
    scheduler = _scheduler(DeficitRoundRobinScheduler, max_batch_size=1)
    a = _request("a", control_hz=20.0, max_execution_horizon=6)
    b = _request("b", control_hz=20.0, max_execution_horizon=10)
    _register(scheduler, [a, b])
    _choose(scheduler, [a, b])

    state_before = (dict(scheduler._deficit), dict(scheduler._request_cost), scheduler._rr_index)
    scheduler.mirror.in_flight_batches_count = 1
    assert scheduler.get_next_batches([a]) == ([], {"reason": "server_busy"})
    scheduler.mirror.in_flight_batches_count = 0
    assert scheduler.get_next_batches([]) == ([], {"reason": "no_candidates"})
    assert (scheduler._deficit, scheduler._request_cost, scheduler._rr_index) == state_before

    scheduler.reset_robot("a")
    assert "a" not in scheduler._deficit
    assert "a" not in scheduler._request_cost
    assert scheduler._rr_robot_order == ["b"]

    scheduler.reset_all()
    assert scheduler._deficit == {}
    assert scheduler._request_cost == {}
    assert scheduler._rr_robot_order == []
    assert scheduler._rr_index == 0
