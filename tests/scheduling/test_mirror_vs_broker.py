"""Comparison tests: replay broker traces into ``Mirror`` and assert agreement.

For each canonical scenario we run two checks:

1. **State tracking** — feed all events (SlotRequest / schedule_pending_chunk /
   AckNotification) into the mirror in time order. For each broker observation,
   the mirror's ``ControlStep.action_step`` should equal the broker's
   ``next_action_step`` at that obs (these are the same number — the
   ``action_start_step`` the broker put on the wire).

2. **Forward simulation** — split the trace after scripted chunks are known,
   feed only the pre-split events to the mirror, then call
   ``Mirror.fast_forward`` to project to the end of the scenario. The mirror's
   per-step ``action_step`` (including ``None`` for pauses) should match the
   broker's recorded actions.
"""

from __future__ import annotations

import pytest

from armory.scheduling.mirror import Mirror
from tests.scheduling.driver import TraceEvent, TraceRecord, run_scenario
from tests.scheduling.fixtures import (  # noqa: F401  (used as pytest fixtures)
    ALL_SCENARIOS,
    back_to_back,
    late_chunk_with_pauses,
    no_overlap,
    overriding_chunk,
    queue_exhaustion,
)

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _replay(mirror: Mirror, events: list[TraceEvent], control_hz: float) -> None:
    for ev in events:
        if ev.kind == "obs":
            mirror.receive_request(ev.payload, control_hz=control_hz)
        elif ev.kind == "schedule":
            chunk, robot_id = ev.payload
            mirror.schedule_pending_chunk(robot_id, chunk)
        elif ev.kind == "ack":
            mirror.receive_response(ev.payload)
        else:  # pragma: no cover
            raise AssertionError(f"unknown event kind {ev.kind}")


def _apply_event(mirror: Mirror, ev: TraceEvent, control_hz: float) -> None:
    _replay(mirror, [ev], control_hz)


def _control_step_by_obs(mirror: Mirror, robot_id: str, obs_step: int):
    for s in mirror.robots[robot_id].steps:
        if s.observation_step == obs_step:
            return s
    raise AssertionError(f"no control step for obs={obs_step}")


# ---------------------------------------------------------------------------
# state tracking
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("scenario_name", ALL_SCENARIOS)
def test_state_tracking_matches_broker(scenario_name: str, request: pytest.FixtureRequest):
    spec = request.getfixturevalue(scenario_name)
    trace: TraceRecord = run_scenario(spec)

    mirror = Mirror()
    _replay(mirror, trace.events, control_hz=spec.control_hz)

    for step_record in trace.per_step_actions:
        cs = _control_step_by_obs(mirror, spec.robot_id, step_record.observation_step)
        # mirror.action_step == request.action_start_step == core._next_action_step
        # at the moment the broker sent that observation. That value is what we
        # store as ``next_action_step`` on the trace step.
        assert cs.action_step == step_record.next_action_step, (
            f"obs={step_record.observation_step}: mirror.action_step={cs.action_step} "
            f"vs broker.next_action_step={step_record.next_action_step}"
        )
        assert cs.next_action_step == step_record.next_action_step + 1


@pytest.mark.parametrize("scenario_name", ALL_SCENARIOS)
def test_deadlines_match_broker_when_no_future_chunk_is_pending(
    scenario_name: str, request: pytest.FixtureRequest
):
    spec = request.getfixturevalue(scenario_name)
    trace: TraceRecord = run_scenario(spec)

    mirror = Mirror()
    for ev in trace.events:
        _apply_event(mirror, ev, control_hz=spec.control_hz)
        if ev.kind != "obs":
            continue

        slot_request = ev.payload
        has_pending_future_chunk = any(
            chunk.arrival_time > slot_request.request_timestamp
            for chunk in mirror.robots[spec.robot_id].chunks
        )
        if has_pending_future_chunk:
            continue

        assert mirror.deadlines()[spec.robot_id] == pytest.approx(slot_request.deadline), (
            f"obs={slot_request.observation_step}: "
            f"mirror.deadline={mirror.deadlines()[spec.robot_id]} "
            f"vs broker.deadline={slot_request.deadline}"
        )


def test_deadline_can_look_ahead_through_pending_future_chunk():
    trace: TraceRecord = run_scenario(late_chunk_with_pauses)

    mirror = Mirror()
    for ev in trace.events:
        _apply_event(mirror, ev, control_hz=late_chunk_with_pauses.control_hz)
        if ev.kind != "obs":
            continue

        slot_request = ev.payload
        if slot_request.observation_step != 4:
            continue

        assert slot_request.deadline == pytest.approx(slot_request.request_timestamp)
        assert mirror.deadlines()[late_chunk_with_pauses.robot_id] > slot_request.deadline
        return

    raise AssertionError("late_chunk_with_pauses did not produce obs=4")


# ---------------------------------------------------------------------------
# forward simulation
# ---------------------------------------------------------------------------


def _split_indices(trace: TraceRecord) -> range:
    """Pick split points after all scripted chunks have been scheduled.

    Earlier splits cannot reproduce the broker trace without feeding future
    server work into the mirror, because the mirror has no knowledge of chunks
    whose triggering observations have not happened yet.
    """
    n = len(trace.per_step_actions)
    assert n >= 2
    latest_schedule_time = max(
        (ev.time for ev in trace.events if ev.kind == "schedule"),
        default=trace.per_step_actions[0].time,
    )
    start = next(
        i
        for i, step_record in enumerate(trace.per_step_actions)
        if step_record.time >= latest_schedule_time
    )
    return range(start, n - 1)


@pytest.mark.parametrize("scenario_name", ALL_SCENARIOS)
def test_fast_forward_matches_broker(scenario_name: str, request: pytest.FixtureRequest):
    spec = request.getfixturevalue(scenario_name)
    trace: TraceRecord = run_scenario(spec)

    for split_obs in _split_indices(trace):
        split_time = trace.per_step_actions[split_obs].time

        mirror = Mirror()
        pre_split = [ev for ev in trace.events if ev.time <= split_time]
        _replay(mirror, pre_split, control_hz=spec.control_hz)

        end_time = trace.per_step_actions[-1].time
        mirror.fast_forward(time=end_time, robot_ids=[], chunks=[])

        for step_record in trace.per_step_actions[split_obs + 1 :]:
            cs = _control_step_by_obs(mirror, spec.robot_id, step_record.observation_step)
            assert cs.action_step == step_record.action_step, (
                f"split_obs={split_obs} obs={step_record.observation_step}: "
                f"mirror.action_step={cs.action_step} vs broker.action_step={step_record.action_step}"
            )
