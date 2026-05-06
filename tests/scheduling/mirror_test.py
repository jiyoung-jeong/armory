from __future__ import annotations

from dataclasses import replace

import pytest

from armory.scheduling.mirror import Mirror, Robot
from armory.serving.schemas import AckNotification, SlotRequest
from armory_client.messages import InferType
from tests.scheduling._cases import ALL_SCENARIOS, CONTROL_HZ, EPS, Scenario

ROBOT_ID = "test"


def _make_request(
    observation_step: int,
    action_index_start: int,
    request_timestamp: float,
    execution_horizon: int,
) -> SlotRequest:
    return SlotRequest(
        slot_index=0,
        robot_id=ROBOT_ID,
        request_id=observation_step,
        arrival_timestamp=request_timestamp,
        observation_step=observation_step,
        action_index_start=action_index_start,
        request_timestamp=request_timestamp,
        deadline=0.0,
        execution_horizon=execution_horizon,
        infer_type=InferType.SYNC,
        params=None,
        noise=None,
        control_hz=CONTROL_HZ,
    )


def _make_robot(scenario: Scenario) -> Robot:
    """Create a Robot seeded with the initial obs=0 control step and acked chunks."""
    horizon = scenario.chunks[0].execution_horizon
    robot = Robot(CONTROL_HZ, horizon)
    robot.step(_make_request(0, 0, 0.0, horizon))
    for chunk in scenario.chunks:
        robot.queue_chunk(chunk)
        robot.confirm_chunk(
            AckNotification(
                robot_id=ROBOT_ID,
                request_id=0,
                observation_step=chunk.observation_step,
                receive_time=chunk.arrival_time,
                server_send_time=0.0,
            )
        )
    return robot


@pytest.mark.parametrize("scenario", ALL_SCENARIOS, ids=lambda s: s.name)
def test_chunk_tracking(scenario: Scenario) -> None:
    robot = _make_robot(scenario)

    assert len(robot.chunks) == len(scenario.chunks)
    for actual, expected in zip(robot.chunks, scenario.chunks):
        assert actual.observation_step == expected.observation_step
        assert actual.action_index_start == expected.action_index_start
        assert actual.execution_horizon == expected.execution_horizon
        assert actual.arrival_time == pytest.approx(expected.arrival_time)
        assert actual.arrived is True

    assert robot.max_arrived_action_step == max(
        c.action_index_start + c.execution_horizon - 1 for c in scenario.chunks
    )


@pytest.mark.parametrize("scenario", ALL_SCENARIOS, ids=lambda s: s.name)
def test_step_forward(scenario: Scenario) -> None:
    robot = _make_robot(scenario)
    expected = scenario.control_steps()

    robot.step_forward(expected[-1].time)

    assert len(robot.steps) == len(expected)
    for actual, exp in zip(robot.steps, expected):
        assert actual.observation_step == exp.observation_step
        assert actual.time == pytest.approx(exp.time)
        assert actual.action_step == exp.action_step
        assert actual.next_action_step == exp.next_action_step


@pytest.mark.parametrize("scenario", ALL_SCENARIOS, ids=lambda s: s.name)
def test_deadline(scenario: Scenario) -> None:
    robot = _make_robot(scenario)
    expected_deadline = scenario.control_steps()[-1].time
    assert robot.deadline() == pytest.approx(expected_deadline)


def _seeded_mirror(scenario: Scenario) -> Mirror:
    horizon = scenario.chunks[0].execution_horizon
    mirror = Mirror()
    mirror.receive_request(_make_request(0, 0, 0.0, horizon), CONTROL_HZ)
    return mirror


def _drip_chunks(mirror: Mirror, scenario: Scenario) -> None:
    """Feed chunks to the mirror via fast_forward, advancing time to just past
    each chunk's arrival."""
    for chunk in scenario.chunks:
        mirror.fast_forward(
            chunk.arrival_time + EPS,
            [ROBOT_ID],
            [replace(chunk, arrived=True)],
        )


@pytest.mark.parametrize("scenario", ALL_SCENARIOS, ids=lambda s: s.name)
def test_mirror_fast_forward(scenario: Scenario) -> None:
    mirror = _seeded_mirror(scenario)
    expected = scenario.control_steps()

    _drip_chunks(mirror, scenario)
    # Advance past the last chunk's coverage to the scenario's final step.
    mirror.fast_forward(expected[-1].time, [], [])

    robot = mirror.robots[ROBOT_ID]
    assert len(robot.steps) == len(expected)
    for actual, exp in zip(robot.steps, expected):
        assert actual.observation_step == exp.observation_step
        assert actual.time == pytest.approx(exp.time)
        assert actual.action_step == exp.action_step
        assert actual.next_action_step == exp.next_action_step


@pytest.mark.parametrize("scenario", ALL_SCENARIOS, ids=lambda s: s.name)
def test_mirror_deadlines_after_fast_forward(scenario: Scenario) -> None:
    """Deadline should match the scenario's final step time even when the
    mirror has only been stepped partway through (deadline projects forward
    using virtual steps)."""
    mirror = _seeded_mirror(scenario)
    _drip_chunks(mirror, scenario)

    expected = scenario.control_steps()[-1].time
    assert mirror.deadlines() == {ROBOT_ID: pytest.approx(expected)}


def test_mirror_fast_forward_multiple_robots() -> None:
    """fast_forward routes chunks per-robot but advances all robots' clocks."""
    from tests.scheduling._cases import LONG_RUN

    rids = ["a", "b"]
    mirror = Mirror()
    horizon = LONG_RUN.chunks[0].execution_horizon
    for rid in rids:
        mirror.receive_request(replace(_make_request(0, 0, 0.0, horizon), robot_id=rid), CONTROL_HZ)

    for chunk in LONG_RUN.chunks:
        arrived = replace(chunk, arrived=True)
        # Send the same chunk to both robots so they stay in lockstep.
        mirror.fast_forward(chunk.arrival_time + EPS, rids, [arrived, arrived])

    expected_deadline = LONG_RUN.control_steps()[-1].time
    deadlines = mirror.deadlines()
    assert deadlines == {rid: pytest.approx(expected_deadline) for rid in rids}


def test_mirror_fast_forward_advances_robot_without_new_chunk() -> None:
    """A robot not named in the chunk list still gets stepped forward."""
    from tests.scheduling._cases import LONG_RUN

    mirror = Mirror()
    horizon = LONG_RUN.chunks[0].execution_horizon
    for rid in ["a", "b"]:
        mirror.receive_request(replace(_make_request(0, 0, 0.0, horizon), robot_id=rid), CONTROL_HZ)
    # Only "a" gets the chunk, but both should advance.
    mirror.fast_forward(
        LONG_RUN.chunks[0].arrival_time + EPS,
        ["a"],
        [replace(LONG_RUN.chunks[0], arrived=True)],
    )

    assert mirror.robots["a"].steps[-1].observation_step == 2
    assert mirror.robots["b"].steps[-1].observation_step == 2
    # "b" has no chunks, so its action_step should be None on every step.
    assert all(s.action_step is None for s in mirror.robots["b"].steps)
