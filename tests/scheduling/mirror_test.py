from __future__ import annotations

import pytest

from armory.scheduling.mirror import Robot
from armory.serving.schemas import AckNotification, SlotRequest
from armory_client.messages import InferType
from tests.scheduling._cases import ALL_SCENARIOS, CONTROL_HZ, Scenario

ROBOT_ID = "test"


def _make_request(
    observation_step: int,
    action_start_step: int,
    request_timestamp: float,
    execution_horizon: int,
) -> SlotRequest:
    return SlotRequest(
        slot_index=0,
        robot_id=ROBOT_ID,
        request_id=observation_step,
        arrival_timestamp=request_timestamp,
        observation_step=observation_step,
        action_start_step=action_start_step,
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
        robot.send_response(chunk)
        robot.receive_response(
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
        assert actual.action_start_step == expected.action_start_step
        assert actual.execution_horizon == expected.execution_horizon
        assert actual.arrival_time == pytest.approx(expected.arrival_time)
        assert actual.arrived is True

    assert robot.max_arrived_action_step == max(
        c.action_start_step + c.execution_horizon - 1 for c in scenario.chunks
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
