"""Unit tests for ``armory.scheduling.mirror``.

These pin down the mirror's behavior in isolation (no broker involvement) so we
can iterate on it. A few tests target known suspicious spots:
``test_actions_executed_with_pause`` and ``test_deadline_returns_last_action_time``
document current (possibly off-by-one) semantics rather than asserting the
ideal ones.
"""

from __future__ import annotations

import pytest

from armory.scheduling.latency import LatencyTracker
from armory.scheduling.mirror import ActionChunk, ControlStep, Mirror, Robot
from armory.serving.schemas import AckNotification, SlotRequest
from armory_client.messages import InferType

CONTROL_HZ = 10.0
DT = 1 / CONTROL_HZ


def _slot_request(
    *,
    robot_id: str = "r0",
    request_id: int = 0,
    request_timestamp: float,
    observation_step: int,
    action_start_step: int,
    deadline: float = 0.0,
    execution_horizon: int = 4,
) -> SlotRequest:
    return SlotRequest(
        slot_index=0,
        robot_id=robot_id,
        request_id=request_id,
        arrival_timestamp=request_timestamp,
        observation_step=observation_step,
        action_start_step=action_start_step,
        request_timestamp=request_timestamp,
        deadline=deadline,
        execution_horizon=execution_horizon,
        infer_type=InferType.SYNC,
        params=None,
        noise=None,
        control_hz=CONTROL_HZ,
    )


# ---------------------------------------------------------------------------
# Robot.actions_executed
# ---------------------------------------------------------------------------


def test_actions_executed_empty():
    robot = Robot(control_hz=CONTROL_HZ, execution_horizon=4)
    assert robot.actions_executed() == 0


def test_actions_executed_contiguous():
    robot = Robot(control_hz=CONTROL_HZ, execution_horizon=4)
    for i, t in enumerate([0.0, 0.1, 0.2]):
        robot.step(ControlStep(time=t, observation_step=i, action_step=i, next_action_step=i + 1))
    assert robot.actions_executed() == 3


def test_actions_executed_with_pause_overcounts():
    """Documents current behavior: ``actions_executed`` returns last-first+1, not count.

    With one action_step=None step interleaved, the count returned exceeds the
    number of non-None steps. Flagged as a known semantic mismatch in the plan;
    this test pins the current behavior so a future fix is intentional.
    """
    robot = Robot(control_hz=CONTROL_HZ, execution_horizon=4)
    schedule = [(0.0, 0, 0), (0.1, 1, 1), (0.2, 2, None), (0.3, 3, 4)]
    for t, obs, act in schedule:
        robot.step(
            ControlStep(
                time=t,
                observation_step=obs,
                action_step=act,
                next_action_step=(act + 1) if act is not None else 2,
            )
        )
    # Three non-None actions, but the range is 4 - 0 + 1.
    assert robot.actions_executed() == 5


# ---------------------------------------------------------------------------
# Robot.action_is_available
# ---------------------------------------------------------------------------


def test_action_is_available_arrived_chunk():
    robot = Robot(control_hz=CONTROL_HZ, execution_horizon=4)
    chunk = ActionChunk(
        observation_step=0,
        arrival_time=0.5,
        action_start_step=1,
        execution_horizon=4,
        arrived=True,
    )
    robot.send_response(chunk)
    assert robot.action_is_available(action_step=1, time=0.5)
    assert robot.action_is_available(action_step=4, time=0.6)
    # Out of range
    assert not robot.action_is_available(action_step=0, time=0.6)
    assert not robot.action_is_available(action_step=5, time=0.6)
    # Not yet arrived
    assert not robot.action_is_available(action_step=2, time=0.4)


# ---------------------------------------------------------------------------
# Robot.advance_step + Robot.deadline
# ---------------------------------------------------------------------------


def test_advance_step_executes_when_action_available():
    robot = Robot(control_hz=CONTROL_HZ, execution_horizon=4)
    chunk = ActionChunk(
        observation_step=0,
        arrival_time=0.0,
        action_start_step=1,
        execution_horizon=4,
        arrived=True,
    )
    robot.send_response(chunk)
    prev = ControlStep(time=0.0, observation_step=0, action_step=0, next_action_step=1)
    nxt = robot.advance_step(prev)
    assert nxt.time == pytest.approx(0.1)
    assert nxt.action_step == 1
    assert nxt.next_action_step == 2


def test_advance_step_pauses_when_chunk_late():
    robot = Robot(control_hz=CONTROL_HZ, execution_horizon=4)
    chunk = ActionChunk(
        observation_step=0,
        arrival_time=10.0,  # not yet arrived
        action_start_step=1,
        execution_horizon=4,
        arrived=False,
    )
    robot.send_response(chunk)
    prev = ControlStep(time=0.0, observation_step=0, action_step=0, next_action_step=1)
    nxt = robot.advance_step(prev)
    assert nxt.action_step is None
    assert nxt.next_action_step == 1  # unchanged on pause


def test_deadline_no_chunks_returns_last_step_time():
    robot = Robot(control_hz=CONTROL_HZ, execution_horizon=4)
    robot.step(ControlStep(time=0.0, observation_step=0, action_step=0, next_action_step=1))
    robot.step(ControlStep(time=0.1, observation_step=1, action_step=1, next_action_step=2))
    assert robot.deadline() == pytest.approx(0.1)


def test_deadline_returns_time_of_last_executed_action():
    """Pin down the off-by-one in ``Robot.deadline``.

    Loop runs while ``next_action_step <= max_overall_action_step``; on exit,
    ``step.time`` is the time the *last available* action just ran. The broker's
    ``deadline = now + len(queue) * dt`` represents one step *later* — when the
    next action would be needed. These differ by 1/control_hz.
    """
    robot = Robot(control_hz=CONTROL_HZ, execution_horizon=4)
    chunk = ActionChunk(
        observation_step=0,
        arrival_time=0.0,
        action_start_step=1,
        execution_horizon=3,  # actions 1,2,3
        arrived=True,
    )
    robot.send_response(chunk)
    robot.step(ControlStep(time=0.0, observation_step=0, action_step=0, next_action_step=1))
    # advance_step internally goes:
    #   step(t=0.1, action=1, next=2)
    #   step(t=0.2, action=2, next=3)
    #   step(t=0.3, action=3, next=4)  # exits loop after this (next > 3)
    assert robot.deadline() == pytest.approx(0.3)


# ---------------------------------------------------------------------------
# Robot.receive_response
# ---------------------------------------------------------------------------


def test_receive_response_marks_arrived_and_updates_arrival_time():
    robot = Robot(control_hz=CONTROL_HZ, execution_horizon=4)
    chunk = ActionChunk(
        observation_step=3,
        arrival_time=99.0,  # original (predicted) arrival
        action_start_step=4,
        execution_horizon=4,
        arrived=False,
    )
    robot.send_response(chunk)
    ack = AckNotification(
        robot_id="r0",
        request_id=1,
        observation_step=3,
        receive_time=0.5,
        server_send_time=0.4,
    )
    robot.receive_response(ack)
    assert len(robot.chunks) == 1
    assert robot.chunks[0].arrived is True
    assert robot.chunks[0].arrival_time == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Mirror.get_chunks
# ---------------------------------------------------------------------------


class _StubLatencyTracker(LatencyTracker):
    """Returns fixed latencies; bypasses the EMA logic."""

    def __init__(
        self,
        observation: float = 0.01,
        infer: float = 0.05,
        action: float = 0.02,
    ) -> None:
        super().__init__()
        self._obs = observation
        self._infer = infer
        self._act = action

    def _update_measurement(self, d, key, value):
        d[key] = value

    def observation_latency(self, robot_id: str) -> float:
        return self._obs

    def infer_latency(self, batch_size: int) -> float:
        return self._infer

    def action_latency(self, robot_id: str) -> float:
        return self._act


def test_get_chunks_arrival_time_math():
    m = Mirror()
    request = _slot_request(request_timestamp=0.0, observation_step=0, action_start_step=0)
    m.receive_request(request, control_hz=CONTROL_HZ)
    # add a second control step at t=0.5 so observation lookup picks the latest before now
    m.receive_request(
        _slot_request(request_id=2, request_timestamp=0.5, observation_step=5, action_start_step=5),
        control_hz=CONTROL_HZ,
    )

    tracker = _StubLatencyTracker(observation=0.01, infer=0.05, action=0.02)
    now = 1.0
    chunks = m.get_chunks(["r0"], tracker, time=now)
    assert len(chunks) == 1
    chunk = chunks[0]
    # observation cutoff = now - obs_latency = 0.99 → latest control step before that is the one at 0.5
    assert chunk.observation_step == 5
    assert chunk.arrival_time == pytest.approx(now + 0.05 + 0.02)
    assert chunk.action_start_step == 6  # next_action_step from the latest step
    assert chunk.execution_horizon == 4


# ---------------------------------------------------------------------------
# Mirror.fast_forward
# ---------------------------------------------------------------------------


def test_fast_forward_advances_all_robots():
    m = Mirror()
    for rid in ("r0", "r1"):
        m.receive_request(
            _slot_request(
                robot_id=rid, request_timestamp=0.0, observation_step=0, action_start_step=0
            ),
            control_hz=CONTROL_HZ,
        )

    m.fast_forward(time=0.3, robot_ids=[], chunks=[])
    for rid in ("r0", "r1"):
        # advance_step uses next_time = prev.time + dt, so steps land at 0.1, 0.2, 0.3
        last = m.robots[rid].steps[-1]
        assert last.time == pytest.approx(0.3)


def test_fast_forward_registers_chunks_before_stepping():
    m = Mirror()
    m.receive_request(
        _slot_request(request_timestamp=0.0, observation_step=0, action_start_step=0),
        control_hz=CONTROL_HZ,
    )
    chunk = ActionChunk(
        observation_step=0,
        arrival_time=0.05,
        action_start_step=1,
        execution_horizon=4,
        arrived=True,
    )
    m.fast_forward(time=0.3, robot_ids=["r0"], chunks=[chunk])
    steps = m.robots["r0"].steps
    # at t=0.1, 0.2, 0.3 chunk has arrived → action_step assigned
    assert steps[1].action_step == 1
    assert steps[2].action_step == 2
    assert steps[3].action_step == 3
