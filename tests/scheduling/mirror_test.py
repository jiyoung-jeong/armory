from __future__ import annotations

from collections import deque
from dataclasses import replace

import pytest

from armory.scheduling.latency import LatencyTracker
from armory.scheduling.mirror import Batch, Mirror, Robot
from armory.serving.rtc import InferType
from armory.serving.schemas import AckNotification, ResponseBatch, SlotRequest
from armory_client.messages import InferResponse, ResponseAck
from tests.scheduling._cases import ALL_SCENARIOS, CONTROL_HZ, EPS, LONG_RUN, Scenario

ROBOT_ID = "test"


class _StubLatencyTracker(LatencyTracker):
    """Constant-latency tracker for testing mirror timing."""

    def __init__(
        self, *, observation: float = 0.0, infer: float = 0.0, action: float = 0.0
    ) -> None:
        super().__init__()
        self._observation = observation
        self._action = action
        self._infer = infer

    def _update_measurement(self, d: dict, key: object, value: float) -> None:  # noqa: ARG002
        pass

    def observation_latency(self, robot_id: str) -> float:  # noqa: ARG002
        return self._observation

    def infer_latency(self, batch_size: int) -> float:  # noqa: ARG002
        return self._infer

    def action_latency(self, robot_id: str) -> float:  # noqa: ARG002
        return self._action


def _make_request(
    observation_step: int,
    action_index_start: int,
    request_timestamp: float,
    max_execution_horizon: int,
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
        min_execution_horizon=0,
        max_execution_horizon=max_execution_horizon,
        infer_type=InferType.SYNC,
        params=None,
        noise=None,
        control_hz=CONTROL_HZ,
    )


def _make_robot(scenario: Scenario) -> Robot:
    """Create a Robot seeded with the initial obs=0 control step and acked chunks."""
    horizon = scenario.chunks[0].max_execution_horizon
    robot = Robot(ROBOT_ID, CONTROL_HZ, 0, horizon, _StubLatencyTracker())
    robot.step(_make_request(0, 0, 0.0, horizon))
    for chunk in scenario.chunks:
        robot.queue_chunk(replace(chunk, origin="confirmed"))
    return robot


@pytest.mark.parametrize("scenario", ALL_SCENARIOS, ids=lambda s: s.name)
def test_chunk_tracking(scenario: Scenario) -> None:
    robot = _make_robot(scenario)

    assert len(robot.chunks) == len(scenario.chunks)
    for actual, expected in zip(robot.chunks, scenario.chunks):
        assert actual.observation_step == expected.observation_step
        assert actual.action_index_start == expected.action_index_start
        assert actual.max_execution_horizon == expected.max_execution_horizon
        assert actual.arrival_time == pytest.approx(expected.arrival_time)
        assert actual.origin == "confirmed"

    assert robot.max_overall_action_step == max(
        c.action_index_start + c.max_execution_horizon - 1 for c in scenario.chunks
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
    horizon = scenario.chunks[0].max_execution_horizon
    mirror = Mirror(_StubLatencyTracker())
    mirror.receive_request(_make_request(0, 0, 0.0, horizon))
    return mirror


def _drip_chunks(mirror: Mirror, scenario: Scenario) -> None:
    """Feed chunks to the mirror via fast_forward, advancing time to just past
    each chunk's arrival."""
    for chunk in scenario.chunks:
        mirror.robots[ROBOT_ID].queue_chunk(chunk)
        mirror.fast_forward(chunk.arrival_time + EPS)


@pytest.mark.parametrize("scenario", ALL_SCENARIOS, ids=lambda s: s.name)
def test_mirror_fast_forward(scenario: Scenario) -> None:
    mirror = _seeded_mirror(scenario)
    expected = scenario.control_steps()

    _drip_chunks(mirror, scenario)
    # Advance past the last chunk's coverage to the scenario's final step.
    mirror.fast_forward(expected[-1].time)

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
    mirror = Mirror(_StubLatencyTracker())
    horizon = LONG_RUN.chunks[0].max_execution_horizon
    for rid in rids:
        mirror.receive_request(replace(_make_request(0, 0, 0.0, horizon), robot_id=rid))

    for chunk in LONG_RUN.chunks:
        # Send the same chunk to both robots so they stay in lockstep.
        for rid in rids:
            mirror.robots[rid].queue_chunk(chunk)
        mirror.fast_forward(chunk.arrival_time + EPS)

    expected_deadline = LONG_RUN.control_steps()[-1].time
    deadlines = mirror.deadlines()
    assert deadlines == {rid: pytest.approx(expected_deadline) for rid in rids}


def test_mirror_fast_forward_advances_robot_without_new_chunk() -> None:
    """A robot not named in the chunk list still gets stepped forward."""

    rids = ["a", "b"]
    mirror = Mirror(_StubLatencyTracker())
    horizon = LONG_RUN.chunks[0].max_execution_horizon
    for rid in rids:
        mirror.receive_request(replace(_make_request(0, 0, 0.0, horizon), robot_id=rid))
    # Only "a" gets the chunk, but both should advance.
    mirror.robots["a"].queue_chunk(LONG_RUN.chunks[0])
    mirror.fast_forward(LONG_RUN.chunks[0].arrival_time + EPS)

    assert mirror.robots["a"].steps[-1].observation_step == 2
    assert mirror.robots["b"].steps[-1].observation_step == 2
    # "b" has no chunks, so its action_step should be None on every step.
    assert all(s.action_step is None for s in mirror.robots["b"].steps)


def test_mirror_twin_preserves_divergent_branch_contents() -> None:
    """A twin preserves branch contents independent of later mirror mutations."""
    horizon = LONG_RUN.chunks[0].max_execution_horizon
    mirror = Mirror(_StubLatencyTracker())
    mirror.receive_request(_make_request(0, 0, 0.0, horizon))
    mirror.robots[ROBOT_ID].queue_chunk(LONG_RUN.chunks[0])

    branch_a_chunk = replace(LONG_RUN.chunks[1], chunk_id=101, action_index_start=5)
    mirror.robots[ROBOT_ID].queue_chunk(branch_a_chunk)
    branch_a_twin = mirror.get_twin()

    mirror.robots[ROBOT_ID].chunks = deque([LONG_RUN.chunks[0]], maxlen=5)
    branch_b_chunk = replace(LONG_RUN.chunks[1], chunk_id=202, action_index_start=4)
    mirror.robots[ROBOT_ID].queue_chunk(branch_b_chunk)

    assert list(branch_a_twin.robots[ROBOT_ID].chunks) == [LONG_RUN.chunks[0], branch_a_chunk]


def test_mirror_update_completion_refines_arrival() -> None:
    """update_batch_completion sets arrival_time to completion + action_latency."""
    tracker = _StubLatencyTracker(observation=0.05, infer=0.1, action=0.02)
    mirror = Mirror(tracker)
    horizon = LONG_RUN.chunks[0].max_execution_horizon
    mirror.receive_request(_make_request(0, 0, 0.0, horizon))
    chunk = LONG_RUN.chunks[0]
    mirror.robots[ROBOT_ID].queue_chunk(chunk)
    mirror.fast_forward(chunk.arrival_time + EPS)

    mirror.in_flight_batches.append(
        Batch(batch_id=0, robot_ids=[ROBOT_ID], chunk_ids=[chunk.chunk_id])
    )
    import numpy as np

    notification = ResponseBatch(
        responses=[
            InferResponse(
                robot_id=ROBOT_ID,
                request_id=0,
                chunk_id=chunk.chunk_id,
                observation_step=chunk.observation_step,
                action_index_start=chunk.action_index_start,
                request_timestamp=0.0,
                actions=np.zeros((1, chunk.max_execution_horizon, 7)),
                min_execution_horizon=chunk.min_execution_horizon,
                max_execution_horizon=chunk.max_execution_horizon,
            )
        ],
        batch_id=0,
        batch_size=1,
        inference_start_time=9.9,
        inference_duration=0.1,
    )
    mirror.update_batch_completion(notification)

    refined = mirror.robots[ROBOT_ID].chunks[0]
    assert refined.arrival_time == pytest.approx(10.02)
    assert refined.origin == "completed"


def test_mirror_confirm_chunk_by_chunk_id() -> None:
    """confirm_chunk matches by chunk_id and sets arrival_time=ack.receive_time."""
    horizon = LONG_RUN.chunks[0].max_execution_horizon
    mirror = Mirror(_StubLatencyTracker())
    mirror.receive_request(_make_request(0, 0, 0.0, horizon))
    chunk = LONG_RUN.chunks[1]  # chunk_id=1
    mirror.robots[ROBOT_ID].queue_chunk(chunk)
    mirror.fast_forward(chunk.arrival_time + EPS)

    ack = AckNotification(
        ack=ResponseAck(
            request_id=0,
            chunk_id=chunk.chunk_id,
            observation_step=chunk.observation_step,
            receive_time=42.0,
            action_index_start=chunk.action_index_start,
            min_execution_horizon=chunk.min_execution_horizon,
            max_execution_horizon=chunk.max_execution_horizon,
            execution_start_step=3,
            first_executed_index=1,
        ),
        robot_id=ROBOT_ID,
        server_send_time=0.0,
    )
    mirror.confirm_chunk(ack)

    confirmed = mirror.robots[ROBOT_ID].chunks[0]
    assert confirmed.arrival_time == pytest.approx(42.0)
    assert confirmed.origin == "confirmed"
    assert confirmed.execution_start_step == 3
    assert confirmed.first_executed_index == 1


def test_mirror_get_chunks_basic() -> None:
    """queue_batch builds chunks with the latency-projected arrival time."""
    from unittest.mock import patch

    tracker = _StubLatencyTracker(observation=0.05, infer=0.1, action=0.02)
    mirror = Mirror(tracker)
    horizon = 5
    mirror.receive_request(_make_request(0, 0, 0.0, horizon))
    # Step the snapshot forward so a control step exists before the obs cutoff.
    mirror.fast_forward(2.0)

    with patch("armory.scheduling.mirror.time") as mock_time:
        mock_time.time.return_value = 2.0
        chunks = mirror.queue_batch([ROBOT_ID], 0)

    assert len(chunks) == 1
    [chunk] = chunks
    # arrival_time = dispatch + infer + action = 2.0 + 0.1 + 0.02
    assert chunk.arrival_time == pytest.approx(2.12)
    assert chunk.origin == "queued"
    assert chunk.max_execution_horizon == horizon
    # observation_step: latest step before (dispatch_time - obs_latency) = 1.95;
    # steps tick at integer times, so the latest step before 1.95 is step 1.
    assert chunk.observation_step == 1
