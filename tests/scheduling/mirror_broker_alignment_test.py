"""Compare real client action selection and broker queues against mirror forecasts."""

from dataclasses import replace
from threading import Lock
from types import SimpleNamespace

import numpy as np
import pytest

from armory.scheduling.mirror import Robot
from armory.serving.schemas import ActionChunk
from armory_client.action_chunk_broker import ActionChunkBroker
from armory_client.messages import InferResponse
from armory_client.schemas import Action, Observation
from evaluation.agents.policy_agent import PolicyAgent
from tests.scheduling.mirror_test import _make_request, _StubLatencyTracker


def response(start, chunk_id=1, observation_step=0, horizon=5):
    return InferResponse(
        robot_id="test",
        request_id=chunk_id,
        chunk_id=chunk_id,
        observation_step=observation_step,
        action_index_start=start,
        request_timestamp=0.0,
        actions=np.zeros((horizon, 7)),
        min_execution_horizon=0,
        max_execution_horizon=horizon,
    )


def setup_pair(control_hz=1.0):
    broker = ActionChunkBroker(0, 5)
    robot = Robot("test", control_hz, 0, 5, _StubLatencyTracker())
    sent = []
    agent = PolicyAgent.__new__(PolicyAgent)
    agent._broker = broker
    agent._lock = Lock()
    agent._event_log = None
    agent._ws_client = SimpleNamespace(
        send=lambda obs, index, *a, **kw: sent.append((obs.step, index))
    )
    agent._create_null_action = lambda obs, chunk: Action(obs.step, np.zeros(7), None, None)
    agent.get_action(Observation(step=0))
    robot.step(_make_request(0, sent[-1][1], 0.0, 5))
    broker.receive_response(response(0))
    robot.queue_chunk(ActionChunk(1, 0, 0, 0, 5, 0.1 / control_hz, origin="confirmed"))
    return broker, robot, agent, sent


@pytest.mark.parametrize("control_hz", [1.0, 20.0])
@pytest.mark.parametrize("dispatch_tick", [1, 3, 6])
@pytest.mark.parametrize("arrival_offset", [-0.001, 0.001])
def test_same_observation_forecast_matches_suffix_queue_and_future_starvation(
    dispatch_tick, arrival_offset, control_hz
):
    broker, robot, agent, sent = setup_pair(control_hz)
    for tick in range(1, dispatch_tick + 1):
        agent.get_action(Observation(step=tick))
        robot.step(_make_request(tick, sent[-1][1], tick / control_hz, 5))
    # Dispatch between ticks; no slot update between prediction and GPU read.
    start = sent[-1][1]
    arrival_tick = dispatch_tick + 2 + arrival_offset
    arrival = arrival_tick / control_hz
    ctx = robot.calculate_chunk_context((dispatch_tick + 0.2) / control_hz, arrival)
    assert ctx.observation_step == sent[-1][0]
    assert ctx.action_index_start == start
    for tick in range(dispatch_tick + 1, int(arrival_tick) + 1):
        agent.get_action(Observation(step=tick))
    expected_skip = max(0, broker.next_action_step - start)
    before = broker.num_actions_available
    broker.receive_response(response(start, 2, dispatch_tick))
    assert ctx.first_executed_index == expected_skip
    assert ctx.execution_start_step == int(arrival_tick) + 1
    suffix = max(0, 5 - expected_skip)
    assert broker.num_actions_available == suffix
    predicted = ActionChunk(
        2,
        ctx.observation_step,
        ctx.action_index_start,
        0,
        5,
        arrival,
        ctx.execution_start_step,
        ctx.first_executed_index,
        "searched",
    )
    robot.queue_chunk(predicted)
    # Forecast from the dispatch snapshot, then compare every future control tick.
    robot.step_forward((dispatch_tick + 10.00001) / control_hz)
    future = {s.observation_step: s for s in robot.steps}
    predicted_next = future[int(arrival_tick)].next_action_step
    predicted_queue = sum(
        robot.action_is_available(i, arrival)
        for i in range(predicted_next, predicted.last_action_index + 1)
    )
    assert predicted_queue - before == broker.num_actions_available - before
    for tick in range(int(arrival_tick) + 1, dispatch_tick + 11):
        action = broker.get_action(tick)
        assert future[tick].action_step == (action.step if action else None)
        assert future[tick].next_action_step == broker.next_action_step


def test_completed_chunk_keeps_actual_observation_and_action_origin():
    broker, robot, agent, sent = setup_pair()
    agent.get_action(Observation(step=1))
    robot.step(_make_request(1, sent[-1][1], 1.0, 5))
    robot.queue_chunk(ActionChunk(2, 1, 1, 0, 5, 3.2))
    actual = response(1, 2, observation_step=1)
    index = robot.apply_response(2, actual, 3.3)
    robot.recompute_and_check(index)
    completed = robot.get_chunk(2)
    assert completed.observation_step == actual.observation_step
    assert completed.action_index_start == actual.action_index_start


def test_newer_slot_is_separate_from_same_observation_origin():
    broker, robot, agent, sent = setup_pair()
    agent.get_action(Observation(step=1))
    robot.step(_make_request(1, sent[-1][1], 1.0, 5))
    selected = _make_request(1, sent[-1][1], 1.0, 5)
    predicted = robot.calculate_chunk_context(1.2, 3.3)
    assert predicted.action_index_start == selected.action_index_start
    agent.get_action(Observation(step=2))
    processed = replace(selected, request_id=2, observation_step=2, action_index_start=sent[-1][1])
    assert processed.request_id != selected.request_id
    assert processed.action_index_start == selected.action_index_start + 1


@pytest.mark.parametrize("last_tick_consumed", [False, True])
def test_sparse_request_resynchronizes_exact_post_pop_counter(last_tick_consumed):
    broker, robot, agent, sent = setup_pair(20.0)
    # Five actions were available at the initial snapshot. Whether the next
    # reported tick is 4 (still consuming) or 10 (already starved) is explicit.
    last = 4 if last_tick_consumed else 10
    for tick in range(1, last + 1):
        agent.get_action(Observation(step=tick))
    request = replace(
        _make_request(last, broker.next_action_step, last / 20.0, 5),
        action_executed=last_tick_consumed,
    )
    robot.step(request)
    assert robot.executed_steps == broker.next_action_step
    assert robot.steps[-1].action_step == (
        broker.next_action_step - 1 if last_tick_consumed else None
    )
    assert len(robot.steps) == 2  # Intermediate action timings are not fabricated.
    robot.step_forward((last + 4.001) / 20.0)
    forecast = {step.observation_step: step for step in robot.steps}
    for tick in range(last + 1, last + 5):
        action = broker.get_action(tick)
        assert forecast[tick].action_step == (action.step if action else None)
        assert forecast[tick].next_action_step == broker.next_action_step


def test_sparse_legacy_request_fails_instead_of_fabricating_progress():
    _, robot, _, _ = setup_pair()
    with pytest.raises(ValueError, match="Sparse observations"):
        robot.step(_make_request(10, 5, 10.0, 5))


def test_sparse_counter_cannot_advance_more_than_control_ticks():
    _, robot, _, _ = setup_pair()
    request = replace(_make_request(2, 4, 2.0, 5), action_executed=True)
    with pytest.raises(ValueError, match="exceeds elapsed"):
        robot.step(request)
