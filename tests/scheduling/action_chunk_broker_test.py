from __future__ import annotations

from typing import NamedTuple

import numpy as np

from armory.scheduling.mirror import ActionChunk, ControlStep
from armory_client.action_chunkers.action_chunk_broker import ActionChunkBrokerBase
from armory_client.messages import InferResponse

CONTROL_HZ = 1
CONTROL_PERIOD = 1 / CONTROL_HZ
EPS = 0.001


def arrives_before(observation_step: int) -> float:
    return observation_step * CONTROL_PERIOD - EPS


class TimedResponse(NamedTuple):
    infer_response: InferResponse
    action_chunk: ActionChunk
    arrival_time: float


def _make_responses(action_chunks: list[ActionChunk]) -> list[TimedResponse]:
    return [
        TimedResponse(
            InferResponse(
                robot_id="test",
                request_id=i,
                observation_step=chunk.observation_step,
                action_start_step=chunk.action_start_step,
                request_timestamp=0.0,
                actions=np.zeros((chunk.execution_horizon, 7)),
                execution_horizon=chunk.execution_horizon,
            ),
            chunk,
            chunk.arrival_time,
        )
        for i, chunk in enumerate(action_chunks)
    ]


def _with_arrival_time(action_chunk: ActionChunk, arrival_time: float) -> ActionChunk:
    return ActionChunk(
        observation_step=action_chunk.observation_step,
        arrival_time=arrival_time,
        action_start_step=action_chunk.action_start_step,
        execution_horizon=action_chunk.execution_horizon,
        arrived=False,
    )


def _control_steps(
    obs_action_next: list[tuple[int, int | None, int]],
) -> list[ControlStep]:
    assert [x[0] for x in obs_action_next] == list(range(len(obs_action_next)))
    return [
        ControlStep(
            time=obs_step * CONTROL_PERIOD,
            observation_step=obs_step,
            action_step=action_step,
            next_action_step=next_action_step,
        )
        for obs_step, action_step, next_action_step in obs_action_next
    ]


def run_broker(
    action_chunks: list[ActionChunk],
    obs_action_next: list[tuple[int, int | None, int]],
) -> None:
    broker = ActionChunkBrokerBase()
    infer_responses = _make_responses(action_chunks)
    for control_step in _control_steps(obs_action_next):
        while infer_responses and infer_responses[0].arrival_time < control_step.time:
            infer_response, expected_chunk, arrival_time = infer_responses.pop(0)
            received = broker.receive_response(infer_response)
            assert _with_arrival_time(received, arrival_time) == expected_chunk
        action = broker.get_action(control_step.observation_step)
        assert action.step == control_step.action_step


def test_chunk_overlap():
    action_chunks = [
        ActionChunk(
            observation_step=0,
            arrival_time=arrives_before(2),
            action_start_step=0,
            execution_horizon=5,
            arrived=False,
        ),
        ActionChunk(
            observation_step=4,
            arrival_time=arrives_before(4),
            action_start_step=2,
            execution_horizon=5,
            arrived=False,
        ),
    ]
    obs_action_next = [
        (0, None, 0),
        (1, None, 0),
        (2, 0, 1),
        (3, 1, 2),
        (4, 2, 3),
        (5, 3, 4),
        (6, 4, 5),
        (7, 5, 6),
        (8, 6, 7),
    ]
    run_broker(action_chunks, obs_action_next)


def test_pause_before_inference():
    action_chunks = [
        ActionChunk(
            observation_step=0,
            arrival_time=arrives_before(2),
            action_start_step=0,
            execution_horizon=5,
            arrived=False,
        ),
        ActionChunk(
            observation_step=7,
            arrival_time=arrives_before(9),
            action_start_step=5,
            execution_horizon=5,
            arrived=False,
        ),
    ]
    obs_action_next = [
        (0, None, 0),
        (1, None, 0),
        (2, 0, 1),
        (3, 1, 2),
        (4, 2, 3),
        (5, 3, 4),
        (6, 4, 5),
        (7, None, 5),
        (8, None, 5),
        (9, 5, 6),
        (10, 6, 7),
        (11, 7, 8),
        (12, 8, 9),
        (13, 9, 10),
    ]
    run_broker(action_chunks, obs_action_next)


def test_pause_during_inference():
    action_chunks = [
        ActionChunk(
            observation_step=0,
            arrival_time=arrives_before(2),
            action_start_step=0,
            execution_horizon=5,
            arrived=False,
        ),
        ActionChunk(
            observation_step=5,
            arrival_time=arrives_before(9),
            action_start_step=3,
            execution_horizon=5,
            arrived=False,
        ),
    ]
    obs_action_next = [
        (0, None, 0),
        (1, None, 0),
        (2, 0, 1),
        (3, 1, 2),
        (4, 2, 3),
        (5, 3, 4),
        (6, 4, 5),
        (7, None, 5),
        (8, None, 5),
        (9, 5, 6),
        (10, 6, 7),
        (11, 7, 8),
    ]
    run_broker(action_chunks, obs_action_next)
