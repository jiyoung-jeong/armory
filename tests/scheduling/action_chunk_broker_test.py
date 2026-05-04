from __future__ import annotations

from typing import NamedTuple

import numpy as np
import pytest

from armory.scheduling.mirror import ActionChunk, ControlStep
from armory_client.action_chunkers.action_chunk_broker import ActionChunkBrokerBase
from armory_client.messages import InferResponse

# NOTE: control hz is 1
CONTROL_HZ = 1

from typing import NamedTuple as TypingNamedTuple


class TimedResponse(TypingNamedTuple):
    infer_response: InferResponse
    action_chunk: ActionChunk
    arrival_time: float


def broker_harness(
    action_chunks: list[ActionChunk], expected_control_steps: list[ControlStep]
) -> None:
    def create_responses(action_chunks: list[ActionChunk]) -> list[TimedResponse]:
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

    def convert_to_mirror_action_chunk(
        action_chunk: ActionChunk, arrival_time: float
    ) -> ActionChunk:
        return ActionChunk(
            observation_step=action_chunk.observation_step,
            arrival_time=arrival_time,
            action_start_step=action_chunk.action_start_step,
            execution_horizon=action_chunk.execution_horizon,
            arrived=False,
        )

    broker = ActionChunkBrokerBase()
    infer_responses = create_responses(action_chunks)
    for control_step in expected_control_steps:
        while infer_responses and infer_responses[0].arrival_time < control_step.time:
            infer_response, expected_action_chunk, arrival_time = infer_responses.pop(0)
            action_chunk = broker.receive_response(infer_response)
            assert (
                convert_to_mirror_action_chunk(action_chunk, arrival_time) == expected_action_chunk
            )
        action = broker.get_action(control_step.observation_step)
        assert action.step == control_step.action_step


def test_chunk_overlap():
    # FIXME: write logic for this, maybe factor out next_action_step in mirror if we can

    obs_and_action_steps_and_next_action_steps = [
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
    assert [x[0] for x in obs_and_action_steps_and_next_action_steps] == list(
        range(len(obs_and_action_steps_and_next_action_steps))
    )

    def arrives_before(observation_step: int) -> float:
        EPS = 0.001
        return observation_step * (1 / CONTROL_HZ) - EPS

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
    expected_control_steps = [
        ControlStep(
            time=obs_step * (1 / CONTROL_HZ),
            observation_step=obs_step,
            action_step=action_step,
            next_action_step=next_action_step,
        )
        for obs_step, action_step, next_action_step in obs_and_action_steps_and_next_action_steps
    ]

    broker_harness(action_chunks, expected_control_steps)


def test_pause_before_inference():
    # FIXME: write logic for this, maybe factor out next_action_step in mirror if we can

    obs_and_action_steps_and_next_action_steps = [
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
    assert [x[0] for x in obs_and_action_steps_and_next_action_steps] == list(
        range(len(obs_and_action_steps_and_next_action_steps))
    )

    def arrives_before(observation_step: int) -> float:
        EPS = 0.001
        return observation_step * (1 / CONTROL_HZ) - EPS

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
    expected_control_steps = [
        ControlStep(
            time=obs_step * (1 / CONTROL_HZ),
            observation_step=obs_step,
            action_step=action_step,
            next_action_step=next_action_step,
        )
        for obs_step, action_step, next_action_step in obs_and_action_steps_and_next_action_steps
    ]

    broker_harness(action_chunks, expected_control_steps)


def test_pause_during_inference():
    # FIXME: write logic for this, maybe factor out next_action_step in mirror if we can

    obs_and_action_steps_and_next_action_steps = [
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
    assert [x[0] for x in obs_and_action_steps_and_next_action_steps] == list(
        range(len(obs_and_action_steps_and_next_action_steps))
    )

    def arrives_before(observation_step: int) -> float:
        EPS = 0.001
        return observation_step * (1 / CONTROL_HZ) - EPS

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
    expected_control_steps = [
        ControlStep(
            time=obs_step * (1 / CONTROL_HZ),
            observation_step=obs_step,
            action_step=action_step,
            next_action_step=next_action_step,
        )
        for obs_step, action_step, next_action_step in obs_and_action_steps_and_next_action_steps
    ]

    broker_harness(action_chunks, expected_control_steps)
