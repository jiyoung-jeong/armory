from __future__ import annotations

from typing import NamedTuple

import numpy as np
import pytest

from armory.scheduling.mirror import ActionChunk
from armory_client.action_chunkers.action_chunk_broker import ActionChunkBrokerBase
from armory_client.messages import InferResponse
from tests.scheduling._cases import ALL_SCENARIOS, Scenario


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
                chunk_id=chunk.chunk_id,
                observation_step=chunk.observation_step,
                action_index_start=chunk.action_index_start,
                request_timestamp=0.0,
                actions=np.zeros((chunk.max_execution_horizon, 7)),
                max_execution_horizon=chunk.max_execution_horizon,
            ),
            chunk,
            chunk.arrival_time,
        )
        for i, chunk in enumerate(action_chunks)
    ]


def _with_arrival_time(action_chunk: ActionChunk, arrival_time: float) -> ActionChunk:
    return ActionChunk(
        chunk_id=action_chunk.chunk_id,
        observation_step=action_chunk.observation_step,
        arrival_time=arrival_time,
        action_index_start=action_chunk.action_index_start,
        max_execution_horizon=action_chunk.max_execution_horizon,
    )


@pytest.mark.parametrize("scenario", ALL_SCENARIOS, ids=lambda s: s.name)
def test_broker(scenario: Scenario) -> None:
    broker = ActionChunkBrokerBase()
    pending = _make_responses(scenario.chunks)
    for control_step in scenario.control_steps():
        while pending and pending[0].arrival_time < control_step.time:
            infer_response, expected_chunk, arrival_time = pending.pop(0)
            received = broker.receive_response(infer_response)
            assert _with_arrival_time(received, arrival_time) == expected_chunk
        action = broker.get_action(control_step.observation_step)
        assert action.step == control_step.action_step
