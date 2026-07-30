from __future__ import annotations

import threading
from types import SimpleNamespace
from typing import NamedTuple

import numpy as np
import pytest

from armory.scheduling.mirror import ActionChunk
from armory_client.action_chunk_broker import ActionChunkBroker
from armory_client.messages import InferResponse
from armory_client.schemas import Action
from evaluation.agents.policy_agent import PolicyAgent
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
                min_execution_horizon=chunk.min_execution_horizon,
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
        min_execution_horizon=action_chunk.min_execution_horizon,
        max_execution_horizon=action_chunk.max_execution_horizon,
    )


@pytest.mark.parametrize("scenario", ALL_SCENARIOS, ids=lambda s: s.name)
def test_broker(scenario: Scenario) -> None:
    broker = ActionChunkBroker()
    pending = _make_responses(scenario.chunks)
    for control_step in scenario.control_steps():
        while pending and pending[0].arrival_time < control_step.time:
            infer_response, expected_chunk, arrival_time = pending.pop(0)
            received = broker.receive_response(infer_response)
            assert _with_arrival_time(received, arrival_time) == expected_chunk
        action = broker.get_action(control_step.observation_step)
        if control_step.action_step is None:
            assert action is None
        else:
            assert action is not None
            assert action.step == control_step.action_step


class _FakeWebsocket:
    def __init__(self) -> None:
        self.sent: list[dict] = []
        self._closed = threading.Event()

    def send(self, *args, **kwargs) -> None:
        self.sent.append(kwargs)

    def receive(self):
        self._closed.wait()
        raise RuntimeError("closed")

    def reset(self) -> None:
        pass

    def close(self) -> None:
        self._closed.set()


def test_action_chunk_broker_sends_configured_min_execution_horizon() -> None:
    ws = _FakeWebsocket()
    broker = ActionChunkBroker(
        min_execution_horizon=3,
        max_execution_horizon=5,
    )
    agent = PolicyAgent(
        ws_client=ws,
        broker=broker,
        create_null_action=lambda observation, _: Action(
            step=observation.step,
            action=np.zeros(7),
            action_chunk_index=None,
            index_in_chunk=None,
        ),
    )

    agent.get_action(SimpleNamespace(step=0))
    agent.close()

    assert ws.sent[-1]["min_execution_horizon"] == 3
    assert ws.sent[-1]["max_execution_horizon"] == 5
