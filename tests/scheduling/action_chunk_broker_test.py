from __future__ import annotations

import logging
import threading
from types import SimpleNamespace
from typing import NamedTuple

import numpy as np
import pytest

from armory.scheduling.mirror import ActionChunk
from armory_client.action_chunkers.action_chunk_broker import (
    ActionChunkBroker,
    ActionChunkBrokerBase,
)
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
    broker = ActionChunkBrokerBase()
    pending = _make_responses(scenario.chunks)
    for control_step in scenario.control_steps():
        while pending and pending[0].arrival_time < control_step.time:
            infer_response, expected_chunk, arrival_time = pending.pop(0)
            received = broker.receive_response(infer_response)
            assert _with_arrival_time(received, arrival_time) == expected_chunk
        action = broker.get_action(control_step.observation_step)
        assert action.step == control_step.action_step


def test_broker_preserves_min_execution_horizon_from_response() -> None:
    broker = ActionChunkBrokerBase()
    response = InferResponse(
        robot_id="test",
        request_id=0,
        chunk_id=10,
        observation_step=4,
        action_index_start=8,
        request_timestamp=0.0,
        actions=np.zeros((5, 7)),
        min_execution_horizon=3,
        max_execution_horizon=5,
    )

    chunk = broker.receive_response(response)

    assert chunk.min_execution_horizon == 3


class _FakeWebsocket:
    def __init__(self) -> None:
        self.sent: list[dict] = []
        self._closed = threading.Event()
        self.receive_entered = threading.Event()
        self.close_count = 0

    def send(self, *args, **kwargs) -> None:
        self.sent.append(kwargs)

    def receive(self):
        self.receive_entered.set()
        if not self._closed.wait(timeout=1):
            raise TimeoutError("test websocket was not closed")
        raise RuntimeError("websocket closed")

    def reset(self) -> None:
        pass

    def close(self) -> None:
        self.close_count += 1
        self._closed.set()


def test_action_chunk_broker_sends_configured_min_execution_horizon() -> None:
    ws = _FakeWebsocket()
    broker = ActionChunkBroker(
        ws,
        control_hz=10,
        min_execution_horizon=3,
        max_execution_horizon=5,
    )

    try:
        broker._infer(SimpleNamespace(step=0))

        assert ws.sent[-1]["min_execution_horizon"] == 3
        assert ws.sent[-1]["max_execution_horizon"] == 5
    finally:
        broker.close()


def test_action_chunk_broker_close_stops_receiver_without_error(caplog) -> None:
    ws = _FakeWebsocket()
    broker = ActionChunkBroker(ws, control_hz=10)
    assert ws.receive_entered.wait(timeout=1)

    with caplog.at_level(logging.ERROR):
        broker.close()
        broker.close()

    assert not broker._background_thread.is_alive()
    assert ws.close_count == 1
    assert "Action response receiver stopped unexpectedly" not in caplog.text


class _BrokenWebsocket(_FakeWebsocket):
    def receive(self):
        self.receive_entered.set()
        raise RuntimeError("receive failed")


def test_action_chunk_broker_logs_unexpected_receive_failure(caplog) -> None:
    ws = _BrokenWebsocket()
    with caplog.at_level(logging.ERROR):
        broker = ActionChunkBroker(ws, control_hz=10)
        broker._background_thread.join(timeout=1)

    try:
        assert not broker._background_thread.is_alive()
        assert "Action response receiver stopped unexpectedly" in caplog.text
        assert "receive failed" in caplog.text
    finally:
        broker.close()
