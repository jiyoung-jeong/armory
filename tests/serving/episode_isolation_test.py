"""Regressions for responses racing with an episode reset."""

from __future__ import annotations

import asyncio
import contextlib
import io
import json
import queue
import threading
import uuid
from types import SimpleNamespace

import numpy as np

from armory.serving.session import _send_loop
from armory_client import msgpack_numpy
from armory_client.action_chunk_broker import ActionChunkBroker
from armory_client.client import BidirectionalWebsocket
from armory_client.messages import InferResponse
from armory_client.schemas import Observation
from evaluation.agents.policy_agent import PolicyAgent


def response(episode_id, request_id):
    return InferResponse(
        robot_id="robot",
        request_id=request_id,
        chunk_id=request_id,
        observation_step=0,
        action_index_start=0,
        request_timestamp=0,
        actions=np.ones((4, 2)),
        min_execution_horizon=1,
        max_execution_horizon=4,
        episode_id=episode_id,
    )


class FakeTransport:
    def __init__(self):
        self.sent = []

    def send(self, payload):
        self.sent.append(msgpack_numpy.unpackb(payload))


def test_reset_generation_propagates_to_wire_messages():
    client = BidirectionalWebsocket("robot")
    client._ws = transport = FakeTransport()
    client.reset()
    first = client.episode_id
    client.reset()
    second = client.episode_id
    assert first and second and first != second
    client.send(Observation(step=0), 0, 0)
    client.send_ack(1, 1, 0, 1.0, 0, 1, 4, 0)
    assert [m["episode_id"] for m in transport.sent] == [first, second, second, second]


class FakeWebsocket:
    def __init__(self):
        self.responses = queue.Queue()
        self.acks = []
        self.accepted = threading.Event()

    def reset(self):
        self.episode_id = uuid.uuid4().hex

    def receive(self):
        value = self.responses.get(timeout=5)
        if value is None:
            raise RuntimeError("closed")
        return value

    def send_ack(self, **fields):
        self.acks.append(fields)
        self.accepted.set()

    def close(self):
        self.responses.put(None)


def test_agent_discards_old_episode_and_only_acks_current(tmp_path):
    ws = FakeWebsocket()
    agent = PolicyAgent(
        ws, ActionChunkBroker(), lambda *_: None, event_log_path=tmp_path / "events.jsonl"
    )
    try:
        old = ws.episode_id
        agent.reset()
        current = ws.episode_id
        ws.responses.put(response(old, 1))
        ws.responses.put(response(current, 2))
        assert ws.accepted.wait(5)
        assert [ack["request_id"] for ack in ws.acks] == [2]
        assert [chunk.episode_id for chunk in agent.action_chunks] == [current]
        assert agent.broker.num_actions_available == 4
    finally:
        agent.close()
    events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    assert [event["request_id"] for event in events if event["kind"] == "response_discarded"] == [1]


def test_server_filters_queued_old_responses_and_records_send_completion():
    async def scenario():
        class Websocket:
            def __init__(self):
                self.sent = []
                self.done = asyncio.Event()

            async def send_bytes(self, data):
                self.sent.append(msgpack_numpy.unpackb(data))
                self.done.set()

        ws = Websocket()
        pending = asyncio.Queue()
        pending.put_nowait(response("old", 1))
        pending.put_nowait(response("current", 2))
        state = SimpleNamespace(events_log=io.StringIO())
        send_times = {}
        task = asyncio.create_task(_send_loop(ws, pending, send_times, {"id": "current"}, state))
        await asyncio.wait_for(ws.done.wait(), 5)
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        assert [r["request_id"] for r in ws.sent] == [2]
        assert set(send_times) == {2}
        events = [json.loads(line) for line in state.events_log.getvalue().splitlines()]
        assert [(event["kind"], event["request_id"]) for event in events] == [
            ("response_discarded", 1),
            ("response_sent", 2),
        ]
        assert events[-1]["send_complete"] >= events[-1]["send_start"]

    asyncio.run(scenario())


def test_response_received_before_reset_is_checked_after_waiting_for_agent_lock():
    ws = FakeWebsocket()
    agent = PolicyAgent(ws, ActionChunkBroker(), lambda *_: None)

    class PauseReceiverLock:
        def __init__(self):
            self.lock = threading.Lock()
            self.waiting = threading.Event()
            self.proceed = threading.Event()
            self.paused = False

        def __enter__(self):
            if threading.current_thread() is agent._background_thread and not self.paused:
                self.paused = True
                self.waiting.set()
                assert self.proceed.wait(5)
            self.lock.acquire()

        def __exit__(self, *_):
            self.lock.release()

    gate = PauseReceiverLock()
    agent._lock = gate
    try:
        ws.responses.put(response(ws.episode_id, 1))
        assert gate.waiting.wait(5)  # Decoded old response, not yet holding the lock.
        agent.reset()
        current = ws.episode_id
        gate.proceed.set()
        ws.responses.put(response(current, 2))
        assert ws.accepted.wait(5)
        assert [ack["request_id"] for ack in ws.acks] == [2]
        assert [chunk.episode_id for chunk in agent.action_chunks] == [current]
    finally:
        gate.proceed.set()
        agent.close()
