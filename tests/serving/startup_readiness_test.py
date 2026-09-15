"""Exercise the startup barrier using real ZMQ subscription handshakes."""

import asyncio
import uuid

import pytest
import zmq.asyncio

from armory.serving.server_runtime import _wait_for_subscribers


def test_broadcast_waits_for_both_workers_before_first_control_message():
    async def scenario():
        context = zmq.asyncio.Context()
        publisher = context.socket(zmq.XPUB)
        publisher.setsockopt(zmq.XPUB_VERBOSER, 1)
        endpoint = f"inproc://readiness-{uuid.uuid4().hex}"
        publisher.bind(endpoint)
        subscribers = [context.socket(zmq.SUB) for _ in range(2)]
        for subscriber in subscribers:
            subscriber.setsockopt(zmq.SUBSCRIBE, b"")
        ready = asyncio.create_task(_wait_for_subscribers(publisher, expected=2, timeout=2))
        try:
            subscribers[0].connect(endpoint)
            with pytest.raises(TimeoutError):
                await asyncio.wait_for(asyncio.shield(ready), timeout=0.05)
            subscribers[1].connect(endpoint)
            await ready
            await publisher.send_pyobj({"control": "warmup_seed"})
            received = await asyncio.wait_for(
                asyncio.gather(*(s.recv_pyobj() for s in subscribers)), timeout=2
            )
            assert received == [{"control": "warmup_seed"}] * 2
        finally:
            ready.cancel()
            await asyncio.gather(ready, return_exceptions=True)
            for socket in [publisher, *subscribers]:
                socket.close(linger=0)
            context.term()

    asyncio.run(scenario())


def test_missing_worker_subscription_times_out():
    async def scenario():
        context = zmq.asyncio.Context()
        publisher = context.socket(zmq.XPUB)
        publisher.setsockopt(zmq.XPUB_VERBOSER, 1)
        publisher.bind(f"inproc://readiness-{uuid.uuid4().hex}")
        try:
            with pytest.raises(TimeoutError):
                await _wait_for_subscribers(publisher, expected=2, timeout=0.05)
        finally:
            publisher.close(linger=0)
            context.term()

    asyncio.run(scenario())


def test_lifespan_cleans_up_workers_when_subscription_barrier_fails(monkeypatch, tmp_path):
    from types import SimpleNamespace
    from unittest.mock import Mock

    from armory.serving import server_runtime
    from armory.serving.config import ServerConfig

    context = zmq.asyncio.Context()
    monkeypatch.setattr(server_runtime.zmq.asyncio, "Context", lambda: context)
    monkeypatch.setattr(
        server_runtime,
        "socket_addresses",
        {name: f"inproc://{name}-{uuid.uuid4().hex}" for name in ("server_out_ep", "gpu_out_ep")},
    )

    async def missing_subscription(*args, **kwargs):
        raise TimeoutError("missing worker subscription")

    monkeypatch.setattr(server_runtime, "_wait_for_subscribers", missing_subscription)
    workers = [Mock(), Mock()]
    for worker in workers:
        worker.is_alive.return_value = False
    ready = Mock()
    resources = (*workers, object(), ready, ready, object())
    lifespan = server_runtime.create_lifespan(
        None, None, ServerConfig(output_dir=tmp_path), None, start_backend=lambda *args: resources
    )

    async def scenario():
        with pytest.raises(TimeoutError, match="missing worker subscription"):
            async with lifespan(SimpleNamespace()):
                pytest.fail("Server must not accept clients before subscriptions are ready")

    asyncio.run(scenario())
    assert context.closed
    for worker in workers:
        worker.terminate.assert_called_once()
        worker.join.assert_called_once_with(5)
