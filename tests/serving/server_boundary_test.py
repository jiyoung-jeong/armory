"""Module-boundary checks for the serving composition facade."""

from __future__ import annotations

import pytest
from starlette.routing import Mount

import armory.serving.runtime as runtime
import armory.serving.server as server
import armory.serving.session as session
from armory.serving.protocol import ServerMetadata


class _UnusedPolicyFactory:
    def __call__(self) -> None:
        raise AssertionError("policy construction belongs inside the app lifespan")


def _metadata() -> ServerMetadata:
    return ServerMetadata(
        config_name="boundary-test",
        checkpoint_dir="",
        action_horizon=4,
        action_dim=2,
        num_steps=1,
        max_batch_size=2,
        env="TEST",
        scheduling_algorithm="round-robin",
    )


def test_server_facade_retains_runtime_compatibility_exports() -> None:
    assert server.MAX_ROBOTS == runtime.MAX_ROBOTS
    assert server.ServerState is runtime.ServerState
    assert server.socket_addresses is runtime.socket_addresses
    assert server._start_backend is runtime._start_backend
    assert server._router_task is runtime._router_task
    assert server._scheduler_metrics_task is runtime._scheduler_metrics_task
    assert server._watchdog_task is runtime._watchdog_task
    assert server._ws_handshake is session._handshake


def test_create_app_only_composes_runtime_until_lifespan_starts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_if_started(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise AssertionError("create_app started backend processes before ASGI lifespan")

    monkeypatch.setattr(server, "_start_backend", fail_if_started)
    app = server.create_app(_metadata(), _UnusedPolicyFactory())

    assert not hasattr(app.state, "server")
    application_paths = [
        route.path
        for route in app.routes
        if route.path in {"/ws", "/metadata", "/reconfigure", "/", "/save-metrics", "/reset"}
    ]
    assert application_paths == [
        "/ws",
        "/metadata",
        "/reconfigure",
        "/",
        "/save-metrics",
        "/reset",
    ]
    assert isinstance(app.routes[-1], Mount)
    assert app.routes[-1].path == ""
