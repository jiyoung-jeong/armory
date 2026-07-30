"""Module-boundary checks for the serving composition facade."""

from __future__ import annotations

import armory.serving.server as server
from armory.serving.config import ServerConfig
from armory.serving.protocol import SchedulerConfig, ServerMetadata


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


def test_create_app_only_composes_runtime_until_lifespan_starts() -> None:
    app = server.create_app(
        _metadata(),
        _UnusedPolicyFactory(),
        ServerConfig(scheduler=SchedulerConfig(scheduling_algorithm="round-robin")),
    )

    assert not hasattr(app.state, "server")
    application_paths = [
        route.path
        for route in app.routes
        if route.path in {"/ws", "/metadata", "/reconfigure", "/reset"}
    ]
    assert application_paths == [
        "/ws",
        "/metadata",
        "/reconfigure",
        "/reset",
    ]
