"""
3 processes:
    WS main process     - FastAPI ASGI app
    Scheduler process   - collects requests from WS main; runs scheduler; dispatches batches to GPU
    GPU process         - loads weights; runs batches; sends responses directly to WS main

ZMQ topology (all ipc://, unique per server instance):
    WS main  ──[PUB: SlotRequest / ResetRequest / AckNotification / WarmupSeed / control]──► Scheduler [binds server_out_ep]
    WS main  ──slots.write()───────────────────────► mp.RawArray shared memory
    GPU      ──[PUB: ResponseBatch]──────────────► WS main, Scheduler   [binds gpu_out_ep]
    Scheduler ──[mp.Queue: RequestBatch]────────────► GPU
    GPU      ──slots.read()──────────────────────────► mp.RawArray shared memory

    A single _router_task in WS main reads from gpu_out_ep and dispatches to per-robot queues.
    Large numpy arrays (observations) cross zero process boundaries via ZMQ.

This module is the stable composition facade. Per-connection protocol handling
lives in ``session.py``; process/IPC lifecycle in ``runtime.py``; and HTTP
control-plane routes in ``routes.py``.
"""

from __future__ import annotations

import itertools
import multiprocessing as mp

import uvicorn
from fastapi import FastAPI, WebSocket

from armory.backends.types import PolicyFactory
from armory.serving.metrics import MetricsStore
from armory.serving.protocol import ServerMetadata
from armory.serving.routes import register_routes
from armory.serving.runtime import (
    MAX_ROBOTS,
    BackendResources,
    ServerState,
    _router_task,  # noqa: F401 - explicit legacy import path
    _scheduler_metrics_task,  # noqa: F401 - explicit legacy import path
    _start_backend,
    _watchdog_task,  # noqa: F401 - explicit legacy import path
    create_lifespan,
    socket_addresses,
)
from armory.serving.schemas import RobotID
from armory.serving.session import (
    _handshake as _ws_handshake,  # noqa: F401 - explicit legacy import path
)
from armory.serving.session import (
    _warmup,
    serve_websocket_session,
)

NUM_WARMUP = 100
_request_id_counter = itertools.count(1)


def _deferred_start_backend(
    metadata: ServerMetadata,
    policy_factory: PolicyFactory,
    scheduler_kwargs: dict[str, object] | None,
    log_queue: mp.Queue | None,
) -> BackendResources:
    """Resolve the compatibility binding when ASGI lifespan actually starts."""
    return _start_backend(metadata, policy_factory, scheduler_kwargs, log_queue)


async def _ws_warmup(
    websocket: WebSocket,
    state: ServerState,
    robot_id: RobotID,
    action_payload_size: int,
) -> None:
    """Compatibility wrapper for the former server-local warmup helper."""
    await _warmup(websocket, state, robot_id, action_payload_size, NUM_WARMUP)


def create_app(
    metadata: ServerMetadata,
    policy_factory: PolicyFactory,
    scheduler_kwargs: dict[str, object] | None = None,
    log_queue: mp.Queue | None = None,
) -> FastAPI:
    """Compose the server runtime, WebSocket transport, and HTTP routes."""
    metrics_store = MetricsStore()
    lifespan = create_lifespan(
        metadata,
        policy_factory,
        scheduler_kwargs,
        log_queue,
        metrics_store,
        start_backend=_deferred_start_backend,
    )
    app = FastAPI(lifespan=lifespan)

    @app.websocket("/ws")
    async def ws_handler(websocket: WebSocket) -> None:
        await websocket.accept()
        state: ServerState = websocket.app.state.server
        await serve_websocket_session(
            websocket,
            state,
            action_payload_size=metadata.action_horizon * metadata.action_dim * 4,
            num_warmup=NUM_WARMUP,
            request_ids=_request_id_counter,
        )

    register_routes(app, metadata, metrics_store)
    return app


class PolicyServer:
    def __init__(
        self,
        metadata: ServerMetadata,
        policy_factory: PolicyFactory,
        scheduler_kwargs: dict[str, object] | None = None,
        log_queue: mp.Queue | None = None,
    ):
        self._metadata = metadata
        self._policy_factory = policy_factory
        self._scheduler_kwargs = scheduler_kwargs
        self._log_queue = log_queue

    def serve_forever(self, host="0.0.0.0", port=8000):
        app = create_app(
            self._metadata,
            self._policy_factory,
            self._scheduler_kwargs,
            self._log_queue,
        )
        uvicorn.run(app, host=host, port=port)


__all__ = [
    "MAX_ROBOTS",
    "NUM_WARMUP",
    "PolicyServer",
    "ServerState",
    "create_app",
    "socket_addresses",
]
