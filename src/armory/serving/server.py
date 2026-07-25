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
lives in ``session.py``; process/IPC lifecycle in ``server_runtime.py``; and HTTP
control-plane routes in ``routes.py``.
"""

from __future__ import annotations

import itertools
import multiprocessing as mp

import uvicorn
from fastapi import FastAPI, WebSocket

from armory.backends.types import PolicyFactory
from armory.serving import server_runtime as _runtime
from armory.serving.protocol import ServerMetadata
from armory.serving.routes import register_routes
from armory.serving.session import serve_websocket_session

NUM_WARMUP = 100
_request_id_counter = itertools.count(1)


def create_app(
    metadata: ServerMetadata,
    policy_factory: PolicyFactory,
    scheduler_kwargs: dict[str, object] | None = None,
    log_queue: mp.Queue | None = None,
) -> FastAPI:
    """Compose the server runtime, WebSocket transport, and HTTP routes."""
    lifespan = _runtime.create_lifespan(
        metadata,
        policy_factory,
        scheduler_kwargs,
        log_queue,
    )
    app = FastAPI(lifespan=lifespan)

    @app.websocket("/ws")
    async def ws_handler(websocket: WebSocket) -> None:
        await websocket.accept()
        state: _runtime.ServerState = websocket.app.state.server
        await serve_websocket_session(
            websocket,
            state,
            action_payload_size=metadata.action_horizon * metadata.action_dim * 4,
            num_warmup=NUM_WARMUP,
            request_ids=_request_id_counter,
        )

    register_routes(app, metadata)
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
    "NUM_WARMUP",
    "PolicyServer",
    "create_app",
]
