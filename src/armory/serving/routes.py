"""HTTP control-plane routes and dashboard mounting for the policy server."""

from __future__ import annotations

import logging
import queue
from dataclasses import asdict

from fastapi import FastAPI, HTTPException, Request
from starlette.middleware.wsgi import WSGIMiddleware

from armory.serving.metrics import MetricsStore
from armory.serving.metrics.dash_app import create_dash_app
from armory.serving.protocol import SchedulerConfig, ServerMetadata
from armory.serving.runtime import ServerState
from armory.serving.scheduler import SCHEDULER_REGISTRY
from armory.serving.schemas import Reconfigure, ResetAll

# Keep existing log attribution while this code moves out of server.py.
logger = logging.getLogger("armory.serving.server")


def register_routes(
    app: FastAPI,
    metadata: ServerMetadata,
    metrics_store: MetricsStore,
) -> None:
    """Register HTTP APIs before mounting the dashboard at the root path."""

    # can also be used for health check
    @app.get("/metadata")
    async def server_metadata(request: Request) -> dict:
        state: ServerState | None = getattr(request.app.state, "server", None)
        payload = asdict(metadata)
        if state is not None:
            payload["scheduling_algorithm"] = state.current_algorithm
            payload["scheduler_kwargs"] = dict(state.current_scheduler_kwargs)
        return payload

    @app.post("/reconfigure")
    async def reconfigure(request: Request) -> dict:
        """Swap the scheduler's algorithm and/or multipliers in place.

        Body: ``{"scheduling_algorithm": str?, "action_horizon_multipliers": dict?}``.
        Either field is optional; omitted fields preserve the current value.
        Returns the resulting effective scheduler_kwargs.
        """
        state: ServerState = request.app.state.server
        body = await request.json() if await request.body() else {}
        algorithm = body.get("scheduling_algorithm") or state.current_algorithm
        if algorithm not in SCHEDULER_REGISTRY:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown scheduling_algorithm {algorithm!r}; "
                f"available: {sorted(SCHEDULER_REGISTRY)}",
            )

        if "action_horizon_multipliers" in body and body["action_horizon_multipliers"] is not None:
            try:
                multipliers = {
                    int(k): float(v) for k, v in body["action_horizon_multipliers"].items()
                }
            except (TypeError, ValueError, AttributeError) as e:
                raise HTTPException(
                    status_code=400,
                    detail=f"action_horizon_multipliers must be a dict of int->float pairs ({e})",
                ) from e
        else:
            multipliers = dict(
                state.current_scheduler_kwargs.get("action_horizon_multipliers")
                or state.boot_action_horizon_multipliers
            )

        config = SchedulerConfig(
            scheduling_algorithm=algorithm,
            alpha=state.boot_alpha,
            action_horizon_multipliers=multipliers,
        )
        kwargs = config.to_scheduler_kwargs() or {}

        await state.scheduler_sock.send_pyobj(
            Reconfigure(algorithm=algorithm, scheduler_kwargs=dict(kwargs))
        )
        state.current_algorithm = algorithm
        state.current_scheduler_kwargs = dict(kwargs)
        logger.info(
            "Reconfigure requested: algorithm=%s scheduler_kwargs=%s",
            algorithm,
            kwargs,
        )
        return {
            "status": "ok",
            "scheduling_algorithm": algorithm,
            "scheduler_kwargs": dict(kwargs),
        }

    @app.get("/")
    async def get_metrics(
        request: Request, window_s: float | None = None, sla_pct: float = 10.0
    ) -> dict:
        return request.app.state.server.metrics_store.snapshot(window_s, sla_pct=sla_pct)

    @app.get("/save-metrics")
    async def save_metrics(request: Request) -> dict:
        # TODO: server-side normalization may be needed if this endpoint becomes
        # the source for Prometheus-style metric export.
        return asdict(request.app.state.server.metrics_store)

    @app.post("/reset")
    async def reset_metrics(request: Request) -> dict:
        state: ServerState = request.app.state.server
        # Drain queued GPU work before clearing scheduler in-flight state.
        drained = 0
        while True:
            try:
                state.batch_queue.get_nowait()
                drained += 1
            except queue.Empty:
                break
        if drained:
            logger.info("Reset: drained %d pending batches from queue", drained)
        await state.scheduler_sock.send_pyobj(ResetAll())
        state.metrics_store.reset()
        return {"status": "ok", "drained_batches": drained}

    dash_app = create_dash_app(metadata, metrics_store)
    app.mount("/", WSGIMiddleware(dash_app.server))
