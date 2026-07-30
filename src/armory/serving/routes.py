"""HTTP control-plane routes and dashboard mounting for the policy server."""

from __future__ import annotations

import logging
import queue
from dataclasses import asdict

from fastapi import FastAPI, HTTPException, Request

from armory.serving.protocol import SchedulerConfig, ServerMetadata
from armory.serving.scheduler import SCHEDULER_REGISTRY
from armory.serving.schemas import Reconfigure, ResetAll
from armory.serving.server_runtime import ServerState

# Keep existing log attribution while this code moves out of server.py.
logger = logging.getLogger("armory.serving.server")


def register_routes(
    app: FastAPI,
    metadata: ServerMetadata,
) -> None:
    """Register server metadata and scheduler control APIs."""

    # can also be used for health check
    @app.get("/metadata")
    async def server_metadata(request: Request) -> dict:
        state: ServerState | None = getattr(request.app.state, "server", None)
        payload = asdict(metadata)
        if state is not None:
            payload["scheduling_algorithm"] = state.current_scheduler.scheduling_algorithm
            payload["scheduler"] = state.current_scheduler.model_dump()
        return payload

    @app.post("/reconfigure")
    async def reconfigure(request: Request) -> dict:
        """Swap the scheduler's algorithm and/or multipliers in place.

        Body: ``{"scheduling_algorithm": str?, "action_horizon_multipliers": dict?}``.
        Either field is optional; omitted fields preserve the current value.
        Returns the resulting effective SchedulerConfig.
        """
        state: ServerState = request.app.state.server
        body = await request.json() if await request.body() else {}
        algorithm = body.get("scheduling_algorithm") or state.current_scheduler.scheduling_algorithm
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
            multipliers = dict(state.current_scheduler.action_horizon_multipliers)

        config = SchedulerConfig(
            scheduling_algorithm=algorithm,
            alpha=state.current_scheduler.alpha,
            action_horizon_multipliers=multipliers,
        )

        await state.scheduler_sock.send_pyobj(Reconfigure(config=config))
        state.current_scheduler = config
        logger.info("Reconfigure requested: %s", config)
        return {
            "status": "ok",
            "scheduling_algorithm": algorithm,
            "scheduler": config.model_dump(),
        }

    @app.post("/reset")
    async def reset_server(request: Request) -> dict:
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
        return {"status": "ok", "drained_batches": drained}
