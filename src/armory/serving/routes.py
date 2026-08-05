"""HTTP control-plane routes and dashboard mounting for the policy server."""

from __future__ import annotations

import logging
import queue
from dataclasses import asdict

from fastapi import FastAPI, HTTPException, Request

from armory.serving.protocol import SchedulerConfig, ServerMetadata
from armory.serving.scheduler import SCHEDULER_REGISTRY
from armory.serving.schemas import Reconfigure, ResetAll
from armory.serving.server_runtime import ServerState, write_metadata

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
            payload["scheduling_algorithm"] = state.config.scheduler.scheduling_algorithm
            payload["scheduler"] = state.config.scheduler.model_dump()
        return payload

    @app.post("/reconfigure")
    async def reconfigure(request: Request) -> dict:
        """Swap the scheduler's algorithm in place.

        Body: ``{"scheduling_algorithm": str?}``, optional; an omitted field
        preserves the current value. Returns the effective SchedulerConfig.
        """
        state: ServerState = request.app.state.server
        body = await request.json() if await request.body() else {}
        algorithm = body.get("scheduling_algorithm") or state.config.scheduler.scheduling_algorithm
        if algorithm not in SCHEDULER_REGISTRY:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown scheduling_algorithm {algorithm!r}; "
                f"available: {sorted(SCHEDULER_REGISTRY)}",
            )

        config = state.config.model_copy(
            update={
                "scheduler": SchedulerConfig(
                    scheduling_algorithm=algorithm,
                )
            }
        )

        await state.scheduler_sock.send_pyobj(Reconfigure(config=config))
        state.config = config
        write_metadata(state, metadata)
        logger.info("Reconfigure requested: %s", config.scheduler)
        return {
            "status": "ok",
            "scheduling_algorithm": algorithm,
            "scheduler": config.scheduler.model_dump(),
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
