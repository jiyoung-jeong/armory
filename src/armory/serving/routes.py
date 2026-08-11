"""HTTP control-plane routes and dashboard mounting for the policy server."""

from __future__ import annotations

import json
import logging
import math
import queue
import time
from dataclasses import asdict, replace

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

    @app.patch("/robots/{robot_id}/weight")
    async def set_robot_weight(robot_id: str, request: Request) -> dict:
        state: ServerState = request.app.state.server
        metadata = state.robot_metadata.get(robot_id)
        if metadata is None:
            raise HTTPException(status_code=404, detail=f"Unknown robot {robot_id!r}")

        body = await request.json()
        try:
            weight = float(body["weight"])
        except (KeyError, TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail="weight must be a number") from exc
        if not math.isfinite(weight) or weight <= 0.0:
            raise HTTPException(status_code=400, detail="weight must be positive and finite")

        old_weight = metadata.weight
        applied_at = time.time()
        state.robot_metadata[robot_id] = replace(metadata, weight=weight)
        state.events_log.write(
            json.dumps(
                {
                    "kind": "weight_switch",
                    "robot_id": robot_id,
                    "old_weight": old_weight,
                    "weight": weight,
                    "applied_at": applied_at,
                }
            )
            + "\n"
        )
        state.events_log.flush()
        logger.info("Updated %s weight: %g -> %g", robot_id, old_weight, weight)
        return {
            "status": "ok",
            "robot_id": robot_id,
            "old_weight": old_weight,
            "weight": weight,
            "applied_at": applied_at,
        }
