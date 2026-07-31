"""HTTP control-plane routes and dashboard mounting for the policy server."""

from __future__ import annotations

import asyncio
import logging
import queue
import uuid
from dataclasses import asdict
from typing import Any

from fastapi import FastAPI, HTTPException, Request

from armory.serving.protocol import SchedulerConfig, ServerMetadata
from armory.serving.scheduler import SCHEDULER_REGISTRY
from armory.serving.schemas import PrepareAck, PrepareScheduler, Reconfigure
from armory.serving.server_runtime import ServerState, write_metadata

# Keep existing log attribution while this code moves out of server.py.
logger = logging.getLogger("armory.serving.server")
RESET_TIMEOUT_S = 60.0


def _resolve_scheduler_config(
    body: dict[str, Any], state: ServerState, *, allow_alpha: bool = False
) -> SchedulerConfig:
    algorithm = body.get("scheduling_algorithm") or state.config.scheduler.scheduling_algorithm
    if not isinstance(algorithm, str) or algorithm not in SCHEDULER_REGISTRY:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown scheduling_algorithm {algorithm!r}; "
            f"available: {sorted(SCHEDULER_REGISTRY)}",
        )

    alpha = state.config.scheduler.alpha
    if allow_alpha:
        try:
            alpha = float(body.get("alpha", alpha))
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=f"alpha must be a float ({exc})") from exc

    return SchedulerConfig(
        scheduling_algorithm=algorithm,
        alpha=alpha,
    )


def _drain_batches(state: ServerState) -> int:
    drained = 0
    while True:
        try:
            state.batch_queue.get_nowait()
            drained += 1
        except queue.Empty:
            return drained


async def _wait_for_reset(state: ServerState, operation_id: str) -> list[str]:
    pending = {"scheduler", "gpu", "router"}
    deadline = asyncio.get_running_loop().time() + RESET_TIMEOUT_S
    while pending:
        remaining = deadline - asyncio.get_running_loop().time()
        if remaining <= 0:
            raise HTTPException(
                status_code=504,
                detail=f"Reset operation {operation_id} timed out waiting for {sorted(pending)}",
            )
        try:
            ack = state.control_ack_queue.get_nowait()
        except queue.Empty:
            await asyncio.sleep(min(0.05, remaining))
            continue
        if not isinstance(ack, PrepareAck) or ack.operation_id != operation_id:
            logger.warning("Discarding stale reset acknowledgment: %r", ack)
            continue
        if ack.error is not None:
            raise HTTPException(
                status_code=500,
                detail=f"{ack.worker} failed reset operation {operation_id}: {ack.error}",
            )
        pending.discard(ack.worker)
    return ["scheduler", "gpu", "router"]


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
        if not isinstance(body, dict):
            raise HTTPException(status_code=400, detail="Request body must be a JSON object")

        async with state.control_lock:
            scheduler_config = _resolve_scheduler_config(body, state)
            config = state.config.model_copy(update={"scheduler": scheduler_config})
            await state.scheduler_sock.send_pyobj(Reconfigure(config=config))
            state.config = config
            write_metadata(state, metadata)
        logger.info("Reconfigure requested: %s", config.scheduler)
        return {
            "status": "ok",
            "scheduling_algorithm": scheduler_config.scheduling_algorithm,
            "scheduler": config.scheduler.model_dump(),
        }

    @app.post("/reset")
    async def reset_server(request: Request) -> dict:
        """Clear all run state and optionally reconfigure, with worker acknowledgments."""
        state: ServerState = request.app.state.server
        body = await request.json() if await request.body() else {}
        if not isinstance(body, dict):
            raise HTTPException(status_code=400, detail="Request body must be a JSON object")

        async with state.control_lock:
            if state.response_queues:
                raise HTTPException(
                    status_code=409,
                    detail="Cannot reset while robot sessions are active: "
                    f"{sorted(state.response_queues)}",
                )
            scheduler_config = _resolve_scheduler_config(body, state, allow_alpha=True)
            config = state.config.model_copy(update={"scheduler": scheduler_config})
            operation_id = uuid.uuid4().hex
            drained = _drain_batches(state)
            await state.scheduler_sock.send_pyobj(
                PrepareScheduler(
                    operation_id=operation_id,
                    config=config,
                )
            )
            acknowledged = await _wait_for_reset(state, operation_id)
            state.config = config
            write_metadata(state, metadata)

        logger.info(
            "Reset server for next run: operation_id=%s scheduler=%s",
            operation_id,
            scheduler_config,
        )
        return {
            "status": "ok",
            "operation_id": operation_id,
            "scheduling_algorithm": scheduler_config.scheduling_algorithm,
            "scheduler": scheduler_config.model_dump(),
            "drained_batches": drained,
            "acknowledged": acknowledged,
        }
