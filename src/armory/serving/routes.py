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
from armory.serving.schemas import PrepareAck, PrepareScheduler, Reconfigure, ResetAll
from armory.serving.server_runtime import ServerState

# Keep existing log attribution while this code moves out of server.py.
logger = logging.getLogger("armory.serving.server")
PREPARE_TIMEOUT_S = 60.0


def _resolve_scheduler_config(
    body: dict[str, Any], state: ServerState, *, allow_alpha: bool = False
) -> tuple[str, dict[str, Any]]:
    algorithm = body.get("scheduling_algorithm") or state.current_algorithm
    if algorithm not in SCHEDULER_REGISTRY:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown scheduling_algorithm {algorithm!r}; "
            f"available: {sorted(SCHEDULER_REGISTRY)}",
        )

    if "action_horizon_multipliers" in body and body["action_horizon_multipliers"] is not None:
        try:
            multipliers = {int(k): float(v) for k, v in body["action_horizon_multipliers"].items()}
        except (TypeError, ValueError, AttributeError) as exc:
            raise HTTPException(
                status_code=400,
                detail=f"action_horizon_multipliers must be a dict of int->float pairs ({exc})",
            ) from exc
    else:
        multipliers = dict(
            state.current_scheduler_kwargs.get("action_horizon_multipliers")
            or state.boot_action_horizon_multipliers
        )

    alpha = state.boot_alpha
    if allow_alpha:
        try:
            alpha = float(body.get("alpha", state.current_scheduler_kwargs.get("alpha", alpha)))
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=f"alpha must be a float ({exc})") from exc

    config = SchedulerConfig(
        scheduling_algorithm=algorithm,
        alpha=alpha,
        action_horizon_multipliers=multipliers,
    )
    return algorithm, dict(config.to_scheduler_kwargs() or {})


def _drain_batches(state: ServerState) -> int:
    drained = 0
    while True:
        try:
            state.batch_queue.get_nowait()
            drained += 1
        except queue.Empty:
            return drained


async def _wait_for_prepare(state: ServerState, operation_id: str) -> list[str]:
    pending = {"scheduler", "gpu", "router"}
    deadline = asyncio.get_running_loop().time() + PREPARE_TIMEOUT_S
    while pending:
        remaining = deadline - asyncio.get_running_loop().time()
        if remaining <= 0:
            raise HTTPException(
                status_code=504,
                detail=f"Prepare operation {operation_id} timed out waiting for {sorted(pending)}",
            )
        try:
            ack = state.control_ack_queue.get_nowait()
        except queue.Empty:
            await asyncio.sleep(min(0.05, remaining))
            continue
        if not isinstance(ack, PrepareAck) or ack.operation_id != operation_id:
            logger.warning("Discarding stale prepare acknowledgment: %r", ack)
            continue
        if ack.error is not None:
            raise HTTPException(
                status_code=500,
                detail=f"{ack.worker} failed prepare operation {operation_id}: {ack.error}",
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
        if not isinstance(body, dict):
            raise HTTPException(status_code=400, detail="Request body must be a JSON object")

        async with state.control_lock:
            algorithm, kwargs = _resolve_scheduler_config(body, state)
            await state.scheduler_sock.send_pyobj(
                Reconfigure(algorithm=algorithm, scheduler_kwargs=kwargs)
            )
            state.current_algorithm = algorithm
            state.current_scheduler_kwargs = kwargs
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

    @app.post("/prepare")
    async def prepare_run(request: Request) -> dict:
        """Reconfigure and clear all between-run state with worker acknowledgments."""
        state: ServerState = request.app.state.server
        body = await request.json() if await request.body() else {}
        if not isinstance(body, dict):
            raise HTTPException(status_code=400, detail="Request body must be a JSON object")

        async with state.control_lock:
            if state.response_queues:
                raise HTTPException(
                    status_code=409,
                    detail="Cannot prepare while robot sessions are active: "
                    f"{sorted(state.response_queues)}",
                )
            algorithm, kwargs = _resolve_scheduler_config(body, state, allow_alpha=True)
            operation_id = uuid.uuid4().hex
            drained = _drain_batches(state)
            await state.scheduler_sock.send_pyobj(
                PrepareScheduler(
                    operation_id=operation_id,
                    algorithm=algorithm,
                    scheduler_kwargs=kwargs,
                )
            )
            acknowledged = await _wait_for_prepare(state, operation_id)
            state.current_algorithm = algorithm
            state.current_scheduler_kwargs = kwargs

        logger.info(
            "Prepared next run: operation_id=%s algorithm=%s scheduler_kwargs=%s",
            operation_id,
            algorithm,
            kwargs,
        )
        return {
            "status": "ok",
            "operation_id": operation_id,
            "scheduling_algorithm": algorithm,
            "scheduler_kwargs": kwargs,
            "drained_batches": drained,
            "acknowledged": acknowledged,
        }

    @app.post("/reset")
    async def reset_server(request: Request) -> dict:
        state: ServerState = request.app.state.server
        async with state.control_lock:
            # Drain queued GPU work before clearing scheduler in-flight state.
            drained = _drain_batches(state)
            await state.scheduler_sock.send_pyobj(ResetAll())
        if drained:
            logger.info("Reset: drained %d pending batches from queue", drained)
        return {"status": "ok", "drained_batches": drained}
