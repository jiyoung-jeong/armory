"""
3 processes:
    WS main process     - FastAPI ASGI app
    Scheduler process   - collects requests from WS main; runs scheduler; dispatches batches to GPU
    GPU process         - loads weights; runs batches; sends responses directly to WS main

ZMQ topology (all ipc://, unique per server instance):
    WS main  ──[PUB: SlotRequest / ResetRequest / AckNotification / WarmupPing]──► Scheduler [binds server_out_ep]
    WS main  ──slots.write()───────────────────────► mp.RawArray shared memory
    GPU      ──[PUB: ResponseBatch]──────────────► WS main, Scheduler   [binds gpu_out_ep]
    Scheduler ──[mp.Queue: list[SlotRequest]]───────► GPU
    GPU      ──slots.read()──────────────────────────► mp.RawArray shared memory

    A single _router_task in WS main reads from gpu_out_ep and dispatches to per-robot queues.
    Large numpy arrays (observations) cross zero process boundaries via ZMQ.
"""

from __future__ import annotations

import asyncio
import dataclasses
import itertools
import logging
import multiprocessing as mp
import os
import queue
import signal
import time
import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass
from multiprocessing.synchronize import Event
from typing import Any

import uvicorn
import zmq.asyncio
from fastapi import FastAPI, HTTPException, Request, WebSocket
from fastapi.concurrency import asynccontextmanager
from starlette.middleware.wsgi import WSGIMiddleware
from starlette.websockets import WebSocketDisconnect

from armory.serving.engine import GpuWorker
from armory.serving.metrics import MetricsStore
from armory.serving.metrics.dash_app import create_dash_app
from armory.serving.scheduler import SCHEDULER_REGISTRY, SchedulerWorker
from armory.serving.schemas import (
    AckNotification,
    BatchProfile,
    Reconfigure,
    ResetAll,
    ResponseBatch,
    RobotID,
    SchedulerDecision,
    SlotRequest,
    WarmupSeed,
)
from armory.serving.slots import RobotSlots, SlotData
from armory_client import msgpack_numpy
from armory_client.messages import (
    ConnectRequest,
    ConnectResponse,
    EpisodeEnd,
    EpisodeStart,
    EpisodeStep,
    InferRequest,
    InferResponse,
    ResetRequest,
    ResponseAck,
    WarmupPong,
)
from armory_client.protocol import SchedulerConfig, ServerMetadata

MAX_ROBOTS = 100
NUM_WARMUP = 100
logger = logging.getLogger(__name__)

_uid = uuid.uuid4().hex[:8]
socket_addresses = {
    "server_out_ep": f"ipc:///tmp/openpi_server_out_{_uid}",
    "gpu_out_ep": f"ipc:///tmp/openpi_gpu_out_{_uid}",
}

_request_id_counter = itertools.count(1)


@dataclass
class ServerState:
    scheduler_sock: zmq.asyncio.Socket  # PUSH to scheduler
    response_queues: dict[str, asyncio.Queue]
    slots: RobotSlots  # WS manages slot allocation
    gpu_proc: mp.Process
    scheduler_proc: mp.Process
    metrics_store: MetricsStore
    robot_metadata: dict[str, ConnectRequest]
    batch_queue: mp.Queue  # exposed so /reset can drain stale work between trials
    # Current effective scheduler config. Mutated by POST /reconfigure so
    # /metadata always reports what the scheduler subprocess is actually using.
    # boot_alpha is preserved across reconfigures (alpha is server-startup-only).
    current_algorithm: str
    current_scheduler_kwargs: dict[str, Any]
    boot_alpha: float
    boot_action_horizon_multipliers: dict[int, float]


async def _router_task(
    response_sock: zmq.asyncio.Socket,
    response_queues: dict[str, asyncio.Queue],
    metrics_store: MetricsStore,
) -> None:
    """Reads batches of InferResponses directly from GPU and dispatches to per-robot queues."""
    logger.info("Router task starting")
    while True:
        try:
            msg: ResponseBatch | BatchProfile = await response_sock.recv_pyobj()
            if isinstance(msg, BatchProfile):
                continue
            assert isinstance(msg, ResponseBatch)
            logger.debug("Received response batch: %s", msg)

            metrics_store.record_batch(msg)
            for response in msg.responses:
                queue = response_queues.get(response.robot_id)
                if queue is not None:
                    await queue.put(response)
                    logger.debug("Put response in queue: %s", response)
                else:
                    logger.info(
                        "No active connection for robot %s, dropping response",
                        response.robot_id,
                    )
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Router task error")


async def _ws_handshake(
    websocket: WebSocket,
    state: ServerState,
) -> tuple[str, int, ConnectRequest] | None:
    """Phase 1: receive ConnectRequest (with client-provided robot_id), confirm connection.

    Returns (robot_id, slot_index, connect_req) on success, or None if the
    client sent an unexpected first message (websocket is closed before returning).
    """
    raw = await websocket.receive_bytes()
    msg = msgpack_numpy.unpackb(raw)
    if msg.get("type") != "connect":
        await websocket.close(code=1002, reason="expected connect message")
        return None
    connect_req = ConnectRequest(**{k: v for k, v in msg.items() if k != "type"})

    robot_id = connect_req.robot_id
    slot_index = state.slots.register(robot_id)
    state.response_queues[robot_id] = asyncio.Queue()
    state.robot_metadata[robot_id] = connect_req

    await websocket.send_bytes(msgpack_numpy.packb(ConnectResponse()))
    logger.info("Robot %s connected (control_hz=%.1f)", robot_id, connect_req.control_hz)
    return robot_id, slot_index, connect_req


async def _ws_warmup(
    websocket: WebSocket,
    state: ServerState,
    robot_id: RobotID,
    action_payload_size: int,
) -> None:
    """Phase 2: NUM_WARMUP ping/pong round trips to seed LatencyTracker."""
    obs_samples: list[tuple[float, float]] = []
    delivery_samples: list[tuple[float, float]] = []

    for _ in range(NUM_WARMUP):
        raw = await websocket.receive_bytes()
        server_receive_time = time.time()
        msg = msgpack_numpy.unpackb(raw)
        if msg.get("type") != "warmup_ping":
            break

        server_send_time = time.time()
        pong = WarmupPong(
            client_timestamp=msg["client_timestamp"],
            server_receive_time=server_receive_time,
            server_send_time=server_send_time,
            payload=bytes(action_payload_size),
        )
        await websocket.send_bytes(msgpack_numpy.packb(pong))
        obs_samples.append((server_receive_time, msg["client_timestamp"]))

        ack_raw = await websocket.receive_bytes()
        ack_msg = msgpack_numpy.unpackb(ack_raw)
        if ack_msg.get("type") == "warmup_ack":
            delivery_samples.append((ack_msg["client_receive_time"], ack_msg["server_send_time"]))

    if obs_samples or delivery_samples:
        await state.scheduler_sock.send_pyobj(
            WarmupSeed(
                robot_id=robot_id,
                obs_samples=obs_samples,
                delivery_samples=delivery_samples,
            )
        )
        logger.info(
            "Robot %s warmup complete (%d obs, %d delivery samples)",
            robot_id,
            len(obs_samples),
            len(delivery_samples),
        )


async def _watchdog_task(gpu_proc: mp.Process, scheduler_proc: mp.Process) -> None:
    """Crashes the server if either backend process dies unexpectedly."""
    while True:
        await asyncio.sleep(1)
        for proc in (gpu_proc, scheduler_proc):
            if not proc.is_alive():
                logger.critical(
                    "Backend process %s died (exit code %s), crashing server",
                    proc.name,
                    proc.exitcode,
                )
                os.kill(os.getpid(), signal.SIGTERM)
                return


async def _scheduler_metrics_task(
    scheduler_metrics_queue: mp.Queue,
    metrics_store: MetricsStore,
) -> None:
    """Drain scheduler timing samples from the scheduler subprocess into MetricsStore."""
    while True:
        drained = False
        while True:
            try:
                samples: list[SchedulerDecision] = scheduler_metrics_queue.get_nowait()
            except queue.Empty:
                break
            metrics_store.record_scheduler_decisions(samples)
            drained = True
        await asyncio.sleep(0 if drained else 0.05)


def _start_backend(
    metadata: ServerMetadata,
    policy_factory: Callable,
    scheduler_kwargs: dict[str, object] | None,
    log_queue: mp.Queue | None,
) -> tuple[mp.Process, mp.Process, RobotSlots, Event, Event, mp.Queue, mp.Queue]:
    slots = RobotSlots(max_robots=MAX_ROBOTS)
    batch_queue: mp.Queue = mp.Queue()
    scheduler_metrics_queue: mp.Queue = mp.Queue()

    gpu_ready = mp.Event()
    sched_ready = mp.Event()

    gpu_proc = mp.Process(
        target=GpuWorker(
            policy_factory,
            metadata.max_batch_size,
            slots,
            batch_queue,
            socket_addresses["server_out_ep"],
            socket_addresses["gpu_out_ep"],
            gpu_ready,
            log_queue,
        ).run,
        daemon=True,
    )

    scheduler_proc = mp.Process(
        target=SchedulerWorker(
            socket_addresses["server_out_ep"],
            socket_addresses["gpu_out_ep"],
            batch_queue,
            scheduler_metrics_queue,
            metadata.max_batch_size,
            metadata.scheduling_algorithm,
            scheduler_kwargs,
            sched_ready,
            log_queue,
        ).run,
        daemon=True,
    )

    logger.info("Starting GPU subprocess…")
    gpu_proc.start()
    logger.info("Starting scheduler subprocess…")
    scheduler_proc.start()

    return (
        scheduler_proc,
        gpu_proc,
        slots,
        sched_ready,
        gpu_ready,
        scheduler_metrics_queue,
        batch_queue,
    )


def create_app(
    metadata: ServerMetadata,
    policy_factory: Callable,
    scheduler_kwargs: dict[str, object] | None = None,
    log_queue: mp.Queue | None = None,
) -> FastAPI:
    metrics_store = MetricsStore()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        (
            scheduler_proc,
            gpu_proc,
            slots,
            sched_ready,
            gpu_ready,
            scheduler_metrics_queue,
            batch_queue,
        ) = _start_backend(
            metadata,
            policy_factory,
            scheduler_kwargs,
            log_queue,
        )

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, sched_ready.wait)
        logger.info("Scheduler ready")
        await loop.run_in_executor(None, gpu_ready.wait)
        logger.info("GPU worker ready")

        zmq_ctx = zmq.asyncio.Context()

        scheduler_sock = zmq_ctx.socket(zmq.PUB)
        scheduler_sock.bind(socket_addresses["server_out_ep"])

        response_sock = zmq_ctx.socket(zmq.SUB)
        response_sock.setsockopt(zmq.SUBSCRIBE, b"")
        response_sock.connect(socket_addresses["gpu_out_ep"])

        response_queues: dict[str, asyncio.Queue] = {}

        boot_kwargs = dict(scheduler_kwargs or {})
        boot_alpha = float(boot_kwargs.get("alpha", 1.0))
        boot_multipliers = {
            int(k): float(v)
            for k, v in (boot_kwargs.get("action_horizon_multipliers") or {}).items()
        }
        app.state.server = ServerState(
            scheduler_sock=scheduler_sock,
            response_queues=response_queues,
            slots=slots,
            gpu_proc=gpu_proc,
            scheduler_proc=scheduler_proc,
            metrics_store=metrics_store,
            robot_metadata={},
            batch_queue=batch_queue,
            current_algorithm=metadata.scheduling_algorithm,
            current_scheduler_kwargs=dict(boot_kwargs),
            boot_alpha=boot_alpha,
            boot_action_horizon_multipliers=boot_multipliers,
        )

        router = asyncio.create_task(_router_task(response_sock, response_queues, metrics_store))
        scheduler_metrics = asyncio.create_task(
            _scheduler_metrics_task(scheduler_metrics_queue, metrics_store)
        )
        watchdog = asyncio.create_task(_watchdog_task(gpu_proc, scheduler_proc))

        yield

        watchdog.cancel()
        scheduler_metrics.cancel()
        router.cancel()
        gpu_proc.terminate()
        scheduler_proc.terminate()

        loop = asyncio.get_event_loop()
        for proc in (gpu_proc, scheduler_proc):
            await loop.run_in_executor(None, proc.join, 5)
            if proc.is_alive():
                logger.warning("Process %s did not exit cleanly, killing", proc.name)
                proc.kill()
                await loop.run_in_executor(None, proc.join)

        scheduler_sock.close()
        response_sock.close()
        scheduler_metrics_queue.close()
        zmq_ctx.term()

    app = FastAPI(lifespan=lifespan)

    @app.websocket("/ws")
    async def ws_handler(websocket: WebSocket):
        await websocket.accept()
        state: ServerState = websocket.app.state.server

        result = await _ws_handshake(websocket, state)
        if result is None:
            return
        robot_id, slot_index, _connect_req = result

        action_payload_size = metadata.action_horizon * metadata.action_dim * 4  # float32 bytes
        await _ws_warmup(websocket, state, robot_id, action_payload_size)

        # Normal operation
        response_queue: asyncio.Queue = state.response_queues[robot_id]
        pending_responses: dict[int, InferResponse] = {}

        async def recv():
            try:
                while True:
                    raw = await websocket.receive_bytes()
                    msg = msgpack_numpy.unpackb(raw)

                    match msg.get("type"):
                        case "reset":
                            await state.scheduler_sock.send_pyobj(ResetRequest(robot_id=robot_id))
                            continue
                        case "ack":
                            ack = ResponseAck(**msg)
                            response = pending_responses.pop(ack.request_id, None)
                            if response is None:
                                # ACK arrived for a request whose pending entry
                                # was already consumed or never registered. Most
                                # commonly this is a late ACK from before a
                                # client-side broker.reset() — the client's
                                # background _receive_actions thread keeps
                                # ACKing in-flight responses while the main
                                # thread resets. Drop quietly instead of
                                # tearing down the websocket.
                                logger.debug(
                                    "ACK for unknown request_id=%s on %s; ignoring (likely post-reset)",
                                    ack.request_id,
                                    robot_id,
                                )
                                continue
                            state.metrics_store.record_response(robot_id, response, ack)
                            await state.scheduler_sock.send_pyobj(
                                AckNotification(
                                    robot_id=robot_id,
                                    request_id=ack.request_id,
                                    chunk_id=ack.chunk_id,
                                    observation_step=ack.observation_step,
                                    action_index_start=ack.action_index_start,
                                    min_execution_horizon=ack.min_execution_horizon,
                                    max_execution_horizon=ack.max_execution_horizon,
                                    execution_start_step=ack.execution_start_step,
                                    first_executed_index=ack.first_executed_index,
                                    receive_time=ack.receive_time,
                                    server_send_time=response.server_send_time,
                                )
                            )
                            continue
                        case "episode_start":
                            state.metrics_store.record_episode_start(robot_id, EpisodeStart(**msg))
                            continue
                        case "episode_step":
                            state.metrics_store.record_episode_step(robot_id, EpisodeStep(**msg))
                            continue
                        case "episode_end":
                            state.metrics_store.record_episode_end(robot_id, EpisodeEnd(**msg))
                            continue
                        case "infer":
                            pass
                        case unknown:
                            logger.warning("Unknown message type %r, dropping", unknown)
                            continue

                    req = InferRequest(**msg)

                    # Write obs + request metadata atomically to shared memory so the
                    # GPU worker always reads metadata that matches the observation it infers.
                    request_id = next(_request_id_counter)
                    arrival_timestamp = time.time()
                    state.slots.write(
                        slot_index,
                        SlotData(
                            robot_id=robot_id,
                            obs=req.observation,
                            request_id=request_id,
                            arrival_timestamp=arrival_timestamp,
                            observation_step=req.observation_step,
                            action_index_start=req.action_index_start,
                            request_timestamp=req.request_timestamp,
                            deadline=req.deadline,
                            min_execution_horizon=req.min_execution_horizon,
                            max_execution_horizon=req.max_execution_horizon,
                            infer_type=req.infer_type,
                            params=req.params,
                            noise=req.noise,
                            control_hz=state.robot_metadata[robot_id].control_hz,
                        ),
                    )

                    slot_req = SlotRequest(
                        slot_index=slot_index,
                        robot_id=robot_id,
                        request_id=request_id,
                        arrival_timestamp=arrival_timestamp,
                        observation_step=req.observation_step,
                        action_index_start=req.action_index_start,
                        request_timestamp=req.request_timestamp,
                        deadline=req.deadline,
                        min_execution_horizon=req.min_execution_horizon,
                        max_execution_horizon=req.max_execution_horizon,
                        infer_type=req.infer_type,
                        params=req.params,
                        noise=req.noise,
                        control_hz=state.robot_metadata[robot_id].control_hz,
                    )
                    await state.scheduler_sock.send_pyobj(slot_req)
                    state.metrics_store.record_request(robot_id, slot_req)
            except WebSocketDisconnect:
                logger.debug("Robot %s disconnected", robot_id)

        async def send():
            while True:
                response: InferResponse = await response_queue.get()
                stamped = dataclasses.replace(response, server_send_time=time.time())
                pending_responses[response.request_id] = stamped
                await websocket.send_bytes(msgpack_numpy.packb(stamped))
                logger.debug("Sent response: %s", stamped)

        recv_task = asyncio.create_task(recv())
        send_task = asyncio.create_task(send())
        try:
            await recv_task
        finally:
            send_task.cancel()
            await state.scheduler_sock.send_pyobj(ResetRequest(robot_id=robot_id))
            state.slots.free(robot_id, expected_idx=slot_index)
            state.response_queues.pop(robot_id, None)

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
        # TODO: removed client-side normalization, should be done server-side and added back? might be different if we follow vllm pattern of prometheus logging
        return asdict(request.app.state.server.metrics_store)

    @app.post("/reset")
    async def reset_metrics(request: Request) -> dict:
        state: ServerState = request.app.state.server
        # 1. Drain any pending batches the scheduler queued for the GPU.
        # If we don't, GPU keeps processing them after the reset and the
        # ResponseBatches arrive at the (now-empty) scheduler in_flight queue,
        # tripping the batch_id assertion. Drain BEFORE telling the scheduler
        # to clear in_flight to minimise the race window.
        drained = 0
        while True:
            try:
                state.batch_queue.get_nowait()
                drained += 1
            except queue.Empty:
                break
        if drained:
            logger.info("Reset: drained %d pending batches from queue", drained)
        # 2. Tell scheduler + engine to clear all per-robot AND mirror-wide state.
        await state.scheduler_sock.send_pyobj(ResetAll())
        # 3. Reset metrics last so the post-reset state has nothing recorded.
        state.metrics_store.reset()
        return {"status": "ok", "drained_batches": drained}

    dash_app = create_dash_app(metadata, metrics_store)
    app.mount("/", WSGIMiddleware(dash_app.server))

    return app


class PolicyServer:
    def __init__(
        self,
        metadata: ServerMetadata,
        policy_factory: Callable,
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
