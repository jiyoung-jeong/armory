"""
3 processes:
    WS main process     - FastAPI ASGI app
    Scheduler process   - collects requests from WS main; runs ILP; dispatches batches to GPU
    GPU process         - loads weights; runs batches; sends responses directly to WS main

ZMQ topology (all ipc://, unique per server instance):
    WS main  ──[PUSH: SlotRequest / ResetRequest]──► Scheduler [binds sched_in_ep]
    WS main  ──slots.write()───────────────────────► mp.RawArray shared memory
    GPU      ──[PUSH: list[InferResponse]]──────────► WS main   [binds gpu_out_ep]
    GPU      ──[PUSH: list[CompletionNotification]]► Scheduler [binds result_ep]
    Scheduler ──[mp.Queue: list[SlotRequest]]───────► GPU
    GPU      ──slots.read()──────────────────────────► mp.RawArray shared memory

    GPU responses bypass the scheduler entirely so ILP solving cannot delay client delivery.
    A single _router_task in WS main reads from gpu_out_ep and dispatches to per-robot queues.
    Large numpy arrays (observations) cross zero process boundaries via ZMQ.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
import dataclasses
from dataclasses import asdict
from dataclasses import dataclass
import logging
import multiprocessing as mp
from multiprocessing.synchronize import Event
import os
import queue
import signal
import time
import uuid

from fastapi import FastAPI
from fastapi import Request
from fastapi import WebSocket
from fastapi.concurrency import asynccontextmanager
from openpi_client import msgpack_numpy
from openpi_client.messages import ConnectRequest
from openpi_client.messages import ConnectResponse
from openpi_client.messages import EpisodeEnd
from openpi_client.messages import EpisodeStart
from openpi_client.messages import InferRequest
from openpi_client.messages import InferResponse
from openpi_client.messages import ResetRequest
from openpi_client.messages import ResponseAck
from openpi_client.messages import WarmupPong
from openpi_client.schemas import ServerMetadata
from starlette.middleware.wsgi import WSGIMiddleware
from starlette.websockets import WebSocketDisconnect
import uvicorn
import zmq.asyncio

from openpi.serving.engine import _run_gpu_worker
from openpi.serving.metrics import MetricsStore
from openpi.serving.metrics.dash_app import create_dash_app
from openpi.serving.scheduler import _run_scheduler
from openpi.serving.schemas import AckNotification
from openpi.serving.schemas import SchedulerDecision
from openpi.serving.schemas import SlotRequest
from openpi.serving.schemas import WarmupSeed
from openpi.serving.schemas import _request_id_counter
from openpi.serving.slots import RobotSlots
from openpi.serving.slots import SlotData

MAX_ROBOTS = 100
NUM_WARMUP = 10
logger = logging.getLogger(__name__)

_uid = uuid.uuid4().hex[:8]
socket_addresses = {
    "sched_in_ep": f"ipc:///tmp/openpi_sched_in_{_uid}",
    "gpu_out_ep": f"ipc:///tmp/openpi_gpu_out_{_uid}",
    "result_ep": f"ipc:///tmp/openpi_result_{_uid}",
}


@dataclass
class ServerState:
    scheduler_sock: zmq.asyncio.Socket  # PUSH to scheduler
    response_queues: dict[str, asyncio.Queue]
    slots: RobotSlots  # WS manages slot allocation
    gpu_proc: mp.Process
    scheduler_proc: mp.Process
    metrics_store: MetricsStore
    robot_metadata: dict[str, ConnectRequest]


async def _router_task(
    response_sock: zmq.asyncio.Socket,
    response_queues: dict[str, asyncio.Queue],
    metrics_store: MetricsStore,
) -> None:
    """Reads batches of InferResponses directly from GPU and dispatches to per-robot queues."""
    logger.info("Router task starting")
    while True:
        try:
            responses: list[InferResponse] = await response_sock.recv_pyobj()
            metrics_store.record_batch(responses)
            for response in responses:
                queue = response_queues.get(response.robot_id)
                if queue is not None:
                    await queue.put(response)
                else:
                    logger.info("No active connection for robot %s, dropping response", response.robot_id)
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

    await websocket.send_bytes(msgpack_numpy.packb(dataclasses.asdict(ConnectResponse())))
    logger.info("Robot %s connected (control_hz=%.1f)", robot_id, connect_req.control_hz)
    return robot_id, slot_index, connect_req


async def _ws_warmup(
    websocket: WebSocket,
    state: ServerState,
    robot_id: str,
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
        await websocket.send_bytes(msgpack_numpy.packb(dataclasses.asdict(pong)))
        obs_samples.append((server_receive_time, msg["client_timestamp"]))

        ack_raw = await websocket.receive_bytes()
        ack_msg = msgpack_numpy.unpackb(ack_raw)
        if ack_msg.get("type") == "warmup_ack":
            delivery_samples.append((ack_msg["client_receive_time"], ack_msg["server_send_time"]))

    if obs_samples or delivery_samples:
        await state.scheduler_sock.send_pyobj(
            WarmupSeed(robot_id=robot_id, obs_samples=obs_samples, delivery_samples=delivery_samples)
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
                logger.critical("Backend process %s died (exit code %s), crashing server", proc.name, proc.exitcode)
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
) -> tuple[mp.Process, mp.Process, RobotSlots, Event, Event, mp.Queue]:
    slots = RobotSlots(max_robots=MAX_ROBOTS)
    batch_queue: mp.Queue = mp.Queue()
    scheduler_metrics_queue: mp.Queue = mp.Queue()

    gpu_ready = mp.Event()
    sched_ready = mp.Event()

    gpu_proc = mp.Process(
        target=_run_gpu_worker,
        args=(
            policy_factory,
            metadata.max_batch_size,
            slots,
            batch_queue,
            socket_addresses["gpu_out_ep"],
            socket_addresses["result_ep"],
            gpu_ready,
            log_queue,
        ),
        daemon=True,
    )

    scheduler_proc = mp.Process(
        target=_run_scheduler,
        args=(
            socket_addresses["sched_in_ep"],
            socket_addresses["result_ep"],
            batch_queue,
            scheduler_metrics_queue,
            metadata.max_batch_size,
            metadata.scheduling_algorithm,
            scheduler_kwargs,
            sched_ready,
            log_queue,
        ),
        daemon=True,
    )

    logger.info("Starting GPU subprocess…")
    gpu_proc.start()
    logger.info("Starting scheduler subprocess…")
    scheduler_proc.start()

    return scheduler_proc, gpu_proc, slots, sched_ready, gpu_ready, scheduler_metrics_queue


def create_app(
    metadata: ServerMetadata,
    policy_factory: Callable,
    scheduler_kwargs: dict[str, object] | None = None,
    log_queue: mp.Queue | None = None,
) -> FastAPI:
    metrics_store = MetricsStore()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        scheduler_proc, gpu_proc, slots, sched_ready, gpu_ready, scheduler_metrics_queue = _start_backend(
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

        scheduler_sock = zmq_ctx.socket(zmq.PUSH)
        scheduler_sock.connect(socket_addresses["sched_in_ep"])

        # WS main binds gpu_out_ep so GPU can connect to us
        response_sock = zmq_ctx.socket(zmq.PULL)
        response_sock.bind(socket_addresses["gpu_out_ep"])

        response_queues: dict[str, asyncio.Queue] = {}

        app.state.server = ServerState(
            scheduler_sock=scheduler_sock,
            response_queues=response_queues,
            slots=slots,
            gpu_proc=gpu_proc,
            scheduler_proc=scheduler_proc,
            metrics_store=metrics_store,
            robot_metadata={},
        )

        router = asyncio.create_task(_router_task(response_sock, response_queues, metrics_store))
        scheduler_metrics = asyncio.create_task(_scheduler_metrics_task(scheduler_metrics_queue, metrics_store))
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
                            response = pending_responses.pop(ack.request_id)
                            state.metrics_store.record_response(robot_id, response, ack)
                            await state.scheduler_sock.send_pyobj(
                                AckNotification(
                                    robot_id=robot_id,
                                    request_id=ack.request_id,
                                    receive_time=ack.receive_time,
                                    server_send_time=response.server_send_time,
                                )
                            )
                            continue
                        case "episode_start":
                            state.metrics_store.record_episode_start(robot_id, EpisodeStart(**msg))
                            continue
                        case "episode_step":
                            state.metrics_store.record_episode_step(robot_id, time.time())
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
                            obs=req.observation,
                            request_id=request_id,
                            arrival_timestamp=arrival_timestamp,
                            observation_step=req.observation_step,
                            action_start_step=req.action_start_step,
                            request_timestamp=req.request_timestamp,
                            deadline=req.deadline,
                            execution_horizon=req.execution_horizon,
                            infer_type=req.infer_type,
                            params=req.params,
                            noise=req.noise,
                        ),
                    )

                    slot_req = SlotRequest(
                        slot_index=slot_index,
                        robot_id=robot_id,
                        request_id=request_id,
                        arrival_timestamp=arrival_timestamp,
                        observation_step=req.observation_step,
                        action_start_step=req.action_start_step,
                        request_timestamp=req.request_timestamp,
                        deadline=req.deadline,
                        execution_horizon=req.execution_horizon,
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
                await websocket.send_bytes(msgpack_numpy.packb(asdict(stamped)))

        recv_task = asyncio.create_task(recv())
        send_task = asyncio.create_task(send())
        try:
            await recv_task
        finally:
            send_task.cancel()
            await state.scheduler_sock.send_pyobj(ResetRequest(robot_id=robot_id))
            state.slots.free(robot_id)
            state.response_queues.pop(robot_id, None)

    # can also be used for health check
    @app.get("/metadata")
    async def server_metadata() -> dict:
        return asdict(metadata)

    @app.get("/")
    async def get_metrics(request: Request, window_s: float | None = None, sla_pct: float = 10.0) -> dict:
        return request.app.state.server.metrics_store.snapshot(window_s, sla_pct=sla_pct)

    @app.get("/save-metrics")
    async def save_metrics(request: Request) -> dict:
        return asdict(request.app.state.server.metrics_store)

    @app.post("/reset")
    async def reset_metrics(request: Request) -> dict:
        request.app.state.server.metrics_store.reset()
        # TODO: reset server state too
        return {"status": "ok"}

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
        app = create_app(self._metadata, self._policy_factory, self._scheduler_kwargs, self._log_queue)
        uvicorn.run(app, host=host, port=port)
