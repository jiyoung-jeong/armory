"""Multiprocess and IPC lifecycle for the policy server.

The order in this module is intentional: fork workers first, create the
main-process ZMQ state, and then wait for the scheduler and GPU readiness
events. ZMQ contexts are not fork-safe, and binding before the waits gives
worker subscribers their whole startup window to connect.
"""

from __future__ import annotations

import asyncio
import json
import logging
import multiprocessing as mp
import os
import pathlib
import signal
import uuid
from collections.abc import Callable
from contextlib import AbstractAsyncContextManager
from dataclasses import asdict, dataclass
from multiprocessing.synchronize import Event
from typing import Protocol, TextIO, TypeAlias

import zmq.asyncio
from fastapi import FastAPI
from fastapi.concurrency import asynccontextmanager

from armory.backends.types import PolicyFactory
from armory.serving.config import ServerConfig
from armory.serving.engine import GpuWorker
from armory.serving.protocol import ServerMetadata
from armory.serving.scheduler import SchedulerWorker
from armory.serving.schemas import BatchProfile, GpuPrepared, PrepareAck, ResponseBatch
from armory.serving.slots import RobotSlots
from armory_client.messages import ConnectRequest

MAX_ROBOTS = 100

_uid = uuid.uuid4().hex[:8]
socket_addresses = {
    "server_out_ep": f"ipc:///tmp/openpi_server_out_{_uid}",
    "gpu_out_ep": f"ipc:///tmp/openpi_gpu_out_{_uid}",
}

# Keep existing log attribution while this code moves out of server.py.
logger = logging.getLogger("armory.serving.server")


@dataclass
class ServerState:
    scheduler_sock: zmq.asyncio.Socket  # PUB to scheduler
    response_queues: dict[str, asyncio.Queue]
    slots: RobotSlots  # WS manages slot allocation
    robot_metadata: dict[str, ConnectRequest]
    batch_queue: mp.Queue  # control routes drain stale work between trials
    control_ack_queue: mp.Queue
    control_lock: asyncio.Lock
    # Current effective server config. Mutated by POST /reconfigure so
    # /metadata always reports what the subprocesses are actually using.
    config: ServerConfig
    metrics_dir: pathlib.Path
    events_log: TextIO


def metrics_dir(config: ServerConfig) -> pathlib.Path:
    path = config.output_dir / "server"
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_metadata(state: ServerState, metadata: ServerMetadata) -> None:
    payload = asdict(metadata)
    payload["scheduling_algorithm"] = state.config.scheduler.scheduling_algorithm
    payload["scheduler"] = state.config.scheduler.model_dump()
    (state.metrics_dir / "metadata.json").write_text(json.dumps(payload, indent=2))


async def _router_task(
    response_sock: zmq.asyncio.Socket,
    response_queues: dict[str, asyncio.Queue],
    control_ack_queue: mp.Queue | None = None,
) -> None:
    """Dispatch GPU response batches to their per-robot queues."""
    logger.info("Router task starting")
    while True:
        try:
            msg: ResponseBatch | BatchProfile | GpuPrepared = await response_sock.recv_pyobj()
            if isinstance(msg, BatchProfile):
                continue
            if isinstance(msg, GpuPrepared):
                if control_ack_queue is not None:
                    control_ack_queue.put_nowait(
                        PrepareAck(operation_id=msg.operation_id, worker="router")
                    )
                continue
            assert isinstance(msg, ResponseBatch)
            logger.debug("Received response batch: %s", msg)

            for response in msg.responses:
                response_queue = response_queues.get(response.robot_id)
                if response_queue is not None:
                    await response_queue.put(response)
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


async def _watchdog_task(gpu_proc: mp.Process, scheduler_proc: mp.Process) -> None:
    """Crash the server if either backend process dies unexpectedly."""
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


BackendResources: TypeAlias = tuple[
    mp.Process,
    mp.Process,
    RobotSlots,
    Event,
    Event,
    mp.Queue,
    mp.Queue,
]


class BackendStarter(Protocol):
    def __call__(
        self,
        metadata: ServerMetadata,
        policy_factory: PolicyFactory,
        config: ServerConfig,
        log_queue: mp.Queue | None,
    ) -> BackendResources: ...


Lifespan: TypeAlias = Callable[[FastAPI], AbstractAsyncContextManager[None]]


def _start_backend(
    metadata: ServerMetadata,
    policy_factory: PolicyFactory,
    config: ServerConfig,
    log_queue: mp.Queue | None,
) -> BackendResources:
    slots = RobotSlots(max_robots=MAX_ROBOTS)
    batch_queue: mp.Queue = mp.Queue()
    control_ack_queue: mp.Queue = mp.Queue()
    gpu_ready = mp.Event()
    sched_ready = mp.Event()

    gpu_proc = mp.Process(
        target=GpuWorker(
            policy_factory,
            config,
            slots,
            batch_queue,
            socket_addresses["server_out_ep"],
            socket_addresses["gpu_out_ep"],
            gpu_ready,
            metrics_dir(config),
            log_queue,
            control_ack_queue,
        ).run,
        daemon=True,
    )

    scheduler_proc = mp.Process(
        target=SchedulerWorker(
            socket_addresses["server_out_ep"],
            socket_addresses["gpu_out_ep"],
            batch_queue,
            metrics_dir(config),
            config,
            sched_ready,
            log_queue,
            control_ack_queue,
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
        batch_queue,
        control_ack_queue,
    )


def create_lifespan(
    metadata: ServerMetadata,
    policy_factory: PolicyFactory,
    config: ServerConfig,
    log_queue: mp.Queue | None,
    *,
    start_backend: BackendStarter = _start_backend,
) -> Lifespan:
    """Build the FastAPI lifespan without creating pre-fork ZMQ state."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        (
            scheduler_proc,
            gpu_proc,
            slots,
            sched_ready,
            gpu_ready,
            batch_queue,
            control_ack_queue,
        ) = start_backend(
            metadata,
            policy_factory,
            config,
            log_queue,
        )

        # The worker processes have already forked, so creating ZMQ state here
        # is safe. Bind before waiting for their readiness events so their SUB
        # sockets have the whole model/profile startup window to connect.
        zmq_ctx = zmq.asyncio.Context()

        scheduler_sock = zmq_ctx.socket(zmq.PUB)
        scheduler_sock.bind(socket_addresses["server_out_ep"])

        response_sock = zmq_ctx.socket(zmq.SUB)
        response_sock.setsockopt(zmq.SUBSCRIBE, b"")
        response_sock.connect(socket_addresses["gpu_out_ep"])

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, sched_ready.wait)
        logger.info("Scheduler ready")
        await loop.run_in_executor(None, gpu_ready.wait)
        logger.info("GPU worker ready")

        response_queues: dict[str, asyncio.Queue] = {}

        record_dir = metrics_dir(config)
        state = ServerState(
            scheduler_sock=scheduler_sock,
            response_queues=response_queues,
            slots=slots,
            robot_metadata={},
            batch_queue=batch_queue,
            control_ack_queue=control_ack_queue,
            control_lock=asyncio.Lock(),
            config=config,
            metrics_dir=record_dir,
            events_log=open(record_dir / "events.jsonl", "w"),
        )
        app.state.server = state
        write_metadata(state, metadata)

        router = asyncio.create_task(
            _router_task(response_sock, response_queues, control_ack_queue)
        )
        watchdog = asyncio.create_task(_watchdog_task(gpu_proc, scheduler_proc))

        yield

        watchdog.cancel()
        router.cancel()
        state.events_log.close()
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
        zmq_ctx.term()

    return lifespan
