"""Multiprocess and IPC lifecycle for the policy server.

The order in this module is intentional: fork workers first, wait for the
scheduler and GPU readiness events, and only then create main-process ZMQ
state. ZMQ contexts are not fork-safe.
"""

from __future__ import annotations

import asyncio
import logging
import multiprocessing as mp
import os
import queue
import signal
import uuid
from collections.abc import Callable
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass
from multiprocessing.synchronize import Event
from typing import Any, Protocol, TypeAlias

import zmq.asyncio
from fastapi import FastAPI
from fastapi.concurrency import asynccontextmanager

from armory.backends.types import PolicyFactory
from armory.serving.engine import GpuWorker
from armory.serving.metrics import MetricsStore
from armory.serving.protocol import ServerMetadata
from armory.serving.scheduler import SchedulerWorker
from armory.serving.schemas import BatchProfile, ResponseBatch, SchedulerDecision
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
    """Dispatch GPU response batches to their per-robot queues."""
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


async def _scheduler_metrics_task(
    scheduler_metrics_queue: mp.Queue,
    metrics_store: MetricsStore,
) -> None:
    """Drain scheduler timing samples into the main-process metrics store."""
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
        scheduler_kwargs: dict[str, object] | None,
        log_queue: mp.Queue | None,
    ) -> BackendResources: ...


Lifespan: TypeAlias = Callable[[FastAPI], AbstractAsyncContextManager[None]]


def _start_backend(
    metadata: ServerMetadata,
    policy_factory: PolicyFactory,
    scheduler_kwargs: dict[str, object] | None,
    log_queue: mp.Queue | None,
) -> BackendResources:
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


def create_lifespan(
    metadata: ServerMetadata,
    policy_factory: PolicyFactory,
    scheduler_kwargs: dict[str, object] | None,
    log_queue: mp.Queue | None,
    metrics_store: MetricsStore,
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
            scheduler_metrics_queue,
            batch_queue,
        ) = start_backend(
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

    return lifespan
