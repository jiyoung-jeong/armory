"""WebSocket connection lifecycle for one robot.

This module owns client protocol handling only. Process lifecycle, shared
router state, and HTTP control-plane routes live in neighboring modules.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import Iterator

from fastapi import WebSocket
from starlette.websockets import WebSocketDisconnect

from armory.serving.rtc import InferType
from armory.serving.schemas import AckNotification, RobotID, SlotData, WarmupSeed
from armory.serving.server_runtime import ServerState
from armory_client import msgpack_numpy
from armory_client.messages import (
    ConnectRequest,
    ConnectResponse,
    InferRequest,
    InferResponse,
    ResetRequest,
    ResponseAck,
    WarmupPong,
)

# Keep existing log attribution while this code moves out of server.py.
logger = logging.getLogger("armory.serving.server")


def _log_event(state: ServerState, record: dict) -> None:
    state.events_log.write(json.dumps(record) + "\n")
    state.events_log.flush()


async def _handshake(
    websocket: WebSocket,
    state: ServerState,
) -> tuple[RobotID, int, ConnectRequest, asyncio.Queue[InferResponse]] | None:
    """Receive and confirm the client-provided robot identity."""
    raw = await websocket.receive_bytes()
    msg = msgpack_numpy.unpackb(raw)
    if msg.get("type") != "connect":
        await websocket.close(code=1002, reason="expected connect message")
        return None
    connect_req = ConnectRequest(**{k: v for k, v in msg.items() if k != "type"})

    robot_id = connect_req.robot_id
    async with state.control_lock:
        slot_index = state.slots.register(robot_id)
        response_queue: asyncio.Queue[InferResponse] = asyncio.Queue()
        try:
            state.response_queues[robot_id] = response_queue
            state.robot_metadata[robot_id] = connect_req

            await websocket.send_bytes(msgpack_numpy.packb(ConnectResponse()))
        except BaseException:
            _release_registration(
                state,
                robot_id=robot_id,
                slot_index=slot_index,
                response_queue=response_queue,
                connect_req=connect_req,
            )
            raise
    logger.info("Robot %s connected (control_hz=%.1f)", robot_id, connect_req.control_hz)
    return robot_id, slot_index, connect_req, response_queue


def _release_registration(
    state: ServerState,
    *,
    robot_id: RobotID,
    slot_index: int,
    response_queue: asyncio.Queue[InferResponse],
    connect_req: ConnectRequest,
) -> bool:
    """Release this connection without disturbing a newer one with the same ID."""
    is_current = state.response_queues.get(robot_id) is response_queue
    state.slots.free(robot_id, expected_idx=slot_index)
    if is_current:
        state.response_queues.pop(robot_id, None)
    if state.robot_metadata.get(robot_id) is connect_req:
        state.robot_metadata.pop(robot_id, None)
    return is_current


async def _warmup(
    websocket: WebSocket,
    state: ServerState,
    robot_id: RobotID,
    action_payload_size: int,
    num_warmup: int,
) -> None:
    """Run ping/pong samples and seed the scheduler's latency tracker."""
    obs_samples: list[tuple[float, float]] = []
    delivery_samples: list[tuple[float, float]] = []

    for _ in range(num_warmup):
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


async def _receive_loop(
    websocket: WebSocket,
    state: ServerState,
    *,
    robot_id: RobotID,
    slot_index: int,
    control_hz: float,
    weight: float,
    send_times: dict[int, float],
    request_ids: Iterator[int],
) -> None:
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
                    send_time = send_times.pop(ack.request_id, None)
                    if send_time is None:
                        # A client reset can leave an in-flight response whose
                        # background receiver still ACKs after local state reset.
                        logger.debug(
                            "ACK for unknown request_id=%s on %s; ignoring (likely post-reset)",
                            ack.request_id,
                            robot_id,
                        )
                        continue
                    await state.scheduler_sock.send_pyobj(
                        AckNotification(ack=ack, robot_id=robot_id, server_send_time=send_time)
                    )
                    _log_event(
                        state,
                        {
                            "kind": "ack",
                            "robot_id": robot_id,
                            "request_id": ack.request_id,
                            "chunk_id": ack.chunk_id,
                            "receive_time": ack.receive_time,
                            "server_send_time": send_time,
                        },
                    )
                    continue
                case "infer":
                    pass
                case unknown:
                    logger.warning("Unknown message type %r, dropping", unknown)
                    continue

            req = InferRequest(**msg)

            # Write observation and request metadata atomically so the GPU
            # always reads metadata corresponding to the same observation.
            slot_data = SlotData(
                slot_index=slot_index,
                robot_id=robot_id,
                request_id=next(request_ids),
                arrival_timestamp=time.time(),
                observation=req.observation,
                observation_step=req.observation_step,
                action_index_start=req.action_index_start,
                request_timestamp=req.request_timestamp,
                deadline=req.deadline,
                min_execution_horizon=req.min_execution_horizon,
                max_execution_horizon=req.max_execution_horizon,
                infer_type=InferType.SYNC,
                params=None,
                noise=req.noise,
                control_hz=control_hz,
                weight=weight,
            )
            state.slots.write(slot_index, slot_data)
            await state.scheduler_sock.send_pyobj(slot_data.request)
            _log_event(
                state,
                {
                    "kind": "request",
                    "robot_id": robot_id,
                    "request_id": slot_data.request_id,
                    "observation_step": req.observation_step,
                    "request_timestamp": req.request_timestamp,
                    "arrival_time": slot_data.arrival_timestamp,
                },
            )
    except WebSocketDisconnect:
        logger.debug("Robot %s disconnected", robot_id)


async def _send_loop(
    websocket: WebSocket,
    response_queue: asyncio.Queue[InferResponse],
    send_times: dict[int, float],
) -> None:
    while True:
        response = await response_queue.get()
        send_times[response.request_id] = time.time()
        await websocket.send_bytes(msgpack_numpy.packb(response))
        logger.debug("Sent response: %s", response)


async def serve_websocket_session(
    websocket: WebSocket,
    state: ServerState,
    *,
    action_payload_size: int,
    num_warmup: int,
    request_ids: Iterator[int],
) -> None:
    """Serve one accepted robot WebSocket until its receive loop exits."""
    result = await _handshake(websocket, state)
    if result is None:
        return
    robot_id, slot_index, connect_req, response_queue = result

    try:
        await _warmup(websocket, state, robot_id, action_payload_size, num_warmup)

        send_times: dict[int, float] = {}
        recv_task = asyncio.create_task(
            _receive_loop(
                websocket,
                state,
                robot_id=robot_id,
                slot_index=slot_index,
                control_hz=connect_req.control_hz,
                weight=connect_req.weight,
                send_times=send_times,
                request_ids=request_ids,
            )
        )
        send_task = asyncio.create_task(_send_loop(websocket, response_queue, send_times))
        try:
            await recv_task
        finally:
            send_task.cancel()
            try:
                await asyncio.gather(send_task, return_exceptions=True)
            except asyncio.CancelledError:
                # ASGI shutdown may cancel this handler while it is already
                # unwinding. The send task has still been cancelled; continue
                # through the registration/reset teardown below.
                pass
    finally:
        # Keep the registration visible until its reset is published. Sharing
        # the control lock with /prepare also prevents a new registration from
        # entering halfway through a run-boundary transition.
        async with state.control_lock:
            is_current = state.response_queues.get(robot_id) is response_queue
            try:
                if is_current:
                    await state.scheduler_sock.send_pyobj(ResetRequest(robot_id=robot_id))
            finally:
                _release_registration(
                    state,
                    robot_id=robot_id,
                    slot_index=slot_index,
                    response_queue=response_queue,
                    connect_req=connect_req,
                )
