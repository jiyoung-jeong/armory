"""Minimal raw websocket server to isolate handshake issues (no FastAPI/ASGI).

Usage:
    uv run scripts/test_ws_server.py
    uv run scripts/test_ws_client.py
"""

import asyncio
import dataclasses
import json
import logging

import websockets.asyncio.server
from websockets.datastructures import Headers
from websockets.http11 import Request, Response

from armory_client import msgpack_numpy
from armory_client.messages import ConnectResponse
from armory_client.schemas import ServerMetadata

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger(__name__)

import logging
import websockets

# This will log the raw bytes of every frame
logger = logging.getLogger('websockets')
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler())

METADATA = ServerMetadata(
    config_name="mock",
    checkpoint_dir="",
    num_steps=100,
    env="mock",
    max_batch_size=8,
    action_horizon=50,
    action_dim=7,
    scheduling_algorithm="fifo",
    tunnel_url=None,
)


async def process_request(connection, request: Request):
    """Intercept plain HTTP requests before the WebSocket upgrade."""
    if request.path == "/metadata":
        body = json.dumps(dataclasses.asdict(METADATA)).encode()
        headers = Headers()
        headers["Content-Type"] = "application/json"
        headers["Content-Length"] = str(len(body))
        return Response(
            status_code=200,
            reason_phrase="OK",
            headers=headers,
            body=body,
        )


async def handler(ws):
    logger.info("Client connected: %s", ws.remote_address)
    try:
        raw = await ws.recv()
        logger.info("Received %d bytes", len(raw))
        msg = msgpack_numpy.unpackb(raw)
        logger.info("Decoded message: %s", msg)

        if msg.get("type") != "connect":
            logger.error("Expected type=connect, got: %r", msg.get("type"))
            await ws.close(1002, "expected connect message")
            return

        logger.info("Handshake OK, robot_id=%s", msg.get("robot_id"))
        await ws.send(msgpack_numpy.packb(dataclasses.asdict(ConnectResponse())))
        logger.info("Sent ConnectResponse — waiting for warmup pings")

        count = 0
        async for raw in ws:
            msg = msgpack_numpy.unpackb(raw)
            logger.info("Received message type=%r", msg.get("type"))
            count += 1
            if count >= 5:
                break

    except Exception:
        logger.exception("Handler error")


async def main():
    async with websockets.asyncio.server.serve(
        handler, "0.0.0.0", 8080, process_request=process_request, compression=None
    ):
        logger.info("Listening on ws://0.0.0.0:8080")
        await asyncio.get_event_loop().create_future()


if __name__ == "__main__":
    asyncio.run(main())
