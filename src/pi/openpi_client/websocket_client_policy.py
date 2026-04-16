import asyncio
import logging
import threading
import time
from typing import Any, Dict, Optional, Tuple

import numpy as np
from dataclasses import asdict
from typing_extensions import override
import websockets.sync.client
import websockets.asyncio.client

from openpi_client import base_policy as _base_policy
from openpi_client import msgpack_numpy
from openpi_client import messages
from openpi_client.schemas import Observation, ServerMetadata


class WebsocketClientPolicy(_base_policy.BasePolicy):
    """Implements the Policy interface by communicating with a server over websocket.

    See WebsocketPolicyServer for a corresponding server implementation.
    """

    def __init__(
        self,
        robot_id: str,
        host: str = "0.0.0.0",
        port: Optional[int] = None,
        api_key: Optional[str] = None,
    ) -> None:
        self._robot_id = robot_id
        self._uri = f"ws://{host}"
        if port is not None:
            self._uri += f":{port}"
        self._packer = msgpack_numpy.Packer()
        self._api_key = api_key
        self._ws_lock = threading.Lock()  # Thread-safe WebSocket access
        self._ws, self._server_metadata = self._wait_for_server()

    @property
    def server_metadata(self) -> ServerMetadata:
        return self._server_metadata

    def _wait_for_server(
        self,
    ) -> Tuple[websockets.sync.client.ClientConnection, ServerMetadata]:
        logging.info(f"Waiting for server at {self._uri}...")
        while True:
            try:
                headers = {"Authorization": f"Api-Key {self._api_key}"} if self._api_key else None
                conn = websockets.sync.client.connect(
                    self._uri,
                    compression=None,
                    max_size=None,
                    additional_headers=headers,
                )
                metadata_dict = msgpack_numpy.unpackb(conn.recv())
                metadata = ServerMetadata(**metadata_dict)
                return conn, metadata
            except ConnectionRefusedError:
                logging.info("Still waiting for server...")
                time.sleep(5)

    @override
    def infer(
        self,
        obs: Dict,
        use_rtc: bool = False,
        deadline: Optional[float] = None,
        s_param: Optional[int] = None,
        d_param: Optional[int] = None,
        noise: Optional[np.ndarray] = None,
    ) -> Dict:  # noqa: UP006
        infer_type = messages.InferType.SYNC
        params = None
        if use_rtc:
            infer_type = messages.InferType.INFERENCE_TIME_RTC
            params = messages.RTCParams(s_param=s_param, d_param=d_param)  # type: ignore
        request = messages.InferRequest(
            request_timestamp=time.time(),
            observation_step=obs.step,
            robot_id=self._robot_id,
            observation=asdict(obs),
            deadline=deadline,
            infer_type=infer_type,
            params=params,
            noise=noise,
        )
        data = msgpack_numpy.packb(asdict(request))

        # Use lock to ensure thread-safe WebSocket communication
        with self._ws_lock:
            self._ws.send(data)
            response = self._ws.recv()

        if isinstance(response, str):
            # we're expecting bytes; if the server sends a string, it's an error.
            raise RuntimeError(f"Error in inference server:\n{response}")

        result = msgpack_numpy.unpackb(response)
        receive_time = time.time()
        ack = messages.ResponseAck(request_id=result["request_id"], receive_time=receive_time)
        with self._ws_lock:
            self._ws.send(msgpack_numpy.packb(asdict(ack)))
        return result


class AsyncWebsocketClientPolicy:
    """Async version of WebsocketClientPolicy for high-performance concurrent requests.

    This class uses async websockets with a fixed-size connection pool to enable true concurrent
    requests without blocking. Each request gets its own connection from the pool.
    """

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: Optional[int] = None,
        api_key: Optional[str] = None,
        num_connections: int = 100,
    ) -> None:
        self._uri = f"ws://{host}"
        if port is not None:
            self._uri += f":{port}"
        self._packer = msgpack_numpy.Packer()
        self._api_key = api_key
        self._server_metadata = None
        self._connection_pool: list[Any] = []
        self._pool_lock = asyncio.Lock()
        self._num_connections = num_connections

    async def connect(self) -> ServerMetadata:
        """Connect to the server and retrieve metadata."""
        results = await asyncio.gather(*[self._create_connection() for _ in range(self._num_connections)])
        self._connection_pool = [conn for conn, _ in results]
        self._server_metadata = results[0][1]
        return self._server_metadata

    async def _create_connection(
        self,
    ) -> Tuple[websockets.asyncio.client.ClientConnection, ServerMetadata]:
        """Create a new websocket connection and retrieve metadata."""
        logging.info(f"Waiting for server at {self._uri}...")
        start = time.time()
        while True:
            try:
                headers = {"Authorization": f"Api-Key {self._api_key}"} if self._api_key else None
                conn = await websockets.asyncio.client.connect(
                    self._uri,
                    compression=None,
                    max_size=None,
                    additional_headers=headers,
                )
                metadata_bytes = await conn.recv()
                metadata_dict = msgpack_numpy.unpackb(metadata_bytes)
                metadata = ServerMetadata(**metadata_dict)
                return conn, metadata

            except ConnectionRefusedError:
                timeout = 300
                if time.time() - start > timeout:
                    raise RuntimeError(f"Failed to connect to server after {timeout} seconds")
                logging.info("Still waiting for server...")
                await asyncio.sleep(5)

    async def _get_connection(self) -> Any:
        """Get a connection from the pool."""
        async with self._pool_lock:
            assert self._connection_pool, (
                "No connections left in pool. Either allocate more connections or reduce the number of concurrent requests."
            )
            return self._connection_pool.pop()

    async def _return_connection(self, conn: Any) -> None:
        """Return a connection to the pool."""
        async with self._pool_lock:
            self._connection_pool.append(conn)

    async def infer(
        self,
        obs: Observation,
        use_rtc: bool = False,
        s_param: Optional[int] = None,
        d_param: Optional[int] = None,
    ) -> Dict:
        """Send an observation and receive an action asynchronously.

        Each request uses its own connection from the pool to avoid
        concurrent recv() conflicts.
        """
        if self._server_metadata is None:
            raise RuntimeError("Client not connected. Call connect() first.")

        infer_type = messages.InferType.SYNC
        params = None
        if use_rtc:
            infer_type = messages.InferType.INFERENCE_TIME_RTC
            params = messages.RTCParams(s_param=s_param, d_param=d_param)  # type: ignore
        request = messages.InferRequest(
            robot_id=self._robot_id,
            observation=asdict(obs),
            infer_type=infer_type,
            params=params,
        )
        data = msgpack_numpy.packb(asdict(request))

        conn = await self._get_connection()
        try:
            await conn.send(data)
            response = await conn.recv()
            if isinstance(response, str):
                # we're expecting bytes; if the server sends a string, it's an error.
                raise RuntimeError(f"Error in inference server:\n{response}")
            result = msgpack_numpy.unpackb(response)
            await self._return_connection(conn)
            return result
        except Exception:
            # Don't return broken connections to pool
            await conn.close()
            raise

    async def close(self) -> None:
        """Close all connections in the pool."""
        async with self._pool_lock:
            for conn in self._connection_pool:
                await conn.close()
            self._connection_pool.clear()

    async def __aenter__(self):
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
