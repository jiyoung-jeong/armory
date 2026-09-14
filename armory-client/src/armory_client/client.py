import logging
import time
import uuid
from collections.abc import Callable

import numpy as np
import websockets.sync.client

from armory_client import messages, msgpack_numpy
from armory_client.messages import (
    ConnectRequest,
    WarmupAck,
    WarmupPing,
    WarmupPong,
)
from armory_client.schemas import Observation

logger = logging.getLogger(__name__)

NUM_WARMUP = 100
WARMUP_OBS_BYTES = 3 * 224 * 224 * 3  # 3 channels, 224x224 pixels, 3 bytes per pixel


def _parse_ws_url(host: str, port: int | None) -> str:
    """Parse a direct websocket endpoint from host/port."""
    explicit_scheme = False
    if host.startswith("https://"):
        ws_scheme = "wss"
        host = host[len("https://") :]
        explicit_scheme = True
    elif host.startswith("http://"):
        ws_scheme = "ws"
        host = host[len("http://") :]
        explicit_scheme = True
    else:
        ws_scheme = "ws"
    base = host if (port is None or explicit_scheme) else f"{host}:{port}"
    return f"{ws_scheme}://{base}/ws"


class BidirectionalWebsocket:
    def __init__(
        self,
        robot_id: str,
        host: str = "0.0.0.0",
        port: int | None = None,
        api_key: str | None = None,
        control_hz: float = 10.0,
        weight: float = 1.0,
        pre_send_hook: Callable[[], None] | None = None,
    ) -> None:
        self._robot_id = robot_id
        self._ws_uri = _parse_ws_url(host, port)
        self._api_key = api_key
        self._pre_send_hook = pre_send_hook
        self._control_hz = control_hz
        self._weight = weight
        self._episode_id = ""

    @property
    def episode_id(self) -> str:
        return self._episode_id

    def connect(self):
        self._ws = self._connect_ws()
        self._handshake()
        self._warmup()

    def _handshake(self) -> None:
        """Send ConnectRequest with robot_id, wait for server acknowledgment."""
        self._ws.send(
            msgpack_numpy.packb(
                ConnectRequest(
                    robot_id=self._robot_id, control_hz=self._control_hz, weight=self._weight
                )
            )
        )
        msgpack_numpy.unpackb(self._ws.recv())  # ConnectResponse ack
        logger.info("Connected as robot_id=%s", self._robot_id)

    def _warmup(self) -> None:
        """Perform num_warmup ping/pong round trips to seed server LatencyTracker."""
        for _ in range(NUM_WARMUP):
            ping = WarmupPing(client_timestamp=time.time(), payload=bytes(WARMUP_OBS_BYTES))
            self._ws.send(msgpack_numpy.packb(ping))
            pong = WarmupPong(**msgpack_numpy.unpackb(self._ws.recv()))
            ack = WarmupAck(server_send_time=pong.server_send_time, client_receive_time=time.time())
            self._ws.send(msgpack_numpy.packb(ack))

    def _connect_ws(self) -> websockets.sync.client.ClientConnection:
        headers = {"Authorization": f"Api-Key {self._api_key}"} if self._api_key else None
        return websockets.sync.client.connect(
            self._ws_uri,
            compression=None,
            max_size=None,
            additional_headers=headers,
        )

    def close(self) -> None:
        self._ws.close()

    def send(
        self,
        obs: Observation,
        action_index_start: int,
        actions_left: int,
        min_execution_horizon: int = 5,
        max_execution_horizon: int = 100,
        noise: np.ndarray | None = None,
    ) -> dict[str, float | int]:
        if self._pre_send_hook is not None:
            self._pre_send_hook()

        request_timestamp = time.time()
        deadline = request_timestamp + actions_left / self._control_hz
        data = msgpack_numpy.packb(
            messages.InferRequest(
                robot_id=self._robot_id,
                episode_id=self._episode_id,
                observation=obs,  # type: ignore[arg-type]
                observation_step=obs.step,
                action_index_start=action_index_start,
                request_timestamp=request_timestamp,
                deadline=deadline,
                min_execution_horizon=min_execution_horizon,
                max_execution_horizon=max_execution_horizon,
                noise=noise,
            )
        )
        serialize_end = time.time()
        self._ws.send(data)  # type: ignore
        return dict(
            request_timestamp=request_timestamp,
            serialize_end=serialize_end,
            send_end=time.time(),
            payload_bytes=len(data),
        )

    def receive(
        self,
    ) -> messages.InferResponse:  # noqa: UP006
        response = self._ws.recv()

        response = msgpack_numpy.unpackb(response)
        if isinstance(response, str):
            # we're expecting bytes; if the server sends a string, it's an error.
            raise RuntimeError(f"Error in inference server:\n{response}")

        return messages.InferResponse(**response)

    def send_ack(
        self,
        request_id: int,
        chunk_id: int,
        observation_step: int,
        receive_time: float,
        action_index_start: int,
        min_execution_horizon: int,
        max_execution_horizon: int,
        execution_start_step: int,
        first_executed_index: int = 0,
    ) -> None:
        ack = messages.ResponseAck(
            request_id=request_id,
            chunk_id=chunk_id,
            observation_step=observation_step,
            receive_time=receive_time,
            action_index_start=action_index_start,
            min_execution_horizon=min_execution_horizon,
            max_execution_horizon=max_execution_horizon,
            execution_start_step=execution_start_step,
            first_executed_index=first_executed_index,
            episode_id=self._episode_id,
        )
        self._ws.send(msgpack_numpy.packb(ack))

    def reset(self) -> None:
        self._episode_id = uuid.uuid4().hex
        data = msgpack_numpy.packb(
            messages.ResetRequest(robot_id=self._robot_id, episode_id=self._episode_id)
        )
        try:
            self._ws.send(data)
        except Exception as exc:  # noqa: BLE001
            # If the websocket is already torn down (e.g. server cleaned up the
            # session after a stale ACK), don't bring the worker down here —
            # the next inference attempt will surface the real failure with a
            # clearer signal. ResetRequest is best-effort scheduler hygiene.
            logger.warning(
                "ws_client.reset for %s skipped (connection closed): %s",
                self._robot_id,
                exc,
            )
