import logging
import time
from typing import Callable
from typing import Optional

import numpy as np
import requests
from dataclasses import asdict
import websockets.sync.client

from openpi_client import messages
from openpi_client import msgpack_numpy
from openpi_client.messages import (
    ConnectRequest,
    WarmupAck,
    WarmupPing,
    WarmupPong,
)
from openpi_client.schemas import Observation, ServerMetadata
from typing import Tuple

logger = logging.getLogger(__name__)

NUM_WARMUP = 10
WARMUP_OBS_BYTES = 3 * 224 * 224 * 3  # 3 channels, 224x224 pixels, 3 bytes per pixel


# FIXME: need Tuple and not tuple to be backwards compatible with Python 3.8 (libero environment)
def _parse_urls(host: str, port: Optional[int]) -> Tuple[str, str]:
    """Parse host/port into (ws_uri, http_base) tuple."""
    explicit_scheme = False
    if host.startswith("https://"):
        ws_scheme, http_scheme = "wss", "https"
        host = host[len("https://") :]
        explicit_scheme = True
    elif host.startswith("http://"):
        ws_scheme, http_scheme = "ws", "http"
        host = host[len("http://") :]
        explicit_scheme = True
    else:
        ws_scheme, http_scheme = "ws", "http"
    base = host if (port is None or explicit_scheme) else f"{host}:{port}"
    return f"{ws_scheme}://{base}/ws", f"{http_scheme}://{base}"


class BidirectionalWebsocket:
    """Implements the Policy interface by communicating with a server over websocket.

    See WebsocketPolicyServer for a corresponding server implementation.
    """

    def __init__(
        self,
        robot_id: str,
        host: str = "0.0.0.0",
        port: Optional[int] = None,
        api_key: Optional[str] = None,
        control_hz: float = 10.0,
        pre_send_hook: Optional[Callable[[], None]] = None,
    ) -> None:
        self._robot_id = robot_id
        self._ws_uri, self._http_base = _parse_urls(host, port)
        self._api_key = api_key
        self._pre_send_hook = pre_send_hook
        self._server_metadata = self._wait_for_server()
        if self._server_metadata.tunnel_url:
            tunnel_host = self._server_metadata.tunnel_url.replace("https://", "", 1)
            self._ws_uri = f"wss://{tunnel_host}/ws"
        self._ws = self._connect_ws()
        self._handshake(control_hz)
        self._warmup()

    @property
    def server_metadata(self) -> ServerMetadata:
        return self._server_metadata

    def _wait_for_server(self) -> ServerMetadata:
        logging.info(f"Waiting for server at {self._http_base}...")
        while True:
            try:
                resp = requests.get(
                    f"{self._http_base}/metadata",
                    headers={"Authorization": f"Api-Key {self._api_key}"} if self._api_key else None,
                    timeout=5,
                )
                resp.raise_for_status()
                return ServerMetadata(**resp.json())
            except requests.exceptions.RequestException:
                logging.info("Still waiting for server...")
                time.sleep(5)

    def _handshake(self, control_hz: float) -> None:
        """Send ConnectRequest with robot_id, wait for server acknowledgment."""
        self._ws.send(msgpack_numpy.packb(asdict(ConnectRequest(robot_id=self._robot_id, control_hz=control_hz))))
        msgpack_numpy.unpackb(self._ws.recv())  # ConnectResponse ack
        logger.info("Connected as robot_id=%s", self._robot_id)

    def _warmup(self) -> None:
        """Perform num_warmup ping/pong round trips to seed server LatencyTracker."""
        for _ in range(NUM_WARMUP):
            ping = WarmupPing(client_timestamp=time.time(), payload=bytes(WARMUP_OBS_BYTES))
            self._ws.send(msgpack_numpy.packb(asdict(ping)))
            pong = WarmupPong(**msgpack_numpy.unpackb(self._ws.recv()))
            ack = WarmupAck(server_send_time=pong.server_send_time, client_receive_time=time.time())
            self._ws.send(msgpack_numpy.packb(asdict(ack)))

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
        deadline: float,
        action_start_step: int,
        infer_type: messages.InferType = messages.InferType.SYNC,
        execution_horizon: int = 0,
        noise: Optional[np.ndarray] = None,
    ) -> None:
        if self._pre_send_hook is not None:
            self._pre_send_hook()

        request = messages.InferRequest(
            request_timestamp=time.time(),
            observation_step=obs.step,
            action_start_step=action_start_step,
            robot_id=self._robot_id,
            observation=asdict(obs),
            deadline=deadline,
            infer_type=infer_type,
            noise=noise,
            execution_horizon=execution_horizon,
        )
        data = msgpack_numpy.packb(asdict(request))
        self._ws.send(data)  # type: ignore

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
        receive_time: float,
        execution_start_step: int,
        first_executed_index: int = 0,
    ) -> None:
        ack = messages.ResponseAck(
            request_id=request_id,
            receive_time=receive_time,
            execution_start_step=execution_start_step,
            first_executed_index=first_executed_index,
        )
        self._ws.send(msgpack_numpy.packb(asdict(ack)))

    def reset(self) -> None:
        data = msgpack_numpy.packb(asdict(messages.ResetRequest(robot_id=self._robot_id)))
        self._ws.send(data)

    def send_episode_start(
        self,
        task_suite_name: str,
        task_id: int,
        episode_idx: int,
        max_episode_steps: int,
        task_language: str,
    ) -> None:
        payload = messages.EpisodeStart(
            task_suite_name=task_suite_name,
            task_id=task_id,
            episode_idx=episode_idx,
            max_episode_steps=max_episode_steps,
            task_language=task_language,
        )
        self._ws.send(msgpack_numpy.packb(asdict(payload)))

    def send_episode_step(self) -> None:
        payload = messages.EpisodeStep()
        self._ws.send(msgpack_numpy.packb(asdict(payload)))

    def send_episode_end(
        self,
        task_suite_name: str,
        task_id: int,
        episode_idx: int,
        success: bool,
        duration_s: float,
        steps_taken: int,
    ) -> None:
        payload = messages.EpisodeEnd(
            task_suite_name=task_suite_name,
            task_id=task_id,
            episode_idx=episode_idx,
            success=success,
            duration_s=duration_s,
            steps_taken=steps_taken,
        )
        self._ws.send(msgpack_numpy.packb(asdict(payload)))
