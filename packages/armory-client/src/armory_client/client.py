import logging
import time
from collections.abc import Callable

import numpy as np
import requests
import websockets.sync.client

from armory_client import messages, msgpack_numpy
from armory_client.messages import (
    ConnectRequest,
    WarmupAck,
    WarmupPing,
    WarmupPong,
)
from armory_client.schemas import Observation, ServerMetadata

logger = logging.getLogger(__name__)

NUM_WARMUP = 100
WARMUP_OBS_BYTES = 3 * 224 * 224 * 3  # 3 channels, 224x224 pixels, 3 bytes per pixel


# FIXME: need Tuple and not tuple to be backwards compatible with Python 3.8 (libero environment)
def _parse_urls(host: str, port: int | None) -> tuple[str, str]:
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
        port: int | None = None,
        api_key: str | None = None,
        control_hz: float = 10.0,
        pre_send_hook: Callable[[], None] | None = None,
    ) -> None:
        self._robot_id = robot_id
        self._ws_uri, self._http_base = _parse_urls(host, port)
        self._api_key = api_key
        self._pre_send_hook = pre_send_hook
        self._control_hz = control_hz

    def connect(self):
        self._server_metadata = self._wait_for_server()
        if self._server_metadata.tunnel_url:
            tunnel_host = self._server_metadata.tunnel_url.replace("https://", "", 1)
            self._ws_uri = f"wss://{tunnel_host}/ws"
        self._ws = self._connect_ws()
        self._handshake(self._control_hz)
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
                    headers={"Authorization": f"Api-Key {self._api_key}"}
                    if self._api_key
                    else None,
                    timeout=5,
                )
                resp.raise_for_status()
                return ServerMetadata.from_http_metadata(resp.json())
            except requests.exceptions.RequestException:
                logging.info("Still waiting for server...")
                time.sleep(5)

    def _handshake(self, control_hz: float) -> None:
        """Send ConnectRequest with robot_id, wait for server acknowledgment."""
        self._ws.send(
            msgpack_numpy.packb(ConnectRequest(robot_id=self._robot_id, control_hz=control_hz))
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
        deadline: float,
        action_index_start: int,
        infer_type: messages.InferType = messages.InferType.SYNC,
        min_execution_horizon: int = 5,
        max_execution_horizon: int = 100,
        noise: np.ndarray | None = None,
    ) -> None:
        if self._pre_send_hook is not None:
            self._pre_send_hook()

        request_timestamp = time.time()
        data = msgpack_numpy.packb(
            messages.InferRequest(
                robot_id=self._robot_id,
                observation=obs,  # type: ignore[arg-type]
                observation_step=obs.step,
                action_index_start=action_index_start,
                request_timestamp=request_timestamp,
                deadline=deadline,
                min_execution_horizon=min_execution_horizon,
                max_execution_horizon=max_execution_horizon,
                infer_type=infer_type,
                noise=noise,
            )
        )
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
        )
        self._ws.send(msgpack_numpy.packb(ack))

    def reset(self) -> None:
        data = msgpack_numpy.packb(messages.ResetRequest(robot_id=self._robot_id))
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

    # TODO: rip these out and don't have the server calculate metrics
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
        self._ws.send(msgpack_numpy.packb(payload))

    def send_episode_step(self) -> None:
        payload = messages.EpisodeStep(client_timestamp=time.time())
        self._ws.send(msgpack_numpy.packb(payload))

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
        self._ws.send(msgpack_numpy.packb(payload))

    # TODO: need server args
    def fetch_server_metadata(args: Args, timeout_s: float = 300.0) -> ServerMetadata:
        """Fetch server metadata, retrying until timeout_s seconds have elapsed."""
        deadline = time.monotonic() + timeout_s
        while True:
            try:
                resp = requests.get(f"{args.http_base}/metadata", timeout=5.0)
                resp.raise_for_status()
                return ServerMetadata(**resp.json())
            except Exception as e:
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        f"Server at {args.http_base} did not respond within {timeout_s:.0f}s"
                    ) from e
                logging.info("Waiting for server to be ready (%s); retrying in 5s...", e)
                time.sleep(5.0)

    def reset_server(args: Args) -> None:
        try:
            requests.post(f"{args.http_base}/reset", timeout=5.0)
            logging.info("Reset server metrics")
        except Exception as e:
            logging.warning(f"Could not reset server metrics: {e}")

    def reconfigure_server(args: Args, server_metadata: ServerMetadata) -> None:
        """Push per-run scheduler config to the server via POST /reconfigure.

        Skipped if both ``scheduling_algorithm`` and ``action_horizon_multipliers``
        are ``None`` on ``args`` (i.e. the client didn't request an override).
        On success, mutates ``server_metadata`` in place so the on-disk
        ``server_metadata.json`` reflects what the scheduler is actually using
        for this run.
        """
        if args.scheduling_algorithm is None and args.action_horizon_multipliers is None:
            return
        body: dict[str, Any] = {}
        if args.scheduling_algorithm is not None:
            body["scheduling_algorithm"] = args.scheduling_algorithm
        if args.action_horizon_multipliers is not None:
            body["action_horizon_multipliers"] = {
                str(k): float(v) for k, v in args.action_horizon_multipliers.items()
            }
        resp = requests.post(f"{args.http_base}/reconfigure", json=body, timeout=10.0)
        if not resp.ok:
            raise RuntimeError(f"POST /reconfigure {resp.status_code}: {resp.text}")
        result = resp.json()
        server_metadata.scheduling_algorithm = result.get(
            "scheduling_algorithm", server_metadata.scheduling_algorithm
        )
        if "scheduler_kwargs" in result:
            server_metadata.scheduler_kwargs = result["scheduler_kwargs"]
        logging.info(
            "Reconfigured server: scheduling_algorithm=%s scheduler_kwargs=%s",
            server_metadata.scheduling_algorithm,
            server_metadata.scheduler_kwargs,
        )
