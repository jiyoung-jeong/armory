from __future__ import annotations

import json
import math
import pathlib
import subprocess
import time
from typing import Any

import numpy as np
import requests

DEFAULT_TOXIC_UPSTREAM = "latency_upstream"
DEFAULT_TOXIC_DOWNSTREAM = "latency_downstream"

ExperimentConfig = Any
NetworkEmulationConfig = ExperimentConfig
WorkerNetworkContext = dict[str, Any]

_DEFAULT_TOXIPROXY_CONFIG: dict[str, Any] = {
    "api_url": "http://localhost:8474",
    "listen_host": "localhost",
    "listen_port_base": 8888,
    "server_args": [],
}

_DEFAULT_SAMPLING_CONFIG: dict[str, Any] = {
    "default_seed": 0,
    "resample_every_requests": 10,
}


def robot_profile_disables_network_emulation(robot_cfg: Any) -> bool:
    return (
        float(robot_cfg.observation_latency.median) == 0.0
        and float(robot_cfg.observation_latency.sigma) == 0.0
        and float(robot_cfg.action_latency.median) == 0.0
        and float(robot_cfg.action_latency.sigma) == 0.0
    )


def experiment_requires_network_emulation(
    config: ExperimentConfig,
    worker_count: int | None = None,
) -> bool:
    robots = config.robots
    max_workers = len(robots) if worker_count is None else int(worker_count)
    for idx in range(max(0, max_workers)):
        if idx >= len(robots):
            continue
        if not robot_profile_disables_network_emulation(robots[idx]):
            return True
    return False


class ToxiproxyController:
    """Small helper for toxiproxy API control and optional local server lifecycle."""

    def __init__(
        self,
        api_url: str,
        *,
        server_args: list[str] | None = None,
        session: requests.Session | None = None,
        timeout_s: float = 2.0,
    ) -> None:
        self._api_url = api_url.rstrip("/")
        self._server_args = list(server_args or [])
        self._session = session or requests.Session()
        self._timeout_s = timeout_s
        self._proc: subprocess.Popen | None = None

    def _url(self, path: str) -> str:
        return f"{self._api_url}{path}"

    def _request(
        self, method: str, path: str, *, expected: tuple[int, ...], **kwargs
    ) -> requests.Response:
        try:
            response = self._session.request(
                method, self._url(path), timeout=self._timeout_s, **kwargs
            )
        except requests.RequestException as exc:
            raise RuntimeError(f"toxiproxy request failed: {method} {path}: {exc}") from exc

        if response.status_code not in expected:
            raise RuntimeError(
                f"toxiproxy request failed: {method} {path} -> {response.status_code}: {response.text[:400]}"
            )
        return response

    def wait_until_ready(self, timeout_s: float = 10.0, poll_interval_s: float = 0.1) -> None:
        deadline = time.time() + timeout_s
        last_error: Exception | None = None
        while time.time() < deadline:
            try:
                self._request("GET", "/proxies", expected=(200,))
                return
            except RuntimeError as exc:
                last_error = exc
                time.sleep(poll_interval_s)

        raise TimeoutError(f"Timed out waiting for toxiproxy API at {self._api_url}: {last_error}")

    def start_server(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            return

        # We own lifecycle for this run and should not reuse an already-running local API
        try:
            self.wait_until_ready(timeout_s=0.25, poll_interval_s=0.05)
        except TimeoutError:
            pass
        else:
            raise RuntimeError(
                "toxiproxy API is already reachable; refusing to reuse an existing server instance"
            )

        # NOTE: toxiproxy command is hardcoded for simplicity
        self._proc = subprocess.Popen(
            ["toxiproxy-server", *self._server_args],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        try:
            self.wait_until_ready(timeout_s=15.0, poll_interval_s=0.1)
        except Exception:
            self.stop_server()
            raise

    def stop_server(self) -> None:
        if self._proc is None:
            return
        if self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.kill()
                self._proc.wait(timeout=5)
        self._proc = None

    def create_proxy(self, name: str, listen: str, upstream: str) -> None:
        payload = {"name": name, "listen": listen, "upstream": upstream, "enabled": True}
        response = self._request("POST", "/proxies", expected=(200, 201, 409), json=payload)
        if response.status_code == 409:
            self.delete_proxy(name)
            self._request("POST", "/proxies", expected=(200, 201), json=payload)

    def delete_proxy(self, name: str) -> None:
        self._request("DELETE", f"/proxies/{name}", expected=(200, 204, 404))

    def _upsert_latency_toxic(
        self, proxy_name: str, toxic_name: str, stream: str, latency_ms: int
    ) -> None:
        payload = {
            "name": toxic_name,
            "type": "latency",
            "stream": stream,
            "toxicity": 1.0,
            "attributes": {
                "latency": int(latency_ms),
                "jitter": 0,
            },
        }

        response = self._request(
            "POST",
            f"/proxies/{proxy_name}/toxics/{toxic_name}",
            expected=(200, 201, 404, 405),
            json=payload,
        )
        if response.status_code in (404, 405):
            self._request(
                "POST", f"/proxies/{proxy_name}/toxics", expected=(200, 201), json=payload
            )

    def set_latency(self, proxy_name: str, upstream_ms: int, downstream_ms: int) -> None:
        self._upsert_latency_toxic(proxy_name, DEFAULT_TOXIC_UPSTREAM, "upstream", upstream_ms)
        self._upsert_latency_toxic(
            proxy_name, DEFAULT_TOXIC_DOWNSTREAM, "downstream", downstream_ms
        )


class RobotNetworkHook:
    """Worker-local hook that updates toxics before each inference request."""

    def __init__(self, context: WorkerNetworkContext) -> None:
        self._context = context
        self._controller = ToxiproxyController(str(context["api_url"]))

        self._uplink_median_ms = float(context["uplink_median_ms"])
        self._uplink_sigma = float(context["uplink_sigma"])
        self._downlink_median_ms = float(context["downlink_median_ms"])
        self._downlink_sigma = float(context["downlink_sigma"])
        self._rng = np.random.default_rng(int(context["seed"]))
        self._uplink_mu = math.log(self._uplink_median_ms)
        self._downlink_mu = math.log(self._downlink_median_ms)

        self._request_index = 0
        self._resample_every = max(1, int(context["resample_every_requests"]))
        self._last_upstream_ms = max(0, int(round(self._uplink_median_ms)))
        self._last_downstream_ms = max(0, int(round(self._downlink_median_ms)))

        self._trace: list[dict[str, Any]] = []
        self._flushed_count = 0

    def _sample_latency(self, median_ms: float, sigma: float, mu: float) -> float:
        if sigma == 0:
            return median_ms
        return float(self._rng.lognormal(mu, sigma))

    def before_send(self) -> None:
        self._request_index += 1
        should_resample = self._request_index == 1 or (
            (self._request_index - 1) % self._resample_every == 0
        )

        if should_resample:
            sampled_uplink = self._sample_latency(
                self._uplink_median_ms,
                self._uplink_sigma,
                self._uplink_mu,
            )
            sampled_downlink = self._sample_latency(
                self._downlink_median_ms,
                self._downlink_sigma,
                self._downlink_mu,
            )
            self._last_upstream_ms = max(0, int(round(sampled_uplink)))
            self._last_downstream_ms = max(0, int(round(sampled_downlink)))
            self._controller.set_latency(
                str(self._context["proxy_name"]),
                self._last_upstream_ms,
                self._last_downstream_ms,
            )
        else:
            sampled_uplink = float(self._last_upstream_ms)
            sampled_downlink = float(self._last_downstream_ms)

        self._trace.append(
            {
                "request_index": self._request_index,
                "sampled_uplink_ms": sampled_uplink,
                "sampled_downlink_ms": sampled_downlink,
                "sampled_rtt_ms": sampled_uplink + sampled_downlink,
                "upstream_latency_ms": self._last_upstream_ms,
                "downstream_latency_ms": self._last_downstream_ms,
                "resampled": should_resample,
                "timestamp": time.time(),
            }
        )

    def flush_trace(self) -> None:
        if self._flushed_count >= len(self._trace):
            return

        path = pathlib.Path(str(self._context["trace_path"]))
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            for entry in self._trace[self._flushed_count :]:
                handle.write(json.dumps(entry, sort_keys=True) + "\n")
        self._flushed_count = len(self._trace)

    def close(self) -> None:
        self.flush_trace()


class NetworkEmulationManager:
    """Main-process manager: start server, create proxies, and build worker contexts."""

    def __init__(
        self,
        config: NetworkEmulationConfig,
        *,
        upstream_host: str,
        upstream_port: int,
        worker_count: int,
        output_dir: str | pathlib.Path,
        toxiproxy_config: dict[str, Any] | None = None,
        sampling_config: dict[str, Any] | None = None,
    ) -> None:
        self._config = config
        self._upstream_host = upstream_host
        self._upstream_port = upstream_port
        self._worker_count = worker_count
        self._output_dir = pathlib.Path(output_dir)

        self._toxi_cfg = toxiproxy_config or _DEFAULT_TOXIPROXY_CONFIG
        self._sampling_cfg = sampling_config or _DEFAULT_SAMPLING_CONFIG

        self._controller = ToxiproxyController(
            str(self._toxi_cfg["api_url"]),
            server_args=list(self._toxi_cfg.get("server_args", [])),
        )

        self._worker_contexts: dict[str, WorkerNetworkContext] = {}
        self._active_proxy_names: list[str] = []

    @property
    def worker_contexts(self) -> dict[str, WorkerNetworkContext]:
        return dict(self._worker_contexts)

    def start(self) -> dict[str, WorkerNetworkContext]:
        self._worker_contexts = {}
        self._active_proxy_names = []

        if self._worker_count == 0:
            self._write_resolved_config()
            return {}

        self._controller.start_server()

        robots = self._config.robots
        toxi_cfg = self._toxi_cfg
        sampling_cfg = self._sampling_cfg

        for idx in range(self._worker_count):
            robot_id = f"robot_{idx}"
            if idx >= len(robots):
                raise ValueError(
                    f"Missing robots[{idx}] in experiment config for worker_count={self._worker_count}"
                )
            robot = robots[idx]

            seed = int(sampling_cfg["default_seed"]) + idx

            context: WorkerNetworkContext = {
                "robot_id": robot_id,
                "proxy_name": f"openpi_{robot_id}_proxy",
                "proxy_host": str(toxi_cfg["listen_host"]),
                "proxy_port": int(toxi_cfg["listen_port_base"]) + idx,
                "api_url": str(toxi_cfg["api_url"]),
                "uplink_median_ms": float(robot.observation_latency.median),
                "uplink_sigma": float(robot.observation_latency.sigma),
                "downlink_median_ms": float(robot.action_latency.median),
                "downlink_sigma": float(robot.action_latency.sigma),
                "seed": int(seed),
                "resample_every_requests": int(sampling_cfg["resample_every_requests"]),
                "trace_path": str(self._output_dir / f"{robot_id}_latency_trace.jsonl"),
                "emulate_network": not robot_profile_disables_network_emulation(robot),
            }
            self._worker_contexts[robot_id] = context

        upstream = f"{self._upstream_host}:{self._upstream_port}"
        for context in self._worker_contexts.values():
            if not bool(context.get("emulate_network", True)):
                continue
            proxy_name = str(context["proxy_name"])
            listen = f"{context['proxy_host']}:{context['proxy_port']}"
            self._controller.create_proxy(proxy_name, listen=listen, upstream=upstream)

            initial_uplink = max(0, int(round(float(context["uplink_median_ms"]))))
            initial_downlink = max(0, int(round(float(context["downlink_median_ms"]))))
            self._controller.set_latency(proxy_name, initial_uplink, initial_downlink)
            self._active_proxy_names.append(proxy_name)

        self._write_resolved_config()
        return dict(self._worker_contexts)

    def close(self) -> None:
        for proxy_name in self._active_proxy_names:
            try:
                self._controller.delete_proxy(proxy_name)
            except Exception:
                pass
        self._active_proxy_names = []
        self._controller.stop_server()

    def _write_resolved_config(self) -> None:
        self._output_dir.mkdir(parents=True, exist_ok=True)
        if hasattr(self._config, "model_dump"):
            experiment_data = self._config.model_dump()
        else:
            experiment_data = dict(self._config)
        payload = {
            "experiment": experiment_data,
            "toxiproxy": self._toxi_cfg,
            "sampling": self._sampling_cfg,
            "upstream": {
                "host": self._upstream_host,
                "port": self._upstream_port,
            },
            "worker_contexts": self._worker_contexts,
        }
        (self._output_dir / "resolved_config.json").write_text(json.dumps(payload, indent=2))
