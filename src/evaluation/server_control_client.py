import logging
import time

import requests

from armory.serving.protocol import SchedulerConfig, ServerMetadata

logger = logging.getLogger(__name__)


def _http_base_url(host: str, port: int | None) -> str:
    """Parse host/port into an HTTP base URL."""
    explicit_scheme = False
    if host.startswith("https://"):
        scheme = "https"
        host = host[len("https://") :]
        explicit_scheme = True
    elif host.startswith("http://"):
        scheme = "http"
        host = host[len("http://") :]
        explicit_scheme = True
    else:
        scheme = "http"
    base = host if (port is None or explicit_scheme) else f"{host}:{port}"
    return f"{scheme}://{base}"


class ServerControlClient:
    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int | None = None,
    ) -> None:
        self._http_base = _http_base_url(host, port)

    def fetch_server_metadata(self, timeout_s: float = 300.0) -> ServerMetadata:
        """Fetch server metadata, retrying until timeout_s seconds have elapsed."""
        deadline = time.monotonic() + timeout_s
        while True:
            try:
                resp = requests.get(f"{self._http_base}/metadata", timeout=5.0)
                resp.raise_for_status()
                return ServerMetadata.from_http_metadata(resp.json())
            except Exception as e:
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        f"Server at {self._http_base} did not respond within {timeout_s:.0f}s"
                    ) from e
                logger.info("Waiting for server to be ready (%s); retrying in 5s...", e)
                time.sleep(5.0)

    def reset_server(self) -> None:
        try:
            requests.post(f"{self._http_base}/reset", timeout=5.0)
            logger.info("Reset server metrics")
        except Exception as e:
            logger.warning(f"Could not reset server metrics: {e}")

    def reconfigure_server(self, config: SchedulerConfig) -> None:
        """Push per-run scheduler config to the server via POST /reconfigure."""
        resp = requests.post(
            f"{self._http_base}/reconfigure", json=config.to_reconfigure_body(), timeout=10.0
        )
        if not resp.ok:
            raise RuntimeError(f"POST /reconfigure {resp.status_code}: {resp.text}")

    # TODO: metric saving should be cleaner
    def fetch_server_metrics(self) -> dict:
        try:
            return requests.get(f"{self._http_base}/save-metrics", timeout=10.0).json()
        except Exception as e:
            logging.warning(f"Could not fetch server metrics history: {e}", exc_info=True)
