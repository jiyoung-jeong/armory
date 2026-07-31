import json
import pathlib
from dataclasses import asdict, dataclass, fields
from typing import Any

from pydantic import BaseModel, ConfigDict


class SchedulerConfig(BaseModel):
    """Scheduler configuration shared between server boot and client reconfigure.

    The server consumes it at startup (scripts/serve.py) and accepts it via
    POST /reconfigure or POST /prepare; clients send it per run.
    """

    model_config = ConfigDict(extra="forbid")

    scheduling_algorithm: str = "greedy-deadline"
    # POST /reconfigure preserves the current alpha; acknowledged POST /prepare
    # may change it between runs.
    alpha: float = 1.0

    def to_reconfigure_body(self) -> dict:
        """JSON body for POST /reconfigure (alpha is preserved, not sent)."""
        return {"scheduling_algorithm": self.scheduling_algorithm}

    def to_prepare_body(self) -> dict:
        """JSON body for the acknowledged POST /prepare run boundary."""
        return {**self.to_reconfigure_body(), "alpha": float(self.alpha)}


@dataclass
class ServerMetadata:
    """Metadata about the policy server and model configuration.

    Sent from server to clients at connection time.
    """

    config_name: str  # e.g., "pi0_aloha_sim", "pi05_libero"
    checkpoint_dir: str
    action_horizon: int
    action_dim: int
    num_steps: int  # sampling steps
    max_batch_size: int
    env: str  # environment mode (ALOHA, LIBERO, etc.)
    scheduling_algorithm: str
    tunnel_url: str | None = None
    location: str | None = None

    @classmethod
    def from_http_metadata(cls, payload: dict[str, Any]) -> "ServerMetadata":
        """Create metadata from server JSON, ignoring newer server-only fields."""
        allowed = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in payload.items() if k in allowed})

    def to_json(self, filepath: pathlib.Path, indent: int = 4) -> None:
        with open(filepath, "w") as f:
            json.dump(asdict(self), f, indent=indent)
