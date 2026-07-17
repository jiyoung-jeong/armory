import json
import pathlib
from dataclasses import asdict, dataclass, fields
from typing import Any

from pydantic import BaseModel


class SchedulerConfig(BaseModel):
    """Scheduler configuration shared between server boot and client reconfigure.

    The server consumes it at startup (scripts/serve.py) and accepts it via
    POST /reconfigure; clients send it to override scheduling per run.
    """

    scheduling_algorithm: str = "greedy-deadline"
    # Server-startup-only: POST /reconfigure preserves the boot-time alpha.
    alpha: float = 1.0
    action_horizon_multipliers: dict[int, float] = {}

    # TODO: nuke these functions
    def to_scheduler_kwargs(self) -> dict | None:
        if self.scheduling_algorithm == "dynamic-action":
            return {"alpha": self.alpha}
        if self.scheduling_algorithm in ("lookahead-actions"):
            return {"action_horizon_multipliers": self.action_horizon_multipliers}
        return None

    def to_reconfigure_body(self) -> dict:
        """JSON body for POST /reconfigure (alpha is boot-only, not sent)."""
        return {
            "scheduling_algorithm": self.scheduling_algorithm,
            "action_horizon_multipliers": {
                str(k): float(v) for k, v in self.action_horizon_multipliers.items()
            },
        }


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
    scheduler_kwargs: dict | None = None
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

    @classmethod
    def from_json(cls, filepath: pathlib.Path) -> "ServerMetadata":
        with open(filepath) as f:
            return cls(**json.load(f))
