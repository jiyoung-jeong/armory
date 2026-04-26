"""Server-level schemas shared between armory core and armory_client."""

from __future__ import annotations

import json
import pathlib
from dataclasses import asdict
from dataclasses import dataclass
from typing import Optional
from typing import Type
from typing import TypeVar

T = TypeVar("T", bound="JSONDataclass")


class JSONDataclass:
    """Mixin that adds JSON serialization to dataclasses."""

    def to_json(self, filepath: pathlib.Path, indent: int = 4) -> None:
        with open(filepath, "w") as f:
            json.dump(asdict(self), f, indent=indent)

    @classmethod
    def from_json(cls: Type[T], filepath: pathlib.Path) -> T:
        with open(filepath, "r") as f:
            data = json.load(f)
            return cls(**data)


@dataclass
class ServerMetadata(JSONDataclass):
    """Metadata the server exposes at /metadata; also consumed by clients and the dashboard."""

    config_name: str
    checkpoint_dir: str
    action_horizon: int
    action_dim: int
    num_steps: int
    max_batch_size: int
    env: str
    scheduling_algorithm: str
    tunnel_url: Optional[str] = None
    location: Optional[str] = None

    def __post_init__(self) -> None:
        try:
            import requests

            info = requests.get("https://ipinfo.io/json", timeout=3).json()
            self.location = f"{info.get('city', '?')}, {info.get('region', '?')}, {info.get('country', '?')}"
        except Exception:
            self.location = "unknown"
