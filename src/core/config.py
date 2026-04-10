"""YAML configuration parser and robot state management."""

import os
import random
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import yaml


class RobotStatus(Enum):
    OFFLINE = "offline"
    BOOTED = "booted"
    ONLINE = "online"


@dataclass
class Robot:
    id: int
    name: str
    ip: str
    status: RobotStatus = RobotStatus.OFFLINE


NAMES = [
    "Shadow", "Neon", "Crimson", "Cobalt", "Phantom", "Onyx",
    "Rogue", "Titan", "Nova", "Blaze", "Frost", "Volt",
    "Echo", "Flux", "Prism", "Viper", "Falcon", "Raven",
    "Lynx", "Mantis",
]


def _generate_name(used: set[str]) -> str:
    """Generate a unique human-readable name."""
    while True:
        name = random.choice(NAMES)
        if name not in used:
            used.add(name)
            return name


class Config:
    """Loads config.yaml and manages robot state."""

    def __init__(self, config_path: str | None = None):
        if config_path is None:
            config_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                "config.yaml",
            )
        with open(config_path) as f:
            raw = yaml.safe_load(f)

        ssh_cfg = raw.get("ssh", {})
        self.ssh_user = ssh_cfg.get("user", "rbansal66")
        self.base_ip = ssh_cfg.get("base_ip", "130.207.121")
        self.ip_offset = ssh_cfg.get("ip_offset", 200)

        used_names: set[str] = set()
        self.robots: list[Robot] = []
        for entry in raw.get("workstations", []):
            wid = entry["id"]
            ip = f"{self.base_ip}.{self.ip_offset + wid}"
            name = _generate_name(used_names)
            self.robots.append(Robot(id=wid, name=name, ip=ip))

        self.robots.sort(key=lambda r: r.id)

    def get_robot(self, robot_id: int) -> Robot | None:
        for r in self.robots:
            if r.id == robot_id:
                return r
        return None

    @property
    def log_dir(self) -> str:
        d = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "logs",
        )
        os.makedirs(d, exist_ok=True)
        return d
