"""Real-robot fleet orchestration: SSH + Docker control plane."""

from armory.real.config import FleetConfig, Robot, RobotStatus
from armory.real.dispatcher import FleetDispatcher
from armory.real.fleet import FleetController

__all__ = [
    "FleetConfig",
    "FleetController",
    "FleetDispatcher",
    "Robot",
    "RobotStatus",
]
