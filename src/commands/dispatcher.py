"""Command dispatcher — broadcast robot commands concurrently."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Callable

from ..core.config import Robot, RobotStatus

if TYPE_CHECKING:
    from ..core.ssh_client import SSHManager


class CommandDispatcher:
    """Dispatches broadcast commands to selected robots."""

    def __init__(self, ssh_manager: SSHManager):
        self.ssh = ssh_manager

    # ── broadcast commands ──────────────────────────────────────

    def enable(self, robots: list[Robot], callback: Callable | None = None):
        """Enable: 'p enable' then 'p goto init'."""
        return self.ssh.submit(self._enable(robots, callback))

    def disable(self, robots: list[Robot], callback: Callable | None = None):
        """Disable: 'p goto reset' then 'p disable'."""
        return self.ssh.submit(self._disable(robots, callback))

    def goto_init(self, robots: list[Robot], callback: Callable | None = None):
        """Goto Init: 'p goto init'."""
        return self.ssh.run_on_robots(robots, "p goto init", callback)

    def goto_zero(self, robots: list[Robot], callback: Callable | None = None):
        """Goto Zero: 'p goto zero'."""
        return self.ssh.run_on_robots(robots, "p goto zero", callback)

    def connect_to_server(self, robots: list[Robot], callback: Callable | None = None):
        """Dummy: set all selected robots to ONLINE status."""
        for r in robots:
            if r.status != RobotStatus.OFFLINE:
                r.status = RobotStatus.ONLINE
        if callback:
            callback({r.id: "Connected (simulated)" for r in robots})

    def boot(self, robots: list[Robot], callback: Callable | None = None):
        """Boot Docker containers on selected robots."""
        return self.ssh.boot_robots(robots, callback)

    def shutdown(self, robots: list[Robot], callback: Callable | None = None):
        """Stop and remove Docker containers on selected robots."""
        return self.ssh.shutdown_robots(robots, callback)

    # ── sequential compound commands ────────────────────────────

    async def _enable(self, robots: list[Robot], callback: Callable | None = None):
        booted = [r for r in robots if r.status != RobotStatus.OFFLINE]
        if not booted:
            if callback:
                callback({r.id: "Skipped — robot offline" for r in robots})
            return

        # Step 1: p enable
        await self.ssh._run_on_robots(booted, "p enable")
        # Step 2: p goto init
        await self.ssh._run_on_robots(booted, "p goto init", callback)

    async def _disable(self, robots: list[Robot], callback: Callable | None = None):
        booted = [r for r in robots if r.status != RobotStatus.OFFLINE]
        if not booted:
            if callback:
                callback({r.id: "Skipped — robot offline" for r in robots})
            return

        # Step 1: p goto reset
        await self.ssh._run_on_robots(booted, "p goto reset")
        # Step 2: p disable
        await self.ssh._run_on_robots(booted, "p disable", callback)
