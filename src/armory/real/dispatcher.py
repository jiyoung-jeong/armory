"""High-level fleet command dispatcher.

Composes ``FleetController`` primitives into the broadcast operations a
fleet operator actually invokes (enable, disable, goto, boot, shutdown,
client/listener lifecycle).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

from armory.real.config import Robot, RobotStatus

if TYPE_CHECKING:
    from armory.real.fleet import FleetController


class FleetDispatcher:
    """Dispatches broadcast commands to selected robots."""

    def __init__(self, fleet: FleetController):
        self.fleet = fleet

    # ── broadcast commands ──────────────────────────────────────

    def enable(self, robots: list[Robot], callback: Callable | None = None):
        """Enable: 'p enable' then 'p goto init'."""
        return self.fleet.submit(self._enable(robots, callback))

    def disable(self, robots: list[Robot], callback: Callable | None = None):
        """Disable: 'p goto reset' then 'p disable'."""
        return self.fleet.submit(self._disable(robots, callback))

    def goto_init(self, robots: list[Robot], callback: Callable | None = None):
        """Goto Init: 'p goto init'."""
        return self.fleet.run_on_robots(robots, "p goto init", callback)

    def goto_zero(self, robots: list[Robot], callback: Callable | None = None):
        """Goto Zero: 'p goto zero'."""
        return self.fleet.run_on_robots(robots, "p goto zero", callback)

    def connect_to_server(self, robots: list[Robot], callback: Callable | None = None):
        """Mark selected robots ONLINE.

        Stub today — flips local status only. Replaced in the next phase by a
        real RPC that registers the fleet with the running armory server, so
        scheduling decisions can be tagged with fleet metadata for fairness/
        starvation metrics.
        """
        # TODO(real-fleet-metrics): replace stub with real server registration.
        for r in robots:
            if r.status != RobotStatus.OFFLINE:
                r.status = RobotStatus.ONLINE
        if callback:
            callback({r.id: "Connected (simulated)" for r in robots})

    def boot(self, robots: list[Robot], callback: Callable | None = None):
        """Boot Docker containers on selected robots."""
        return self.fleet.boot_robots(robots, callback)

    def shutdown(self, robots: list[Robot], callback: Callable | None = None):
        """Disable first, kill tunnels, then stop Docker containers."""
        return self.fleet.submit(self._shutdown_safely(robots, callback))

    def start_tunnel(self, robots: list[Robot], callback: Callable | None = None):
        """Start SSH tunnels on selected workstations, outside Docker."""
        return self.fleet.start_tunnels(robots, callback)

    def kill_tunnel(self, robots: list[Robot], callback: Callable | None = None):
        """Kill SSH tunnels on selected workstations, outside Docker."""
        return self.fleet.kill_tunnels(robots, callback)

    def start_listener(self, robots: list[Robot], callback: Callable | None = None):
        """Start the local data collection listener inside Docker."""
        return self.fleet.start_data_listeners(robots, callback)

    def start_client(self, robots: list[Robot], callback: Callable | None = None):
        """Start the Piper client node inside Docker."""
        return self.fleet.start_clients(robots, callback)

    def kill_listener(self, robots: list[Robot], callback: Callable | None = None):
        """Stop the local data collection listener inside Docker."""
        return self.fleet.kill_data_listeners(robots, callback)

    def kill_client(self, robots: list[Robot], callback: Callable | None = None):
        """Stop the Piper client node inside Docker."""
        return self.fleet.kill_clients(robots, callback)

    # ── sequential compound commands ────────────────────────────

    async def _enable(self, robots: list[Robot], callback: Callable | None = None):
        booted = [r for r in robots if r.status != RobotStatus.OFFLINE]
        if not booted:
            if callback:
                callback({r.id: "Skipped — robot offline" for r in robots})
            return

        # Step 1: p enable
        await self.fleet._run_on_robots(booted, "p enable")
        # Step 2: p goto init
        await self.fleet._run_on_robots(booted, "p goto init", callback)

    async def _disable(self, robots: list[Robot], callback: Callable | None = None):
        booted = [r for r in robots if r.status != RobotStatus.OFFLINE]
        if not booted:
            if callback:
                callback({r.id: "Skipped — robot offline" for r in robots})
            return

        # Step 1: p goto reset
        await self.fleet._run_on_robots(booted, "p goto reset")
        # Step 2: p disable
        await self.fleet._run_on_robots(booted, "p disable", callback)

    async def _shutdown_safely(
        self,
        robots: list[Robot],
        callback: Callable | None = None,
    ):
        active = [r for r in robots if r.status != RobotStatus.OFFLINE]
        if not active:
            if callback:
                callback({r.id: "Skipped — robot offline" for r in robots})
            return

        # Safeguard: never remove Docker until the robot command stack is disabled.
        reset_results = await self.fleet._run_on_robots(active, "p goto reset")
        disable_results = await self.fleet._run_on_robots(active, "p disable")
        disabled = [
            robot
            for robot in active
            if self._result_ok(reset_results.get(robot.id))
            and self._result_ok(disable_results.get(robot.id))
        ]
        tunnel_results = await self.fleet._kill_tunnels(disabled)
        shutdown_results = await self.fleet._shutdown_robots(disabled)

        results = {}
        for robot in active:
            if robot not in disabled:
                results[robot.id] = (
                    f"Reset: {reset_results.get(robot.id, 'unknown')}; "
                    f"Disable: {disable_results.get(robot.id, 'unknown')}; "
                    "Shutdown skipped — disable safeguard failed"
                )
                continue

            results[robot.id] = (
                f"Reset: {reset_results.get(robot.id, 'unknown')}; "
                f"Disable: {disable_results.get(robot.id, 'unknown')}; "
                f"Tunnel: {tunnel_results.get(robot.id, 'unknown')}; "
                f"Shutdown: {shutdown_results.get(robot.id, 'unknown')}"
            )
        if callback:
            callback(results)
        return results

    @staticmethod
    def _result_ok(result) -> bool:
        return result is not None and not str(result).startswith("ERROR")
