"""High-level fleet command dispatcher.

Composes ``FleetController`` primitives into the broadcast operations a
fleet operator actually invokes (enable, disable, goto, boot, shutdown,
client/listener lifecycle, run-trial).
"""

from __future__ import annotations

import asyncio
import pathlib
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

    def run_trial(
        self,
        robots: list[Robot],
        duration_sec: float,
        output_dir: pathlib.Path,
        fetch_video: bool = False,
        grace_sec: float = 5.0,
        remote_subdir: str = "armory_episodes",
        callback: Callable | None = None,
        control_hz_overrides: dict[int, int] | None = None,
    ):
        """Run a bounded client trial then fetch each robot's data via SFTP.

        Sequence:
          1. ``start_clients(robots)``
          2. wait ``duration_sec`` seconds
          3. ``kill_clients(robots, grace_sec=grace_sec)`` (SIGINT, then SIGKILL)
          4. brief settle so RealSaver flushes its background writes
          5. ``fetch_episode_data(robots, output_dir, remote_subdir, fetch_video)``

        ``control_hz_overrides`` (workstation id → control_hz) is forwarded
        per-robot as ``--ros-args -p control_hz:=<N>``. Robots not in the dict
        use the node's compiled-in default.

        Returns a Future whose result is a summary dict with keys
        ``start``, ``kill``, ``fetch``, and ``output_dir``.
        """
        return self.fleet.submit(
            self._run_trial(
                robots, duration_sec, pathlib.Path(output_dir),
                fetch_video, grace_sec, remote_subdir, callback,
                control_hz_overrides,
            )
        )

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

    async def _run_trial(
        self,
        robots: list[Robot],
        duration_sec: float,
        output_dir: pathlib.Path,
        fetch_video: bool,
        grace_sec: float,
        remote_subdir: str,
        callback: Callable | None,
        control_hz_overrides: dict[int, int] | None = None,
    ):
        log = self.fleet.logger
        n = len(robots)

        extra_args_per_robot = (
            {rid: f"--ros-args -p control_hz:={int(hz)}"
             for rid, hz in control_hz_overrides.items()}
            if control_hz_overrides
            else None
        )
        if extra_args_per_robot:
            applied = {r.id: extra_args_per_robot[r.id] for r in robots
                       if r.id in extra_args_per_robot}
            if applied:
                log.info(f"trial: applying control_hz overrides for {len(applied)} robot(s)")

        log.info(
            f"trial: start_clients on {n} robot(s); will run for {duration_sec:.1f}s"
        )
        start_results = await self.fleet._start_clients(
            robots, callback=None, extra_args_per_robot=extra_args_per_robot,
        )

        await asyncio.sleep(max(0.0, float(duration_sec)))

        log.info(f"trial: killing clients (SIGINT, grace={grace_sec:.1f}s)")
        kill_results = await self.fleet._kill_clients(
            robots, callback=None, grace_sec=grace_sec
        )

        # Settle so RealSaver's executor finishes flushing after SIGINT.
        await asyncio.sleep(min(grace_sec, 2.0))

        log.info(f"trial: fetching episode data to {output_dir}")
        fetch_results = await self.fleet._fetch_episode_data(
            robots,
            local_dir=pathlib.Path(output_dir),
            remote_subdir=remote_subdir,
            include_video=fetch_video,
            callback=None,
        )

        summary = {
            "start": start_results,
            "kill": kill_results,
            "fetch": fetch_results,
            "output_dir": str(output_dir),
        }
        log.info(f"trial: complete — {output_dir}")
        if callback:
            callback(summary)
        return summary

    @staticmethod
    def _result_ok(result) -> bool:
        return result is not None and not str(result).startswith("ERROR")
