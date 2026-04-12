"""Curses-based CLI dashboard for ARMory."""

from __future__ import annotations

import curses
import threading
import time
from collections import deque
from typing import TYPE_CHECKING

from ..core.config import RobotStatus

if TYPE_CHECKING:
    from ..commands.dispatcher import CommandDispatcher
    from ..core.config import Config
    from ..core.ssh_client import SSHManager


# ── colour pairs ────────────────────────────────────────────────
PAIR_HEADER = 1
PAIR_OFFLINE = 2
PAIR_BOOTED = 3
PAIR_ONLINE = 4
PAIR_BORDER = 5
PAIR_CMD = 6
PAIR_LOG = 7
PAIR_HIGHLIGHT = 8
PAIR_MUTED = 9
PAIR_NOTICE = 10
PAIR_SURFACE = 11

MIN_WIDTH = 80
MIN_HEIGHT = 24

CORE_COMMANDS = [
    ("6", "Boot Fleet"),
    ("T", "Start Tunnel"),
    ("X", "Kill Tunnel"),
    ("7", "Shutdown"),
    ("R", "Refresh"),
    ("Q", "Quit"),
]

RUNTIME_COMMANDS = [
    ("5", "Connect"),
    ("1", "Enable"),
    ("2", "Disable"),
    ("3", "Goto Init"),
    ("4", "Goto Zero"),
]

RUNTIME_KEYS = {key for key, _ in RUNTIME_COMMANDS}
ENTER_KEYS = {curses.KEY_ENTER, 10, 13}

TARGET_COMMANDS = [
    ("6", "Boot Docker", "boot"),
    ("T", "Start Tunnel", "start_tunnel"),
    ("X", "Kill Tunnel", "kill_tunnel"),
    ("7", "Shutdown Docker", "shutdown"),
    ("5", "Connect to Server", "connect"),
    ("1", "Enable", "enable"),
    ("2", "Disable", "disable"),
    ("3", "Goto Init", "goto_init"),
    ("4", "Goto Zero", "goto_zero"),
]


class Dashboard:
    """Renders the main ARMory curses dashboard."""

    def __init__(self, config: Config, ssh: SSHManager, dispatcher: CommandDispatcher):
        self.config = config
        self.ssh = ssh
        self.dispatcher = dispatcher
        self.log_lines: deque[str] = deque(maxlen=200)
        self._busy = False
        self._lock = threading.Lock()
        self._last_busy_notice = 0.0
        self._runtime_unlocked_announced = False
        self._focused_robot_idx = 0
        self._selected_robot_ids: set[int] = set()

    # ── public entry point ──────────────────────────────────────

    def run(self):
        curses.wrapper(self._main)

    # ── curses main loop ────────────────────────────────────────

    def _main(self, stdscr: curses.window):
        self._stdscr = stdscr
        self._init_colors()
        try:
            curses.curs_set(0)
        except curses.error:
            pass
        stdscr.nodelay(True)
        stdscr.keypad(True)

        self._log("ARMory dashboard started.")
        self._log("Querying workstation status...")

        self._busy = True
        self.ssh.check_all_status(callback=self._on_status_done)

        while True:
            try:
                self._render()
                key = stdscr.getch()
                if key == -1:
                    curses.napms(100)
                    continue
                if self._handle_key(key):
                    break
            except KeyboardInterrupt:
                break

    # ── input handling ──────────────────────────────────────────

    def _handle_key(self, key: int) -> bool:
        """Handle a keypress. Returns True to quit."""
        ch = chr(key).upper() if 0 <= key < 256 else ""

        if ch == "Q":
            return True

        if key in (curses.KEY_UP, curses.KEY_BTAB) or ch == "K":
            self._move_robot_focus(-1)
            return False
        if key in (curses.KEY_DOWN, ord("\t")) or ch == "J":
            self._move_robot_focus(1)
            return False
        if ch == " ":
            self._toggle_focused_robot()
            return False
        if key in ENTER_KEYS:
            if self._busy:
                self._log("Current operation still running. Robot command menu is locked.")
                return False
            self._open_robot_command_menu()
            return False

        if self._busy:
            now = time.time()
            if now - self._last_busy_notice > 1.5:
                self._last_busy_notice = now
                self._log("Current operation still running. Controls will unlock shortly.")
            return False

        if ch in RUNTIME_KEYS and not self._runtime_controls_unlocked():
            self._log(
                "Runtime controls stay locked until every workstation is booted "
                f"({self._ready_robot_count()}/{len(self.config.robots)} ready)."
            )
            return False

        if ch == "R":
            self._refresh_status()
        elif ch == "T":
            self._do_start_tunnel()
        elif ch == "X":
            self._do_kill_tunnel()
        elif ch == "1":
            self._broadcast_with_confirm("Enable", self.dispatcher.enable)
        elif ch == "2":
            self._broadcast_with_confirm("Disable", self.dispatcher.disable)
        elif ch == "3":
            self._broadcast_with_confirm("Goto Init", self.dispatcher.goto_init)
        elif ch == "4":
            self._broadcast_with_confirm("Goto Zero", self.dispatcher.goto_zero)
        elif ch == "5":
            self._do_connect_to_server()
        elif ch == "6":
            self._do_boot()
        elif ch == "7":
            self._do_shutdown()

        return False

    # ── command actions ─────────────────────────────────────────

    def _refresh_status(self):
        self._busy = True
        self._log("Refreshing status...")
        self.ssh.check_all_status(callback=self._on_status_done)

    def _on_status_done(self, *_args):
        with self._lock:
            self._busy = False
        self._announce_runtime_state()
        self._log("Status refresh complete.")

    def _broadcast_with_confirm(self, name: str, action_fn):
        """Show Y/N confirmation, then execute the broadcast command."""
        if not self._confirm(f"Execute '{name}' across the ready fleet?"):
            self._log(f"{name} cancelled.")
            return

        active = [r for r in self.config.robots if r.status != RobotStatus.OFFLINE]
        if not active:
            self._log(f"No active robots for '{name}'.")
            return

        self._busy = True
        self._log(f"Executing '{name}' on {len(active)} robot(s)...")

        def on_done(results=None):
            with self._lock:
                self._busy = False
            if results:
                for rid, out in results.items():
                    self._log(f"  WS-{rid}: {str(out)[:80]}")
            self._announce_runtime_state()
            self._log(f"'{name}' complete.")

        action_fn(active, callback=on_done)

    def _do_boot(self):
        """Boot targets all robots regardless of status."""
        if not self._confirm("Boot Docker on all workstations?"):
            self._log("Boot cancelled.")
            return

        targets = list(self.config.robots)
        self._busy = True
        self._log(f"Booting {len(targets)} workstation(s)...")

        def on_done(results=None):
            with self._lock:
                self._busy = False
            if results:
                for rid, out in results.items():
                    self._log(f"  WS-{rid}: {str(out)[:80]}")
            self._announce_runtime_state()
            self._log("Boot complete.")

        self.dispatcher.boot(targets, callback=on_done)

    def _do_shutdown(self):
        """Disable, kill tunnels, then shutdown all non-offline robots."""
        active = [r for r in self.config.robots if r.status != RobotStatus.OFFLINE]
        if not active:
            self._log("No active robots to shut down.")
            return
        if not self._confirm(
            f"Disable, kill tunnel, then shutdown {len(active)} workstation(s)?"
        ):
            self._log("Shutdown cancelled.")
            return

        self._busy = True
        self._log(
            f"Disabling, killing tunnels, and shutting down {len(active)} workstation(s)..."
        )

        def on_done(results=None):
            with self._lock:
                self._busy = False
            if results:
                for rid, out in results.items():
                    self._log(f"  WS-{rid}: {str(out)[:80]}")
            self._announce_runtime_state()
            self._log("Shutdown complete.")

        self.dispatcher.shutdown(active, callback=on_done)

    def _do_start_tunnel(self):
        """Start SSH tunnels on every configured workstation."""
        targets = list(self.config.robots)
        if not targets:
            self._log("No workstations configured for tunnel start.")
            return
        if not self._confirm(f"Start SSH tunnel on {len(targets)} workstation(s)?"):
            self._log("Start Tunnel cancelled.")
            return

        self._busy = True
        self._log(f"Starting SSH tunnel on {len(targets)} workstation(s)...")

        def on_done(results=None):
            with self._lock:
                self._busy = False
            if results:
                for rid, out in results.items():
                    self._log(f"  WS-{rid}: {str(out)[:80]}")
            self._log("Start Tunnel complete.")

        self.dispatcher.start_tunnel(targets, callback=on_done)

    def _do_kill_tunnel(self):
        """Kill SSH tunnels on every configured workstation."""
        targets = list(self.config.robots)
        if not targets:
            self._log("No workstations configured for tunnel kill.")
            return
        if not self._confirm(f"Kill SSH tunnel on {len(targets)} workstation(s)?"):
            self._log("Kill Tunnel cancelled.")
            return

        self._busy = True
        self._log(f"Killing SSH tunnel on {len(targets)} workstation(s)...")

        def on_done(results=None):
            with self._lock:
                self._busy = False
            if results:
                for rid, out in results.items():
                    self._log(f"  WS-{rid}: {str(out)[:80]}")
            self._log("Kill Tunnel complete.")

        self.dispatcher.kill_tunnel(targets, callback=on_done)

    def _do_connect_to_server(self):
        if not self._confirm("Connect all booted robots to server?"):
            self._log("Connect to Server cancelled.")
            return

        booted = [r for r in self.config.robots if r.status == RobotStatus.BOOTED]
        if not booted:
            self._log("No booted robots to connect.")
            return

        self.dispatcher.connect_to_server(booted)
        self._announce_runtime_state()
        self._log(f"Connected {len(booted)} robot(s) to server (simulated).")

    def _execute_target_command(
        self,
        action: str,
        name: str,
        targets: list,
    ) -> bool:
        """Execute a command against the selected/focused robot set."""
        eligible = self._eligible_targets(action, targets)
        skipped = len(targets) - len(eligible)
        target_label = self._format_robot_ids(eligible)

        if not eligible:
            self._log(
                f"No eligible robot(s) for '{name}' in "
                f"{self._format_robot_ids(targets)}."
            )
            return False

        if skipped:
            self._log(f"Skipped {skipped} ineligible selected robot(s) for '{name}'.")

        if action == "connect":
            self.dispatcher.connect_to_server(eligible)
            self._announce_runtime_state()
            self._log(f"Connected {target_label} to server (simulated).")
            return True

        dispatch = getattr(self.dispatcher, action)
        self._busy = True
        self._log(f"Running '{name}' on {target_label}...")

        def on_done(results=None):
            with self._lock:
                self._busy = False
            if results:
                for rid, out in results.items():
                    self._log(f"  WS-{rid}: {str(out)[:80]}")
            self._announce_runtime_state()
            self._log(f"'{name}' complete for {target_label}.")

        dispatch(eligible, callback=on_done)
        return True

    def _eligible_targets(self, action: str, targets: list) -> list:
        if action in ("boot", "start_tunnel", "kill_tunnel"):
            return targets
        if action == "connect":
            return [r for r in targets if r.status == RobotStatus.BOOTED]
        if action == "shutdown":
            return [r for r in targets if r.status != RobotStatus.OFFLINE]
        return [r for r in targets if r.status != RobotStatus.OFFLINE]

    def _open_robot_command_menu(self):
        targets = self._target_robots()
        if not targets:
            self._log("No robot selected.")
            return

        stdscr = self._stdscr
        h, w = stdscr.getmaxyx()
        box_w = min(74, w - 6)
        box_h = min(max(14, len(TARGET_COMMANDS) + 7), h - 4)
        start_y = max(1, h // 2 - box_h // 2)
        start_x = max(2, w // 2 - box_w // 2)
        win = curses.newwin(box_h, box_w, start_y, start_x)
        win.bkgd(" ", curses.color_pair(PAIR_SURFACE))

        stdscr.nodelay(False)
        notice = ""
        try:
            while True:
                self._draw_robot_command_menu(win, box_h, box_w, targets, notice)
                key = stdscr.getch()
                ch = chr(key).upper() if 0 <= key < 256 else ""

                if ch in ("Q", "N") or key == 27:
                    self._log("Robot command menu closed.")
                    return

                for command_key, label, action in TARGET_COMMANDS:
                    if ch != command_key:
                        continue
                    if not self._eligible_targets(action, targets):
                        curses.beep()
                        notice = f"Unavailable: {label} ({self._target_disabled_reason(action)})"
                        break
                    if self._execute_target_command(action, label, targets):
                        return
        finally:
            stdscr.nodelay(True)

    def _draw_robot_command_menu(
        self,
        win: curses.window,
        h: int,
        w: int,
        targets: list,
        notice: str = "",
    ):
        win.erase()
        win.bkgd(" ", curses.color_pair(PAIR_SURFACE))
        self._draw_box(win, 0, 0, h, w, "Robot Command Menu")

        target_label = self._format_robot_ids(targets)
        self._center_text(
            win,
            2,
            1,
            w - 2,
            f"Target: {target_label}",
            curses.color_pair(PAIR_LOG) | curses.A_BOLD,
        )
        self._center_text(
            win,
            3,
            1,
            w - 2,
            notice or "Press a command key. Disabled commands stay locked.",
            curses.color_pair(PAIR_NOTICE if notice else PAIR_MUTED),
        )

        row = 5
        for key, label, action in TARGET_COMMANDS:
            if row >= h - 2:
                break

            available = bool(self._eligible_targets(action, targets))
            attr = curses.color_pair(PAIR_CMD if available else PAIR_MUTED)
            key_attr = curses.color_pair(PAIR_HIGHLIGHT if available else PAIR_MUTED) | curses.A_BOLD
            suffix = "" if available else f" ({self._target_disabled_reason(action)})"

            self._add_text(win, row, 4, "[", attr)
            self._add_text(win, row, 5, key, key_attr)
            self._add_text(win, row, 6, f"] {label}{suffix}"[: w - 9], attr)
            row += 1

        self._center_text(
            win,
            h - 2,
            1,
            w - 2,
            "[Q] Cancel",
            curses.color_pair(PAIR_NOTICE),
        )
        win.refresh()

    @staticmethod
    def _target_disabled_reason(action: str) -> str:
        if action == "connect":
            return "needs booted"
        if action in ("boot", "start_tunnel", "kill_tunnel"):
            return "unavailable"
        return "needs active"

    # ── confirmation dialog ─────────────────────────────────────

    def _confirm(self, message: str) -> bool:
        """Show a blocking Y/N confirmation prompt."""
        stdscr = self._stdscr
        h, w = stdscr.getmaxyx()
        box_w = min(max(40, len(message) + 8), w - 6)
        box_h = 7
        start_y = max(1, h // 2 - box_h // 2)
        start_x = max(2, w // 2 - box_w // 2)

        win = curses.newwin(box_h, box_w, start_y, start_x)
        win.bkgd(" ", curses.color_pair(PAIR_SURFACE))
        self._draw_box(win, 0, 0, box_h, box_w, "Confirm Action")
        self._center_text(
            win,
            2,
            1,
            box_w - 2,
            message,
            curses.color_pair(PAIR_LOG) | curses.A_BOLD,
        )
        self._center_text(
            win,
            4,
            1,
            box_w - 2,
            "[Y] Confirm   [N] Cancel",
            curses.color_pair(PAIR_NOTICE),
        )
        win.refresh()

        stdscr.nodelay(False)
        while True:
            ch = stdscr.getch()
            if ch in (ord("y"), ord("Y")):
                stdscr.nodelay(True)
                return True
            if ch in (ord("n"), ord("N"), 27):  # 27 = ESC
                stdscr.nodelay(True)
                return False

    # ── rendering ───────────────────────────────────────────────

    def _render(self):
        stdscr = self._stdscr
        try:
            h, w = stdscr.getmaxyx()
        except Exception:
            return

        if h < MIN_HEIGHT or w < MIN_WIDTH:
            stdscr.erase()
            self._center_text(
                stdscr,
                h // 2,
                0,
                w,
                f"Resize to at least {MIN_WIDTH}x{MIN_HEIGHT} for the polished ARMory layout.",
                curses.color_pair(PAIR_NOTICE) | curses.A_BOLD,
            )
            stdscr.refresh()
            return

        stdscr.erase()
        self._clamp_robot_focus()

        content_w = min(w - 2, 124)
        content_h = min(h - 1, 34)
        origin_x = max(0, (w - content_w) // 2)
        origin_y = max(0, (h - content_h) // 2)

        header_h = 2
        gap = 1
        desired_log_h = max(7, min(10, content_h // 3))
        min_body_h = 12 if self._runtime_controls_unlocked() else 10
        log_h = min(desired_log_h, content_h - header_h - (gap * 2) - min_body_h)
        body_h = content_h - header_h - log_h - (gap * 2)

        side_w = min(44, max(36, content_w // 3))
        right_w = content_w - side_w - gap

        self._draw_header(origin_y, origin_x, content_w)
        self._draw_side_panel(origin_y + header_h + gap, origin_x, body_h, side_w)

        right_x = origin_x + side_w + gap
        right_y = origin_y + header_h + gap
        if self._runtime_controls_unlocked():
            fleet_h = min(6, body_h - gap - 6)
            fleet_h = max(5, fleet_h)
            runtime_h = body_h - fleet_h - gap
            self._draw_core_panel(right_y, right_x, fleet_h, right_w)
            self._draw_runtime_panel(right_y + fleet_h + gap, right_x, runtime_h, right_w)
        else:
            self._draw_core_panel(right_y, right_x, body_h, right_w, show_notice=True)

        log_y = right_y + body_h + gap
        self._draw_log_panel(log_y, origin_x, log_h, content_w)

        stdscr.refresh()

    def _draw_header(self, y: int, x: int, w: int):
        stdscr = self._stdscr
        title = "ARMory"
        self._center_text(
            stdscr,
            y,
            x,
            w,
            title,
            curses.color_pair(PAIR_HIGHLIGHT) | curses.A_BOLD,
        )

        badge = (
            "SYNCING"
            if self._busy
            else "RUNTIME READY"
            if self._runtime_controls_unlocked()
            else f"BOOT {self._ready_robot_count()}/{len(self.config.robots)}"
        )
        badge_attr = curses.color_pair(
            PAIR_BOOTED if self._busy else PAIR_ONLINE if self._runtime_controls_unlocked() else PAIR_NOTICE
        ) | curses.A_BOLD
        self._right_text(stdscr, y, x, w, badge, badge_attr)

        offline, booted, online = self._status_counts()
        segments = [
            ("offline ", curses.color_pair(PAIR_MUTED)),
            (str(offline), curses.color_pair(PAIR_OFFLINE) | curses.A_BOLD),
            ("   booted ", curses.color_pair(PAIR_MUTED)),
            (str(booted), curses.color_pair(PAIR_BOOTED) | curses.A_BOLD),
            ("   online ", curses.color_pair(PAIR_MUTED)),
            (str(online), curses.color_pair(PAIR_ONLINE) | curses.A_BOLD),
        ]
        self._draw_centered_segments(stdscr, y + 1, x, w, segments)

    def _draw_side_panel(self, y: int, x: int, h: int, w: int):
        stdscr = self._stdscr
        self._draw_box(stdscr, y, x, h, w, "Fleet Matrix")

        total = len(self.config.robots)
        ready = self._ready_robot_count()
        selected = len(self._selected_robot_ids)
        self._center_text(
            stdscr,
            y + 1,
            x + 1,
            w - 2,
            f"{ready}/{total} ready | {selected} selected",
            curses.color_pair(PAIR_NOTICE),
        )

        show_ip = h >= (len(self.config.robots) * 2 + 5)
        row = y + 3
        for idx, robot in enumerate(self.config.robots):
            if row >= y + h - 2:
                break

            focused = idx == self._focused_robot_idx
            selected = robot.id in self._selected_robot_ids
            status_text = robot.status.value.upper()
            pair = self._status_pair(robot.status)
            focus_marker = ">" if focused else " "
            select_marker = "*" if selected else " "
            label = f"{focus_marker}{select_marker} WS-{robot.id}  {robot.name}"
            status_x = x + w - len(status_text) - 3
            label_w = max(0, status_x - (x + 2) - 1)
            row_attr = curses.A_REVERSE if focused else 0
            label_pair = PAIR_HIGHLIGHT if selected else PAIR_CMD

            self._add_text(
                stdscr,
                row,
                x + 2,
                label[:label_w],
                curses.color_pair(label_pair) | curses.A_BOLD | row_attr,
            )
            self._add_text(
                stdscr,
                row,
                status_x,
                status_text,
                curses.color_pair(pair) | curses.A_BOLD | row_attr,
            )
            row += 1

            if show_ip and row < y + h - 2:
                self._add_text(
                    stdscr,
                    row,
                    x + 2,
                    robot.ip[: w - 4],
                    curses.color_pair(PAIR_MUTED) | row_attr,
                )
                row += 1

        if h >= 8:
            self._center_text(
                stdscr,
                y + h - 2,
                x + 1,
                w - 2,
                "Up/Down focus | Space select | Enter command",
                curses.color_pair(PAIR_MUTED),
            )

    def _draw_core_panel(self, y: int, x: int, h: int, w: int, show_notice: bool = False):
        stdscr = self._stdscr
        self._draw_box(stdscr, y, x, h, w, "Core Controls")
        self._draw_command_grid(y + 2, x + 2, w - 4, h - 3, CORE_COMMANDS)

        if not show_notice or h < 8:
            return

        ready = self._ready_robot_count()
        total = len(self.config.robots)
        self._center_text(
            stdscr,
            y + h - 3,
            x + 1,
            w - 2,
            "Runtime controls appear once the fleet is fully booted.",
            curses.color_pair(PAIR_MUTED),
        )
        self._draw_centered_segments(
            stdscr,
            y + h - 2,
            x + 1,
            w - 2,
            [
                ("ready ", curses.color_pair(PAIR_MUTED)),
                (f"{ready}/{total}", curses.color_pair(PAIR_HIGHLIGHT) | curses.A_BOLD),
                ("  press [6] to boot remaining workstations", curses.color_pair(PAIR_NOTICE)),
            ],
        )

    def _draw_runtime_panel(self, y: int, x: int, h: int, w: int):
        stdscr = self._stdscr
        self._draw_box(stdscr, y, x, h, w, "Runtime Controls")
        self._draw_command_grid(y + 2, x + 2, w - 4, h - 3, RUNTIME_COMMANDS)

    def _draw_command_grid(
        self,
        y: int,
        x: int,
        w: int,
        max_rows: int,
        commands: list[tuple[str, str]],
    ):
        stdscr = self._stdscr
        cols = 2 if w >= 30 and len(commands) > 1 else 1
        gap = 3
        col_w = max(12, (w - (gap * (cols - 1))) // cols)

        for idx, (key, label) in enumerate(commands):
            row = y + (idx // cols)
            if row >= y + max_rows:
                break
            col = idx % cols
            cell_x = x + col * (col_w + gap)

            self._add_text(stdscr, row, cell_x, "[", curses.color_pair(PAIR_CMD))
            self._add_text(
                stdscr,
                row,
                cell_x + 1,
                key,
                curses.color_pair(PAIR_HIGHLIGHT) | curses.A_BOLD,
            )
            self._add_text(stdscr, row, cell_x + 1 + len(key), "] ", curses.color_pair(PAIR_CMD))
            text_x = cell_x + len(key) + 3
            label_w = max(0, col_w - (text_x - cell_x))
            self._add_text(
                stdscr,
                row,
                text_x,
                label[:label_w],
                curses.color_pair(PAIR_CMD),
            )

    def _draw_log_panel(self, y: int, x: int, h: int, w: int):
        stdscr = self._stdscr
        self._draw_box(stdscr, y, x, h, w, "Signal Log")

        visible = list(self.log_lines)[-max(0, h - 2) :]
        if not visible:
            self._center_text(
                stdscr,
                y + 2,
                x + 1,
                w - 2,
                "No activity yet.",
                curses.color_pair(PAIR_MUTED),
            )
            return

        row = y + 1
        for line in visible:
            if row >= y + h - 1:
                break
            self._add_text(
                stdscr,
                row,
                x + 2,
                line[: w - 4],
                curses.color_pair(PAIR_LOG),
            )
            row += 1

    # ── helpers ─────────────────────────────────────────────────

    def _runtime_controls_unlocked(self) -> bool:
        return bool(self.config.robots) and self._ready_robot_count() == len(self.config.robots)

    def _clamp_robot_focus(self):
        if not self.config.robots:
            self._focused_robot_idx = 0
            self._selected_robot_ids.clear()
            return

        self._focused_robot_idx = max(
            0,
            min(self._focused_robot_idx, len(self.config.robots) - 1),
        )
        valid_ids = {robot.id for robot in self.config.robots}
        self._selected_robot_ids.intersection_update(valid_ids)

    def _move_robot_focus(self, delta: int):
        self._clamp_robot_focus()
        if not self.config.robots:
            return
        self._focused_robot_idx = (self._focused_robot_idx + delta) % len(self.config.robots)

    def _toggle_focused_robot(self):
        robot = self._focused_robot()
        if robot is None:
            self._log("No robot available to select.")
            return

        if robot.id in self._selected_robot_ids:
            self._selected_robot_ids.remove(robot.id)
            self._log(f"Unselected WS-{robot.id}.")
            return

        self._selected_robot_ids.add(robot.id)
        self._log(f"Selected WS-{robot.id}.")

    def _focused_robot(self):
        self._clamp_robot_focus()
        if not self.config.robots:
            return None
        return self.config.robots[self._focused_robot_idx]

    def _target_robots(self) -> list:
        self._clamp_robot_focus()
        if self._selected_robot_ids:
            return [
                robot
                for robot in self.config.robots
                if robot.id in self._selected_robot_ids
            ]

        robot = self._focused_robot()
        return [robot] if robot is not None else []

    @staticmethod
    def _format_robot_ids(robots: list) -> str:
        if not robots:
            return "none"
        labels = [f"WS-{robot.id}" for robot in robots]
        if len(labels) <= 5:
            return ", ".join(labels)
        return f"{', '.join(labels[:5])}, +{len(labels) - 5} more"

    def _ready_robot_count(self) -> int:
        return sum(
            robot.status in (RobotStatus.BOOTED, RobotStatus.ONLINE)
            for robot in self.config.robots
        )

    def _status_counts(self) -> tuple[int, int, int]:
        offline = sum(robot.status == RobotStatus.OFFLINE for robot in self.config.robots)
        booted = sum(robot.status == RobotStatus.BOOTED for robot in self.config.robots)
        online = sum(robot.status == RobotStatus.ONLINE for robot in self.config.robots)
        return offline, booted, online

    def _announce_runtime_state(self):
        unlocked = self._runtime_controls_unlocked()
        if unlocked and not self._runtime_unlocked_announced:
            self._runtime_unlocked_announced = True
            self._log("All workstations are ready. Runtime controls unlocked.")
        elif not unlocked:
            self._runtime_unlocked_announced = False

    def _log(self, msg: str):
        ts = time.strftime("%H:%M:%S")
        self.log_lines.append(f"[{ts}] {msg}")
        if self.ssh is not None:
            self.ssh.system_logger.info(msg)

    def _draw_box(
        self,
        target: curses.window,
        y: int,
        x: int,
        h: int,
        w: int,
        title: str,
    ):
        if h < 3 or w < 6:
            return

        border_attr = curses.color_pair(PAIR_BORDER)
        self._add_text(target, y, x, f"╭{'─' * (w - 2)}╮", border_attr)
        for row in range(y + 1, y + h - 1):
            self._add_text(target, row, x, "│", border_attr)
            self._add_text(target, row, x + w - 1, "│", border_attr)
        self._add_text(target, y + h - 1, x, f"╰{'─' * (w - 2)}╯", border_attr)

        label = f" {title} "
        label_x = x + max(2, min(w - len(label) - 2, 3))
        self._add_text(
            target,
            y,
            label_x,
            label[: max(0, w - 4)],
            border_attr | curses.A_BOLD,
        )

    def _draw_centered_segments(
        self,
        target: curses.window,
        y: int,
        x: int,
        w: int,
        segments: list[tuple[str, int]],
    ):
        total = sum(len(text) for text, _ in segments)
        cursor = x + max(0, (w - total) // 2)
        for text, attr in segments:
            self._add_text(target, y, cursor, text, attr)
            cursor += len(text)

    def _center_text(
        self,
        target: curses.window,
        y: int,
        x: int,
        w: int,
        text: str,
        attr: int = 0,
    ):
        if w <= 0:
            return
        clipped = text[:w]
        start_x = x + max(0, (w - len(clipped)) // 2)
        self._add_text(target, y, start_x, clipped, attr)

    def _right_text(
        self,
        target: curses.window,
        y: int,
        x: int,
        w: int,
        text: str,
        attr: int = 0,
    ):
        start_x = x + max(0, w - len(text))
        self._add_text(target, y, start_x, text[:w], attr)

    def _status_pair(self, status: RobotStatus) -> int:
        if status == RobotStatus.ONLINE:
            return PAIR_ONLINE
        if status == RobotStatus.BOOTED:
            return PAIR_BOOTED
        return PAIR_OFFLINE

    @staticmethod
    def _add_text(target: curses.window, y: int, x: int, text: str, attr: int = 0):
        if not text:
            return
        try:
            target.addstr(y, x, text, attr)
        except curses.error:
            pass

    @staticmethod
    def _init_colors():
        curses.start_color()
        curses.use_default_colors()

        if curses.COLORS >= 256:
            curses.init_pair(PAIR_HEADER, 16, 225)
            curses.init_pair(PAIR_OFFLINE, 210, -1)
            curses.init_pair(PAIR_BOOTED, 223, -1)
            curses.init_pair(PAIR_ONLINE, 151, -1)
            curses.init_pair(PAIR_BORDER, 183, -1)
            curses.init_pair(PAIR_CMD, 225, -1)
            curses.init_pair(PAIR_LOG, 252, -1)
            curses.init_pair(PAIR_HIGHLIGHT, 219, -1)
            curses.init_pair(PAIR_MUTED, 246, -1)
            curses.init_pair(PAIR_NOTICE, 117, -1)
            curses.init_pair(PAIR_SURFACE, 255, 236)
            return

        curses.init_pair(PAIR_HEADER, curses.COLOR_BLACK, curses.COLOR_MAGENTA)
        curses.init_pair(PAIR_OFFLINE, curses.COLOR_RED, -1)
        curses.init_pair(PAIR_BOOTED, curses.COLOR_YELLOW, -1)
        curses.init_pair(PAIR_ONLINE, curses.COLOR_GREEN, -1)
        curses.init_pair(PAIR_BORDER, curses.COLOR_MAGENTA, -1)
        curses.init_pair(PAIR_CMD, curses.COLOR_WHITE, -1)
        curses.init_pair(PAIR_LOG, curses.COLOR_WHITE, -1)
        curses.init_pair(PAIR_HIGHLIGHT, curses.COLOR_CYAN, -1)
        curses.init_pair(PAIR_MUTED, curses.COLOR_WHITE, -1)
        curses.init_pair(PAIR_NOTICE, curses.COLOR_CYAN, -1)
        curses.init_pair(PAIR_SURFACE, curses.COLOR_WHITE, curses.COLOR_BLACK)
