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

COMMANDS = [
    ("1", "Enable"),
    ("2", "Disable"),
    ("3", "Goto Init"),
    ("4", "Goto Zero"),
    ("5", "Connect to Server"),
    ("6", "Boot Workstation"),
    ("7", "Shutdown Workstation"),
    ("R", "Refresh Status"),
    ("Q", "Quit"),
]


class Dashboard:
    """Renders the main ARMory curses dashboard."""

    def __init__(self, config: Config, ssh: SSHManager, dispatcher: CommandDispatcher):
        self.config = config
        self.ssh = ssh
        self.dispatcher = dispatcher
        self.log_lines: deque[str] = deque(maxlen=200)
        self._pending_future = None
        self._busy = False
        self._lock = threading.Lock()

    # ── public entry point ──────────────────────────────────────

    def run(self):
        curses.wrapper(self._main)

    # ── curses main loop ────────────────────────────────────────

    def _main(self, stdscr: curses.window):
        self._stdscr = stdscr
        self._init_colors()
        curses.curs_set(0)
        stdscr.nodelay(True)

        self._log("ARMory dashboard started.")
        self._log("Querying workstation status...")

        # Initial status check
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

        if self._busy:
            self._log("Please wait — operation in progress...")
            return False

        if ch == "R":
            self._refresh_status()
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
        self._log("Status refresh complete.")

    def _broadcast_with_confirm(self, name: str, action_fn):
        """Show Y/N confirmation, then execute the broadcast command."""
        if not self._confirm(f"Execute '{name}' on all active robots?"):
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
            self._log("Boot complete.")

        self.dispatcher.boot(targets, callback=on_done)

    def _do_shutdown(self):
        """Shutdown targets all non-offline robots."""
        active = [r for r in self.config.robots if r.status != RobotStatus.OFFLINE]
        if not active:
            self._log("No active robots to shut down.")
            return
        if not self._confirm(f"Shutdown Docker on {len(active)} workstation(s)?"):
            self._log("Shutdown cancelled.")
            return

        self._busy = True
        self._log(f"Shutting down {len(active)} workstation(s)...")

        def on_done(results=None):
            with self._lock:
                self._busy = False
            if results:
                for rid, out in results.items():
                    self._log(f"  WS-{rid}: {str(out)[:80]}")
            self._log("Shutdown complete.")

        self.dispatcher.shutdown(active, callback=on_done)

    def _do_connect_to_server(self):
        if not self._confirm("Connect all booted robots to server?"):
            self._log("Connect to Server cancelled.")
            return
        booted = [r for r in self.config.robots if r.status == RobotStatus.BOOTED]
        if not booted:
            self._log("No booted robots to connect.")
            return
        self.dispatcher.connect_to_server(booted)
        self._log(f"Connected {len(booted)} robot(s) to server (simulated).")

    # ── confirmation dialog ─────────────────────────────────────

    def _confirm(self, message: str) -> bool:
        """Show a blocking Y/N confirmation prompt."""
        stdscr = self._stdscr
        h, w = stdscr.getmaxyx()
        box_w = min(len(message) + 6, w - 4)
        box_h = 5
        start_y = h // 2 - box_h // 2
        start_x = w // 2 - box_w // 2

        win = curses.newwin(box_h, box_w, start_y, start_x)
        win.bkgd(" ", curses.color_pair(PAIR_HEADER))
        win.border()
        win.addstr(1, 2, message[: box_w - 4])
        win.addstr(3, 2, "[Y] Yes   [N] No")
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
        if h < 12 or w < 50:
            stdscr.clear()
            stdscr.addstr(0, 0, "Terminal too small. Resize to at least 50x12.")
            stdscr.refresh()
            return

        stdscr.erase()

        side_w = max(24, w // 3)
        main_w = w - side_w
        log_h = max(4, h // 4)
        panel_h = h - log_h - 2  # -2 for header + separator

        # Header bar
        title = " ARMory Dashboard "
        stdscr.attron(curses.color_pair(PAIR_HEADER) | curses.A_BOLD)
        stdscr.addstr(0, 0, " " * (w - 1))
        stdscr.addstr(0, max(0, w // 2 - len(title) // 2), title)
        if self._busy:
            stdscr.addstr(0, w - 14, " ⟳ WORKING  ")
        stdscr.attroff(curses.color_pair(PAIR_HEADER) | curses.A_BOLD)

        # Side panel — robot status
        self._draw_side_panel(1, 0, panel_h, side_w)

        # Main panel — commands
        self._draw_main_panel(1, side_w, panel_h, main_w)

        # Separator
        sep_y = 1 + panel_h
        stdscr.attron(curses.color_pair(PAIR_BORDER))
        stdscr.addstr(sep_y, 0, "─" * (w - 1))
        stdscr.attroff(curses.color_pair(PAIR_BORDER))

        # Log panel
        self._draw_log_panel(sep_y + 1, 0, log_h, w)

        stdscr.refresh()

    def _draw_side_panel(self, y: int, x: int, h: int, w: int):
        stdscr = self._stdscr
        stdscr.attron(curses.color_pair(PAIR_BORDER) | curses.A_BOLD)
        stdscr.addstr(y, x + 1, "Robot Status")
        stdscr.attroff(curses.color_pair(PAIR_BORDER) | curses.A_BOLD)

        # Vertical separator
        for row in range(y, y + h):
            try:
                stdscr.addch(row, w - 1, "│", curses.color_pair(PAIR_BORDER))
            except curses.error:
                pass

        row = y + 2
        for robot in self.config.robots:
            if row >= y + h - 1:
                break
            status = robot.status
            if status == RobotStatus.ONLINE:
                pair = PAIR_ONLINE
            elif status == RobotStatus.BOOTED:
                pair = PAIR_BOOTED
            else:
                pair = PAIR_OFFLINE

            label = f" {robot.id}. {robot.name}"
            status_str = f"    ({status.value})"
            try:
                stdscr.addstr(row, x + 1, label[: w - 3], curses.A_BOLD)
                row += 1
                stdscr.addstr(row, x + 1, status_str[: w - 3], curses.color_pair(pair))
                row += 2
            except curses.error:
                break

    def _draw_main_panel(self, y: int, x: int, h: int, w: int):
        stdscr = self._stdscr
        stdscr.attron(curses.color_pair(PAIR_BORDER) | curses.A_BOLD)
        stdscr.addstr(y, x + 2, "Commands")
        stdscr.attroff(curses.color_pair(PAIR_BORDER) | curses.A_BOLD)

        row = y + 2
        for key, label in COMMANDS:
            if row >= y + h - 1:
                break
            try:
                stdscr.addstr(row, x + 3, f"[", curses.color_pair(PAIR_CMD))
                stdscr.addstr(f"{key}", curses.color_pair(PAIR_HIGHLIGHT) | curses.A_BOLD)
                stdscr.addstr(f"] {label}", curses.color_pair(PAIR_CMD))
            except curses.error:
                pass
            row += 2

    def _draw_log_panel(self, y: int, x: int, h: int, w: int):
        stdscr = self._stdscr
        stdscr.attron(curses.color_pair(PAIR_BORDER) | curses.A_BOLD)
        try:
            stdscr.addstr(y, x + 1, "Log")
        except curses.error:
            pass
        stdscr.attroff(curses.color_pair(PAIR_BORDER) | curses.A_BOLD)

        # Show most recent log lines that fit
        visible = list(self.log_lines)[-max(0, h - 1) :]
        row = y + 1
        for line in visible:
            if row >= y + h:
                break
            try:
                stdscr.addstr(row, x + 2, line[: w - 4], curses.color_pair(PAIR_LOG))
            except curses.error:
                pass
            row += 1

    # ── helpers ─────────────────────────────────────────────────

    def _log(self, msg: str):
        ts = time.strftime("%H:%M:%S")
        self.log_lines.append(f"[{ts}] {msg}")
        self.ssh.system_logger.info(msg)

    @staticmethod
    def _init_colors():
        curses.start_color()
        curses.use_default_colors()
        curses.init_pair(PAIR_HEADER, curses.COLOR_BLACK, curses.COLOR_CYAN)
        curses.init_pair(PAIR_OFFLINE, curses.COLOR_RED, -1)
        curses.init_pair(PAIR_BOOTED, curses.COLOR_YELLOW, -1)
        curses.init_pair(PAIR_ONLINE, curses.COLOR_GREEN, -1)
        curses.init_pair(PAIR_BORDER, curses.COLOR_CYAN, -1)
        curses.init_pair(PAIR_CMD, curses.COLOR_WHITE, -1)
        curses.init_pair(PAIR_LOG, curses.COLOR_WHITE, -1)
        curses.init_pair(PAIR_HIGHLIGHT, curses.COLOR_CYAN, -1)
