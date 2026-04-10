#!/usr/bin/env python3
"""ARMory — Advanced Robotic Manipulation CLI tool."""

import os
import re
import shutil
import signal
import sys

# Allow running as `python -m src.main` or `python src/main.py`
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.commands.dispatcher import CommandDispatcher
from src.core.config import Config
from src.core.ssh_client import SSHManager
from src.ui.dashboard import Dashboard

# ── ASCII splash ────────────────────────────────────────────────

ESC = chr(27)
RESET = f"{ESC}[0m"
BOLD = f"{ESC}[1m"
PASTEL_PINK = f"{ESC}[38;5;225m"
PASTEL_LAVENDER = f"{ESC}[38;5;183m"
PASTEL_SKY = f"{ESC}[38;5;153m"
MUTED = f"{ESC}[38;5;250m"

ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
BANNER_WIDTH = 54


def _visible_len(text: str) -> int:
    return len(ANSI_RE.sub("", text))


def _center_line(text: str, width: int) -> str:
    return " " * max(0, (width - _visible_len(text)) // 2) + text


def _banner_line(text: str) -> str:
    inner_width = BANNER_WIDTH - 2
    padding = max(0, inner_width - _visible_len(text))
    left = padding // 2
    right = padding - left
    return f"{PASTEL_LAVENDER}│{RESET}{' ' * left}{text}{' ' * right}{PASTEL_LAVENDER}│{RESET}"


def _splash_lines() -> list[str]:
    brand = f"{PASTEL_PINK}{BOLD}ARM{RESET}{PASTEL_SKY}{BOLD}ory{RESET}"
    subtitle = f"{MUTED}Advanced Robotic Manipulation{RESET}"
    console = f"{MUTED}fleet console{RESET}"
    return [
        f"{PASTEL_LAVENDER}╭{'─' * (BANNER_WIDTH - 2)}╮{RESET}",
        _banner_line(brand),
        _banner_line(subtitle),
        _banner_line(console),
        f"{PASTEL_LAVENDER}╰{'─' * (BANNER_WIDTH - 2)}╯{RESET}",
    ]


def bootup():
    """Display the ASCII splash screen and welcome message."""
    print(f"{ESC}[2J{ESC}[H", end="")

    try:
        user = os.getlogin()
    except OSError:
        user = os.environ.get("USER", "operator")

    cols, rows = shutil.get_terminal_size(fallback=(100, 30))
    splash = _splash_lines()
    welcome = f"{PASTEL_SKY}Welcome, {BOLD}{user}{RESET}"
    prompt = f"{MUTED}Press Enter to proceed{RESET}"

    block_height = len(splash) + 4
    top_padding = max(0, (rows - block_height) // 2)
    print("\n" * top_padding, end="")

    for line in splash:
        print(_center_line(line, cols))
    print()
    print(_center_line(welcome, cols))
    print()
    input(_center_line(prompt, cols))


def main():
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "config.yaml")
    config = Config(config_path)

    dashboard = Dashboard(config, ssh=None, dispatcher=None)
    ssh = SSHManager(config, on_log=dashboard._log)
    ssh.start()

    dispatcher = CommandDispatcher(ssh)
    dashboard.ssh = ssh
    dashboard.dispatcher = dispatcher

    # Graceful shutdown on SIGINT
    original_sigint = signal.getsignal(signal.SIGINT)

    def _shutdown(sig, frame):
        ssh.stop()
        signal.signal(signal.SIGINT, original_sigint)
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)

    try:
        dashboard.run()
    finally:
        ssh.stop()


if __name__ == "__main__":
    bootup()
    main()
