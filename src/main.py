#!/usr/bin/env python3
"""ARMory — Advanced Robotic Manipulation CLI tool."""

import os
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
CYAN = f"{ESC}[38;5;51m"
GRAY = f"{ESC}[38;5;245m"
RESET = f"{ESC}[0m"

LOGO_ART = """
     █████╗ ██████╗ ███╗   ███╗ ██████╗ ██████╗ ██╗   ██╗
    ██╔══██╗██╔══██╗████╗ ████║██╔═══██╗██╔══██╗╚██╗ ██╔╝
    ███████║██████╔╝██╔████╔██║██║   ██║██████╔╝ ╚████╔╝
    ██╔══██║██╔══██╗██║╚██╔╝██║██║   ██║██╔══██╗  ╚██╔╝
    ██║  ██║██║  ██║██║ ╚═╝ ██║╚██████╔╝██║  ██║   ██║
    ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝     ╚═╝ ╚═════╝ ╚═╝  ╚═╝   ╚═╝
"""

LOGO_SUB = """    ═══════════════════════════════════════════════════
                Advanced  Robotic  Manipulation
    ═══════════════════════════════════════════════════"""

LOGO = f"{CYAN}{LOGO_ART}{RESET}\n{GRAY}{LOGO_SUB}{RESET}"


def bootup():
    """Display the ASCII splash screen and welcome message."""
    os.system("clear")
    print(LOGO)

    try:
        user = os.getlogin()
    except OSError:
        user = os.environ.get("USER", "operator")

    print(f"    \033[38;5;51mWelcome, \033[1m{user}\033[0m")
    print()
    input("    \033[38;5;245mPress Enter to proceed...\033[0m")


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
