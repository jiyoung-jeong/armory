"""Centralized logging configuration for openpi."""

import logging
import logging.handlers
import multiprocessing as mp
from pathlib import Path

from rich.logging import RichHandler


def _make_handlers(log_path: Path | None) -> list[logging.Handler]:
    console = RichHandler(rich_tracebacks=True, show_time=True, show_path=False)
    console.setFormatter(logging.Formatter("[%(processName)s] %(message)s", datefmt="[%X]"))

    handlers: list[logging.Handler] = [console]
    if log_path is not None:
        fh = logging.FileHandler(log_path)
        fh.setFormatter(logging.Formatter("%(asctime)s [%(processName)s] %(levelname)s %(name)s: %(message)s"))
        handlers.append(fh)
    return handlers


def setup_logging(
    level: int = logging.INFO, log_path: Path | None = None
) -> tuple[mp.Queue, logging.handlers.QueueListener]:
    """Configure logging for the main process.

    Returns a (log_queue, listener) pair. Pass log_queue to worker processes via
    setup_worker_logging(). Keep a reference to listener for its lifetime; call
    listener.stop() on shutdown.
    """
    handlers = _make_handlers(log_path)
    logging.basicConfig(level=level, format="%(message)s", datefmt="[%X]", handlers=handlers, force=True)

    log_queue: mp.Queue = mp.Queue(-1)
    listener = logging.handlers.QueueListener(log_queue, *handlers, respect_handler_level=True)
    listener.start()
    return log_queue, listener


def setup_worker_logging(log_queue: mp.Queue, process_name: str, level: int = logging.INFO) -> None:
    """Configure a worker subprocess to forward all log records to the main process queue."""
    mp.current_process().name = process_name
    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()
    root.addHandler(logging.handlers.QueueHandler(log_queue))
