"""Shared Rich Live display + helpers for the interactive sweep scripts.

Two visual roles share one layout: SERVER tails ``server.stdout.log`` and CLIENT
tails ``client.stdout.log``. The non-display fallback (``PrintDisplay``) has the
same API and is used under ``--no-display`` or when stdout is not a TTY.
"""

from __future__ import annotations

import contextlib
import json
import pathlib
import signal
import sys
import threading
import time
from collections.abc import Iterator
from typing import Any

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table
from rich.text import Text


def _tail(path: pathlib.Path, n: int) -> list[str]:
    if not path.exists():
        return []
    try:
        with path.open("rb") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - 4096))
            data = f.read()
    except OSError:
        return []
    return data.decode("utf-8", errors="replace").splitlines()[-n:]


def _classify(case_dir: pathlib.Path) -> str:
    r = case_dir / "result.json"
    if not r.exists():
        return "missing"
    try:
        return "ok" if json.loads(r.read_text()).get("status") == "ok" else "failed"
    except (OSError, json.JSONDecodeError):
        return "failed"


def _format_elapsed(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _case_dirs_from_cases(run_root: pathlib.Path, cases: list[Any]) -> list[pathlib.Path]:
    return [run_root / c.run_id for c in cases]


@contextlib.contextmanager
def install_signal_handlers() -> Iterator[threading.Event]:
    stopping = threading.Event()

    def handler(signum, frame):  # noqa: ARG001
        stopping.set()

    old_int = signal.signal(signal.SIGINT, handler)
    old_term = signal.signal(signal.SIGTERM, handler)
    try:
        yield stopping
    finally:
        signal.signal(signal.SIGINT, old_int)
        signal.signal(signal.SIGTERM, old_term)


class BaseLiveDisplay:
    """Rich Live layout shared by server and client.

    Subclasses set ``role`` and ``log_filename``. Event-driven mutators
    (``set_current``, ``set_status``) come from the main loop; the refresh
    thread polls log tails + totals every ``refresh_interval`` seconds.
    """

    role: str = "SWEEP"
    log_filename: str = "server.stdout.log"

    def __init__(
        self,
        run_root: pathlib.Path,
        cases: list[Any],
        *,
        host: str = "",
        port: int = 0,
        log_tail_lines: int = 10,
        refresh_interval: float = 0.5,
    ) -> None:
        self.run_root = run_root
        self.cases = cases
        self.case_dirs = _case_dirs_from_cases(run_root, cases)
        self.host = host
        self.port = port
        self.log_tail_lines = log_tail_lines
        self.refresh_interval = refresh_interval

        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._live: Live | None = None
        self._console = Console()
        self._start_time = time.monotonic()

        self._current_idx: int | None = None
        self._current_case_dir: pathlib.Path | None = None
        self._status_msg: str = "starting"

    # ------------------------------------------------------------------ enter/exit

    def __enter__(self) -> BaseLiveDisplay:
        self._live = Live(
            self._render(),
            console=self._console,
            refresh_per_second=4,
            screen=False,
            redirect_stdout=False,
            redirect_stderr=False,
        )
        self._live.__enter__()
        self._thread = threading.Thread(target=self._refresh_loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._live is not None:
            try:
                self._live.update(self._render())
            except Exception:  # noqa: BLE001
                pass
            self._live.__exit__(exc_type, exc, tb)

    # ------------------------------------------------------------------ event API

    def set_current(
        self,
        idx: int,
        case_dir: pathlib.Path,
        *,
        case_budget_s: float | None = None,  # noqa: ARG002
    ) -> None:
        with self._lock:
            self._current_idx = idx
            self._current_case_dir = case_dir
            self._status_msg = "starting case"
        self._push()

    def set_status(self, msg: str) -> None:
        with self._lock:
            self._status_msg = msg
        self._push()

    # ------------------------------------------------------------------ refresh thread

    def _refresh_loop(self) -> None:
        while not self._stop.wait(self.refresh_interval):
            self._push()

    def _push(self) -> None:
        if self._live is None:
            return
        try:
            self._live.update(self._render())
        except Exception:  # noqa: BLE001
            pass

    # ------------------------------------------------------------------ render

    def _render(self) -> Layout:
        with self._lock:
            idx = self._current_idx
            case_dir = self._current_case_dir
            status_msg = self._status_msg

        layout = Layout()
        layout.split_column(
            Layout(name="header", size=5),
            Layout(name="params", size=10),
            Layout(name="status", size=3),
            Layout(name="log", ratio=1),
            Layout(name="totals", size=3),
        )
        layout["header"].update(self._render_header(idx, case_dir))
        layout["params"].update(self._render_params(case_dir, idx))
        layout["status"].update(self._render_status(status_msg))
        layout["log"].update(self._render_log(case_dir))
        layout["totals"].update(self._render_totals())
        return layout

    def _render_header(self, idx: int | None, case_dir: pathlib.Path | None) -> Panel:
        stamp = self.run_root.name
        n_total = len(self.cases)
        position = f"[{(idx or 0) + 1} / {n_total}]" if idx is not None else f"[- / {n_total}]"
        case_label = case_dir.name if case_dir is not None else "(none)"
        host_port = f"{self.host}:{self.port}" if self.port else "(unset)"
        body = Text.from_markup(
            f"[bold]run_root[/bold] {self.run_root}\n"
            f"[bold]stamp[/bold]    {stamp}    [bold]host[/bold] {host_port}\n"
            f"[bold]case[/bold]     {position}  {case_label}"
        )
        return Panel(
            body,
            title=f"Armory Interactive Sweep — {self.role}",
            border_style="cyan",
        )

    def _render_params(self, case_dir: pathlib.Path | None, idx: int | None) -> Panel:
        table = Table.grid(padding=(0, 2), expand=True)
        table.add_column(style="bold", justify="right", ratio=1)
        table.add_column(ratio=4)
        if case_dir is None or idx is None or idx >= len(self.cases):
            table.add_row("(no case selected)", "")
        else:
            case = self.cases[idx]
            table.add_row("scheduler", str(case.scheduler))
            table.add_row("seed", str(case.seed))
            table.add_row("max_batch_size", str(case.max_batch_size))
            ahm = getattr(case, "action_horizon_multipliers", None) or {}
            table.add_row(
                "action_horizon_multipliers",
                ", ".join(f"{k}={v}" for k, v in ahm.items()) if ahm else "-",
            )
            table.add_row(
                "experiment",
                f"{case.experiment_name}   num_robots={case.num_robots}   "
                f"server_variant={case.server_variant or '-'}",
            )
        return Panel(table, title="Current Case", border_style="magenta")

    def _render_status(self, msg: str) -> Panel:
        return Panel(Text(msg, style="yellow"), title="Status", border_style="yellow")

    def _render_log(self, case_dir: pathlib.Path | None) -> Panel:
        title = f"Recent {self.log_filename} (tail {self.log_tail_lines})"
        if case_dir is None:
            return Panel(Text("(no case selected)", style="dim"), title=title, border_style="blue")
        log_path = case_dir / "logs" / self.log_filename
        lines = _tail(log_path, self.log_tail_lines)
        body = (
            Text("\n".join(lines)) if lines else Text("(empty)", style="dim")
        )
        return Panel(body, title=title, border_style="blue")

    def _render_totals(self) -> Panel:
        ok = failed = missing = 0
        for d in self.case_dirs:
            kind = _classify(d)
            if kind == "ok":
                ok += 1
            elif kind == "failed":
                failed += 1
            else:
                missing += 1
        elapsed = _format_elapsed(time.monotonic() - self._start_time)
        text = Text.from_markup(
            f"[green]ok {ok}[/green]   [red]failed {failed}[/red]   "
            f"[dim]pending {missing}[/dim]           elapsed {elapsed}"
        )
        return Panel(text, title="Totals", border_style="green")


class ServerLiveDisplay(BaseLiveDisplay):
    role = "SERVER"
    log_filename = "server.stdout.log"


class ClientLiveDisplay(BaseLiveDisplay):
    role = "CLIENT"
    log_filename = "client.stdout.log"


class PrintDisplay:
    """Plain-print fallback with the same API as ``BaseLiveDisplay``."""

    def __init__(
        self,
        run_root: pathlib.Path,
        cases: list[Any],
        *,
        role: str = "SWEEP",
        host: str = "",
        port: int = 0,
        log_tail_lines: int = 10,  # noqa: ARG002
        refresh_interval: float = 0.5,  # noqa: ARG002
    ) -> None:
        self.run_root = run_root
        self.cases = cases
        self.case_dirs = _case_dirs_from_cases(run_root, cases)
        self.role = role
        self.host = host
        self.port = port

    def __enter__(self) -> PrintDisplay:
        host_port = f"{self.host}:{self.port}" if self.port else "(unset)"
        print(
            f"[{self.role}] run_root={self.run_root}  host={host_port}  total_cases={len(self.cases)}",
            flush=True,
        )
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def set_current(
        self,
        idx: int,
        case_dir: pathlib.Path,
        *,
        case_budget_s: float | None = None,  # noqa: ARG002
    ) -> None:
        case = self.cases[idx]
        print(
            f"[{self.role}] case [{idx + 1}/{len(self.cases)}] {case.run_id}",
            flush=True,
        )

    def set_status(self, msg: str) -> None:
        print(f"[{self.role}] {msg}", flush=True)


def make_display(
    *,
    role: str,
    run_root: pathlib.Path,
    cases: list[Any],
    host: str = "",
    port: int = 0,
    log_tail_lines: int = 10,
    no_display: bool = False,
) -> BaseLiveDisplay | PrintDisplay:
    use_plain = no_display or not sys.stdout.isatty()
    if use_plain:
        return PrintDisplay(
            run_root, cases, role=role, host=host, port=port, log_tail_lines=log_tail_lines
        )
    cls = ServerLiveDisplay if role == "SERVER" else ClientLiveDisplay
    return cls(run_root, cases, host=host, port=port, log_tail_lines=log_tail_lines)


# ---------------------------------------------------------------- resume prompt


def resume_prompt(
    run_root: pathlib.Path,
    cases: list[Any],
    *,
    role: str,
    non_interactive: bool = False,
) -> list[pathlib.Path]:
    case_dirs = _case_dirs_from_cases(run_root, cases)
    buckets: dict[str, list[pathlib.Path]] = {"ok": [], "failed": [], "missing": []}
    for d in case_dirs:
        buckets[_classify(d)].append(d)

    if not buckets["ok"] and not buckets["failed"]:
        return case_dirs

    console = Console()
    table = Table(title=f"Existing case status under {run_root.name}", show_header=True)
    table.add_column("status", style="bold")
    table.add_column("count", justify="right")
    for k in ("ok", "failed", "missing"):
        style = {"ok": "green", "failed": "red", "missing": "dim"}[k]
        table.add_row(Text(k, style=style), str(len(buckets[k])))
    console.print(table)
    console.print(
        f"[dim]role={role}: type the same choice in both terminals so the selections match.[/dim]"
    )

    if non_interactive:
        choice = "resume"
        console.print(f"[dim]non-interactive: defaulting to '{choice}'[/dim]")
    else:
        choice = Prompt.ask(
            "Resume mode",
            choices=["resume", "all", "failed-only"],
            default="resume",
        )

    if choice == "all":
        return case_dirs
    # "resume" and "failed-only" both skip ok
    return buckets["failed"] + buckets["missing"]


# ---------------------------------------------------------------- final summary


def print_final_summary(run_root: pathlib.Path, cases: list[Any], *, role: str) -> None:
    console = Console()
    case_dirs = _case_dirs_from_cases(run_root, cases)
    table = Table(
        title=f"Final case status — {role} ({run_root})",
        show_header=True,
        header_style="bold",
    )
    table.add_column("run_id", overflow="fold")
    table.add_column("status")
    table.add_column("duration", justify="right")
    table.add_column("error", overflow="fold")

    ok = failed = missing = 0
    for case, case_dir in zip(cases, case_dirs):
        kind = _classify(case_dir)
        result_path = case_dir / "result.json"
        duration = ""
        error = ""
        if result_path.exists():
            try:
                row = json.loads(result_path.read_text())
                error = (row.get("error") or "")[:80]
            except (OSError, json.JSONDecodeError):
                pass
        done_path = case_dir / "client_done.json"
        if done_path.exists():
            try:
                done = json.loads(done_path.read_text())
                if "duration_sec" in done:
                    duration = _format_elapsed(done["duration_sec"])
            except (OSError, json.JSONDecodeError):
                pass

        if kind == "ok":
            ok += 1
            style = "green"
        elif kind == "failed":
            failed += 1
            style = "red"
        else:
            missing += 1
            style = "dim"
        table.add_row(case.run_id, Text(kind, style=style), duration, error)
    console.print(table)
    console.print(
        f"[bold]Summary[/bold]: [green]ok {ok}[/green]   [red]failed {failed}[/red]   "
        f"[dim]pending {missing}[/dim]   total {len(cases)}"
    )


# ---------------------------------------------------------------- smoke test


class _FakeCase:
    def __init__(self, run_id: str, scheduler: str, seed: int, mbs: int, alpha: float) -> None:
        self.run_id = run_id
        self.scheduler = scheduler
        self.seed = seed
        self.max_batch_size = mbs
        self.alpha = alpha
        self.experiment_name = "smoke"
        self.num_robots = 1
        self.server_variant = ""


def _smoke_main() -> None:
    """Tiny visual smoke test: ``python scripts/interactive/_display.py``."""
    run_root = pathlib.Path("/tmp/armory_display_smoke")
    run_root.mkdir(parents=True, exist_ok=True)
    cases = [
        _FakeCase(f"scheduler=greedy-deadline__seed=7__mbs={i}", "greedy-deadline", 7, i, 1.0)
        for i in (1, 2, 4)
    ]
    for c in cases:
        d = run_root / c.run_id
        (d / "logs").mkdir(parents=True, exist_ok=True)
        (d / "logs" / "server.stdout.log").write_text(
            "\n".join(f"line {i}" for i in range(20)) + "\n"
        )

    role = sys.argv[1] if len(sys.argv) > 1 else "SERVER"
    with make_display(role=role, run_root=run_root, cases=cases, host="ice-1", port=8451) as d:
        for i, c in enumerate(cases):
            d.set_current(i, run_root / c.run_id)
            for s in ("starting server", "waiting for /metadata (5s)", "running run_libero"):
                d.set_status(s)
                time.sleep(0.4)
            (run_root / c.run_id / "result.json").write_text(
                json.dumps({"status": "ok"}) + "\n"
            )
            time.sleep(0.3)
    print_final_summary(run_root, cases, role=role)


if __name__ == "__main__":
    _smoke_main()
