"""Interactive sweep driver: server side.

Run this in one terminal of an interactive allocation (e.g. ``salloc`` with one
L40S + one V100). It materializes the same case grid as
``scripts/sbatch/launch_sweep.py`` and then, for each case, launches
``scripts/serve.py`` to completion-of-the-client. Sync with the client side
(``run_sweep_client.py``) happens via filesystem sentinels inside each case_dir.

Two modes:

  Fresh sweep — pass the full launch_sweep arg set. The script materializes a
  timestamped ``<output_dir>/<stamp>/`` run_root with case_dirs and a
  ``manifest.json``, then iterates.

  Resume — pass ``--run-root <path>``. The script reads the existing
  ``manifest.json``, re-patches host/port into each case's JSONs, and iterates.

Note: case-generation utilities are imported from ``launch_sweep.py`` directly
to keep the case grid DRY. A future refactor can pull them into a shared
``sweep_common`` module.

Running multiple pairs at once: each invocation picks its own free port and
writes to its own ``<output_dir>/<stamp>/`` run_root, so two pairs can coexist
on the same node. Distinguish them with ``--stamp pair_a_<id>`` (or different
``--output-dir`` roots).

Optimization: the script keeps ``serve.py`` running across consecutive cases
that share the same ``server_args`` (modulo ``log_dir``). Server stdout/stderr
go to a shared ``run_root/logs/server-group-NN.{stdout,stderr}.log`` and each
case_dir gets a symlink to that file. The client's existing ``POST /reset``
call resets per-case metrics, so this reuse is transparent to result.json.
"""

# pyright: reportMissingImports=false
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import pathlib
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from typing import Any

_HERE = pathlib.Path(__file__).resolve().parent
SCRIPTS_DIR = _HERE.parent
REPO_ROOT = SCRIPTS_DIR.parent
sys.path.insert(0, str(SCRIPTS_DIR / "sbatch"))
sys.path.insert(0, str(SCRIPTS_DIR / "modal"))
sys.path.insert(0, str(_HERE))

from _display import (  # noqa: E402
    install_signal_handlers,
    make_display,
    print_final_summary,
    resume_prompt,
)
from launch_sweep import (  # noqa: E402
    Case,
    _client_config_paths,
    _experiment_name,
    _make_cases,
    _materialize_case,
    _read_experiment_config,
    _resolve_path,
    _server_config_paths,
    _server_variant_name,
    parse_list_args,
)

# ---------------------------------------------------------------- helpers


def _utc_now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")  # noqa: UP017


def _write_json(path: pathlib.Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")


def _patch_json(path: pathlib.Path, overrides: dict[str, Any]) -> None:
    data = json.loads(path.read_text())
    data.update(overrides)
    path.write_text(json.dumps(data, indent=2) + "\n")


def _pick_free_port(lo: int = 8000, hi: int = 9000) -> int:
    """Pick a free TCP port. Tries random ports in [lo, hi] first; falls back to
    asking the kernel for any ephemeral free port."""
    import random

    ports = list(range(lo, hi + 1))
    random.shuffle(ports)
    for port in ports:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("", port))
                return port
        except OSError:
            continue
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


# ---------------------------------------------------------------- case I/O


class _ResumeCase:
    """Lightweight Case stand-in built from a materialized ``case.json``."""

    def __init__(self, case_json: dict[str, Any]) -> None:
        self.run_id: str = case_json["run_id"]
        self.scheduler: str = case_json.get("scheduler", "")
        self.experiment_name: str = case_json.get("experiment", "")
        self.num_robots: int = int(case_json.get("num_robots", 0) or 0)
        self.seed: int = int(case_json.get("seed", 0) or 0)
        self.max_batch_size: int = int(case_json.get("max_batch_size", 0) or 0)
        self.alpha: float = float(case_json.get("alpha", 0.0) or 0.0)
        self.server_variant: str = case_json.get("server_variant", "") or ""
        self.stamp: str = case_json.get("stamp", "")
        self.action_horizon_multipliers: dict[str, Any] = (
            case_json.get("action_horizon_multipliers") or {}
        )


def _load_existing_run(run_root: pathlib.Path) -> tuple[pathlib.Path, list[_ResumeCase]]:
    if not run_root.is_dir():
        raise SystemExit(f"--run-root is not a directory: {run_root}")
    manifest = run_root / "manifest.json"
    if manifest.exists():
        order = json.loads(manifest.read_text()).get("run_ids", [])
    else:
        order = [p.parent.name for p in sorted(run_root.glob("*/case.json"))]
    cases: list[_ResumeCase] = []
    for run_id in order:
        case_json = run_root / run_id / "case.json"
        if not case_json.exists():
            continue
        cases.append(_ResumeCase(json.loads(case_json.read_text())))
    if not cases:
        raise SystemExit(f"No case.json files under {run_root}")
    return run_root, cases


def _materialize_new_run(
    args: argparse.Namespace,
) -> tuple[pathlib.Path, list[Case]]:
    stamp = args.stamp or dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d_%H%M%S")  # noqa: UP017
    run_root = pathlib.Path(args.output_dir) / stamp
    run_root.mkdir(parents=True, exist_ok=True)

    server_paths = _server_config_paths(args.server_config)
    server_variants = [
        (_server_variant_name(path), json.loads(path.read_text())) for path in server_paths
    ]
    client_paths = _client_config_paths(args.client_config)
    resolved_client_config = _resolve_path(args.client_config)
    config_root = resolved_client_config if resolved_client_config.is_dir() else None
    experiment_configs = [
        (_experiment_name(path, root=config_root), _read_experiment_config(path))
        for path in client_paths
    ]
    if args.server_policy == "default":
        server_variants = [
            (variant, {**server_args, "policy": {"type": "default"}})
            for variant, server_args in server_variants
        ]

    client_args = {"experiment_config": "", "progress_type": "logging", "overwrite": True}
    cases = _make_cases(
        server_variants=server_variants,
        client_args=client_args,
        experiment_configs=experiment_configs,
        schedulers=parse_list_args(args.schedulers),
        seeds=parse_list_args(args.seeds, cast=int),
        max_batch_sizes=parse_list_args(args.max_batch_size, cast=int),
        alphas=parse_list_args(args.alpha, cast=float),
        stamp=stamp,
    )

    for case in cases:
        _materialize_case(case, run_root=run_root)

    _write_manifest(run_root, cases)
    return run_root, cases


def _write_manifest(run_root: pathlib.Path, cases: list[Any]) -> None:
    _write_json(
        run_root / "manifest.json",
        {
            "stamp": run_root.name,
            "written_at": _utc_now_iso(),
            "run_ids": [c.run_id for c in cases],
        },
    )


def _apply_host_port(run_root: pathlib.Path, cases: list[Any], host: str, port: int) -> None:
    for c in cases:
        case_dir = run_root / c.run_id
        _patch_json(case_dir / "server_args.json", {"port": port})
        _patch_json(case_dir / "client_args.json", {"host": host, "port": port})


def _write_port_json(run_root: pathlib.Path, host: str, port: int) -> None:
    _write_json(
        run_root / "port.json",
        {"host": host, "port": port, "started_at": _utc_now_iso(), "server_pid_marker": os.getpid()},
    )


def _write_stopped_sentinel(run_root: pathlib.Path, reason: str) -> None:
    path = run_root / "sweep_stopped"
    if path.exists():
        return
    path.write_text(json.dumps({"stopped_at": _utc_now_iso(), "reason": reason}) + "\n")


# ---------------------------------------------------------------- per-case loop


def _wait_for_metadata(
    *,
    host: str,
    port: int,
    timeout: float,
    proc: subprocess.Popen,
    display: Any,
    stopping: threading.Event,
) -> None:
    url = f"http://{host}:{port}/metadata"
    start = time.monotonic()
    last_status = 0.0
    while True:
        if stopping.is_set():
            raise KeyboardInterrupt("stopped while waiting for /metadata")
        rc = proc.poll()
        if rc is not None:
            raise RuntimeError(f"serve.py exited rc={rc} before /metadata responded")
        try:
            with urllib.request.urlopen(url, timeout=5) as resp:  # noqa: S310
                if 200 <= resp.status < 500:
                    return
        except (urllib.error.URLError, urllib.error.HTTPError, OSError, TimeoutError):
            pass
        elapsed = time.monotonic() - start
        if elapsed >= timeout:
            raise TimeoutError(f"/metadata not ready after {int(elapsed)}s")
        if time.monotonic() - last_status > 5:
            display.set_status(f"waiting for /metadata ({int(elapsed)}s)")
            last_status = time.monotonic()
        time.sleep(2.0)


def _wait_for_client_done(
    case_dir: pathlib.Path,
    *,
    proc: subprocess.Popen,
    display: Any,
    stopping: threading.Event,
) -> None:
    done = case_dir / "client_done.json"
    start = time.monotonic()
    last_status = 0.0
    while not done.exists():
        if stopping.is_set():
            raise KeyboardInterrupt("stopped while waiting for client_done")
        rc = proc.poll()
        if rc is not None:
            raise RuntimeError(f"serve.py exited rc={rc} while waiting for client")
        if time.monotonic() - last_status > 10:
            elapsed = int(time.monotonic() - start)
            display.set_status(f"client running ({elapsed}s elapsed)")
            last_status = time.monotonic()
        time.sleep(0.1)


def _terminate(proc: subprocess.Popen, *, grace: float) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=grace)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass


def _server_key(case_dir: pathlib.Path) -> str:
    """Hashable key identifying a server config. Two cases share a key iff
    their ``server_args.json`` differ only in fields that don't change server
    behavior (currently just ``log_dir``)."""
    data = json.loads((case_dir / "server_args.json").read_text())
    data.pop("log_dir", None)
    return json.dumps(data, sort_keys=True)


def _link_or_copy(src: pathlib.Path, dst: pathlib.Path) -> None:
    """Symlink dst -> src using a relative path. No-op on FS without symlinks."""
    if dst.is_symlink() or dst.exists():
        try:
            dst.unlink()
        except OSError:
            return
    try:
        dst.symlink_to(os.path.relpath(src, dst.parent))
    except OSError:
        # Filesystem without symlink support: leave the case_dir entry absent;
        # users can find the shared log at run_root/logs/server-group-NN.*.
        pass


class _ServerSession:
    """Manages a single ``serve.py`` process across multiple cases.

    The server is kept up across consecutive cases whose ``server_args.json``
    (sans ``log_dir``) is identical. The client's existing ``POST /reset`` call
    at the start of ``run_libero`` clears per-case state, so reuse is
    transparent to the result.
    """

    def __init__(self, *, args: argparse.Namespace, run_root: pathlib.Path) -> None:
        self.args = args
        self.run_root = run_root
        self.run_logs_dir = run_root / "logs"
        self.run_logs_dir.mkdir(parents=True, exist_ok=True)

        self.proc: subprocess.Popen | None = None
        self._current_key: str | None = None
        self._stdout_path: pathlib.Path | None = None
        self._stderr_path: pathlib.Path | None = None
        self._stdout_fh: Any = None
        self._stderr_fh: Any = None
        self.group_idx: int = 0
        self.servers_spawned: int = 0
        self.cases_run: int = 0
        self.cases_reused: int = 0

    # ------------------------------------------------------------------ public

    def run_case(
        self,
        case_dir: pathlib.Path,
        *,
        display: Any,
        stopping: threading.Event,
    ) -> None:
        reused = self._ensure_server_for(case_dir, display=display, stopping=stopping)
        if self._stdout_path and self._stderr_path:
            logs = case_dir / "logs"
            logs.mkdir(parents=True, exist_ok=True)
            _link_or_copy(self._stdout_path, logs / "server.stdout.log")
            _link_or_copy(self._stderr_path, logs / "server.stderr.log")

        server_ready = case_dir / "server_ready.json"
        server_ready.unlink(missing_ok=True)
        _write_json(
            server_ready,
            {
                "started_at": _utc_now_iso(),
                "pid": self.proc.pid if self.proc is not None else 0,
                "host": self.args.host,
                "port": self.args.port,
                "group_idx": self.group_idx,
                "reused": reused,
            },
        )
        try:
            display.set_status(
                f"server ready (group {self.group_idx}{', reused' if reused else ''})"
                "; waiting for client_done.json"
            )
            _wait_for_client_done(
                case_dir, proc=self.proc, display=display, stopping=stopping
            )
        finally:
            server_ready.unlink(missing_ok=True)
        self.cases_run += 1
        if reused:
            self.cases_reused += 1

    def close(self) -> None:
        if self.proc is not None:
            _terminate(self.proc, grace=self.args.shutdown_grace)
            self.proc = None
        self._close_handles()

    # ------------------------------------------------------------------ internal

    def _metadata_alive(self) -> bool:
        url = f"http://{self.args.host}:{self.args.port}/metadata"
        try:
            with urllib.request.urlopen(url, timeout=2) as resp:  # noqa: S310
                return 200 <= resp.status < 500
        except (urllib.error.URLError, urllib.error.HTTPError, OSError, TimeoutError):
            return False

    def _ensure_server_for(
        self,
        case_dir: pathlib.Path,
        *,
        display: Any,
        stopping: threading.Event,
    ) -> bool:
        """Start (or restart) the server if needed. Return True iff the existing
        process was reused."""
        key = _server_key(case_dir)
        proc_alive = self.proc is not None and self.proc.poll() is None
        if proc_alive and key == self._current_key and self._metadata_alive():
            return True

        if self.proc is not None:
            _terminate(self.proc, grace=self.args.shutdown_grace)
            self.proc = None
            self._close_handles()

        self.group_idx += 1
        self._stdout_path = self.run_logs_dir / f"server-group-{self.group_idx:02d}.stdout.log"
        self._stderr_path = self.run_logs_dir / f"server-group-{self.group_idx:02d}.stderr.log"
        self._stdout_fh = self._stdout_path.open("ab")
        self._stderr_fh = self._stderr_path.open("ab")
        banner = f"\n----- {_utc_now_iso()} starting server (group {self.group_idx}) -----\n".encode()
        self._stdout_fh.write(banner)
        self._stdout_fh.flush()
        self._stderr_fh.write(banner)
        self._stderr_fh.flush()

        display.set_status(f"starting serve.py (group {self.group_idx})")
        self.proc = subprocess.Popen(
            [
                "uv",
                "run",
                "python",
                "scripts/serve.py",
                "--json-path",
                str(case_dir / "server_args.json"),
            ],
            cwd=REPO_ROOT,
            stdout=self._stdout_fh,
            stderr=self._stderr_fh,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
            start_new_session=True,
        )
        self.servers_spawned += 1
        _wait_for_metadata(
            host=self.args.host,
            port=self.args.port,
            timeout=self.args.metadata_timeout,
            proc=self.proc,
            display=display,
            stopping=stopping,
        )
        self._current_key = key
        return False

    def _close_handles(self) -> None:
        for attr in ("_stdout_fh", "_stderr_fh"):
            fh = getattr(self, attr, None)
            if fh is not None:
                try:
                    fh.close()
                except OSError:
                    pass
                setattr(self, attr, None)


# ---------------------------------------------------------------- CLI


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--run-root",
        default="",
        help="Resume an existing materialized sweep dir. If set, the launch_sweep flags are ignored.",
    )
    parser.add_argument("--server-config", default="configs/server/mock.json")
    parser.add_argument("--client-config", default="")
    parser.add_argument(
        "--server-policy",
        choices=["default", "config"],
        default="default",
    )
    parser.add_argument("--output-dir", default="experiments/sweeps/interactive")
    parser.add_argument(
        "--schedulers",
        default="fixed-max-batch,greedy-deadline,round-robin,lookahead-actions,dynamic-action",
    )
    parser.add_argument("--seeds", default="7")
    parser.add_argument("--max-batch-size", default="")
    parser.add_argument("--alpha", default="")
    parser.add_argument("--stamp", default="")
    parser.add_argument(
        "--host",
        default="",
        help="Server host to advertise to the client. Default: socket.gethostname().",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=0,
        help="Server port. Default: pick a free port in 8000-9000.",
    )
    parser.add_argument("--metadata-timeout", type=float, default=900.0)
    parser.add_argument("--shutdown-grace", type=float, default=10.0)
    parser.add_argument("--log-tail-lines", type=int, default=10)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-display", action="store_true")
    parser.add_argument(
        "--non-interactive",
        action="store_true",
        help="Skip the resume prompt and default to 'resume' (skip cases with status=ok).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.run_root:
        run_root = pathlib.Path(args.run_root).resolve()
        run_root, cases = _load_existing_run(run_root)
        print(f"Resuming sweep from {run_root}  ({len(cases)} case(s))")
    else:
        if not args.client_config:
            raise SystemExit("--client-config is required (unless --run-root is set).")
        run_root, cases = _materialize_new_run(args)
        print(f"Materialized {len(cases)} case(s) under {run_root}")

    if not args.host:
        args.host = socket.gethostname()
    if not args.port:
        args.port = _pick_free_port()

    # Clear stale per-sweep sentinels from a previous run before announcing
    # this server's port, so the client doesn't see a leftover sweep_stopped
    # or pre-existing server_ready.json files.
    (run_root / "sweep_stopped").unlink(missing_ok=True)
    for c in cases:
        (run_root / c.run_id / "server_ready.json").unlink(missing_ok=True)

    _apply_host_port(run_root, cases, args.host, args.port)
    _write_port_json(run_root, args.host, args.port)

    print(f"Server host: {args.host}    port: {args.port}")
    print(f"Manifest:    {run_root / 'manifest.json'}")
    print()
    print("Run this in your second terminal:")
    print(
        f"  uv run python scripts/interactive/run_sweep_client.py --run-root {run_root}"
    )
    print()

    if args.dry_run:
        print("--dry-run set; not spawning serve.py. Exiting.")
        return

    selection = resume_prompt(
        run_root, cases, role="SERVER", non_interactive=args.non_interactive
    )
    if not selection:
        print("No cases to run; exiting.")
        return

    reason = "completed"
    session = _ServerSession(args=args, run_root=run_root)
    try:
        with make_display(
            role="SERVER",
            run_root=run_root,
            cases=cases,
            host=args.host,
            port=args.port,
            log_tail_lines=args.log_tail_lines,
            no_display=args.no_display,
        ) as display:
            with install_signal_handlers() as stopping:
                for case_dir in selection:
                    if stopping.is_set():
                        reason = "sigint"
                        break
                    idx = next(
                        (i for i, c in enumerate(cases) if c.run_id == case_dir.name),
                        None,
                    )
                    if idx is None:
                        continue
                    display.set_current(idx, case_dir)
                    try:
                        session.run_case(case_dir, display=display, stopping=stopping)
                    except KeyboardInterrupt:
                        reason = "sigint"
                        break
                    except Exception as exc:  # noqa: BLE001
                        display.set_status(f"case failed: {exc}")
                        time.sleep(1.0)
    finally:
        session.close()

    _write_stopped_sentinel(run_root, reason=reason)
    print_final_summary(run_root, cases, role="SERVER")
    if session.cases_run:
        print(
            f"Spawned {session.servers_spawned} server process(es) "
            f"across {session.cases_run} case(s) "
            f"({session.cases_reused} reused server)."
        )


if __name__ == "__main__":
    main()
