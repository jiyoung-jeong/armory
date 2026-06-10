"""Interactive sweep driver: client side.

Run this in the second terminal of an interactive allocation. It iterates the
case grid that ``run_sweep_server.py`` already materialized, and for each case:

  1. Waits for the server to write ``case_dir/server_ready.json``.
  2. Runs ``scripts/run_libero.py --json-path case_dir/client_args.json`` to
     completion in the foreground.
  3. Writes ``case_dir/result.json`` via ``collect_results.write_case_result``
     (same schema the Slurm flow produces).
  4. Writes ``case_dir/client_done.json`` so the server can move on.

If the server writes ``run_root/sweep_stopped`` (e.g. on Ctrl-C), the client
exits cleanly between cases.

Running multiple pairs at once: each pair just needs its own ``--run-root``
(the server picks a free port per sweep, so two servers can coexist on the
same node). Use ``--stamp pair_a_<id>`` on the server side to make the run
dirs visually distinct.
"""

# pyright: reportMissingImports=false
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import pathlib
import subprocess
import sys
import threading
import time
from typing import Any

_HERE = pathlib.Path(__file__).resolve().parent
SCRIPTS_DIR = _HERE.parent
REPO_ROOT = SCRIPTS_DIR.parent
sys.path.insert(0, str(SCRIPTS_DIR / "sbatch"))
sys.path.insert(0, str(SCRIPTS_DIR / "modal"))
sys.path.insert(0, str(SCRIPTS_DIR / "visualization"))
sys.path.insert(0, str(_HERE))

from _display import (  # noqa: E402
    install_signal_handlers,
    make_display,
    print_final_summary,
    resume_prompt,
)
from collect_results import write_case_result  # noqa: E402


def _utc_now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")  # noqa: UP017


def _write_json(path: pathlib.Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")


class _ClientCase:
    def __init__(self, case_json: dict[str, Any]) -> None:
        self.run_id: str = case_json["run_id"]
        self.scheduler: str = case_json.get("scheduler", "")
        self.experiment_name: str = case_json.get("experiment", "")
        self.num_robots: int = int(case_json.get("num_robots", 0) or 0)
        self.seed: int = int(case_json.get("seed", 0) or 0)
        self.max_batch_size: int = int(case_json.get("max_batch_size", 0) or 0)
        self.alpha: float = float(case_json.get("alpha", 0.0) or 0.0)
        self.server_variant: str = case_json.get("server_variant", "") or ""
        self.action_horizon_multipliers: dict[str, Any] = (
            case_json.get("action_horizon_multipliers") or {}
        )


def _load_run(run_root: pathlib.Path) -> list[_ClientCase]:
    if not run_root.is_dir():
        raise SystemExit(f"--run-root is not a directory: {run_root}")
    manifest = run_root / "manifest.json"
    if manifest.exists():
        order = json.loads(manifest.read_text()).get("run_ids", [])
    else:
        order = [p.parent.name for p in sorted(run_root.glob("*/case.json"))]
    cases: list[_ClientCase] = []
    for run_id in order:
        case_json_path = run_root / run_id / "case.json"
        if not case_json_path.exists():
            continue
        cases.append(_ClientCase(json.loads(case_json_path.read_text())))
    if not cases:
        raise SystemExit(f"No case.json files under {run_root}")
    return cases


def _wait_for_server_ready(
    case_dir: pathlib.Path,
    *,
    run_root: pathlib.Path,
    timeout: float,
    display: Any,
    stopping: threading.Event,
) -> bool:
    """Return True when ``server_ready.json`` appears; False if we should bail
    (timeout, stop signal, or ``sweep_stopped``)."""
    target = case_dir / "server_ready.json"
    start = time.monotonic()
    last_status = 0.0
    while not target.exists():
        if stopping.is_set():
            return False
        if (run_root / "sweep_stopped").exists():
            return False
        elapsed = time.monotonic() - start
        if elapsed >= timeout:
            return False
        if time.monotonic() - last_status > 5:
            display.set_status(f"waiting for server_ready ({int(elapsed)}s)")
            last_status = time.monotonic()
        time.sleep(0.1)
    return True


def _run_one_case_client(
    case_dir: pathlib.Path,
    *,
    run_root: pathlib.Path,
    args: argparse.Namespace,
    display: Any,
    stopping: threading.Event,
) -> None:
    (case_dir / "client_done.json").unlink(missing_ok=True)
    logs = case_dir / "logs"
    logs.mkdir(parents=True, exist_ok=True)

    ok = _wait_for_server_ready(
        case_dir,
        run_root=run_root,
        timeout=args.server_ready_timeout,
        display=display,
        stopping=stopping,
    )
    if not ok:
        if stopping.is_set() or (run_root / "sweep_stopped").exists():
            return
        display.set_status("timed out waiting for server_ready.json")
        write_case_result(
            case_dir, status="failed", error="server_ready.json never appeared"
        )
        _write_json(
            case_dir / "client_done.json",
            {"exit_code": -1, "finished_at": _utc_now_iso(), "duration_sec": 0},
        )
        return

    display.set_status("running run_libero.py")
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    stdout_path = logs / "client.stdout.log"
    stderr_path = logs / "client.stderr.log"
    t0 = time.monotonic()
    with stdout_path.open("ab") as out, stderr_path.open("ab") as err:
        out.write(f"\n----- {_utc_now_iso()} starting client -----\n".encode())
        out.flush()
        err.write(f"\n----- {_utc_now_iso()} starting client -----\n".encode())
        err.flush()
        proc = subprocess.Popen(
            [
                "uv",
                "run",
                "python",
                "scripts/run_libero.py",
                "--json-path",
                str(case_dir / "client_args.json"),
            ],
            cwd=REPO_ROOT,
            stdout=out,
            stderr=err,
            env=env,
            start_new_session=True,
        )
        try:
            while True:
                try:
                    rc = proc.wait(timeout=1.0)
                    break
                except subprocess.TimeoutExpired:
                    if stopping.is_set():
                        display.set_status("stopping client")
                        proc.terminate()
                        try:
                            rc = proc.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            proc.kill()
                            rc = proc.wait(timeout=5)
                        break
        except Exception:  # noqa: BLE001
            proc.kill()
            rc = proc.wait(timeout=5)
    duration = int(time.monotonic() - t0)

    if stopping.is_set():
        # Don't record a result on operator-initiated stop mid-case.
        return

    status = "ok" if rc == 0 else "failed"
    error = "" if rc == 0 else f"run_libero exited rc={rc}"
    write_case_result(case_dir, status=status, error=error)
    _write_json(
        case_dir / "client_done.json",
        {
            "exit_code": rc,
            "finished_at": _utc_now_iso(),
            "duration_sec": duration,
        },
    )
    display.set_status(f"case finished status={status} in {duration}s")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--server-ready-timeout", type=float, default=900.0)
    parser.add_argument("--log-tail-lines", type=int, default=10)
    parser.add_argument("--no-display", action="store_true")
    parser.add_argument(
        "--non-interactive",
        action="store_true",
        help="Skip the resume prompt and default to 'resume' (skip cases with status=ok).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_root = pathlib.Path(args.run_root).resolve()
    cases = _load_run(run_root)
    print(f"Loaded {len(cases)} case(s) from {run_root}")

    port_json = run_root / "port.json"
    if port_json.exists():
        try:
            payload = json.loads(port_json.read_text())
            print(f"Server target: {payload.get('host')}:{payload.get('port')}")
        except (OSError, json.JSONDecodeError):
            pass
    else:
        print(
            "warning: run_root/port.json not present yet — has the server script started?"
        )

    selection = resume_prompt(
        run_root, cases, role="CLIENT", non_interactive=args.non_interactive
    )
    if not selection:
        print("No cases to run; exiting.")
        return

    with make_display(
        role="CLIENT",
        run_root=run_root,
        cases=cases,
        host="",
        port=0,
        log_tail_lines=args.log_tail_lines,
        no_display=args.no_display,
    ) as display:
        with install_signal_handlers() as stopping:
            for case_dir in selection:
                if stopping.is_set():
                    break
                if (run_root / "sweep_stopped").exists():
                    display.set_status("server signalled sweep_stopped; exiting")
                    break
                idx = next(
                    (i for i, c in enumerate(cases) if c.run_id == case_dir.name),
                    None,
                )
                if idx is None:
                    continue
                display.set_current(idx, case_dir)
                try:
                    _run_one_case_client(
                        case_dir,
                        run_root=run_root,
                        args=args,
                        display=display,
                        stopping=stopping,
                    )
                except KeyboardInterrupt:
                    break

    print_final_summary(run_root, cases, role="CLIENT")


if __name__ == "__main__":
    main()
