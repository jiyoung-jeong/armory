"""Run small server/client scheduler sweeps on Modal.

Example:
    modal run scripts/modal_sweep.py \
        --schedulers fixed-max-batch,greedy-deadline,round-robin \
        --num-robots 2,4,6 \
        --output-dir experiments/sweeps/mock
"""

from __future__ import annotations

import csv
import dataclasses
import datetime as dt
import io
import json
import pathlib
import shlex
import subprocess
import sys
import tarfile
import time
import urllib.request
from typing import Any

import modal

APP_NAME = "armory-scheduler-sweep"
REMOTE_ROOT = pathlib.Path("/root/codex")
REMOTE_OUTPUT_ROOT = pathlib.Path("/tmp/armory_sweep")
PYTHONPATH = ":".join(
    [
        str(REMOTE_ROOT / "src"),
        str(REMOTE_ROOT / "src/backends"),
        str(REMOTE_ROOT / "packages/armory-client/src"),
    ]
)


def _ignore_modal_copy(path: pathlib.Path) -> bool:
    parts = set(path.parts)
    return bool(parts & {".git", ".venv", ".ruff_cache", ".pytest_cache", "__pycache__"})


image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install_from_requirements("requirements-modal-mock.txt")
    .add_local_dir("src", str(REMOTE_ROOT / "src"), copy=True, ignore=_ignore_modal_copy)
    .add_local_dir("scripts", str(REMOTE_ROOT / "scripts"), copy=True, ignore=_ignore_modal_copy)
    .add_local_dir("packages", str(REMOTE_ROOT / "packages"), copy=True, ignore=_ignore_modal_copy)
    .add_local_dir("configs", str(REMOTE_ROOT / "configs"), copy=True, ignore=_ignore_modal_copy)
    .workdir(str(REMOTE_ROOT))
    .env({"PYTHONPATH": PYTHONPATH, "MPLBACKEND": "Agg"})
)

app = modal.App(APP_NAME)


@dataclasses.dataclass(frozen=True)
class SweepCase:
    scheduler: str
    num_robots: int
    seed: int

    @property
    def run_id(self) -> str:
        return f"scheduler={self.scheduler}__robots={self.num_robots}__seed={self.seed}"


def _parse_csv(value: str, *, cast=str) -> list[Any]:
    return [cast(item.strip()) for item in value.split(",") if item.strip()]


def _append_args(base: list[str], extra: str) -> list[str]:
    return base + shlex.split(extra)


def _wait_for_server(port: int, timeout_s: float = 180.0) -> None:
    deadline = time.monotonic() + timeout_s
    url = f"http://127.0.0.1:{port}/metadata"
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=5.0) as resp:
                if resp.status == 200:
                    return
        except Exception as exc:  # noqa: BLE001
            last_error = exc
        time.sleep(2.0)
    raise TimeoutError(f"server did not become ready at {url}: {last_error}")


def _run_subprocess(
    args: list[str], *, log_path: pathlib.Path, timeout_s: int | None = None
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as log_file:
        result = subprocess.run(
            args,
            cwd=REMOTE_ROOT,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout_s,
            env={
                **{k: v for k, v in __import__("os").environ.items()},
                **dict(PYTHONPATH=PYTHONPATH, MPLBACKEND="Agg"),
            },
        )
    if result.returncode != 0:
        raise subprocess.CalledProcessError(result.returncode, args)


def _tar_directory(path: pathlib.Path) -> bytes:
    def compact_filter(info: tarfile.TarInfo) -> tarfile.TarInfo | None:
        artifact_path = pathlib.Path(info.name)
        if artifact_path.suffix in {".mp4", ".parquet", ".npz"}:
            return None
        return info

    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        tar.add(path, arcname=path.name, filter=compact_filter)
    return buffer.getvalue()


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _jain_fairness(values: list[float]) -> float:
    if not values:
        return 0.0
    denom = len(values) * sum(v * v for v in values)
    return (sum(values) ** 2 / denom) if denom > 0 else 0.0


def _summarize_run(output_dir: pathlib.Path, case: SweepCase) -> dict[str, Any]:
    summary_path = output_dir / "summary.csv"
    results_path = output_dir / "results.csv"
    runtime_path = output_dir / "runtime_metadata.json"
    server_path = output_dir / "server_metadata.json"

    total_success = 0.0
    overall_starvation_rate = 0.0
    post_first_starvation_rate = 0.0
    if summary_path.exists():
        with summary_path.open() as f:
            rows = list(csv.DictReader(f))
        if rows:
            total_success = sum(_safe_float(r.get("success")) for r in rows) / len(rows)
            starvation_steps = sum(_safe_float(r.get("starvation_steps")) for r in rows)
            observed_steps = sum(_safe_float(r.get("observed_steps")) for r in rows)
            post_first_starvation_steps = sum(
                _safe_float(r.get("post_first_starvation_steps")) for r in rows
            )
            post_first_observed_steps = sum(
                _safe_float(r.get("post_first_observed_steps")) for r in rows
            )
            overall_starvation_rate = starvation_steps / observed_steps if observed_steps else 0.0
            post_first_starvation_rate = (
                post_first_starvation_steps / post_first_observed_steps
                if post_first_observed_steps
                else 0.0
            )

    robot_rates: list[float] = []
    if results_path.exists():
        by_robot: dict[str, dict[str, float]] = {}
        with results_path.open() as f:
            for row in csv.DictReader(f):
                robot = str(row.get("robot_idx", "unknown"))
                stats = by_robot.setdefault(robot, {"starvation_steps": 0.0, "observed_steps": 0.0})
                stats["starvation_steps"] += _safe_float(row.get("starvation_steps"))
                stats["observed_steps"] += _safe_float(row.get("observed_steps"))
        robot_rates = [
            stats["starvation_steps"] / stats["observed_steps"]
            for stats in by_robot.values()
            if stats["observed_steps"] > 0
        ]

    runtime = json.loads(runtime_path.read_text()) if runtime_path.exists() else {}
    server = json.loads(server_path.read_text()) if server_path.exists() else {}
    service_rates = [1.0 - rate for rate in robot_rates]
    sorted_rates = sorted(robot_rates)
    tail_count = max(1, int(len(sorted_rates) * 0.1)) if sorted_rates else 0

    return {
        "run_id": case.run_id,
        "scheduler": case.scheduler,
        "num_robots": case.num_robots,
        "seed": case.seed,
        "success_rate": total_success,
        "starvation_rate": overall_starvation_rate,
        "post_first_starvation_rate": post_first_starvation_rate,
        "robot_starvation_rate_max": max(robot_rates) if robot_rates else 0.0,
        "robot_starvation_rate_std": _safe_float(__import__("statistics").pstdev(robot_rates))
        if len(robot_rates) > 1
        else 0.0,
        "robot_starvation_rate_cvar90": sum(sorted_rates[-tail_count:]) / tail_count
        if tail_count
        else 0.0,
        "service_jain_fairness": _jain_fairness(service_rates),
        "max_batch_size": server.get("max_batch_size", ""),
        "action_horizon": server.get("action_horizon", ""),
        "max_steps": runtime.get("max_steps", ""),
        "num_trials_per_task": runtime.get("num_trials_per_task", ""),
    }


@app.function(image=image, timeout=60 * 60, cpu=4, memory=16384)
def run_case(
    case: SweepCase,
    *,
    server_extra: str,
    client_extra: str,
    max_batch_size: int,
    max_steps: int,
    num_trials_per_task: int,
    port: int,
) -> dict[str, Any]:
    run_dir = REMOTE_OUTPUT_ROOT / case.run_id
    output_dir = run_dir / "output"
    log_dir = run_dir / "logs"
    run_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    server_cmd = _append_args(
        [
            sys.executable,
            "scripts/serve.py",
            "--port",
            str(port),
            "--max-batch-size",
            str(max_batch_size),
            "--scheduling-algorithm",
            case.scheduler,
        ],
        server_extra,
    )
    client_cmd = _append_args(
        [
            sys.executable,
            "scripts/run_libero.py",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--env",
            "mock",
            "--overwrite",
            "--progress-type",
            "logging",
            "--num-robots",
            str(case.num_robots),
            "--max-steps",
            str(max_steps),
            "--num-trials-per-task",
            str(num_trials_per_task),
            "--seed",
            str(case.seed),
            "--output-dir",
            str(output_dir),
        ],
        client_extra,
    )

    server_log = log_dir / "server.log"
    with server_log.open("w") as log_file:
        server_proc = subprocess.Popen(
            server_cmd,
            cwd=REMOTE_ROOT,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            env={
                **{k: v for k, v in __import__("os").environ.items()},
                **dict(PYTHONPATH=PYTHONPATH, MPLBACKEND="Agg"),
            },
        )
    try:
        _wait_for_server(port)
        _run_subprocess(client_cmd, log_path=log_dir / "client.log", timeout_s=60 * 45)
        summary = _summarize_run(output_dir, case)
        summary["status"] = "ok"
    except Exception as exc:  # noqa: BLE001
        summary = {
            "run_id": case.run_id,
            "scheduler": case.scheduler,
            "num_robots": case.num_robots,
            "seed": case.seed,
            "status": "failed",
            "error": repr(exc),
        }
    finally:
        server_proc.terminate()
        try:
            server_proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            server_proc.kill()
            server_proc.wait(timeout=20)

    (run_dir / "case.json").write_text(json.dumps(dataclasses.asdict(case), indent=2))
    (run_dir / "summary_row.json").write_text(json.dumps(summary, indent=2))
    summary["artifact_tgz"] = _tar_directory(run_dir)
    return summary


def _write_rows(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key != "artifact_tgz" and key not in keys:
                keys.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in keys})


@app.local_entrypoint()
def main(
    schedulers: str = "fixed-max-batch,greedy-deadline,round-robin",
    num_robots: str = "2,4,6",
    seeds: str = "7",
    output_dir: str = "experiments/sweeps/mock",
    max_batch_size: int = 4,
    max_steps: int = 50,
    num_trials_per_task: int = 1,
    server_extra: str = "--env LIBERO policy:mock --policy.action-horizon 10 --policy.action-dim 7 --policy.profile l40s_pi05",
    client_extra: str = "",
    port: int = 8080,
) -> None:
    """Run the Cartesian product of schedulers, num_robots, and seeds."""
    out = pathlib.Path(output_dir)
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d_%H%M%S")  # noqa: UP017
    artifacts_dir = out / "artifacts"
    cases = [
        SweepCase(scheduler=scheduler, num_robots=robots, seed=seed)
        for scheduler in _parse_csv(schedulers)
        for robots in _parse_csv(num_robots, cast=int)
        for seed in _parse_csv(seeds, cast=int)
    ]

    rows: list[dict[str, Any]] = []
    for result in run_case.map(
        cases,
        kwargs={
            "server_extra": server_extra,
            "client_extra": client_extra,
            "max_batch_size": max_batch_size,
            "max_steps": max_steps,
            "num_trials_per_task": num_trials_per_task,
            "port": port,
        },
        order_outputs=False,
    ):
        artifact_bytes = result.pop("artifact_tgz")
        artifact_path = artifacts_dir / f"{result['run_id']}.tgz"
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        artifact_path.write_bytes(artifact_bytes)
        result["artifact_path"] = str(artifact_path)
        rows.append(result)
        print(
            f"{result['status']}: {result['run_id']} "
            f"starvation={_safe_float(result.get('starvation_rate')):.3f} "
            f"fairness={_safe_float(result.get('service_jain_fairness')):.3f}"
        )

    sweep_csv = out / f"sweep_results_{stamp}.csv"
    latest_csv = out / "sweep_results.csv"
    _write_rows(sweep_csv, rows)
    _write_rows(latest_csv, rows)
    print(f"Wrote {latest_csv}")
    print(f"Wrote {sweep_csv}")
