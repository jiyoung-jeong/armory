"""Run server/client scheduler sweeps on Modal.

Example:
    modal run scripts/modal_sweep.py \
        --schedulers fixed-max-batch,greedy-deadline,round-robin \
        --experiment-configs configs/experiments/mock/short.json \
        --num-robots 1,2,3,4,5,6,7,8,9,10 \
        --server-config configs/server/mock.json \
        --seeds 7,42 \
        --output-dir experiments/sweeps/big_mock
"""

from __future__ import annotations

import csv
import dataclasses
import datetime as dt
import json
import pathlib
import shlex
import shutil
import subprocess
import sys
from typing import Any

import modal

APP_NAME = "armory-scheduler-sweep"
ARTIFACTS_VOLUME_NAME = "armory-scheduler-sweep-artifacts"
REMOTE_ROOT = pathlib.Path("/app")
REMOTE_OUTPUT_ROOT = pathlib.Path("/tmp/armory_sweep")
REMOTE_ARTIFACTS_ROOT = pathlib.Path("/artifacts")
ARTIFACT_SKIP_SUFFIXES = {".mp4", ".parquet", ".npz"}
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
    .workdir(str(REMOTE_ROOT))
    .env({"PYTHONPATH": PYTHONPATH, "MPLBACKEND": "Agg"})
    .add_local_dir("packages", str(REMOTE_ROOT / "packages"), copy=True, ignore=_ignore_modal_copy)
    .add_local_dir("src", str(REMOTE_ROOT / "src"), copy=True, ignore=_ignore_modal_copy)
    .add_local_dir("configs", str(REMOTE_ROOT / "configs"), copy=True, ignore=_ignore_modal_copy)
    .add_local_dir("scripts", str(REMOTE_ROOT / "scripts"), copy=True, ignore=_ignore_modal_copy)
)

app = modal.App(APP_NAME)

artifacts_volume = modal.Volume.from_name(ARTIFACTS_VOLUME_NAME, create_if_missing=True)


@dataclasses.dataclass(frozen=True)
class SweepCase:
    scheduler: str
    experiment_config: str  # path relative to repo root
    num_robots: int
    seed: int

    @property
    def run_id(self) -> str:
        config_name = pathlib.Path(self.experiment_config).stem
        return f"scheduler={self.scheduler}__config={config_name}__robots={self.num_robots}__seed={self.seed}"


def _parse_csv(value: str, *, cast=str) -> list[Any]:
    return [cast(item.strip()) for item in value.split(",") if item.strip()]


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


def _write_command_manifest(
    run_dir: pathlib.Path, *, server_cmd: list[str], client_cmd: list[str]
) -> None:
    commands = {
        "cwd": str(REMOTE_ROOT),
        "env": {"PYTHONPATH": PYTHONPATH, "MPLBACKEND": "Agg"},
        "server": {"argv": server_cmd, "shell": shlex.join(server_cmd)},
        "client": {"argv": client_cmd, "shell": shlex.join(client_cmd)},
    }
    (run_dir / "commands.json").write_text(json.dumps(commands, indent=2))
    (run_dir / "commands.sh").write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "set -euo pipefail",
                f"cd {shlex.quote(str(REMOTE_ROOT))}",
                f"export PYTHONPATH={shlex.quote(PYTHONPATH)}",
                "export MPLBACKEND=Agg",
                "",
                "# Start this first, then run the client command in another shell.",
                f"SERVER_CMD={shlex.quote(shlex.join(server_cmd))}",
                f"CLIENT_CMD={shlex.quote(shlex.join(client_cmd))}",
                'printf "server: %s\\n" "$SERVER_CMD"',
                'printf "client: %s\\n" "$CLIENT_CMD"',
            ]
        )
        + "\n"
    )


def _copy_run_dir(src_root: pathlib.Path, dest_root: pathlib.Path) -> None:
    """Copy run_dir into the mounted artifacts volume, skipping bulky binaries."""
    for src in src_root.rglob("*"):
        if src.is_dir() or src.suffix in ARTIFACT_SKIP_SUFFIXES:
            continue
        dst = dest_root / src.relative_to(src_root)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _build_server_cmd(srv_cfg: dict[str, Any], *, port: int, scheduler: str) -> list[str]:
    cmd = [
        sys.executable,
        "scripts/serve.py",
        "--port",
        str(port),
        "--env",
        srv_cfg.get("env", "LIBERO"),
        "--max-batch-size",
        str(srv_cfg.get("max_batch_size", 1)),
        "--scheduling-algorithm",
        scheduler,
        f"policy:{srv_cfg.get('policy_type', 'default')}",
    ]
    for k, v in srv_cfg.get("policy", {}).items():
        cmd += [f"--policy.{k.replace('_', '-')}", str(v)]
    return cmd


def _expand_experiment_config(exp_cfg: dict[str, Any], num_robots: int) -> dict[str, Any]:
    """Return a copy of exp_cfg with robot profiles set for num_robots robots.

    If the config already defines more than one robot explicitly, those profiles are used
    as-is and num_robots is ignored (the config is authoritative).
    Otherwise robot_0's profile is replicated to fill num_robots robots.
    """
    if len(exp_cfg["robots"]) > 1:
        actual = len(exp_cfg["robots"])
        return {**exp_cfg, "experiment": {**exp_cfg["experiment"], "num_robots": actual}}
    robot_template = exp_cfg["robots"]["robot_0"]
    return {
        **exp_cfg,
        "experiment": {**exp_cfg["experiment"], "num_robots": num_robots},
        "robots": {f"robot_{i}": dict(robot_template) for i in range(num_robots)},
    }


def _config_num_robots(cfg_path: str, fallback: int) -> int:
    """Read robot count from a local config file when robots are pre-defined, else fallback."""
    try:
        n = len(json.loads(pathlib.Path(cfg_path).read_text()).get("robots", {}))
        if n > 1:
            return n
    except Exception:
        pass
    return fallback


def _build_client_cmd(
    *,
    port: int,
    seed: int,
    output_dir: pathlib.Path,
    experiment_config_path: pathlib.Path,
    max_steps: int,
) -> list[str]:
    return [
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
        "--max-steps",
        str(max_steps),
        "--seed",
        str(seed),
        "--output-dir",
        str(output_dir),
        "--experiment-config",
        str(experiment_config_path),
    ]


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
    sorted_rates = sorted(robot_rates)
    tail_count = max(1, int(len(sorted_rates) * 0.1)) if sorted_rates else 0

    summary = {
        "run_id": case.run_id,
        "scheduler": case.scheduler,
        "experiment_config": case.experiment_config,
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
        "max_batch_size": server.get("max_batch_size", ""),
        "action_horizon": server.get("action_horizon", ""),
        "max_steps": runtime.get("max_steps", ""),
        "num_trials_per_task": runtime.get("num_trials_per_task", ""),
    }

    try:
        from sims.libero.metrics import compute_server_timing_health  # noqa: PLC0415

        health = compute_server_timing_health(output_dir)
        if health is not None:
            summary.update(health)
    except Exception:  # noqa: BLE001
        pass

    return summary


@app.function(
    image=image,
    timeout=60 * 60,
    cpu=4,
    memory=8192,
    volumes={str(REMOTE_ARTIFACTS_ROOT): artifacts_volume},
)
def run_case(
    case: SweepCase,
    *,
    server_config: str,
    port: int,
    stamp: str,
    max_batch_size_override: int | None = None,
    max_steps_override: int | None = None,
) -> dict[str, Any]:
    exp_cfg: dict[str, Any] = json.loads((REMOTE_ROOT / case.experiment_config).read_text())
    srv_cfg: dict[str, Any] = json.loads((REMOTE_ROOT / server_config).read_text())

    if max_batch_size_override is not None:
        srv_cfg["max_batch_size"] = max_batch_size_override
    if max_steps_override is not None:
        exp_cfg["experiment"]["max_steps"] = max_steps_override

    exp_cfg = _expand_experiment_config(exp_cfg, case.num_robots)
    max_steps = int(exp_cfg["experiment"]["max_steps"])

    run_dir = REMOTE_OUTPUT_ROOT / case.run_id
    output_dir = run_dir / "output"
    log_dir = run_dir / "logs"
    run_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    saved_exp_config = run_dir / "experiment_config.json"
    (run_dir / "case.json").write_text(json.dumps(dataclasses.asdict(case), indent=2))
    saved_exp_config.write_text(json.dumps(exp_cfg, indent=2))
    (run_dir / "server_config.json").write_text(json.dumps(srv_cfg, indent=2))

    server_cmd = _build_server_cmd(srv_cfg, port=port, scheduler=case.scheduler)
    client_cmd = _build_client_cmd(
        port=port,
        seed=case.seed,
        output_dir=output_dir,
        experiment_config_path=saved_exp_config,
        max_steps=max_steps,
    )
    _write_command_manifest(run_dir, server_cmd=server_cmd, client_cmd=client_cmd)

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
        _run_subprocess(client_cmd, log_path=log_dir / "client.log", timeout_s=60 * 45)
        summary = _summarize_run(output_dir, case)
        summary["status"] = "ok"
    except Exception as exc:  # noqa: BLE001
        summary = {
            "run_id": case.run_id,
            "scheduler": case.scheduler,
            "experiment_config": case.experiment_config,
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

    dest = REMOTE_ARTIFACTS_ROOT / stamp / case.run_id
    _copy_run_dir(run_dir, dest)
    artifacts_volume.commit()
    summary["artifact_remote_path"] = str(dest)
    return summary


def _write_rows(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in keys})


@app.local_entrypoint()
def main(
    schedulers: str = "fixed-max-batch,greedy-deadline,round-robin",
    experiment_configs: str = "configs/experiments/mock/short.json",
    num_robots: str = "2,4,6",
    server_config: str = "configs/server/mock.json",
    seeds: str = "7",
    output_dir: str = "experiments/sweeps/mock",
    port: int = 8080,
    max_batch_size: int | None = None,
    max_steps: int | None = None,
) -> None:
    """Run the Cartesian product of schedulers, experiment_configs, num_robots, and seeds."""
    out = pathlib.Path(output_dir)
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d_%H%M%S")  # noqa: UP017
    artifacts_dir = out / "artifacts"
    cases = [
        SweepCase(
            scheduler=scheduler,
            experiment_config=cfg,
            num_robots=_config_num_robots(cfg, n),
            seed=seed,
        )
        for scheduler in _parse_csv(schedulers)
        for cfg in _parse_csv(experiment_configs)
        for n in _parse_csv(num_robots, cast=int)
        for seed in _parse_csv(seeds, cast=int)
    ]

    rows: list[dict[str, Any]] = []
    for result in run_case.map(
        cases,
        kwargs={
            "server_config": server_config,
            "port": port,
            "stamp": stamp,
            "max_batch_size_override": max_batch_size,
            "max_steps_override": max_steps,
        },
        order_outputs=False,
    ):
        rows.append(result)
        print(
            f"{result['status']}: {result['run_id']} "
            f"starvation={_safe_float(result.get('starvation_rate')):.3f} "
        )

    artifacts_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading artifacts from volume '{ARTIFACTS_VOLUME_NAME}/{stamp}' -> {artifacts_dir}")
    subprocess.run(
        [
            "modal",
            "volume",
            "get",
            ARTIFACTS_VOLUME_NAME,
            stamp,
            str(artifacts_dir),
            "--force",
        ],
        check=True,
    )
    for row in rows:
        row["artifact_path"] = str(artifacts_dir / stamp / row["run_id"])

    sweep_csv = out / f"sweep_results_{stamp}.csv"
    latest_csv = out / "sweep_results.csv"
    _write_rows(sweep_csv, rows)
    _write_rows(latest_csv, rows)
    print(f"Wrote {latest_csv}")
    print(f"Wrote {sweep_csv}")

    suspicious = [r for r in rows if r.get("timing_suspicious")]
    if suspicious:
        print(f"WARNING: {len(suspicious)} run(s) flagged for suspicious timings:")
        for r in suspicious:
            print(f"  {r['run_id']}: {r.get('timing_flags', '')}")

    sys.path.insert(0, str(pathlib.Path(__file__).parent))
    from plot_starvation_sweep import DEFAULT_METRICS, plot_results  # noqa: PLC0415

    timing_metrics = [
        "step_interval_p95_ms",
        "inference_p99_ms",
        "inbound_p95_ms",
        "outbound_p95_ms",
    ]
    # Stamp parallels artifacts/<stamp>/ so a re-run never overwrites a prior
    # plot set; the CSV next to it (sweep_results_<stamp>.csv) is the inputs.
    plots_dir = out / "plots" / stamp
    plot_results(latest_csv, plots_dir, metrics=list(DEFAULT_METRICS) + timing_metrics)
