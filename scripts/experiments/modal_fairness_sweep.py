"""Run a fairness-vs-heterogeneity sweep on Modal.

Sweeps n_fast (number of "fast" robots in a 15-robot fleet) for each combination of
(model, scheduler, seed) and reports Jain's index on per-robot starvation rate.

Heterogeneity definition:
    - 15 robots total, partitioned into n_fast fast + (15 - n_fast) slow robots
    - fast robots: execution_horizon = FAST_HORIZON (4)
    - slow robots: execution_horizon = SLOW_HORIZON (10)
    - n_fast is the only knob; per-robot horizons are fixed regardless of model

Plot output:
    - one figure per model (pi05, gr00t)
    - x = n_fast, y = Jain's freshness index, one line per scheduler

Example:
    modal run scripts/modal_fairness_sweep.py \
        --models pi05,gr00t-n1.7 \
        --schedulers fixed-max-batch,greedy-deadline,round-robin,dynamic-action \
        --n-fast-grid 0,1,3,5,7,10,12,14,15 \
        --seeds 7,42 \
        --output-dir experiments/sweeps/fairness_het
"""

from __future__ import annotations

import csv
import dataclasses
import datetime as dt
import io
import json
import pathlib
import subprocess
import sys
import tarfile
from typing import Any

import modal

APP_NAME = "armory-fairness-sweep"
REMOTE_ROOT = pathlib.Path("/app")
REMOTE_OUTPUT_ROOT = pathlib.Path("/tmp/armory_fairness_sweep")
PYTHONPATH = ":".join(
    [
        str(REMOTE_ROOT / "src"),
        str(REMOTE_ROOT / "src/backends"),
        str(REMOTE_ROOT / "packages/armory-client/src"),
    ]
)

NUM_ROBOTS = 15
CONTROL_HZ = 20
MAX_BATCH_SIZE = 4
FAST_HORIZON = 4
SLOW_HORIZON = 10
ALPHA_FOR_DYNAMIC = 1.0
DEFAULT_MAX_STEPS = 200
DEFAULT_TRIALS_PER_ROBOT = 1

MODEL_TO_PROFILE = {
    "pi05": "l40s_pi05",
    "gr00t-n1.7": "l40s_gr00t",
}

# tyro accepts the ModelFamily enum *name*, not its value
MODEL_TO_ENUM_NAME = {
    "pi05": "PI05",
    "gr00t-n1.7": "GROOT_N17",
}

SCHEDULER_DISPLAY = {
    "fixed-max-batch": "fixed-max-batch",
    "greedy-deadline": "greedy-deadline",
    "round-robin": "round-robin",
    "dynamic-action": f"dynamic-action (α={ALPHA_FOR_DYNAMIC})",
}


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


@dataclasses.dataclass(frozen=True)
class SweepCase:
    model: str
    scheduler: str
    n_fast: int
    seed: int

    @property
    def run_id(self) -> str:
        return (
            f"model={self.model}__scheduler={self.scheduler}"
            f"__nfast={self.n_fast:02d}__seed={self.seed}"
        )


def _parse_csv(value: str, *, cast=str) -> list[Any]:
    return [cast(item.strip()) for item in value.split(",") if item.strip()]


def _compute_horizons(n_fast: int, n_total: int = NUM_ROBOTS) -> list[int]:
    """Per-robot execution horizons: n_fast at FAST_HORIZON, the rest at SLOW_HORIZON."""
    n_slow = n_total - n_fast
    return [FAST_HORIZON] * n_fast + [SLOW_HORIZON] * n_slow


def _build_experiment_config(horizons: list[int], max_steps: int) -> dict[str, Any]:
    return {
        "experiment": {
            "action_chunk_broker_type": "naive_async",
            "num_robots": len(horizons),
            "trials_per_robot": DEFAULT_TRIALS_PER_ROBOT,
            "max_steps": max_steps,
        },
        "robots": {
            f"robot_{i}": {
                "execution_horizon": int(h),
                "uplink_median_ms": 0.0,
                "uplink_sigma": 0.0,
                "downlink_median_ms": 0.0,
                "downlink_sigma": 0.0,
            }
            for i, h in enumerate(horizons)
        },
    }


def _build_server_cmd(*, model: str, scheduler: str, port: int) -> list[str]:
    profile = MODEL_TO_PROFILE[model]
    # All top-level Args flags must come BEFORE the policy:mock subcommand,
    # otherwise tyro treats them as belonging to the policy subcommand.
    pre_policy = [
        sys.executable,
        "scripts/serve.py",
        "--port",
        str(port),
        "--env",
        "LIBERO",
        "--model",
        MODEL_TO_ENUM_NAME[model],
        "--max-batch-size",
        str(MAX_BATCH_SIZE),
        "--scheduling-algorithm",
        scheduler,
    ]
    if scheduler == "dynamic-action":
        pre_policy += ["--alpha", str(ALPHA_FOR_DYNAMIC)]
    post_policy = [
        "policy:mock",
        "--policy.action-horizon",
        "10",
        "--policy.action-dim",
        "7",
        "--policy.profile",
        profile,
    ]
    return pre_policy + post_policy


def _build_client_cmd(
    *, port: int, seed: int, output_dir: pathlib.Path, experiment_config_path: pathlib.Path
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
        "--seed",
        str(seed),
        "--output-dir",
        str(output_dir),
        "--experiment-config",
        str(experiment_config_path),
    ]


def _run_subprocess(args: list[str], *, log_path: pathlib.Path, timeout_s: int | None = None) -> None:
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
        if pathlib.Path(info.name).suffix in {".mp4", ".parquet", ".npz"}:
            return None
        return info

    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        tar.add(path, arcname=path.name, filter=compact_filter)
    return buffer.getvalue()


def _summarize_run(output_dir: pathlib.Path, case: SweepCase, horizons: list[int]) -> dict[str, Any]:
    from sims.libero.metrics import compute_fairness_metrics  # noqa: PLC0415

    fairness = compute_fairness_metrics(output_dir)
    summary: dict[str, Any] = {
        "run_id": case.run_id,
        "model": case.model,
        "scheduler": case.scheduler,
        "n_fast": case.n_fast,
        "seed": case.seed,
        "horizons": json.dumps(horizons),
    }
    if fairness is not None:
        summary["alpha"] = fairness.get("alpha")
        summary["jain_freshness"] = fairness["jain_freshness"]
        summary["jain_starvation"] = fairness["jain_starvation"]
        rates = fairness["starvation_rate"]
        if rates:
            summary["mean_starvation"] = float(sum(rates) / len(rates))
            summary["max_starvation"] = float(max(rates))
            summary["min_starvation"] = float(min(rates))
    return summary


@app.function(image=image, timeout=60 * 60, cpu=4, memory=16384)
def run_case(case: SweepCase, *, port: int, max_steps: int) -> dict[str, Any]:
    horizons = _compute_horizons(case.n_fast)
    exp_cfg = _build_experiment_config(horizons, max_steps)

    run_dir = REMOTE_OUTPUT_ROOT / case.run_id
    output_dir = run_dir / "output"
    log_dir = run_dir / "logs"
    run_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    saved_exp_config = run_dir / "experiment_config.json"
    (run_dir / "case.json").write_text(json.dumps(dataclasses.asdict(case), indent=2))
    saved_exp_config.write_text(json.dumps(exp_cfg, indent=2))

    server_cmd = _build_server_cmd(model=case.model, scheduler=case.scheduler, port=port)
    client_cmd = _build_client_cmd(
        port=port, seed=case.seed, output_dir=output_dir, experiment_config_path=saved_exp_config
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
        _run_subprocess(client_cmd, log_path=log_dir / "client.log", timeout_s=60 * 30)
        summary = _summarize_run(output_dir, case, horizons)
        summary["status"] = "ok"
    except Exception as exc:  # noqa: BLE001
        summary = {
            "run_id": case.run_id,
            "model": case.model,
            "scheduler": case.scheduler,
            "n_fast": case.n_fast,
            "seed": case.seed,
            "horizons": json.dumps(horizons),
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


def _plot_fairness_curves(results_csv: pathlib.Path, plots_dir: pathlib.Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    df = pd.read_csv(results_csv)
    if df.empty:
        return
    df = df[df.get("status", "ok") == "ok"]
    if "jain_freshness" not in df.columns:
        return
    df = df.dropna(subset=["jain_freshness"])

    plots_dir.mkdir(parents=True, exist_ok=True)
    schedulers = sorted(df["scheduler"].unique())
    color_cycle = plt.cm.tab10(np.linspace(0, 1, max(len(schedulers), 2)))

    for model in sorted(df["model"].unique()):
        sub_model = df[df["model"] == model]
        fig, ax = plt.subplots(figsize=(9, 5))
        for color, sched in zip(color_cycle, schedulers):
            sub = sub_model[sub_model["scheduler"] == sched]
            if sub.empty:
                continue
            agg = (
                sub.groupby("n_fast")["jain_freshness"]
                .agg(["mean", "std", "count"])
                .reset_index()
                .sort_values("n_fast")
            )
            yerr = agg["std"].fillna(0.0) / agg["count"].clip(lower=1).pow(0.5)
            ax.errorbar(
                agg["n_fast"],
                agg["mean"],
                yerr=yerr,
                marker="o",
                linewidth=1.6,
                capsize=3,
                color=color,
                label=SCHEDULER_DISPLAY.get(sched, sched),
            )
        ax.set_xlabel("Heterogeneity (n_fast / 15)", fontsize=12)
        ax.set_ylabel("Jain's index on freshness rate", fontsize=12)
        ax.set_title(f"Fairness vs heterogeneity — {model}", fontsize=13, fontweight="bold")
        ax.set_ylim(0, 1.02)
        ax.set_xlim(-0.5, NUM_ROBOTS + 0.5)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="lower left", fontsize=9, frameon=False)
        plt.tight_layout()
        out = plots_dir / f"jains_vs_het__{model.replace('.', '_').replace('/', '_')}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Wrote {out}")


@app.local_entrypoint()
def main(
    models: str = "pi05,gr00t-n1.7",
    schedulers: str = "fixed-max-batch,greedy-deadline,round-robin,dynamic-action",
    n_fast_grid: str = "0,1,3,5,7,10,12,14,15",
    seeds: str = "7,42,123",
    output_dir: str = "experiments/sweeps/fairness_het_2",
    port: int = 8080,
    max_steps: int = DEFAULT_MAX_STEPS,
) -> None:
    out = pathlib.Path(output_dir)
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d_%H%M%S")  # noqa: UP017
    artifacts_dir = out / "artifacts"

    cases = [
        SweepCase(model=m, scheduler=s, n_fast=n, seed=seed)
        for m in _parse_csv(models)
        for s in _parse_csv(schedulers)
        for n in _parse_csv(n_fast_grid, cast=int)
        for seed in _parse_csv(seeds, cast=int)
    ]
    print(f"Submitting {len(cases)} cases")

    rows: list[dict[str, Any]] = []
    for result in run_case.map(
        cases,
        kwargs={"port": port, "max_steps": max_steps},
        order_outputs=False,
    ):
        artifact_bytes = result.pop("artifact_tgz", None)
        if artifact_bytes is not None:
            run_dir = artifacts_dir / result["run_id"]
            run_dir.mkdir(parents=True, exist_ok=True)
            with tarfile.open(fileobj=io.BytesIO(artifact_bytes), mode="r:gz") as tar:
                tar.extractall(run_dir)
            result["artifact_path"] = str(run_dir)
        rows.append(result)
        jain = result.get("jain_freshness", "")
        print(
            f"{result['status']}: {result['run_id']} "
            f"jain_freshness={jain if jain == '' else f'{float(jain):.4f}'}"
        )

    sweep_csv = out / f"sweep_results_{stamp}.csv"
    latest_csv = out / "sweep_results.csv"
    _write_rows(sweep_csv, rows)
    _write_rows(latest_csv, rows)
    print(f"Wrote {latest_csv}")
    print(f"Wrote {sweep_csv}")

    _plot_fairness_curves(latest_csv, out / "plots")
