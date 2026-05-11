"""Sweep alpha for dynamic-action vs three baselines, on a list of (n_fast, n_slow) scenarios.

For each (scenario, model):
    - fixed-max-batch, greedy-deadline, round-robin: single point each (seeds averaged)
    - dynamic-action: a curve over alpha = 0..1

Plot output:
    - one figure per (scenario, model)
    - x = mean starvation rate, y = Jain's index on freshness
    - baselines render as labeled scatter points; dynamic-action renders as a connected curve

Example:
    modal run scripts/experiments/modal_alpha_fairness_sweep.py \
        --models pi05,gr00t-n1.7 \
        --scenarios 1f9s,5f5s \
        --alpha-grid 0.0,0.1,0.2,0.3,0.5,0.7,1.0 \
        --seeds 7,42 \
        --output-dir experiments/sweeps/fairness_alpha
"""

from __future__ import annotations

import csv
import dataclasses
import datetime as dt
import io
import json
import pathlib
import re
import subprocess
import sys
import tarfile
import threading
from typing import Any

import modal

APP_NAME = "armory-fairness-alpha-sweep"
REMOTE_ROOT = pathlib.Path("/app")
REMOTE_OUTPUT_ROOT = pathlib.Path("/tmp/armory_fairness_alpha_sweep")
PYTHONPATH = ":".join(
    [
        str(REMOTE_ROOT / "src"),
        str(REMOTE_ROOT / "src/backends"),
        str(REMOTE_ROOT / "packages/armory-client/src"),
    ]
)

CONTROL_HZ = 20
# MAX_BATCH_SIZE = 4
MAX_BATCH_SIZE = 20
FAST_HORIZON = 4
SLOW_HORIZON = 10
DEFAULT_MAX_STEPS = 200
DEFAULT_TRIALS_PER_ROBOT = 1

BASELINE_SCHEDULERS = ("fixed-max-batch", "greedy-deadline", "round-robin")
DYNAMIC_SCHEDULER = "dynamic-action"

MODEL_TO_PROFILE = {
    "pi05": "l40s_pi05",
    "gr00t-n1.7": "l40s_gr00t",
}

MODEL_TO_ENUM_NAME = {
    "pi05": "PI05",
    "gr00t-n1.7": "GROOT_N17",
}

BASELINE_STYLE = {
    "fixed-max-batch": {"marker": "s", "color": "#1f77b4"},
    "greedy-deadline": {"marker": "^", "color": "#2ca02c"},
    "round-robin": {"marker": "D", "color": "#d62728"},
}
DYNAMIC_CMAP = "viridis"


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
class Scenario:
    """A heterogeneity scenario: n_fast fast robots + n_slow slow robots."""

    n_fast: int
    n_slow: int

    @property
    def n_total(self) -> int:
        return self.n_fast + self.n_slow

    @property
    def scenario_id(self) -> str:
        return f"{self.n_fast}f{self.n_slow}s"

    def horizons(self) -> list[int]:
        return [FAST_HORIZON] * self.n_fast + [SLOW_HORIZON] * self.n_slow


@dataclasses.dataclass(frozen=True)
class SweepCase:
    model: str
    scheduler: str
    scenario_id: str
    n_fast: int
    n_slow: int
    seed: int
    # alpha is only meaningful for dynamic-action; None for the three baselines
    alpha: float | None

    @property
    def run_id(self) -> str:
        alpha_part = "" if self.alpha is None else f"__alpha={self.alpha:.3f}"
        return (
            f"model={self.model}__scheduler={self.scheduler}"
            f"__scenario={self.scenario_id}{alpha_part}__seed={self.seed}"
        )


def _parse_csv(value: str, *, cast=str) -> list[Any]:
    return [cast(item.strip()) for item in value.split(",") if item.strip()]


def _parse_scenarios(value: str) -> list[Scenario]:
    """Parse strings like '1f9s,5f5s' into Scenario objects."""
    pattern = re.compile(r"^\s*(\d+)f(\d+)s\s*$")
    out: list[Scenario] = []
    for token in value.split(","):
        if not token.strip():
            continue
        m = pattern.match(token)
        if not m:
            raise ValueError(
                f"Bad scenario token {token!r}; expected '<n_fast>f<n_slow>s' (e.g. '1f9s')"
            )
        out.append(Scenario(n_fast=int(m.group(1)), n_slow=int(m.group(2))))
    return out


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


def _build_server_cmd(
    *, model: str, scheduler: str, port: int, alpha: float | None
) -> list[str]:
    profile = MODEL_TO_PROFILE[model]
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
    if scheduler == DYNAMIC_SCHEDULER:
        if alpha is None:
            raise ValueError("dynamic-action requires alpha")
        pre_policy += ["--alpha", str(alpha)]
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


def _stream_to_log_and_stdout(stream, log_file, prefix: str) -> None:
    """Tee a subprocess text stream line-by-line to both ``log_file`` and ``sys.stdout``.

    Routing subprocess output via this helper makes it visible in the Modal
    container log (which only captures the function's own stdout/stderr) while
    still preserving the per-run log files in the artifact tarball.
    """
    for raw in stream:
        log_file.write(raw)
        log_file.flush()
        line = raw if raw.endswith("\n") else raw + "\n"
        sys.stdout.write(f"[{prefix}] {line}")
        sys.stdout.flush()


def _run_subprocess(
    args: list[str],
    *,
    log_path: pathlib.Path,
    prefix: str,
    timeout_s: int | None = None,
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = log_path.open("w")
    proc = subprocess.Popen(
        args,
        cwd=REMOTE_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env={
            **{k: v for k, v in __import__("os").environ.items()},
            **dict(PYTHONPATH=PYTHONPATH, MPLBACKEND="Agg"),
        },
    )
    reader = threading.Thread(
        target=_stream_to_log_and_stdout,
        args=(proc.stdout, log_file, prefix),
        daemon=True,
    )
    reader.start()
    try:
        rc = proc.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
        reader.join(timeout=5)
        log_file.close()
        raise
    reader.join(timeout=5)
    log_file.close()
    if rc != 0:
        raise subprocess.CalledProcessError(rc, args)


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
    from sims.libero.metrics import (  # noqa: PLC0415
        compute_fairness_metrics,
        compute_starvation_variance_series,
    )

    fairness = compute_fairness_metrics(output_dir)
    summary: dict[str, Any] = {
        "run_id": case.run_id,
        "model": case.model,
        "scheduler": case.scheduler,
        "scenario_id": case.scenario_id,
        "n_fast": case.n_fast,
        "n_slow": case.n_slow,
        "n_total": case.n_fast + case.n_slow,
        "alpha_requested": case.alpha,
        "seed": case.seed,
        "horizons": json.dumps(horizons),
    }
    if fairness is not None:
        summary["alpha_observed"] = fairness.get("alpha")
        summary["jain_freshness"] = fairness["jain_freshness"]
        summary["jain_starvation"] = fairness["jain_starvation"]
        rates = fairness["starvation_rate"]
        if rates:
            summary["mean_starvation"] = float(sum(rates) / len(rates))
            summary["max_starvation"] = float(max(rates))
            summary["min_starvation"] = float(min(rates))

    # Pull cross-robot starvation variance from the same source as the
    # starvation_variance_over_time plot so the sweep summary matches its
    # final value exactly (actions_left<=0 on a wall-clock canvas, not
    # cost_history NaNs aggregated per episode).
    series = compute_starvation_variance_series(output_dir)
    if series is not None:
        summary["starvation_variance"] = series["final_starvation_variance"]
    return summary


# @app.function(image=image, timeout=60 * 60, cpu=25, memory=16384)
@app.function(image=image, timeout=60 * 60, cpu=25, memory=16384)
def run_case(case: SweepCase, *, port: int, max_steps: int) -> dict[str, Any]:
    horizons = [FAST_HORIZON] * case.n_fast + [SLOW_HORIZON] * case.n_slow
    exp_cfg = _build_experiment_config(horizons, max_steps)

    run_dir = REMOTE_OUTPUT_ROOT / case.run_id
    output_dir = run_dir / "output"
    log_dir = run_dir / "logs"
    run_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    saved_exp_config = run_dir / "experiment_config.json"
    (run_dir / "case.json").write_text(json.dumps(dataclasses.asdict(case), indent=2))
    saved_exp_config.write_text(json.dumps(exp_cfg, indent=2))

    server_cmd = _build_server_cmd(
        model=case.model, scheduler=case.scheduler, port=port, alpha=case.alpha
    )
    client_cmd = _build_client_cmd(
        port=port, seed=case.seed, output_dir=output_dir, experiment_config_path=saved_exp_config
    )

    server_log = log_dir / "server.log"
    server_log_file = server_log.open("w")
    server_proc = subprocess.Popen(
        server_cmd,
        cwd=REMOTE_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env={
            **{k: v for k, v in __import__("os").environ.items()},
            **dict(PYTHONPATH=PYTHONPATH, MPLBACKEND="Agg"),
        },
    )
    server_reader = threading.Thread(
        target=_stream_to_log_and_stdout,
        args=(server_proc.stdout, server_log_file, "server"),
        daemon=True,
    )
    server_reader.start()
    try:
        _run_subprocess(
            client_cmd,
            log_path=log_dir / "client.log",
            prefix="client",
            timeout_s=60 * 30,
        )
        summary = _summarize_run(output_dir, case, horizons)
        summary["status"] = "ok"
    except Exception as exc:  # noqa: BLE001
        summary = {
            "run_id": case.run_id,
            "model": case.model,
            "scheduler": case.scheduler,
            "scenario_id": case.scenario_id,
            "n_fast": case.n_fast,
            "n_slow": case.n_slow,
            "n_total": case.n_fast + case.n_slow,
            "alpha_requested": case.alpha,
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
        server_reader.join(timeout=5)
        server_log_file.close()

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


def _plot_starvation_vs_fairness(results_csv: pathlib.Path, plots_dir: pathlib.Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    df = pd.read_csv(results_csv)
    if df.empty:
        return
    df = df[df.get("status", "ok") == "ok"]
    needed = {"starvation_variance", "mean_starvation", "scheduler", "model", "scenario_id"}
    if not needed.issubset(df.columns):
        return
    df = df.dropna(subset=["starvation_variance", "mean_starvation"])

    plots_dir.mkdir(parents=True, exist_ok=True)

    for (scenario_id, model), sub in df.groupby(["scenario_id", "model"]):
        fig, ax = plt.subplots(figsize=(8, 6))

        # Baselines: aggregate across seeds → one point per scheduler
        for sched in BASELINE_SCHEDULERS:
            sub_b = sub[sub["scheduler"] == sched]
            if sub_b.empty:
                continue
            x = float(sub_b["mean_starvation"].mean())
            y = float(sub_b["starvation_variance"].mean())
            xerr = float(sub_b["mean_starvation"].std(ddof=0)) if len(sub_b) > 1 else 0.0
            yerr = float(sub_b["starvation_variance"].std(ddof=0)) if len(sub_b) > 1 else 0.0
            style = BASELINE_STYLE[sched]
            ax.errorbar(
                x, y, xerr=xerr, yerr=yerr,
                marker=style["marker"], markersize=11, color=style["color"],
                linestyle="none", capsize=3, label=sched, zorder=4,
            )

        # Dynamic-action: one curve, points sorted by alpha, colored on a gradient
        sub_d = sub[sub["scheduler"] == DYNAMIC_SCHEDULER]
        if not sub_d.empty:
            agg = (
                sub_d.groupby("alpha_requested")
                .agg(
                    mean_starvation=("mean_starvation", "mean"),
                    starvation_variance=("starvation_variance", "mean"),
                    starvation_std=("mean_starvation", "std"),
                    variance_std=("starvation_variance", "std"),
                    count=("seed", "count"),
                )
                .reset_index()
                .sort_values("alpha_requested")
            )
            xs = agg["mean_starvation"].to_numpy()
            ys = agg["starvation_variance"].to_numpy()
            alphas = agg["alpha_requested"].to_numpy()
            ax.plot(xs, ys, "-", color="0.5", linewidth=1.2, alpha=0.7, zorder=2)
            sc = ax.scatter(
                xs, ys, c=alphas, cmap=DYNAMIC_CMAP, s=70,
                edgecolors="black", linewidths=0.6, zorder=3,
                label=f"{DYNAMIC_SCHEDULER} (alpha sweep)",
                vmin=0.0, vmax=1.0,
            )
            cbar = fig.colorbar(sc, ax=ax, pad=0.02)
            cbar.set_label("alpha", fontsize=10)
            n_seeds = int(agg["count"].max()) if not agg.empty else 1
            if n_seeds > 1:
                xerr = (agg["starvation_std"].fillna(0.0) / np.sqrt(n_seeds)).to_numpy()
                yerr = (agg["variance_std"].fillna(0.0) / np.sqrt(n_seeds)).to_numpy()
                ax.errorbar(
                    xs, ys, xerr=xerr, yerr=yerr,
                    fmt="none", ecolor="0.6", capsize=2, alpha=0.6, zorder=2,
                )

        ax.set_xlabel("Mean starvation rate (lower is better)", fontsize=12)
        ax.set_ylabel("Cross-robot starvation variance (lower is fairer)", fontsize=12)
        ax.set_title(
            f"Starvation vs fairness — scenario={scenario_id}, model={model}",
            fontsize=13, fontweight="bold",
        )
        ax.set_ylim(bottom=0.0)
        ax.set_xlim(left=0.0)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=9, frameon=False)
        plt.tight_layout()

        safe_model = model.replace(".", "_").replace("/", "_")
        out = plots_dir / f"starvation_vs_fairness__{scenario_id}__{safe_model}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Wrote {out}")


@app.local_entrypoint()
def main(
    # models: str = "pi05,gr00t-n1.7",
    models: str = "gr00t-n1.7",
    # scenarios: str = "1f9s,5f5s",
    scenarios: str = "5f20s",
    alpha_grid: str = "0.0,0.1,0.2,0.3,0.5,0.7,1.0",
    seeds: str = "42",
    output_dir: str = "experiments/sweeps/fairness_alpha_gr00t",
    port: int = 8080,
    max_steps: int = DEFAULT_MAX_STEPS,
) -> None:
    out = pathlib.Path(output_dir)
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d_%H%M%S")  # noqa: UP017
    artifacts_dir = out / "artifacts"

    model_list = _parse_csv(models)
    scenario_list = _parse_scenarios(scenarios)
    alpha_list = _parse_csv(alpha_grid, cast=float)
    seed_list = _parse_csv(seeds, cast=int)

    cases: list[SweepCase] = []
    for scenario in scenario_list:
        for model in model_list:
            for seed in seed_list:
                # baselines: one per scheduler
                for sched in BASELINE_SCHEDULERS:
                    cases.append(
                        SweepCase(
                            model=model,
                            scheduler=sched,
                            scenario_id=scenario.scenario_id,
                            n_fast=scenario.n_fast,
                            n_slow=scenario.n_slow,
                            seed=seed,
                            alpha=None,
                        )
                    )
                # dynamic-action: one per alpha
                for alpha in alpha_list:
                    cases.append(
                        SweepCase(
                            model=model,
                            scheduler=DYNAMIC_SCHEDULER,
                            scenario_id=scenario.scenario_id,
                            n_fast=scenario.n_fast,
                            n_slow=scenario.n_slow,
                            seed=seed,
                            alpha=alpha,
                        )
                    )

    print(
        f"Submitting {len(cases)} cases "
        f"(scenarios={[s.scenario_id for s in scenario_list]} "
        f"models={model_list} alphas={alpha_list} seeds={seed_list})"
    )

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
        var = result.get("starvation_variance", "")
        starv = result.get("mean_starvation", "")
        print(
            f"{result['status']}: {result['run_id']} "
            f"starvation_variance={var if var == '' else f'{float(var):.4f}'} "
            f"mean_starvation={starv if starv == '' else f'{float(starv):.4f}'}"
        )

    sweep_csv = out / f"sweep_results_{stamp}.csv"
    latest_csv = out / "sweep_results.csv"
    _write_rows(sweep_csv, rows)
    _write_rows(latest_csv, rows)
    print(f"Wrote {latest_csv}")
    print(f"Wrote {sweep_csv}")

    _plot_starvation_vs_fairness(latest_csv, out / "plots")
