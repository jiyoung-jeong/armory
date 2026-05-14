"""Sweep alpha for dynamic-action vs baselines, on a list of (n_fast, n_slow) scenarios.

Uses the MOCK setup: mock policy server + mock client colocated in one CPU
container per case.

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
import json
import pathlib
import re
import subprocess
import sys
from typing import Any

_HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))  # _setups
sys.path.insert(0, str(_HERE.parent))  # serve, run_libero

import run_libero  # noqa: E402
import serve  # noqa: E402
from _setups import ARTIFACTS_VOLUME_NAME, MOCK, Case, app  # noqa: E402

MAX_BATCH_SIZE = 5
FAST_HORIZON = 4
SLOW_HORIZON = 10
DEFAULT_MAX_STEPS = 200
DEFAULT_TRIALS_PER_ROBOT = 1

BASELINE_SCHEDULERS = ("fixed-max-batch", "greedy-deadline", "round-robin", "lookahead-actions")
DYNAMIC_SCHEDULER = "dynamic-action"

MODEL_TO_PROFILE = {
    "pi05": "l40s_pi05",
    "gr00t-n1.7": "l40s_gr00t",
}
# serve.ModelFamily member names keyed by the sweep's model token.
MODEL_TO_ENUM_NAME = {
    "pi05": "PI05",
    "gr00t-n1.7": "GROOT_N17",
}

BASELINE_STYLE = {
    "fixed-max-batch": {"marker": "s", "color": "#1f77b4"},
    "greedy-deadline": {"marker": "^", "color": "#2ca02c"},
    "round-robin": {"marker": "D", "color": "#d62728"},
    "lookahead-actions": {"marker": "o", "color": "#9467bd"},
}
DYNAMIC_CMAP = "viridis"


@dataclasses.dataclass(frozen=True)
class Scenario:
    """A heterogeneity scenario: n_fast fast robots + n_slow slow robots."""

    n_fast: int
    n_slow: int

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
    # alpha is only meaningful for dynamic-action; None for the baselines.
    alpha: float | None

    @property
    def run_id(self) -> str:
        alpha_part = "" if self.alpha is None else f"__alpha={self.alpha:.3f}"
        return (
            f"model={self.model}__scheduler={self.scheduler}"
            f"__scenario={self.scenario_id}{alpha_part}__seed={self.seed}"
        )

    def horizons(self) -> list[int]:
        return [FAST_HORIZON] * self.n_fast + [SLOW_HORIZON] * self.n_slow


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


def _build_case(case: SweepCase, *, output_dir: pathlib.Path, max_steps: int) -> Case:
    """Resolve a SweepCase into the serve.Args + run_libero.Args the setup runs."""
    horizons = case.horizons()
    exp_cfg = _build_experiment_config(horizons, max_steps)
    run_dir = output_dir / "runs" / case.run_id

    server_args = serve.Args(
        model=serve.ModelFamily[MODEL_TO_ENUM_NAME[case.model]],
        max_batch_size=MAX_BATCH_SIZE,
        scheduling_algorithm=case.scheduler,
        alpha=case.alpha if case.alpha is not None else serve.Args.alpha,
        policy=serve.Mock(action_horizon=10, action_dim=7, profile=MODEL_TO_PROFILE[case.model]),
        log_dir=str(run_dir / "server_logs"),
    )
    client_args = run_libero.Args(
        env="mock",
        overwrite=True,
        progress_type="logging",
        max_steps=max_steps,
        seed=case.seed,
        output_dir=run_dir / "output",
        log_dir=run_dir / "client_logs",
    )
    return Case(
        run_id=case.run_id,
        run_dir=run_dir,
        server_args=server_args,
        client_args=client_args,
        experiment_config=exp_cfg,
    )


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


def _plot_one_yaxis(ax, sub, *, y_col: str, y_label: str, title: str):
    """Scatter (mean_starvation, y_col) onto ``ax`` with baselines + dynamic-action curve.

    Returns the dynamic-action scatter handle (for a shared colorbar) or None.
    """
    import numpy as np  # noqa: PLC0415

    handle = None
    for sched in BASELINE_SCHEDULERS:
        sub_b = sub[sub["scheduler"] == sched].dropna(subset=["mean_starvation", y_col])
        if sub_b.empty:
            continue
        x = float(sub_b["mean_starvation"].mean())
        y = float(sub_b[y_col].mean())
        xerr = float(sub_b["mean_starvation"].std(ddof=0)) if len(sub_b) > 1 else 0.0
        yerr = float(sub_b[y_col].std(ddof=0)) if len(sub_b) > 1 else 0.0
        style = BASELINE_STYLE[sched]
        ax.errorbar(
            x,
            y,
            xerr=xerr,
            yerr=yerr,
            marker=style["marker"],
            markersize=11,
            color=style["color"],
            linestyle="none",
            capsize=3,
            label=sched,
            zorder=4,
        )

    sub_d = sub[sub["scheduler"] == DYNAMIC_SCHEDULER].dropna(subset=["mean_starvation", y_col])
    if not sub_d.empty:
        agg = (
            sub_d.groupby("alpha_requested")
            .agg(
                mean_starvation=("mean_starvation", "mean"),
                y_mean=(y_col, "mean"),
                starvation_std=("mean_starvation", "std"),
                y_std=(y_col, "std"),
                count=("seed", "count"),
            )
            .reset_index()
            .sort_values("alpha_requested")
        )
        xs = agg["mean_starvation"].to_numpy()
        ys = agg["y_mean"].to_numpy()
        alphas = agg["alpha_requested"].to_numpy()
        ax.plot(xs, ys, "-", color="0.5", linewidth=1.2, alpha=0.7, zorder=2)
        handle = ax.scatter(
            xs,
            ys,
            c=alphas,
            cmap=DYNAMIC_CMAP,
            s=70,
            edgecolors="black",
            linewidths=0.6,
            zorder=3,
            label=f"{DYNAMIC_SCHEDULER} (alpha sweep)",
            vmin=0.0,
            vmax=1.0,
        )
        n_seeds = int(agg["count"].max()) if not agg.empty else 1
        if n_seeds > 1:
            xerr = (agg["starvation_std"].fillna(0.0) / np.sqrt(n_seeds)).to_numpy()
            yerr = (agg["y_std"].fillna(0.0) / np.sqrt(n_seeds)).to_numpy()
            ax.errorbar(
                xs,
                ys,
                xerr=xerr,
                yerr=yerr,
                fmt="none",
                ecolor="0.6",
                capsize=2,
                alpha=0.6,
                zorder=2,
            )

    ax.set_xlabel("Mean starvation rate (lower is better)", fontsize=11)
    ax.set_ylabel(y_label, fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8, frameon=False)
    return handle


def _autoscale_with_pad(ax, sub, y_col: str, pad_frac: float = 0.12) -> None:
    """Tighten x/y limits to the data range with a fractional padding."""
    s = sub.dropna(subset=["mean_starvation", y_col])
    if s.empty:
        return
    xs = s["mean_starvation"].to_numpy(dtype=float)
    ys = s[y_col].to_numpy(dtype=float)
    x_lo, x_hi = float(xs.min()), float(xs.max())
    y_lo, y_hi = float(ys.min()), float(ys.max())
    x_pad = max((x_hi - x_lo) * pad_frac, 1e-6)
    y_pad = max((y_hi - y_lo) * pad_frac, 1e-6)
    ax.set_xlim(x_lo - x_pad, x_hi + x_pad)
    ax.set_ylim(y_lo - y_pad, y_hi + y_pad)


def _save_single_panel_plot(
    results_csv: pathlib.Path,
    plots_dir: pathlib.Path,
    *,
    name_prefix: str,
    y_col: str,
    y_label: str,
    panel_title: str,
    pareto: bool,
) -> None:
    """Render one PNG per (scenario, model) for the given y_col.

    ``pareto=True`` auto-scales axes to the data range with padding; otherwise the
    axes are anchored at zero so a fixed-scale "lower is better" plot is easy to
    compare across scenarios.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    df = pd.read_csv(results_csv)
    if df.empty:
        return
    df = df[df.get("status", "ok") == "ok"]
    needed = {"mean_starvation", y_col, "scheduler", "model", "scenario_id"}
    if not needed.issubset(df.columns):
        return
    df = df.dropna(subset=["mean_starvation", y_col])

    plots_dir.mkdir(parents=True, exist_ok=True)
    for (scenario_id, model), sub in df.groupby(["scenario_id", "model"]):
        fig, ax = plt.subplots(figsize=(8, 6))
        handle = _plot_one_yaxis(
            ax,
            sub,
            y_col=y_col,
            y_label=y_label,
            title=panel_title,
        )
        if pareto:
            _autoscale_with_pad(ax, sub, y_col)
        else:
            ax.set_ylim(bottom=0.0)
            ax.set_xlim(left=0.0)
        if handle is not None:
            cbar = fig.colorbar(handle, ax=ax, pad=0.02)
            cbar.set_label("scheduler alpha", fontsize=10)
        fig.suptitle(
            f"scenario={scenario_id}, model={model}",
            fontsize=13,
            fontweight="bold",
        )
        plt.tight_layout()
        safe_model = model.replace(".", "_").replace("/", "_")
        out = plots_dir / f"{name_prefix}__{scenario_id}__{safe_model}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Wrote {out}")


def _plot_starvation_pareto(results_csv: pathlib.Path, plots_dir: pathlib.Path) -> None:
    """Pareto plot: mean vs worst-robot (max) starvation. Bottom-left = best.

    Baselines render as fixed points; dynamic-action sweeps a Pareto frontier
    between utilitarian (low mean, high max) and Rawlsian (slightly higher mean,
    low max).
    """
    _save_single_panel_plot(
        results_csv,
        plots_dir,
        name_prefix="starvation_pareto",
        y_col="max_starvation",
        y_label="Worst-robot starvation rate  (lower is fairer)",
        panel_title="Pareto: mean vs worst-robot starvation",
        pareto=True,
    )


def _plot_starvation_vs_variance(results_csv: pathlib.Path, plots_dir: pathlib.Path) -> None:
    """Mean starvation vs cross-robot starvation variance. Bottom-left = best."""
    _save_single_panel_plot(
        results_csv,
        plots_dir,
        name_prefix="starvation_vs_variance",
        y_col="starvation_variance",
        y_label="Cross-robot starvation variance (lower is fairer)",
        panel_title="Mean starvation vs cross-robot variance",
        pareto=False,
    )


@app.local_entrypoint()
def main(
    models: str = "pi05",
    scenarios: str = "1f9s,5f5s",
    alpha_grid: str = "0.0,0.25,0.5,0.75,1.0",
    seeds: str = "1",
    output_dir: str = "experiments/sweeps/fairness_alpha_sweep_pi05_test",
    max_steps: int = DEFAULT_MAX_STEPS,
) -> None:
    out = pathlib.Path(output_dir)
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d_%H%M%S")  # noqa: UP017

    model_list = _parse_csv(models)
    scenario_list = _parse_scenarios(scenarios)
    alpha_list = _parse_csv(alpha_grid, cast=float)
    seed_list = _parse_csv(seeds, cast=int)

    cases: list[SweepCase] = []
    for scenario in scenario_list:
        for model in model_list:
            for seed in seed_list:
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

    built = [_build_case(case, output_dir=out, max_steps=max_steps) for case in cases]
    case_meta = {
        case.run_id: {
            "run_id": case.run_id,
            "model": case.model,
            "scheduler": case.scheduler,
            "scenario_id": case.scenario_id,
            "n_fast": case.n_fast,
            "n_slow": case.n_slow,
            "n_total": case.n_fast + case.n_slow,
            "alpha_requested": case.alpha,
            "seed": case.seed,
            "horizons": json.dumps(case.horizons()),
        }
        for case in cases
    }

    rows: list[dict[str, Any]] = []
    for result in MOCK.run(built, stamp=stamp, cpu=10, memory=16384):
        row = {**case_meta[result["run_id"]], **result}
        rows.append(row)
        max_s = row.get("max_starvation", "")
        mean_s = row.get("mean_starvation", "")
        print(
            f"{row['status']}: {row['run_id']} "
            f"max_starvation={max_s if max_s == '' else f'{float(max_s):.4f}'} "
            f"mean_starvation={mean_s if mean_s == '' else f'{float(mean_s):.4f}'}"
        )

    artifacts_dir = out / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading artifacts from volume '{ARTIFACTS_VOLUME_NAME}/{stamp}' -> {artifacts_dir}")
    subprocess.run(
        ["modal", "volume", "get", ARTIFACTS_VOLUME_NAME, stamp, str(artifacts_dir), "--force"],
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

    _plot_starvation_pareto(latest_csv, out / "plots")
    _plot_starvation_vs_variance(latest_csv, out / "plots")
