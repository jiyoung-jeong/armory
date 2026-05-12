"""Post-process a fairness sweep produced by scripts/modal_fairness_sweep.py.

Walks the sweep's artifact directory, recomputes per-cell fairness metrics
(Jain's freshness, Jain's starvation, cross-robot starvation variance) from
each run's per-robot starvation rates, writes a richer CSV, and emits two
plots per model: Jain's vs n_fast and starvation variance vs n_fast.

Example:
    uv run python scripts/plot_fairness_sweep.py \
        --sweep-dir experiments/sweeps/fairness_het_2
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "src/backends"))
sys.path.insert(0, str(REPO_ROOT / "packages/armory-client/src"))

from sims.libero.metrics import compute_fairness_metrics  # noqa: E402


def _find_output_dir(artifact_dir: pathlib.Path) -> pathlib.Path | None:
    """Locate the run's output directory inside an artifact tree.

    Modal sweep tarballs nest as <run_id>/<run_id>/output/, but some local
    layouts collapse to <run_id>/output/ — handle both.
    """
    for candidate in (
        artifact_dir / "output",
        artifact_dir / artifact_dir.name / "output",
    ):
        if candidate.exists():
            return candidate
    return None


def _per_cell_metrics(
    output_dir: pathlib.Path, n_fast: int
) -> dict[str, Any] | None:
    fairness = compute_fairness_metrics(output_dir)
    if fairness is None:
        return None
    rates = fairness["starvation_rate"]
    robot_idx = fairness.get("robot_idx") or list(range(len(rates)))
    if not rates:
        return None
    rates_arr = np.asarray(rates, dtype=float)
    # Convention from _compute_horizons: robots [0..n_fast-1] are fast, rest are slow.
    fast_mask = np.asarray([int(i) < n_fast for i in robot_idx])
    fast_rates = rates_arr[fast_mask]
    slow_rates = rates_arr[~fast_mask]
    return {
        "alpha": fairness.get("alpha"),
        "jain_freshness": float(fairness["jain_freshness"]),
        "jain_starvation": float(fairness["jain_starvation"]),
        "starvation_var": float(np.var(rates_arr)),
        "starvation_std": float(np.std(rates_arr)),
        "starvation_max_minus_min": float(rates_arr.max() - rates_arr.min()),
        "starvation_mean": float(rates_arr.mean()),
        "starvation_max": float(rates_arr.max()),
        "starvation_min": float(rates_arr.min()),
        "fast_cohort_mean_starvation": float(fast_rates.mean()) if fast_rates.size else float("nan"),
        "slow_cohort_mean_starvation": float(slow_rates.mean()) if slow_rates.size else float("nan"),
        "n_robots_observed": int(rates_arr.size),
    }


def _load_case(artifact_dir: pathlib.Path) -> dict[str, Any] | None:
    case_path = artifact_dir / "case.json"
    nested = artifact_dir / artifact_dir.name / "case.json"
    if case_path.exists():
        return json.loads(case_path.read_text())
    if nested.exists():
        return json.loads(nested.read_text())
    return None


def collect_metrics(sweep_dir: pathlib.Path) -> pd.DataFrame:
    artifacts_dir = sweep_dir / "artifacts"
    rows: list[dict[str, Any]] = []
    for artifact in sorted(artifacts_dir.iterdir()):
        if not artifact.is_dir():
            continue
        output_dir = _find_output_dir(artifact)
        case = _load_case(artifact)
        if output_dir is None or case is None:
            print(f"[skip] {artifact.name}: missing output/ or case.json")
            continue
        metrics = _per_cell_metrics(output_dir, int(case.get("n_fast", 0)))
        if metrics is None:
            print(f"[skip] {artifact.name}: no fairness metrics (run may have failed)")
            continue
        rows.append({**case, **metrics, "run_id": artifact.name})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.sort_values(["model", "scheduler", "n_fast", "seed"]).reset_index(drop=True)
    return df


def _plot_metric_vs_nfast(
    df: pd.DataFrame,
    metric: str,
    *,
    ylabel: str,
    title_prefix: str,
    out_path: pathlib.Path,
    ylim: tuple[float, float] | None = None,
    legend_loc: str = "best",
) -> None:
    schedulers = sorted(df["scheduler"].unique())
    color_cycle = plt.cm.tab10(np.linspace(0, 1, max(len(schedulers), 2)))

    fig, ax = plt.subplots(figsize=(9, 5))
    for color, sched in zip(color_cycle, schedulers):
        sub = df[df["scheduler"] == sched]
        if sub.empty:
            continue
        agg = (
            sub.groupby("n_fast")[metric]
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
            label=sched,
        )
    ax.set_xlabel("Heterogeneity (n_fast out of 15)", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title_prefix, fontsize=13, fontweight="bold")
    if ylim is not None:
        ax.set_ylim(*ylim)
    n_max = int(df["n_fast"].max()) if not df.empty else 15
    ax.set_xlim(-0.5, n_max + 0.5)
    ax.grid(True, alpha=0.3)
    ax.legend(loc=legend_loc, fontsize=9, frameon=False)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


def _plot_cohort_starvation(df: pd.DataFrame, model: str, out_path: pathlib.Path) -> None:
    """Two-panel plot: fast-cohort starvation (top) and slow-cohort starvation (bottom)."""
    sub_model = df[df["model"] == model]
    if sub_model.empty:
        return
    schedulers = sorted(sub_model["scheduler"].unique())
    color_cycle = plt.cm.tab10(np.linspace(0, 1, max(len(schedulers), 2)))

    fig, (ax_fast, ax_slow) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    for color, sched in zip(color_cycle, schedulers):
        sub = sub_model[sub_model["scheduler"] == sched]
        for ax, col in ((ax_fast, "fast_cohort_mean_starvation"), (ax_slow, "slow_cohort_mean_starvation")):
            cell = sub.dropna(subset=[col])
            if cell.empty:
                continue
            agg = (
                cell.groupby("n_fast")[col]
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
                label=sched,
            )

    n_max = int(sub_model["n_fast"].max())
    for ax, title in ((ax_fast, "Fast cohort (horizon=4)"), (ax_slow, "Slow cohort (horizon=10)")):
        ax.set_ylabel("Mean starvation rate", fontsize=11)
        ax.set_xlim(-0.5, n_max + 0.5)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        ax.set_title(title, fontsize=11)
    ax_slow.set_xlabel("Heterogeneity (n_fast out of 15)", fontsize=12)
    ax_fast.legend(loc="upper left", fontsize=9, frameon=False)
    fig.suptitle(
        f"Trade-off: fast vs slow cohort starvation — {model}",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout(rect=(0, 0, 1, 0.97))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


def make_plots(df: pd.DataFrame, plots_dir: pathlib.Path) -> None:
    if df.empty:
        print("No data to plot")
        return
    plots_dir.mkdir(parents=True, exist_ok=True)
    for model in sorted(df["model"].unique()):
        model_slug = model.replace(".", "_").replace("/", "_")
        sub = df[df["model"] == model]
        _plot_metric_vs_nfast(
            sub,
            "jain_freshness",
            ylabel="Jain's index on freshness rate",
            title_prefix=f"Fairness vs heterogeneity — {model}",
            out_path=plots_dir / f"jains_vs_het__{model_slug}.png",
            ylim=(0, 1.02),
            legend_loc="lower left",
        )
        _plot_metric_vs_nfast(
            sub,
            "starvation_var",
            ylabel="Cross-robot variance of starvation rate",
            title_prefix=f"Starvation variance vs heterogeneity — {model}",
            out_path=plots_dir / f"starvation_var_vs_het__{model_slug}.png",
            legend_loc="upper left",
        )
        _plot_metric_vs_nfast(
            sub,
            "starvation_mean",
            ylabel="Mean starvation rate (all 15 robots)",
            title_prefix=f"Aggregate starvation vs heterogeneity — {model}",
            out_path=plots_dir / f"mean_starvation_vs_het__{model_slug}.png",
            ylim=(0, 1),
            legend_loc="upper left",
        )
        _plot_cohort_starvation(
            df,
            model,
            plots_dir / f"cohort_starvation_vs_het__{model_slug}.png",
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sweep-dir",
        type=pathlib.Path,
        required=True,
        help="Sweep directory (must contain artifacts/ subdir).",
    )
    parser.add_argument(
        "--output-csv",
        type=pathlib.Path,
        default=None,
        help="Output CSV path (default: <sweep-dir>/sweep_metrics_recomputed.csv).",
    )
    parser.add_argument(
        "--plots-dir",
        type=pathlib.Path,
        default=None,
        help="Output plots dir (default: <sweep-dir>/plots).",
    )
    args = parser.parse_args()

    sweep_dir = args.sweep_dir
    if not (sweep_dir / "artifacts").exists():
        raise SystemExit(f"No artifacts/ subdir in {sweep_dir}")

    output_csv = args.output_csv or (sweep_dir / "sweep_metrics_recomputed.csv")
    plots_dir = args.plots_dir or (sweep_dir / "plots")

    df = collect_metrics(sweep_dir)
    if df.empty:
        raise SystemExit("No usable runs found.")
    df.to_csv(output_csv, index=False)
    print(f"Wrote {output_csv} ({len(df)} rows)")

    make_plots(df, plots_dir)


if __name__ == "__main__":
    main()
