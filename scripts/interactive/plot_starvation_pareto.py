"""Per-num_robots fairness/starvation Pareto plots for an interactive sweep run.

Walks ``<run_root>/<case_dir>/{case.json,result.json}`` for all ``status=ok``
cases, groups by ``num_robots``, and writes one PNG per group under
``<run_root>/plots/``:

    starvation_pareto__num_robots={N}.png    mean vs worst-robot starvation
    starvation_vs_jain__num_robots={N}.png   mean starvation vs Jain fairness

Style:
    * Constant baselines (max-batch, round-robin, greedy-deadline) render as
      single scatter points with categorical colors.
    * lookahead-actions renders as a connected gradient curve over the swept
      ``action_horizon_multiplier`` (ahm) — reads as "one method varying ahm"
      rather than several different schedulers.

Run:
    uv run python scripts/interactive/plot_starvation_pareto.py \\
        experiments/sweeps/interactive/20260523_194733
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

BASELINE_SCHEDULERS = ("max-batch", "greedy-deadline", "round-robin")
SWEPT_SCHEDULER = "lookahead-actions"

BASELINE_STYLE: dict[str, dict[str, str]] = {
    "max-batch":       {"marker": "s", "color": "#F94144"},
    "greedy-deadline": {"marker": "^", "color": "#F7B801"},
    "round-robin":     {"marker": "D", "color": "#F9C74F"},
}

# Single-hue blue gradient anchored on #37A3D2 — keeps clear contrast with
# the EDF red and RR yellow baselines.
SWEEP_CMAP = LinearSegmentedColormap.from_list(
    "lookahead_ahm_blue", ["#A8D9EC", "#154A60"]
)


def _load_run(run_root: pathlib.Path) -> pd.DataFrame:
    rows: list[dict] = []
    for case_dir in sorted(run_root.iterdir()):
        case_json = case_dir / "case.json"
        result_json = case_dir / "result.json"
        if not (case_json.is_file() and result_json.is_file()):
            continue
        try:
            case = json.loads(case_json.read_text())
            result = json.loads(result_json.read_text())
        except json.JSONDecodeError as e:
            print(f"WARN: skipping {case_dir.name}: {e}", file=sys.stderr)
            continue
        if result.get("status") != "ok":
            continue
        rows.append({
            "run_id": case["run_id"],
            "case_dir": str(case_dir),
            "scheduler": case["scheduler"],
            "num_robots": int(case["num_robots"]),
            "seed": case["seed"],
            "alpha": float(case.get("alpha", 1.0)),
            "action_horizon_multiplier": float(case.get("action_horizon_multiplier", 0.0) or 0.0),
            "mean_starvation": result.get("mean_starvation"),
            "max_starvation": result.get("max_starvation"),
            "min_starvation": result.get("min_starvation"),
            "starvation_variance": result.get("starvation_variance"),
            "jain_starvation": result.get("jain_starvation"),
            "jain_freshness": result.get("jain_freshness"),
            "success_rate": result.get("success_rate"),
        })
    if not rows:
        raise SystemExit(f"No ok cases found under {run_root}")
    return pd.DataFrame(rows)


def _per_robot_starvation(case_dir: pathlib.Path) -> dict[int, float]:
    """Aggregate ``outputs/results.csv`` to per-robot starvation rates.

    Trial mode runs multiple episodes per robot; per-robot rate is
    ``sum(starvation_steps) / sum(observed_steps)`` across all episodes for
    that ``robot_idx``. Returns an empty dict if results.csv is missing or
    malformed.
    """
    csv_path = case_dir / "outputs" / "results.csv"
    if not csv_path.is_file():
        return {}
    df = pd.read_csv(csv_path)
    needed = {"robot_idx", "starvation_steps", "observed_steps"}
    if not needed.issubset(df.columns):
        return {}
    g = df.groupby("robot_idx", as_index=True)[["starvation_steps", "observed_steps"]].sum()
    rates = (g["starvation_steps"] / g["observed_steps"].where(g["observed_steps"] > 0)).fillna(0.0)
    return {int(k): float(v) for k, v in rates.items()}


def _classify_fast_slow(case_dir: pathlib.Path) -> tuple[list[int], list[int]]:
    """Split robots into (fast_ids, slow_ids) by ``max_execution_horizon``.

    Reads ``experiment_config.json``; the smallest unique horizon = fast
    (shorter horizon ⇒ calls back more often), largest = slow. Any middle
    tier is ignored. Homogeneous workloads (one unique horizon) return
    ``([], [])`` so callers can skip the fast/slow plot for that case.
    """
    cfg_path = case_dir / "experiment_config.json"
    if not cfg_path.is_file():
        return [], []
    try:
        cfg = json.loads(cfg_path.read_text())
    except json.JSONDecodeError:
        return [], []
    by_horizon: dict[int, list[int]] = {}
    for key, robot_cfg in cfg.get("robots", {}).items():
        if not key.startswith("robot_"):
            continue
        try:
            rid = int(key.split("_")[-1])
            horizon = int(robot_cfg["max_execution_horizon"])
        except (ValueError, KeyError, TypeError):
            continue
        by_horizon.setdefault(horizon, []).append(rid)
    if len(by_horizon) < 2:
        return [], []
    horizons_sorted = sorted(by_horizon)
    return sorted(by_horizon[horizons_sorted[0]]), sorted(by_horizon[horizons_sorted[-1]])


def _augment_with_fast_slow(df: pd.DataFrame) -> pd.DataFrame:
    """Per case, compute mean starvation of fast vs slow robots.

    Adds three columns:
        - ``fast_mean_starvation`` — mean per-robot starvation rate of the fast cohort.
        - ``slow_mean_starvation`` — mean per-robot starvation rate of the slow cohort.
        - ``fast_slow_gap`` — ``|fast_mean - slow_mean|``. Lower = the two
          workload tiers are treated more equitably; identity-shift-free
          fairness measure for 2-tier workloads.

    All three are NaN for homogeneous cases or when per-robot data is missing.
    """
    df = df.copy()
    df["fast_mean_starvation"] = float("nan")
    df["slow_mean_starvation"] = float("nan")
    df["fast_slow_gap"] = float("nan")
    df["fast_robot_ids"] = ""
    df["slow_robot_ids"] = ""

    for idx, row in df.iterrows():
        case_dir = pathlib.Path(row["case_dir"])
        fast_ids, slow_ids = _classify_fast_slow(case_dir)
        if not fast_ids or not slow_ids:
            continue
        rates = _per_robot_starvation(case_dir)
        if not rates:
            continue
        fast_rates = [rates[r] for r in fast_ids if r in rates]
        slow_rates = [rates[r] for r in slow_ids if r in rates]
        fast_mean = sum(fast_rates) / len(fast_rates) if fast_rates else float("nan")
        slow_mean = sum(slow_rates) / len(slow_rates) if slow_rates else float("nan")
        df.at[idx, "fast_mean_starvation"] = fast_mean
        df.at[idx, "slow_mean_starvation"] = slow_mean
        if fast_rates and slow_rates:
            df.at[idx, "fast_slow_gap"] = abs(fast_mean - slow_mean)
        df.at[idx, "fast_robot_ids"] = ",".join(str(r) for r in fast_ids)
        df.at[idx, "slow_robot_ids"] = ",".join(str(r) for r in slow_ids)
    return df


def _plot_panel(
    ax,
    sub: pd.DataFrame,
    *,
    x_col: str,
    y_col: str,
    x_label: str,
    y_label: str,
    title: str,
    higher_y_better: bool,
) -> object | None:
    """Render baselines as scatter + the lookahead-actions ahm sweep as a curve."""
    handle = None

    for sched in BASELINE_SCHEDULERS:
        sub_b = sub[sub["scheduler"] == sched].dropna(subset=[x_col, y_col])
        if sub_b.empty:
            continue
        # Baselines should be a single point; if multiple seeds/configs share a
        # scheduler, take the mean so the plot stays uncluttered.
        x = float(sub_b[x_col].mean())
        y = float(sub_b[y_col].mean())
        style = BASELINE_STYLE[sched]
        ax.scatter(
            x, y,
            marker=style["marker"], s=140, color=style["color"],
            edgecolors="black", linewidths=0.7,
            label=sched, zorder=4,
        )

    sub_s = (
        sub[sub["scheduler"] == SWEPT_SCHEDULER]
        .dropna(subset=[x_col, y_col])
        .sort_values("action_horizon_multiplier")
    )
    if not sub_s.empty:
        xs = sub_s[x_col].to_numpy()
        ys = sub_s[y_col].to_numpy()
        ahms = sub_s["action_horizon_multiplier"].to_numpy()
        ax.plot(xs, ys, "-", color="#7a7a7a", linewidth=1.2, alpha=0.55, zorder=2)
        vmin = float(ahms.min()) if len(ahms) else 1.0
        vmax = float(ahms.max()) if len(ahms) else 1.0
        if vmax == vmin:
            vmax = vmin + 1.0  # so the colormap has a non-trivial range
        handle = ax.scatter(
            xs, ys,
            c=ahms, cmap=SWEEP_CMAP, s=95,
            edgecolors="black", linewidths=0.6,
            label=f"{SWEPT_SCHEDULER} (ahm sweep)", zorder=3,
            vmin=vmin, vmax=vmax,
        )
        # Annotate each lookahead-actions point with its ahm value, slightly
        # offset so labels don't sit on the marker.
        for x, y, ahm in zip(xs, ys, ahms, strict=False):
            ax.annotate(
                f"ahm={ahm:g}",
                xy=(x, y), xytext=(6, 6),
                textcoords="offset points",
                fontsize=7, color="#444",
            )

    ax.set_xlabel(x_label, fontsize=11)
    ax.set_ylabel(y_label, fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8, frameon=False)

    # Pad both axes around the data so annotations / markers stay inside.
    s = sub.dropna(subset=[x_col, y_col])
    if not s.empty:
        xs = s[x_col].to_numpy(dtype=float)
        ys = s[y_col].to_numpy(dtype=float)
        xpad = max((xs.max() - xs.min()) * 0.15, 1e-6)
        ypad = max((ys.max() - ys.min()) * 0.15, 1e-6)
        ax.set_xlim(xs.min() - xpad, xs.max() + xpad)
        if higher_y_better:
            ax.set_ylim(ys.min() - ypad, min(1.02, ys.max() + ypad))
        else:
            ax.set_ylim(ys.min() - ypad, ys.max() + ypad)
    return handle


def _save_one_plot(
    df: pd.DataFrame,
    plots_dir: pathlib.Path,
    *,
    name_prefix: str,
    x_col: str = "mean_starvation",
    x_label: str = "Mean starvation rate  (lower is better)",
    y_col: str,
    y_label: str,
    panel_title: str,
    higher_y_better: bool,
    cbar_label: str = "action_horizon_multiplier",
) -> None:
    needed = {x_col, y_col, "scheduler", "num_robots"}
    if df.empty or not needed.issubset(df.columns):
        print(f"WARN: missing required columns for {name_prefix}; skipping", file=sys.stderr)
        return
    plots_dir.mkdir(parents=True, exist_ok=True)
    for num_robots, sub in df.groupby("num_robots"):
        sub = sub.dropna(subset=[x_col, y_col])
        if sub.empty:
            continue
        fig, ax = plt.subplots(figsize=(8, 6))
        handle = _plot_panel(
            ax, sub,
            x_col=x_col,
            y_col=y_col,
            x_label=x_label,
            y_label=y_label,
            title=panel_title,
            higher_y_better=higher_y_better,
        )
        if handle is not None:
            cbar = fig.colorbar(handle, ax=ax, pad=0.02)
            cbar.set_label(cbar_label, fontsize=10)
        fig.suptitle(f"num_robots={num_robots}", fontsize=13, fontweight="bold")
        plt.tight_layout()
        out = plots_dir / f"{name_prefix}__num_robots={num_robots}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
        plt.close(fig)
        print(f"Wrote {out}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "run_root",
        type=pathlib.Path,
        help="Sweep run root (e.g. experiments/sweeps/interactive/20260523_194733)",
    )
    parser.add_argument(
        "--out-dir",
        type=pathlib.Path,
        default=None,
        help="Override output directory (defaults to <run_root>/plots/)",
    )
    args = parser.parse_args()

    run_root = args.run_root.resolve()
    if not run_root.is_dir():
        raise SystemExit(f"{run_root} is not a directory")

    df = _load_run(run_root)
    print(
        f"Loaded {len(df)} ok cases: "
        f"schedulers={sorted(df['scheduler'].unique())}, "
        f"num_robots={sorted(df['num_robots'].unique())}"
    )
    df = _augment_with_fast_slow(df)

    plots_dir = args.out_dir or (run_root / "plots")

    _save_one_plot(
        df, plots_dir,
        name_prefix="starvation_pareto",
        y_col="max_starvation",
        y_label="Worst-robot starvation rate  (lower is fairer)",
        panel_title="Pareto: mean vs worst-robot starvation",
        higher_y_better=False,
    )
    _save_one_plot(
        df, plots_dir,
        name_prefix="starvation_vs_jain",
        y_col="jain_starvation",
        y_label="Jain fairness on starvation  (higher is fairer)",
        panel_title="Mean starvation vs Jain fairness",
        higher_y_better=True,
    )
    # Identity-free fairness for 2-tier workloads: ``|mean(fast) - mean(slow)|``.
    # Doesn't shift as the worst-robot identity flips between cases — captures
    # "does the scheduler equalize outcomes across workload tiers". Homogeneous
    # runs have no fast/slow split and are skipped (NaN).
    _save_one_plot(
        df, plots_dir,
        name_prefix="starvation_vs_gap",
        y_col="fast_slow_gap",
        y_label="|fast_mean − slow_mean| starvation  (lower is fairer)",
        panel_title="Mean starvation vs fast/slow tier gap",
        higher_y_better=False,
    )
    # Fast vs slow: robots are classified per case by ``max_execution_horizon``
    # in experiment_config.json (shortest = fast, longest = slow). Each axis is
    # the mean per-robot starvation rate within that cohort. Homogeneous cases
    # have no fast/slow split and are skipped automatically (NaN).
    _save_one_plot(
        df, plots_dir,
        name_prefix="fast_vs_slow_starvation",
        x_col="fast_mean_starvation",
        x_label="Fast-robot mean starvation rate  (lower is better)",
        y_col="slow_mean_starvation",
        y_label="Slow-robot mean starvation rate  (lower is better)",
        panel_title="Fast vs slow robot starvation",
        higher_y_better=False,
    )

    print(f"\nAll plots written under {plots_dir}")


if __name__ == "__main__":
    main()
