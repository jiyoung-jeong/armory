"""Per-seed plotter for fairness-alpha sweeps.

Unlike ``modal_alpha_fairness_sweep.py``'s built-in plots (which average
across seeds), this writes one figure per (scenario, model, seed).

Color choices:
    * baselines render as fixed scatter points with categorical colors
    * dynamic-action renders as a connected curve whose points share a
      single-hue gradient — readable as "one method varying alpha" rather
      than the multi-hue viridis cmap which made each point look like a
      different scheduler.

Run:
    python scripts/experiments/plot_alpha_fairness_per_seed.py \
        experiments/sweeps/fairness_alpha_sweep_pi05_1f9s_5f5s_3seeds
"""

from __future__ import annotations

import argparse
import datetime as dt
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

# The metrics module lives under src/sims/libero, with sibling backends and
# the armory-client workspace on its sys.path; replicate that layout so this
# standalone script can import it.
_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
for _p in (
    _REPO_ROOT / "src",
    _REPO_ROOT / "src/backends",
    _REPO_ROOT / "armory-client/src",
):
    sys.path.insert(0, str(_p))

# from sims.libero.metrics import compute_goodput_metrics  # noqa: E402

BASELINE_SCHEDULERS = ("max-batch", "greedy-deadline", "round-robin", "lookahead-actions")
DYNAMIC_SCHEDULER = "dynamic-action"

# Categorical palette for baselines — saturated, mutually distinguishable,
# and chromatically separate from the dynamic-action teal gradient below.
BASELINE_STYLE = {
    "max-batch": {"marker": "s", "color": "#1f77b4"},  # blue
    "greedy-deadline": {"marker": "^", "color": "#2ca02c"},  # green
    "round-robin": {"marker": "D", "color": "#d62728"},  # red
    "lookahead-actions": {"marker": "o", "color": "#ff7f0e"},  # orange
}

# Single-hue teal gradient: low alpha = pale, high alpha = deep teal. Reads
# as "the same method shaded by alpha" instead of "four different methods".
DYNAMIC_CMAP = LinearSegmentedColormap.from_list("dynamic_alpha_teal", ["#a8e0dc", "#0c4f4f"])


def _plot_one(ax, sub: pd.DataFrame, *, y_col: str, y_label: str, title: str):
    """Render baselines + dynamic-action curve onto ``ax`` for one (scenario, model, seed)."""
    handle = None

    for sched in BASELINE_SCHEDULERS:
        sub_b = sub[sub["scheduler"] == sched].dropna(subset=["mean_starvation", y_col])
        if sub_b.empty:
            continue
        x = float(sub_b["mean_starvation"].iloc[0])
        y = float(sub_b[y_col].iloc[0])
        style = BASELINE_STYLE[sched]
        ax.scatter(
            x,
            y,
            marker=style["marker"],
            s=130,
            color=style["color"],
            edgecolors="black",
            linewidths=0.7,
            label=sched,
            zorder=4,
        )

    sub_d = (
        sub[sub["scheduler"] == DYNAMIC_SCHEDULER]
        .dropna(subset=["mean_starvation", y_col])
        .sort_values("alpha_requested")
    )
    if not sub_d.empty:
        xs = sub_d["mean_starvation"].to_numpy()
        ys = sub_d[y_col].to_numpy()
        alphas = sub_d["alpha_requested"].to_numpy()
        # Connecting line in muted grey so the gradient on the points stands out.
        ax.plot(xs, ys, "-", color="#7a7a7a", linewidth=1.2, alpha=0.55, zorder=2)
        handle = ax.scatter(
            xs,
            ys,
            c=alphas,
            cmap=DYNAMIC_CMAP,
            s=85,
            edgecolors="black",
            linewidths=0.6,
            label=f"{DYNAMIC_SCHEDULER} (alpha sweep)",
            zorder=3,
            vmin=0.0,
            vmax=1.0,
        )

    ax.set_xlabel("Mean starvation rate  (lower is better)", fontsize=11)
    ax.set_ylabel(y_label, fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8, frameon=False)
    return handle


def _autoscale_with_pad(ax, sub: pd.DataFrame, y_col: str, pad_frac: float = 0.12) -> None:
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


def _save_plot(
    df: pd.DataFrame,
    plots_dir: pathlib.Path,
    *,
    name_prefix: str,
    y_col: str,
    y_label: str,
    panel_title: str,
    pareto: bool,
) -> None:
    needed = {"mean_starvation", y_col, "scheduler", "model", "scenario_id", "seed"}
    if df.empty or not needed.issubset(df.columns):
        return
    df = df.dropna(subset=["mean_starvation", y_col])
    plots_dir.mkdir(parents=True, exist_ok=True)
    for (scenario_id, model, seed), sub in df.groupby(["scenario_id", "model", "seed"]):
        fig, ax = plt.subplots(figsize=(8, 6))
        handle = _plot_one(ax, sub, y_col=y_col, y_label=y_label, title=panel_title)
        if pareto:
            _autoscale_with_pad(ax, sub, y_col)
        else:
            ax.set_ylim(bottom=0.0)
            ax.set_xlim(left=0.0)
        if handle is not None:
            cbar = fig.colorbar(handle, ax=ax, pad=0.02)
            cbar.set_label("scheduler alpha", fontsize=10)
        fig.suptitle(
            f"scenario={scenario_id}, model={model}, seed={seed}",
            fontsize=13,
            fontweight="bold",
        )
        plt.tight_layout()
        safe_model = str(model).replace(".", "_").replace("/", "_")
        out = plots_dir / f"{name_prefix}__{scenario_id}__{safe_model}__seed{seed}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Wrote {out}")


def plot_starvation_pareto(df: pd.DataFrame, plots_dir: pathlib.Path) -> None:
    """Pareto plot: mean vs worst-robot (max) starvation. Bottom-left = best."""
    _save_plot(
        df,
        plots_dir,
        name_prefix="starvation_pareto",
        y_col="max_starvation",
        y_label="Worst-robot starvation rate  (lower is fairer)",
        panel_title="Pareto: mean vs worst-robot starvation",
        pareto=True,
    )


def plot_starvation_vs_variance(df: pd.DataFrame, plots_dir: pathlib.Path) -> None:
    """Mean starvation vs cross-robot starvation variance. Bottom-left = best."""
    _save_plot(
        df,
        plots_dir,
        name_prefix="starvation_vs_variance",
        y_col="starvation_variance",
        y_label="Cross-robot starvation variance  (lower is fairer)",
        panel_title="Mean starvation vs cross-robot variance",
        pareto=False,
    )


# def _augment_with_goodput(df: pd.DataFrame) -> pd.DataFrame:
#     """Add throughput / goodput / goodput_ratio columns by reading each artifact dir.

#     The CSV's ``artifact_path`` points at ``<sweep>/artifacts/<run_id>``; the
#     server history lives one level deeper in ``<run_id>/output``. We tolerate
#     both layouts: a direct ``output/`` child or a single nested run dir
#     containing one.
#     """
#     if "artifact_path" not in df.columns:
#         print("WARNING: 'artifact_path' missing from CSV — skipping goodput augmentation", file=sys.stderr)
#         return df

#     cache: dict[str, dict | None] = {}
#     rows: list[dict | None] = []
#     for path_str in df["artifact_path"].fillna(""):
#         if not path_str:
#             rows.append(None)
#             continue
#         if path_str in cache:
#             rows.append(cache[path_str])
#             continue
#         artifact_dir = pathlib.Path(path_str)
#         # Find the output dir (server_metrics_history.json sits inside it).
#         candidate_outputs = list(artifact_dir.glob("**/output"))
#         # Prefer the shallowest match.
#         candidate_outputs.sort(key=lambda p: len(p.parts))
#         metrics: dict | None = None
#         for out_dir in candidate_outputs:
#             metrics = compute_goodput_metrics(out_dir)
#             if metrics is not None:
#                 break
#         cache[path_str] = metrics
#         rows.append(metrics)

#     df = df.copy()
#     df["throughput_actions_per_s"] = [r["throughput_actions_per_s"] if r else None for r in rows]
#     df["goodput_actions_per_s"] = [r["goodput_actions_per_s"] if r else None for r in rows]
#     df["goodput_ratio"] = [r["goodput_ratio"] if r else None for r in rows]
#     return df


# def _plot_throughput_vs_goodput_one(ax, sub: pd.DataFrame, *, title: str):
#     """Render baselines + dynamic-action curve on (throughput, goodput) axes."""
#     handle = None
#     for sched in BASELINE_SCHEDULERS:
#         sub_b = sub[sub["scheduler"] == sched].dropna(
#             subset=["throughput_actions_per_s", "goodput_actions_per_s"]
#         )
#         if sub_b.empty:
#             continue
#         x = float(sub_b["throughput_actions_per_s"].iloc[0])
#         y = float(sub_b["goodput_actions_per_s"].iloc[0])
#         style = BASELINE_STYLE[sched]
#         ax.scatter(
#             x, y, marker=style["marker"], s=130, color=style["color"],
#             edgecolors="black", linewidths=0.7, label=sched, zorder=4,
#         )

#     sub_d = (
#         sub[sub["scheduler"] == DYNAMIC_SCHEDULER]
#         .dropna(subset=["throughput_actions_per_s", "goodput_actions_per_s"])
#         .sort_values("alpha_requested")
#     )
#     if not sub_d.empty:
#         xs = sub_d["throughput_actions_per_s"].to_numpy()
#         ys = sub_d["goodput_actions_per_s"].to_numpy()
#         alphas = sub_d["alpha_requested"].to_numpy()
#         ax.plot(xs, ys, "-", color="#7a7a7a", linewidth=1.2, alpha=0.55, zorder=2)
#         handle = ax.scatter(
#             xs, ys, c=alphas, cmap=DYNAMIC_CMAP, s=85,
#             edgecolors="black", linewidths=0.6, zorder=3,
#             label=f"{DYNAMIC_SCHEDULER} (alpha sweep)",
#             vmin=0.0, vmax=1.0,
#         )

#     # Diagonal: perfect goodput = throughput. Distance below this line is waste.
#     sub_all = sub.dropna(subset=["throughput_actions_per_s", "goodput_actions_per_s"])
#     if not sub_all.empty:
#         lo = 0.0
#         hi = float(sub_all["throughput_actions_per_s"].max()) * 1.05
#         ax.plot([lo, hi], [lo, hi], "--", color="0.7", linewidth=1.0,
#                 label="goodput = throughput", zorder=1)

#     ax.set_xlabel("Throughput  (actions delivered / s)", fontsize=11)
#     ax.set_ylabel("Goodput  (actions actually consumed / s)", fontsize=11)
#     ax.set_title(title, fontsize=12, fontweight="bold")
#     ax.grid(True, alpha=0.3)
#     ax.legend(loc="best", fontsize=8, frameon=False)
#     return handle


# def plot_throughput_vs_goodput(df: pd.DataFrame, plots_dir: pathlib.Path) -> None:
#     """One PNG per (scenario, model, seed): throughput vs goodput.

#     Distance below the y=x diagonal is GPU work that produced actions the
#     robot never actually executed (chunk arrived too late, or trailing tail
#     of a chunk was superseded). In heterogeneous workloads the leading-late
#     portion is typically the bigger waste.
#     """
#     needed = {
#         "throughput_actions_per_s",
#         "goodput_actions_per_s",
#         "scheduler",
#         "model",
#         "scenario_id",
#         "seed",
#     }
#     if df.empty or not needed.issubset(df.columns):
#         return
#     df = df.dropna(subset=["throughput_actions_per_s", "goodput_actions_per_s"])
#     plots_dir.mkdir(parents=True, exist_ok=True)
#     for (scenario_id, model, seed), sub in df.groupby(["scenario_id", "model", "seed"]):
#         fig, ax = plt.subplots(figsize=(8, 6))
#         handle = _plot_throughput_vs_goodput_one(
#             ax, sub, title="GPU goodput vs throughput",
#         )
#         ax.set_xlim(left=0.0)
#         ax.set_ylim(bottom=0.0)
#         if handle is not None:
#             cbar = fig.colorbar(handle, ax=ax, pad=0.02)
#             cbar.set_label("scheduler alpha", fontsize=10)
#         fig.suptitle(
#             f"scenario={scenario_id}, model={model}, seed={seed}",
#             fontsize=13, fontweight="bold",
#         )
#         plt.tight_layout()
#         safe_model = str(model).replace(".", "_").replace("/", "_")
#         out = plots_dir / f"throughput_vs_goodput__{scenario_id}__{safe_model}__seed{seed}.png"
#         fig.savefig(out, dpi=150, bbox_inches="tight")
#         plt.close(fig)
#         print(f"Wrote {out}")


# def plot_starvation_vs_goodput_ratio(df: pd.DataFrame, plots_dir: pathlib.Path) -> None:
#     """Mean starvation vs goodput ratio. Top-left = lowest waste at lowest starvation."""
#     _save_plot(
#         df, plots_dir,
#         name_prefix="starvation_vs_goodput_ratio",
#         y_col="goodput_ratio",
#         y_label="Goodput / throughput  (1.0 = no wasted GPU work)",
#         panel_title="Mean starvation vs GPU goodput efficiency",
#         pareto=True,
#     )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "output_dir",
        type=pathlib.Path,
        help="Sweep output directory containing sweep_results.csv",
    )
    parser.add_argument(
        "--csv",
        type=pathlib.Path,
        default=None,
        help="Explicit CSV path (defaults to <output_dir>/sweep_results.csv)",
    )
    args = parser.parse_args()

    csv_path = args.csv or (args.output_dir / "sweep_results.csv")
    if not csv_path.exists():
        print(f"ERROR: {csv_path} not found", file=sys.stderr)
        sys.exit(1)

    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d_%H%M%S")  # noqa: UP017
    plots_dir = args.output_dir / "plots_per_seed" / stamp

    df = pd.read_csv(csv_path)
    if "status" in df.columns:
        df = df[df["status"] == "ok"]

    print("Augmenting rows with goodput metrics from artifact dirs...")
    # df = _augment_with_goodput(df)

    plot_starvation_pareto(df, plots_dir)
    plot_starvation_vs_variance(df, plots_dir)
    # plot_throughput_vs_goodput(df, plots_dir)
    # plot_starvation_vs_goodput_ratio(df, plots_dir)

    print(f"\nAll plots written under {plots_dir}")


if __name__ == "__main__":
    main()
