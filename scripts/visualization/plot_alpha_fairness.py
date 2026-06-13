"""Plot alpha-fairness sweep results."""

from __future__ import annotations

import ast
import pathlib

BASELINE_SCHEDULERS = (
    "max-batch",
    "fixed-max-batch",
    "greedy-deadline",
    "round-robin",
    "lookahead-actions",
)
DYNAMIC_SCHEDULERS = ("dynamic-action", "action-deficit", "starvation-fair", "lookahead-actions")
LOOKAHEAD_SCHEDULERS = {"lookahead-actions"}

BASELINE_STYLE = {
    "max-batch": {"marker": "o", "color": "#1f77b4"},
    "fixed-max-batch": {"marker": "s", "color": "#1f77b4"},
    "greedy-deadline": {"marker": "^", "color": "#2ca02c"},
    "round-robin": {"marker": "D", "color": "#d62728"},
    "lookahead-actions": {"marker": "o", "color": "#9467bd"},
}
LOOKAHEAD_COLORS = (
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
    "#ff7f0e",
)
DYNAMIC_STYLE = {
    "dynamic-action": {"marker": "o", "line_color": "0.45"},
    "action-deficit": {"marker": "X", "line_color": "0.20"},
    "starvation-fair": {"marker": "D", "line_color": "0.20"},
    "lookahead-actions": {"marker": "o", "line_color": "0.45"},
}
DYNAMIC_CMAP = "viridis"


def _batch_label(value) -> str:
    try:
        return f"{int(float(value))}"
    except (TypeError, ValueError):
        return str(value)


def _batch_groups(df):
    if "max_batch_size" not in df.columns:
        return [(None, df)]
    unique = df["max_batch_size"].dropna().unique()
    if len(unique) <= 1:
        return [(None, df)]
    return list(df.groupby("max_batch_size", dropna=False))


def _has_value(value) -> bool:
    try:
        if value != value:
            return False
    except TypeError:
        pass
    return value is not None and str(value).strip() != ""


def _ratio_text(value) -> str | None:
    if not _has_value(value):
        return None
    text = str(value).strip()
    if text.startswith("ratio_"):
        text = text.removeprefix("ratio_").replace("_", ".")
        return f"ratio={text}"
    return text.replace("_", ".")


def _ratio_from_multipliers(value) -> str | None:
    if not _has_value(value):
        return None
    try:
        parsed = ast.literal_eval(str(value))
    except (SyntaxError, ValueError):
        return None
    if not isinstance(parsed, dict) or len(parsed) < 2:
        return None
    multipliers = {int(k): float(v) for k, v in parsed.items()}
    horizons = sorted(multipliers)
    base = multipliers[horizons[-1]]
    if base == 0:
        return None
    ratio = multipliers[horizons[0]] / base
    return f"ratio={ratio:g}"


def _lookahead_label(sched: str, sub) -> str:
    if "server_variant" in sub.columns:
        variants = [v for v in sub["server_variant"].dropna().unique() if _has_value(v)]
        if variants:
            ratio = _ratio_text(variants[0])
            if ratio is not None:
                return f"{sched} ({ratio})"
    if "action_horizon_multipliers" in sub.columns:
        for value in sub["action_horizon_multipliers"].dropna().unique():
            ratio = _ratio_from_multipliers(value)
            if ratio is not None:
                return f"{sched} ({ratio})"
    return sched


def _baseline_groups_for_scheduler(sched: str, df):
    group_cols = []
    if "max_batch_size" in df.columns and len(df["max_batch_size"].dropna().unique()) > 1:
        group_cols.append("max_batch_size")
    if sched in LOOKAHEAD_SCHEDULERS:
        if "server_variant" in df.columns and any(_has_value(v) for v in df["server_variant"]):
            group_cols.append("server_variant")
        elif (
            "action_horizon_multipliers" in df.columns
            and len(
                [v for v in df["action_horizon_multipliers"].dropna().unique() if _has_value(v)]
            )
            > 1
        ):
            group_cols.append("action_horizon_multipliers")
    if not group_cols:
        return [(None, df)]
    return list(df.groupby(group_cols, dropna=False))


def _plot_one_yaxis(ax, sub, *, y_col: str, y_label: str, title: str):
    """Scatter (mean_starvation, y_col) onto ``ax`` with baselines + alpha curve."""
    import numpy as np  # noqa: PLC0415

    handle = None
    for sched in BASELINE_SCHEDULERS:
        sub_b = sub[sub["scheduler"] == sched].dropna(subset=["mean_starvation", y_col])
        if sub_b.empty:
            continue
        for group_index, (_group_key, sub_batch) in enumerate(
            _baseline_groups_for_scheduler(sched, sub_b)
        ):
            x = float(sub_batch["mean_starvation"].mean())
            y = float(sub_batch[y_col].mean())
            xerr = float(sub_batch["mean_starvation"].std(ddof=0)) if len(sub_batch) > 1 else 0.0
            yerr = float(sub_batch[y_col].std(ddof=0)) if len(sub_batch) > 1 else 0.0
            style = dict(BASELINE_STYLE[sched])
            label = _lookahead_label(sched, sub_batch) if sched in LOOKAHEAD_SCHEDULERS else sched
            batch_varies = (
                "max_batch_size" in sub_b.columns
                and len(sub_b["max_batch_size"].dropna().unique()) > 1
            )
            if batch_varies:
                max_batch_size = sub_batch["max_batch_size"].iloc[0]
                label = f"{label} (B={_batch_label(max_batch_size)})"
            if sched in LOOKAHEAD_SCHEDULERS:
                style["color"] = LOOKAHEAD_COLORS[group_index % len(LOOKAHEAD_COLORS)]
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
                label=label,
                zorder=4,
            )

    sub_d = sub[sub["scheduler"].isin(DYNAMIC_SCHEDULERS)].dropna(subset=["mean_starvation", y_col])
    for sched, sub_sched_all in sub_d.groupby("scheduler"):
        for max_batch_size, sub_sched in _batch_groups(sub_sched_all):
            agg = (
                sub_sched.groupby("alpha_requested")
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
            if agg.empty:
                continue

            xs = agg["mean_starvation"].to_numpy()
            ys = agg["y_mean"].to_numpy()
            alphas = agg["alpha_requested"].to_numpy()
            style = DYNAMIC_STYLE.get(sched, {"marker": "o", "line_color": "0.5"})
            label = f"{sched} (alpha sweep)"
            if max_batch_size is not None:
                label = f"{sched} (B={_batch_label(max_batch_size)}, alpha sweep)"
            ax.plot(
                xs,
                ys,
                "-",
                color=style["line_color"],
                linewidth=1.2,
                alpha=0.7,
                zorder=2,
            )
            handle = ax.scatter(
                xs,
                ys,
                c=alphas,
                cmap=DYNAMIC_CMAP,
                marker=style["marker"],
                s=70,
                edgecolors="black",
                linewidths=0.6,
                zorder=3,
                label=label,
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
) -> list[pathlib.Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    df = pd.read_csv(results_csv)
    if df.empty:
        return []
    if "status" in df.columns:
        df = df[df["status"] == "ok"].copy()
    if "alpha_requested" not in df.columns and "alpha" in df.columns:
        df["alpha_requested"] = pd.to_numeric(df["alpha"], errors="coerce")
    if "scenario_id" not in df.columns:
        df["scenario_id"] = "all"
    if "model" not in df.columns:
        df["model"] = "all"

    needed = {"mean_starvation", y_col, "scheduler", "model", "scenario_id", "alpha_requested"}
    if not needed.issubset(df.columns):
        return []
    df = df.dropna(subset=["mean_starvation", y_col])

    plots_dir.mkdir(parents=True, exist_ok=True)
    written: list[pathlib.Path] = []
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
        safe_scenario = str(scenario_id).replace(".", "_").replace("/", "_")
        safe_model = str(model).replace(".", "_").replace("/", "_")
        out = plots_dir / f"{name_prefix}__{safe_scenario}__{safe_model}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Wrote {out}")
        written.append(out)
    return written


def plot_starvation_pareto(
    results_csv: pathlib.Path, plots_dir: pathlib.Path
) -> list[pathlib.Path]:
    """Pareto plot: mean vs worst-robot starvation. Bottom-left = best."""
    return _save_single_panel_plot(
        results_csv,
        plots_dir,
        name_prefix="starvation_pareto",
        y_col="max_starvation",
        y_label="Worst-robot starvation rate  (lower is fairer)",
        panel_title="Pareto: mean vs worst-robot starvation",
        pareto=True,
    )


def plot_starvation_vs_variance(
    results_csv: pathlib.Path, plots_dir: pathlib.Path
) -> list[pathlib.Path]:
    """Mean starvation vs cross-robot starvation variance. Bottom-left = best."""
    return _save_single_panel_plot(
        results_csv,
        plots_dir,
        name_prefix="starvation_vs_variance",
        y_col="starvation_variance",
        y_label="Cross-robot starvation variance (lower is fairer)",
        panel_title="Mean starvation vs cross-robot variance",
        pareto=False,
    )


def plot_results(results_csv: pathlib.Path, plots_dir: pathlib.Path) -> list[pathlib.Path]:
    written: list[pathlib.Path] = []
    written.extend(plot_starvation_pareto(results_csv, plots_dir))
    written.extend(plot_starvation_vs_variance(results_csv, plots_dir))
    return written
