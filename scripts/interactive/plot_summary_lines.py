"""Line-chart trends and a fast-vs-slow throughput tradeoff curve from a
summary folder produced by ``summarize_sweep_runs.py``.

Reads ``results_max_batch_size={M}.csv`` files and emits, per ``max_batch_size``:

  1. System throughput vs num_robots
        x = num_robots
        y = cluster total throughput (``thr_total``)
        one panel per scenario (10f / 1f9s / 5f5s), one line per scheduler

  2. Average starvation vs num_robots
        x = num_robots
        y = mean starvation rate (``starv``, in %)
        same layout

  3. Throughput-throughput tradeoff
        x = fast-tier throughput (``thr_fast``)
        y = slow-tier throughput (``thr_slow``)
        one panel per heterogeneous scenario (1f9s, 5f5s)
        one line per scheduler, each point = one num_robots value

Style mirrors the paper-ready bar plots: clean white background, hidden
top/right spines, light horizontal grid.

Run:
    uv run python scripts/interactive/plot_summary_lines.py \\
        experiments/sweeps/interactive/_summary
    uv run python scripts/interactive/plot_summary_lines.py \\
        experiments/sweeps/interactive/_summary --num-robots 2,4,6,8,10
"""

from __future__ import annotations

import argparse
import pathlib
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

# -------- constants & helpers -------------------------------------------------

PREFERRED_SCHEDULER_ORDER = [
    "max-batch",
    "round-robin",
    "lookahead-actions@ahm=1",
    "lookahead-actions@ahm=3",
    "lookahead-actions@ahm=5",
]
SCHEDULER_DISPLAY = {
    "max-batch": "EDF",
    "round-robin": "RR",
}
LA_AHM_RE = re.compile(r"^lookahead-actions@ahm=(\d+(?:\.\d+)?)$")

SCENARIO_DISPLAY = {
    "hom": "10 Fast",
    "1f9s": "One Fast",
    "5f5s": "Half Fast",
}
SCENARIO_ORDER = ["hom", "1f9s", "5f5s"]
TIERED_SCENARIOS = ["1f9s", "5f5s"]

# Categorical palette for baselines + a blue gradient for the lookahead
# sweep (so ahm=1 → ahm=5 reads as "one method varying weight" rather than
# three distinct schedulers). Blue family is anchored on #37A3D2; baseline
# categorical colors are red and yellow for visual contrast.
SCHEDULER_COLORS = {
    "max-batch":   "#F94144",  # red
    "round-robin": "#F9C74F",  # yellow
    "lookahead-actions@ahm=1": "#A8D9EC",  # light blue
    "lookahead-actions@ahm=3": "#37A3D2",  # mid blue
    "lookahead-actions@ahm=5": "#154A60",  # dark blue
}
SCHEDULER_MARKERS = {
    "max-batch":              "s",
    "round-robin":            "D",
    "lookahead-actions@ahm=1": "o",
    "lookahead-actions@ahm=3": "o",
    "lookahead-actions@ahm=5": "o",
}

MBS_RE = re.compile(r"results_max_batch_size=(\d+)\.csv$")
COL_RE = re.compile(
    r"^(?P<scenario>[^_]+(?:_[^_]+)*?)__(?P<scheduler>.+?)__"
    r"(?P<metric>starv|starv_fast|starv_slow|thr_fast|thr_slow|thr_total|"
    r"successes|successes_fast|successes_slow|worst|n)$"
)


def _scheduler_display(name: str) -> str:
    if name in SCHEDULER_DISPLAY:
        return SCHEDULER_DISPLAY[name]
    m = LA_AHM_RE.match(name)
    if m:
        w = m.group(1)
        if w.endswith(".0"):
            w = w.split(".")[0]
        return f"LA@{w}"
    return name


def _ordered_schedulers(present: list[str]) -> list[str]:
    head = [s for s in PREFERRED_SCHEDULER_ORDER if s in present]
    rest = sorted(s for s in present if s not in PREFERRED_SCHEDULER_ORDER)
    return head + rest


def _strip_chrome(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for s in ("bottom", "left"):
        ax.spines[s].set_visible(True)
        ax.spines[s].set_linewidth(1.2)
    ax.set_axisbelow(True)
    ax.grid(axis="y", linestyle="-", color="#dddddd", linewidth=0.8)
    ax.set_facecolor("white")
    ax.tick_params(axis="both", which="major", length=6, width=1.2, labelsize=12)


# -------- summary loading -----------------------------------------------------


def _load_summary(summary_dir: pathlib.Path) -> dict[int, pd.DataFrame]:
    out: dict[int, pd.DataFrame] = {}
    for path in sorted(summary_dir.glob("results_max_batch_size=*.csv")):
        m = MBS_RE.search(path.name)
        if not m:
            continue
        out[int(m.group(1))] = pd.read_csv(path, index_col="num_robots")
    if not out:
        raise SystemExit(f"No results CSVs found under {summary_dir}")
    return out


def _parse_columns(df: pd.DataFrame) -> dict[tuple[str, str], dict[str, str]]:
    out: dict[tuple[str, str], dict[str, str]] = {}
    for col in df.columns:
        m = COL_RE.match(col)
        if not m:
            continue
        key = (m.group("scenario"), m.group("scheduler"))
        out.setdefault(key, {})[m.group("metric")] = col
    return out


def _series_for_metric(
    df: pd.DataFrame,
    cols: dict[tuple[str, str], dict[str, str]],
    scenario: str,
    scheduler: str,
    metric: str,
    nr_filter: set[int] | None,
) -> tuple[list[int], list[float]]:
    """Return (xs=num_robots, ys=metric) for one (scenario, scheduler) series,
    dropping NaN rows. Filters xs by nr_filter if provided."""
    col = cols.get((scenario, scheduler), {}).get(metric)
    if col is None:
        return ([], [])
    xs: list[int] = []
    ys: list[float] = []
    for nr in df.index:
        if nr_filter is not None and int(nr) not in nr_filter:
            continue
        v = df.loc[nr, col]
        if pd.isna(v):
            continue
        xs.append(int(nr))
        ys.append(float(v))
    return xs, ys


# -------- plotting ------------------------------------------------------------


def _plot_metric_vs_num_robots(
    out_path: pathlib.Path,
    df: pd.DataFrame,
    cols: dict[tuple[str, str], dict[str, str]],
    schedulers: list[str],
    metric: str,
    *,
    title_prefix: str,
    ylabel: str,
    y_scale: float = 1.0,
    nr_filter: set[int] | None,
) -> None:
    """One row of 3 subplots (one per scenario). x=num_robots, y=metric.
    One line per scheduler."""
    scenarios = [sc for sc in SCENARIO_ORDER if any(sc == k[0] for k in cols)]
    if not scenarios:
        return
    n = len(scenarios)
    fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 4.5), sharey=True)
    if n == 1:
        axes = [axes]
    fig.set_facecolor("white")

    for i, scenario in enumerate(scenarios):
        ax = axes[i]
        for sch in schedulers:
            xs, ys = _series_for_metric(df, cols, scenario, sch, metric, nr_filter)
            if not xs:
                continue
            ys_scaled = [y * y_scale for y in ys]
            ax.plot(
                xs, ys_scaled,
                marker=SCHEDULER_MARKERS.get(sch, "o"),
                color=SCHEDULER_COLORS.get(sch, "#444444"),
                linewidth=1.8, markersize=6,
                label=_scheduler_display(sch),
            )
        ax.set_title(f"{title_prefix}: {SCENARIO_DISPLAY.get(scenario, scenario)}",
                     fontsize=14)
        ax.set_xlabel("Number of Robots", fontsize=12)
        if i == 0:
            ax.set_ylabel(ylabel, fontsize=12)
        _strip_chrome(ax)
        if i == n - 1:
            ax.legend(loc="best", fontsize=10, framealpha=0.95)

    fig.tight_layout()
    fig.savefig(out_path, dpi=130, facecolor="white")
    fig.savefig(out_path.with_suffix(".pdf"), facecolor="white", bbox_inches="tight")
    plt.close(fig)


def _draw_series_on_axis(
    ax,
    df: pd.DataFrame,
    cols: dict[tuple[str, str], dict[str, str]],
    schedulers: list[str],
    scenario: str,
    metric: str,
    *,
    y_scale: float,
    nr_filter: set[int] | None,
    show_legend: bool = False,
) -> None:
    for sch in schedulers:
        xs, ys = _series_for_metric(df, cols, scenario, sch, metric, nr_filter)
        if not xs:
            continue
        ys_scaled = [y * y_scale for y in ys]
        ax.plot(
            xs, ys_scaled,
            marker=SCHEDULER_MARKERS.get(sch, "o"),
            color=SCHEDULER_COLORS.get(sch, "#444444"),
            linewidth=1.8, markersize=6,
            label=_scheduler_display(sch),
        )
    _strip_chrome(ax)
    if show_legend:
        ax.legend(loc="best", fontsize=10, framealpha=0.95)


def _plot_overview_stack(
    out_path: pathlib.Path,
    df: pd.DataFrame,
    cols: dict[tuple[str, str], dict[str, str]],
    schedulers: list[str],
    nr_filter: set[int] | None,
) -> None:
    """2 rows (system throughput / avg starvation) × N cols (scenarios)."""
    scenarios = [sc for sc in SCENARIO_ORDER if any(sc == k[0] for k in cols)]
    if not scenarios:
        return
    n = len(scenarios)
    fig, axes = plt.subplots(2, n, figsize=(5.5 * n, 8.5), sharex=True)
    if n == 1:
        axes = axes.reshape(2, 1)
    fig.set_facecolor("white")

    row_specs = [
        ("thr_total", 1.0,   "System throughput",         "Cluster throughput (successes / sec)"),
        ("starv",     100.0, "Average starvation",         "Average starvation rate (%)"),
    ]
    for row, (metric, y_scale, title_prefix, ylabel) in enumerate(row_specs):
        for col, scenario in enumerate(scenarios):
            ax = axes[row, col]
            _draw_series_on_axis(
                ax, df, cols, schedulers, scenario, metric,
                y_scale=y_scale, nr_filter=nr_filter,
                show_legend=(col == n - 1 and row == 0),
            )
            if row == 0:
                ax.set_title(SCENARIO_DISPLAY.get(scenario, scenario), fontsize=14)
            if col == 0:
                ax.set_ylabel(ylabel, fontsize=12)
            if row == 1:
                ax.set_xlabel("Number of Robots", fontsize=12)

    fig.tight_layout()
    fig.savefig(out_path, dpi=130, facecolor="white")
    fig.savefig(out_path.with_suffix(".pdf"), facecolor="white", bbox_inches="tight")
    plt.close(fig)


def _plot_tier_breakdown(
    out_path: pathlib.Path,
    df: pd.DataFrame,
    cols: dict[tuple[str, str], dict[str, str]],
    schedulers: list[str],
    scenario: str,
    nr_filter: set[int] | None,
) -> None:
    """Per scenario, 2×2 grid: rows = throughput / starvation, cols = fast / slow tier."""
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.5), sharex=True)
    fig.set_facecolor("white")
    fig.suptitle(SCENARIO_DISPLAY.get(scenario, scenario), fontsize=15, y=0.99)

    panel_specs = [
        # (row, col, metric, y_scale, title, ylabel)
        (0, 0, "thr_fast",   1.0,   "Fast-tier throughput", "Throughput (successes / sec)"),
        (0, 1, "thr_slow",   1.0,   "Slow-tier throughput", "Throughput (successes / sec)"),
        (1, 0, "starv_fast", 100.0, "Fast-tier starvation", "Starvation rate (%)"),
        (1, 1, "starv_slow", 100.0, "Slow-tier starvation", "Starvation rate (%)"),
    ]
    for row, col, metric, y_scale, title, ylabel in panel_specs:
        ax = axes[row, col]
        _draw_series_on_axis(
            ax, df, cols, schedulers, scenario, metric,
            y_scale=y_scale, nr_filter=nr_filter,
            show_legend=(row == 0 and col == 1),
        )
        ax.set_title(title, fontsize=13)
        if col == 0:
            ax.set_ylabel(ylabel, fontsize=12)
        if row == 1:
            ax.set_xlabel("Number of Robots", fontsize=12)

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=130, facecolor="white")
    fig.savefig(out_path.with_suffix(".pdf"), facecolor="white", bbox_inches="tight")
    plt.close(fig)


def _plot_throughput_tradeoff(
    out_path: pathlib.Path,
    df: pd.DataFrame,
    cols: dict[tuple[str, str], dict[str, str]],
    schedulers: list[str],
    nr_filter: set[int] | None,
) -> None:
    """For each heterogeneous scenario: x=fast throughput, y=slow throughput.
    One line per scheduler, points labelled by num_robots, connected in
    num_robots order."""
    scenarios = [sc for sc in TIERED_SCENARIOS if any(sc == k[0] for k in cols)]
    if not scenarios:
        return
    n = len(scenarios)
    fig, axes = plt.subplots(1, n, figsize=(6.0 * n, 5.0), sharex=False, sharey=False)
    if n == 1:
        axes = [axes]
    fig.set_facecolor("white")

    for i, scenario in enumerate(scenarios):
        ax = axes[i]
        for sch in schedulers:
            fast_x, fast_y = _series_for_metric(df, cols, scenario, sch, "thr_fast", nr_filter)
            slow_x, slow_y = _series_for_metric(df, cols, scenario, sch, "thr_slow", nr_filter)
            # Intersect num_robots that have both values.
            by_x = {x: f for x, f in zip(fast_x, fast_y)}
            by_x_slow = {x: s for x, s in zip(slow_x, slow_y)}
            common = sorted(set(by_x) & set(by_x_slow))
            if not common:
                continue
            xs_thr_fast = [by_x[k] for k in common]
            ys_thr_slow = [by_x_slow[k] for k in common]
            color = SCHEDULER_COLORS.get(sch, "#444444")
            ax.plot(
                xs_thr_fast, ys_thr_slow,
                marker=SCHEDULER_MARKERS.get(sch, "o"),
                color=color, linewidth=1.8, markersize=6,
                label=_scheduler_display(sch),
            )
            # Annotate each point with its num_robots (small, near point).
            for nr, xv, yv in zip(common, xs_thr_fast, ys_thr_slow):
                ax.annotate(
                    str(nr), (xv, yv), xytext=(4, 4),
                    textcoords="offset points",
                    fontsize=8, color=color, alpha=0.9,
                )

        ax.set_title(f"Throughput tradeoff: {SCENARIO_DISPLAY.get(scenario, scenario)}",
                     fontsize=14)
        ax.set_xlabel("Fast-tier throughput (successes / sec)", fontsize=12)
        ax.set_ylabel("Slow-tier throughput (successes / sec)", fontsize=12)
        _strip_chrome(ax)
        if i == n - 1:
            ax.legend(loc="best", fontsize=10, framealpha=0.95)

        # Optional: dashed identity line for visual reference.
        lims = (
            min(ax.get_xlim()[0], ax.get_ylim()[0]),
            max(ax.get_xlim()[1], ax.get_ylim()[1]),
        )
        ax.plot(lims, lims, "--", color="#cccccc", linewidth=0.8, zorder=0)

    fig.tight_layout()
    fig.savefig(out_path, dpi=130, facecolor="white")
    fig.savefig(out_path.with_suffix(".pdf"), facecolor="white", bbox_inches="tight")
    plt.close(fig)


# -------- CLI / driver --------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("summary_dir", type=pathlib.Path,
                   help="Directory containing results_max_batch_size=*.csv "
                        "(output of summarize_sweep_runs.py).")
    p.add_argument("--plots-subdir", default="plots",
                   help="Subdir under summary_dir to write PNGs into (default: plots).")
    p.add_argument("--num-robots", type=str, default=None,
                   help="Comma-separated list of num_robots values to include "
                        "(e.g. '2,4,6,8,10'). Default: all present.")
    p.add_argument("--max-batch-size", type=str, default=None,
                   help="Comma-separated list of mbs values to render. Default: all.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    summary_dir = args.summary_dir.resolve()
    if not summary_dir.is_dir():
        raise SystemExit(f"Not a directory: {summary_dir}")
    plots_dir = summary_dir / args.plots_subdir
    plots_dir.mkdir(parents=True, exist_ok=True)

    nr_filter: set[int] | None = None
    if args.num_robots:
        nr_filter = {int(x.strip()) for x in args.num_robots.split(",") if x.strip()}
    mbs_filter: set[int] | None = None
    if args.max_batch_size:
        mbs_filter = {int(x.strip()) for x in args.max_batch_size.split(",") if x.strip()}

    by_mbs = _load_summary(summary_dir)

    n_written = 0
    for mbs, df in by_mbs.items():
        if mbs_filter is not None and mbs not in mbs_filter:
            continue
        cols = _parse_columns(df)
        observed_schedulers = sorted({sch for _, sch in cols})
        schedulers = _ordered_schedulers(observed_schedulers)

        # Plot 1: system throughput vs num_robots.
        _plot_metric_vs_num_robots(
            plots_dir / f"system_throughput_lines__mbs{mbs}.png",
            df, cols, schedulers,
            metric="thr_total",
            title_prefix="System throughput",
            ylabel="Cluster throughput (successes / sec)",
            nr_filter=nr_filter,
        )
        # Plot 2: avg starvation vs num_robots.
        _plot_metric_vs_num_robots(
            plots_dir / f"avg_starvation_lines__mbs{mbs}.png",
            df, cols, schedulers,
            metric="starv",
            title_prefix="Average starvation",
            ylabel="Average starvation rate (%)",
            y_scale=100.0,
            nr_filter=nr_filter,
        )
        # Plot 3: throughput-throughput tradeoff (1f9s + 5f5s only).
        _plot_throughput_tradeoff(
            plots_dir / f"throughput_tradeoff__mbs{mbs}.png",
            df, cols, schedulers,
            nr_filter=nr_filter,
        )
        # Plots 4–5: per-tier throughput vs num_robots.
        # These reveal what gets masked in the cluster total when one tier
        # outnumbers the other (e.g. 1f9s, where +1 fast robot's gain is
        # swamped by tiny -slow per-robot drops × 9).
        _plot_metric_vs_num_robots(
            plots_dir / f"fast_throughput_lines__mbs{mbs}.png",
            df, cols, schedulers,
            metric="thr_fast",
            title_prefix="Fast-tier throughput",
            ylabel="Fast robot throughput (successes / sec)",
            nr_filter=nr_filter,
        )
        _plot_metric_vs_num_robots(
            plots_dir / f"slow_throughput_lines__mbs{mbs}.png",
            df, cols, schedulers,
            metric="thr_slow",
            title_prefix="Slow-tier throughput",
            ylabel="Slow robot throughput (successes / sec)",
            nr_filter=nr_filter,
        )
        # Plots 6–7: per-tier starvation vs num_robots (mirrors per-tier throughput).
        _plot_metric_vs_num_robots(
            plots_dir / f"fast_starvation_lines__mbs{mbs}.png",
            df, cols, schedulers,
            metric="starv_fast",
            title_prefix="Fast-tier starvation",
            ylabel="Fast robot starvation rate (%)",
            y_scale=100.0,
            nr_filter=nr_filter,
        )
        _plot_metric_vs_num_robots(
            plots_dir / f"slow_starvation_lines__mbs{mbs}.png",
            df, cols, schedulers,
            metric="starv_slow",
            title_prefix="Slow-tier starvation",
            ylabel="Slow robot starvation rate (%)",
            y_scale=100.0,
            nr_filter=nr_filter,
        )
        # Plot 8: overview stack — system throughput (top row) + avg starvation
        # (bottom row), one column per scenario. Single composite figure.
        _plot_overview_stack(
            plots_dir / f"overview_lines__mbs{mbs}.png",
            df, cols, schedulers, nr_filter=nr_filter,
        )
        n_written += 8

        # Plots 9+: per-scenario tier breakdown — 2×2 grid of (throughput/starv) ×
        # (fast/slow). Only emitted for scenarios that have both tiers (1f9s, 5f5s).
        for scenario in TIERED_SCENARIOS:
            if not any(scenario == k[0] for k in cols):
                continue
            _plot_tier_breakdown(
                plots_dir / f"tier_breakdown__{scenario}__mbs{mbs}.png",
                df, cols, schedulers, scenario, nr_filter=nr_filter,
            )
            n_written += 1

    print(f"Wrote {n_written} PNGs to {plots_dir}")


if __name__ == "__main__":
    main()
