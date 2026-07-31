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
    uv run python scripts/visualization/plot_summary_lines.py \\
        <summary-dir>
    uv run python scripts/visualization/plot_summary_lines.py \\
        <summary-dir> --num-robots 2,4,6,8,10
"""

from __future__ import annotations

import argparse
import pathlib
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
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
    "max-batch": "#8E6CA8",  # muted purple (no red/yellow)
    "round-robin": "#5FA86F",  # muted green
    "lookahead-actions@ahm=1": "#6FB0D6",  # blue family, tighter spread
    "lookahead-actions@ahm=3": "#3C86B8",
    "lookahead-actions@ahm=5": "#1E5C84",
}
SCHEDULER_MARKERS = {
    "max-batch": "s",
    "round-robin": "D",
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
        return "LA" if w == "1" else f"LA@{w}"
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
                xs,
                ys_scaled,
                marker=SCHEDULER_MARKERS.get(sch, "o"),
                color=SCHEDULER_COLORS.get(sch, "#444444"),
                linewidth=1.8,
                markersize=6,
                label=_scheduler_display(sch),
            )
        ax.set_title(f"{title_prefix}: {SCENARIO_DISPLAY.get(scenario, scenario)}", fontsize=14)
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
            xs,
            ys_scaled,
            marker=SCHEDULER_MARKERS.get(sch, "o"),
            color=SCHEDULER_COLORS.get(sch, "#444444"),
            linewidth=1.8,
            markersize=6,
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
        ("thr_total", 60.0, "System throughput", "Cluster throughput (successes / min)"),
        ("starv", 100.0, "Average starvation", "Average starvation rate (%)"),
    ]
    for row, (metric, y_scale, title_prefix, ylabel) in enumerate(row_specs):
        for col, scenario in enumerate(scenarios):
            ax = axes[row, col]
            _draw_series_on_axis(
                ax,
                df,
                cols,
                schedulers,
                scenario,
                metric,
                y_scale=y_scale,
                nr_filter=nr_filter,
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
        (0, 0, "thr_fast", 60.0, "Fast-tier throughput", "Throughput (successes / min)"),
        (0, 1, "thr_slow", 60.0, "Slow-tier throughput", "Throughput (successes / min)"),
        (1, 0, "starv_fast", 100.0, "Fast-tier starvation", "Starvation rate (%)"),
        (1, 1, "starv_slow", 100.0, "Slow-tier starvation", "Starvation rate (%)"),
    ]
    for row, col, metric, y_scale, title, ylabel in panel_specs:
        ax = axes[row, col]
        _draw_series_on_axis(
            ax,
            df,
            cols,
            schedulers,
            scenario,
            metric,
            y_scale=y_scale,
            nr_filter=nr_filter,
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


def _plot_tier_breakdown_combined(
    out_path: pathlib.Path,
    df: pd.DataFrame,
    cols: dict[tuple[str, str], dict[str, str]],
    schedulers: list[str],
    scenarios: list[str],
    nr_filter: set[int] | None,
    share_y: bool = False,
) -> None:
    """2 rows × (2 × N) cols tier breakdown for multiple tiered scenarios in
    one figure. Rows = throughput / starvation; within each scenario the two
    columns are fast / slow tier. Single shared legend on top. sharex=True
    ties x-axes.

    share_y: when True, every panel in a row is forced to the same y-range
    (the union across that row) so panels are directly comparable. Tick labels
    stay on all panels. Default False leaves each panel auto-scaled."""
    scenarios = [sc for sc in scenarios if any(sc == k[0] for k in cols)]
    if not scenarios:
        return
    n_scen = len(scenarios)
    n_cols = 2 * n_scen
    fig, axes = plt.subplots(
        2,
        n_cols,
        figsize=(4.2 * n_cols, 8.5),
        sharex=True,
    )
    if n_cols == 1:
        axes = axes.reshape(2, 1)
    fig.set_facecolor("white")

    panel_specs = [
        # (row, tier_col_offset, metric, y_scale, tier_label, row_ylabel)
        (0, 0, "thr_fast", 60.0, "Fast tier", "Throughput (successes / min)"),
        (0, 1, "thr_slow", 60.0, "Slow tier", "Throughput (successes / min)"),
        (1, 0, "starv_fast", 100.0, "Fast tier", "Starvation rate (%)"),
        (1, 1, "starv_slow", 100.0, "Slow tier", "Starvation rate (%)"),
    ]
    for s_idx, scenario in enumerate(scenarios):
        scen_label = SCENARIO_DISPLAY.get(scenario, scenario)
        for row, tier_off, metric, y_scale, tier_label, ylabel in panel_specs:
            col = s_idx * 2 + tier_off
            ax = axes[row, col]
            _draw_series_on_axis(
                ax,
                df,
                cols,
                schedulers,
                scenario,
                metric,
                y_scale=y_scale,
                nr_filter=nr_filter,
                show_legend=False,
            )
            if row == 0:
                ax.set_title(f"{scen_label} — {tier_label}", fontsize=13)
            if col == 0:
                ax.set_ylabel(ylabel, fontsize=12)
            if row == 1:
                ax.set_xlabel("Number of Robots", fontsize=12)

    def _fmt_outlier(v: float) -> str:
        return f"{v:.1f}" if abs(v) < 10 else f"{v:.0f}"

    if share_y:
        for row in range(2):
            row_axes = [axes[row, c] for c in range(n_cols)]
            # Gather all plotted y-values across the row's data series.
            all_y: list[float] = []
            for ax in row_axes:
                for line in ax.get_lines():
                    all_y.extend(float(v) for v in line.get_ydata() if np.isfinite(v))
            if not all_y:
                continue
            arr = np.asarray(all_y, dtype=float)
            data_min, data_max = float(arr.min()), float(arr.max())

            # Robust clip (Q1−1.5·IQR, Q3+1.5·IQR) so a single extreme point
            # doesn't squash every other panel in the shared row. Points beyond
            # the fence are annotated off-chart instead of stretching the axis.
            q1, q3 = np.percentile(arr, [25, 75])
            iqr = q3 - q1
            lo_fence, hi_fence = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            within = arr[(arr >= lo_fence) & (arr <= hi_fence)]
            lo = float(within.min()) if within.size else data_min
            hi = float(within.max()) if within.size else data_max
            span = (hi - lo) or 1.0
            low_clipped = lo > data_min
            high_clipped = hi < data_max
            ymin = (lo - 0.10 * span) if low_clipped else min(0.0, data_min)
            ymax = (hi + 0.12 * span) if high_clipped else data_max + 0.05 * span

            for c, ax in enumerate(row_axes):
                ax.set_ylim(ymin, ymax)
                if c > 0:  # aligned scale → drop redundant tick labels
                    ax.tick_params(axis="y", labelleft=False)
                for line in ax.get_lines():
                    color = line.get_color()
                    for xv, yv in zip(line.get_xdata(), line.get_ydata()):
                        if not np.isfinite(yv):
                            continue
                        if high_clipped and yv > ymax:
                            ax.plot(
                                [xv],
                                [ymax],
                                marker="^",
                                color=color,
                                markersize=8,
                                clip_on=False,
                                zorder=6,
                            )
                            ax.annotate(
                                _fmt_outlier(float(yv)),
                                xy=(xv, ymax),
                                xytext=(0, 6),
                                textcoords="offset points",
                                ha="center",
                                va="bottom",
                                fontsize=8,
                                color=color,
                                fontweight="bold",
                                annotation_clip=False,
                            )
                        elif low_clipped and yv < ymin:
                            ax.plot(
                                [xv],
                                [ymin],
                                marker="v",
                                color=color,
                                markersize=8,
                                clip_on=False,
                                zorder=6,
                            )
                            ax.annotate(
                                _fmt_outlier(float(yv)),
                                xy=(xv, ymin),
                                xytext=(0, -6),
                                textcoords="offset points",
                                ha="center",
                                va="top",
                                fontsize=8,
                                color=color,
                                fontweight="bold",
                                annotation_clip=False,
                            )

    # Gather legend handles from any non-empty axis.
    handles, labels = [], []
    for ax in axes.flat:
        axis_handles, axis_labels = ax.get_legend_handles_labels()
        if axis_handles:
            seen = set(labels)
            for handle, label in zip(axis_handles, axis_labels):
                if label not in seen:
                    handles.append(handle)
                    labels.append(label)
                    seen.add(label)
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.0),
            ncol=len(labels),
            fontsize=12,
            framealpha=0.95,
            frameon=False,
        )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=130, facecolor="white", bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), facecolor="white", bbox_inches="tight")
    plt.close(fig)


def _plot_metric_vs_batch_size(
    out_path: pathlib.Path,
    by_mbs: dict[int, pd.DataFrame],
    metric: str,
    *,
    title_prefix: str,
    ylabel: str,
    y_scale: float = 1.0,
    nr_filter: set[int] | None,
) -> None:
    """Pivot the data: x = max_batch_size, y = metric.

    One row of subplots per scenario, one line per scheduler. If multiple
    num_robots are present, draws one line per (scheduler, num_robots) using
    scheduler color and num_robots as the marker (so a sweep across N still
    reads cleanly). Otherwise (single N) draws one line per scheduler.
    """
    # Collect: {(scenario, scheduler, num_robots): [(mbs, value), ...]}
    series: dict[tuple[str, str, int], list[tuple[int, float]]] = {}
    scenarios_seen: set[str] = set()
    for mbs, df in sorted(by_mbs.items()):
        cols = _parse_columns(df)
        for (scenario, scheduler), metric_cols in cols.items():
            col = metric_cols.get(metric)
            if col is None:
                continue
            scenarios_seen.add(scenario)
            for nr in df.index:
                if nr_filter is not None and int(nr) not in nr_filter:
                    continue
                v = df.loc[nr, col]
                if pd.isna(v):
                    continue
                series.setdefault((scenario, scheduler, int(nr)), []).append((mbs, float(v)))
    if not series:
        return

    scenarios = [sc for sc in SCENARIO_ORDER if sc in scenarios_seen] + [
        sc for sc in sorted(scenarios_seen) if sc not in SCENARIO_ORDER
    ]
    n = len(scenarios)

    # Collect num_robots in play to decide marker scheme.
    all_nr = sorted({nr for _, _, nr in series})
    multi_nr = len(all_nr) > 1
    nr_markers = ["o", "s", "D", "^", "v", "P", "X", "*"]
    nr_to_marker = {nr: nr_markers[i % len(nr_markers)] for i, nr in enumerate(all_nr)}

    fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 4.5), sharey=True)
    if n == 1:
        axes = [axes]
    fig.set_facecolor("white")

    all_schedulers_seen: set[str] = set()
    for i, scenario in enumerate(scenarios):
        ax = axes[i]
        scen_schedulers = [sch for (sc, sch, _) in series if sc == scenario]
        scen_schedulers = _ordered_schedulers(sorted(set(scen_schedulers)))
        for sch in scen_schedulers:
            all_schedulers_seen.add(sch)
            for nr in all_nr:
                pts = series.get((scenario, sch, nr))
                if not pts:
                    continue
                pts_sorted = sorted(pts)
                xs = [p[0] for p in pts_sorted]
                ys = [p[1] * y_scale for p in pts_sorted]
                marker = nr_to_marker[nr] if multi_nr else SCHEDULER_MARKERS.get(sch, "o")
                label = (
                    f"{_scheduler_display(sch)} (N={nr})" if multi_nr else _scheduler_display(sch)
                )
                ax.plot(
                    xs,
                    ys,
                    marker=marker,
                    color=SCHEDULER_COLORS.get(sch, "#444444"),
                    linewidth=1.8,
                    markersize=6,
                    label=label,
                )
        title = f"{title_prefix}: {SCENARIO_DISPLAY.get(scenario, scenario)}"
        ax.set_title(title, fontsize=14)
        ax.set_xlabel("Max Batch Size", fontsize=12)
        if i == 0:
            ax.set_ylabel(ylabel, fontsize=12)
        _strip_chrome(ax)
        if i == n - 1:
            ax.legend(loc="best", fontsize=9, framealpha=0.95)

    fig.tight_layout()
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
            by_x = {x: f * 60.0 for x, f in zip(fast_x, fast_y)}
            by_x_slow = {x: s * 60.0 for x, s in zip(slow_x, slow_y)}
            common = sorted(set(by_x) & set(by_x_slow))
            if not common:
                continue
            xs_thr_fast = [by_x[k] for k in common]
            ys_thr_slow = [by_x_slow[k] for k in common]
            color = SCHEDULER_COLORS.get(sch, "#444444")
            ax.plot(
                xs_thr_fast,
                ys_thr_slow,
                marker=SCHEDULER_MARKERS.get(sch, "o"),
                color=color,
                linewidth=1.8,
                markersize=6,
                label=_scheduler_display(sch),
            )
            # Annotate each point with its num_robots (small, near point).
            for nr, xv, yv in zip(common, xs_thr_fast, ys_thr_slow):
                ax.annotate(
                    str(nr),
                    (xv, yv),
                    xytext=(4, 4),
                    textcoords="offset points",
                    fontsize=8,
                    color=color,
                    alpha=0.9,
                )

        ax.set_title(
            f"Throughput tradeoff: {SCENARIO_DISPLAY.get(scenario, scenario)}", fontsize=14
        )
        ax.set_xlabel("Fast-tier throughput (successes / min)", fontsize=12)
        ax.set_ylabel("Slow-tier throughput (successes / min)", fontsize=12)
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
    p.add_argument(
        "summary_dir",
        type=pathlib.Path,
        help="Directory containing results_max_batch_size=*.csv "
        "(output of summarize_sweep_runs.py).",
    )
    p.add_argument(
        "--plots-subdir",
        default="plots",
        help="Subdir under summary_dir to write PNGs into (default: plots).",
    )
    p.add_argument(
        "--num-robots",
        type=str,
        default=None,
        help="Comma-separated list of num_robots values to include "
        "(e.g. '2,4,6,8,10'). Default: all present.",
    )
    p.add_argument(
        "--max-batch-size",
        type=str,
        default=None,
        help="Comma-separated list of mbs values to render. Default: all.",
    )
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

    # Cross-mbs ablation plots: x = max_batch_size, y = metric. Emitted once
    # (not per-mbs) if there are ≥2 batch-size values.
    n_written = 0
    if len([m for m in by_mbs if mbs_filter is None or m in mbs_filter]) >= 2:
        ablation_by_mbs = {
            m: df for m, df in by_mbs.items() if mbs_filter is None or m in mbs_filter
        }
        _plot_metric_vs_batch_size(
            plots_dir / "batch_size_vs_throughput.png",
            ablation_by_mbs,
            metric="thr_total",
            title_prefix="System throughput",
            ylabel="Cluster throughput (successes / min)",
            y_scale=60.0,
            nr_filter=nr_filter,
        )
        _plot_metric_vs_batch_size(
            plots_dir / "batch_size_vs_starvation.png",
            ablation_by_mbs,
            metric="starv",
            title_prefix="Average starvation",
            ylabel="Average starvation rate (%)",
            y_scale=100.0,
            nr_filter=nr_filter,
        )
        n_written += 2

    for mbs, df in by_mbs.items():
        if mbs_filter is not None and mbs not in mbs_filter:
            continue
        cols = _parse_columns(df)
        observed_schedulers = sorted({sch for _, sch in cols})
        schedulers = _ordered_schedulers(observed_schedulers)

        # Plot 1: system throughput vs num_robots.
        _plot_metric_vs_num_robots(
            plots_dir / f"system_throughput_lines__mbs{mbs}.png",
            df,
            cols,
            schedulers,
            metric="thr_total",
            title_prefix="System throughput",
            ylabel="Cluster throughput (successes / min)",
            y_scale=60.0,
            nr_filter=nr_filter,
        )
        # Plot 2: avg starvation vs num_robots.
        _plot_metric_vs_num_robots(
            plots_dir / f"avg_starvation_lines__mbs{mbs}.png",
            df,
            cols,
            schedulers,
            metric="starv",
            title_prefix="Average starvation",
            ylabel="Average starvation rate (%)",
            y_scale=100.0,
            nr_filter=nr_filter,
        )
        # Plot 3: throughput-throughput tradeoff (1f9s + 5f5s only).
        _plot_throughput_tradeoff(
            plots_dir / f"throughput_tradeoff__mbs{mbs}.png",
            df,
            cols,
            schedulers,
            nr_filter=nr_filter,
        )
        # Plots 4–5: per-tier throughput vs num_robots.
        # These reveal what gets masked in the cluster total when one tier
        # outnumbers the other (e.g. 1f9s, where +1 fast robot's gain is
        # swamped by tiny -slow per-robot drops × 9).
        _plot_metric_vs_num_robots(
            plots_dir / f"fast_throughput_lines__mbs{mbs}.png",
            df,
            cols,
            schedulers,
            metric="thr_fast",
            title_prefix="Fast-tier throughput",
            ylabel="Fast robot throughput (successes / min)",
            y_scale=60.0,
            nr_filter=nr_filter,
        )
        _plot_metric_vs_num_robots(
            plots_dir / f"slow_throughput_lines__mbs{mbs}.png",
            df,
            cols,
            schedulers,
            metric="thr_slow",
            title_prefix="Slow-tier throughput",
            ylabel="Slow robot throughput (successes / min)",
            y_scale=60.0,
            nr_filter=nr_filter,
        )
        # Plots 6–7: per-tier starvation vs num_robots (mirrors per-tier throughput).
        _plot_metric_vs_num_robots(
            plots_dir / f"fast_starvation_lines__mbs{mbs}.png",
            df,
            cols,
            schedulers,
            metric="starv_fast",
            title_prefix="Fast-tier starvation",
            ylabel="Fast robot starvation rate (%)",
            y_scale=100.0,
            nr_filter=nr_filter,
        )
        _plot_metric_vs_num_robots(
            plots_dir / f"slow_starvation_lines__mbs{mbs}.png",
            df,
            cols,
            schedulers,
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
            df,
            cols,
            schedulers,
            nr_filter=nr_filter,
        )
        n_written += 8

        # Plots 9+: per-scenario tier breakdown — 2×2 grid of (throughput/starv) ×
        # (fast/slow). Only emitted for scenarios that have both tiers (1f9s, 5f5s).
        tiered_present = [sc for sc in TIERED_SCENARIOS if any(sc == k[0] for k in cols)]
        for scenario in tiered_present:
            _plot_tier_breakdown(
                plots_dir / f"tier_breakdown__{scenario}__mbs{mbs}.png",
                df,
                cols,
                schedulers,
                scenario,
                nr_filter=nr_filter,
            )
            n_written += 1
        # Plot N+1: combined tier breakdown for all tiered scenarios side by
        # side. Two variants: auto-scaled per panel, and y-aligned per row.
        if len(tiered_present) >= 2:
            _plot_tier_breakdown_combined(
                plots_dir / f"tier_breakdown_combined__mbs{mbs}.png",
                df,
                cols,
                schedulers,
                tiered_present,
                nr_filter=nr_filter,
            )
            _plot_tier_breakdown_combined(
                plots_dir / f"tier_breakdown_combined_aligned__mbs{mbs}.png",
                df,
                cols,
                schedulers,
                tiered_present,
                nr_filter=nr_filter,
                share_y=True,
            )
            n_written += 2

    print(f"Wrote {n_written} PNGs to {plots_dir}")


if __name__ == "__main__":
    main()
