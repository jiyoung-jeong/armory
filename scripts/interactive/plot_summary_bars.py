"""Bar-chart plots from a summary folder produced by ``summarize_sweep_runs.py``.

Reads ``results_max_batch_size={M}.csv`` files (columns shaped
``<scenario>__<scheduler>__<metric>``) and emits, per ``(num_robots,
max_batch_size)`` slice, six PNGs under ``<summary_dir>/plots/``:

  Throughput (x=scheduler, y=throughput)
    1. throughput__hom__nr={N}_mbs={M}.png    — single bar per scheduler
                                                 (fast tier only)
    2. throughput__1f9s__nr={N}_mbs={M}.png   — two bars per scheduler
                                                 (fast vs slow)
    3. throughput__5f5s__nr={N}_mbs={M}.png   — same shape as plot 2

  Starvation (x=scheduler, y=starvation rate)
    4. starvation__hom__nr={N}_mbs={M}.png    — two bars per scheduler
                                                 (avg vs worst-robot)
    5. starvation__1f9s__nr={N}_mbs={M}.png   — same shape
    6. starvation__5f5s__nr={N}_mbs={M}.png   — same shape

Schedulers with no data for that slice are skipped. Slices with no usable
schedulers are not written.

Run:
    uv run python scripts/interactive/plot_summary_bars.py \\
        experiments/sweeps/interactive/_summary
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

PREFERRED_SCHEDULER_ORDER = [
    "max-batch",
    "round-robin",
    "lookahead-actions@ahm=1",
    "lookahead-actions@ahm=3",
    "lookahead-actions@ahm=5",
]
HOM_SCENARIO = "hom"
TWO_TIER_SCENARIOS = ("1f9s", "5f5s")
MBS_RE = re.compile(r"results_max_batch_size=(\d+)\.csv$")
COL_RE = re.compile(r"^(?P<scenario>[^_]+(?:_[^_]+)*?)__(?P<scheduler>.+?)__(?P<metric>starv|thr_fast|thr_slow|worst|n)$")


def _load_summary(summary_dir: pathlib.Path) -> dict[int, pd.DataFrame]:
    out: dict[int, pd.DataFrame] = {}
    for path in sorted(summary_dir.glob("results_max_batch_size=*.csv")):
        m = MBS_RE.search(path.name)
        if not m:
            continue
        df = pd.read_csv(path, index_col="num_robots")
        out[int(m.group(1))] = df
    if not out:
        raise SystemExit(f"No results CSVs found in {summary_dir}")
    return out


def _parse_columns(df: pd.DataFrame) -> dict[tuple[str, str], dict[str, str]]:
    """{(scenario, scheduler): {metric: column_name}}"""
    out: dict[tuple[str, str], dict[str, str]] = {}
    for col in df.columns:
        m = COL_RE.match(col)
        if not m:
            continue
        key = (m.group("scenario"), m.group("scheduler"))
        out.setdefault(key, {})[m.group("metric")] = col
    return out


def _ordered_schedulers(present: list[str]) -> list[str]:
    head = [s for s in PREFERRED_SCHEDULER_ORDER if s in present]
    rest = sorted(s for s in present if s not in PREFERRED_SCHEDULER_ORDER)
    return head + rest


def _fmt(v: float) -> str:
    return f"{v:.3f}"


def _annotate(ax, x: float, y: float, val: float | None) -> None:
    if val is None or pd.isna(val):
        return
    ax.text(x, y, _fmt(val), ha="center", va="bottom", fontsize=8)


def _plot_single_bar(
    out_path: pathlib.Path,
    schedulers: list[str],
    values: list[float | None],
    title: str,
    ylabel: str,
    color: str = "#1f77b4",
) -> None:
    keep = [(s, v) for s, v in zip(schedulers, values) if v is not None and not pd.isna(v)]
    if not keep:
        return
    schedulers_k, values_k = zip(*keep)
    fig, ax = plt.subplots(figsize=(max(6, 0.9 * len(schedulers_k) + 3), 4.5))
    xs = np.arange(len(schedulers_k))
    bars = ax.bar(xs, values_k, color=color, edgecolor="black", linewidth=0.6)
    for x, b, v in zip(xs, bars, values_k):
        _annotate(ax, x, b.get_height(), v)
    ax.set_xticks(xs)
    ax.set_xticklabels(schedulers_k, rotation=20, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _plot_grouped_bars(
    out_path: pathlib.Path,
    schedulers: list[str],
    series: list[tuple[str, list[float | None], str]],
    title: str,
    ylabel: str,
) -> None:
    """series = [(legend_label, [value_per_scheduler], color), ...]"""
    # Keep schedulers with at least one non-NaN value across any series.
    keep_idx = [
        i for i in range(len(schedulers))
        if any(s[1][i] is not None and not pd.isna(s[1][i]) for s in series)
    ]
    if not keep_idx:
        return
    schedulers_k = [schedulers[i] for i in keep_idx]
    series_k = [(lab, [vs[i] for i in keep_idx], col) for lab, vs, col in series]

    n_series = len(series_k)
    width = 0.8 / n_series
    xs = np.arange(len(schedulers_k))
    fig, ax = plt.subplots(figsize=(max(7, 1.1 * len(schedulers_k) + 3), 4.8))
    for s_idx, (label, vals, color) in enumerate(series_k):
        offset = (s_idx - (n_series - 1) / 2) * width
        plot_vals = [0.0 if (v is None or pd.isna(v)) else v for v in vals]
        bars = ax.bar(
            xs + offset, plot_vals, width=width,
            label=label, color=color, edgecolor="black", linewidth=0.6,
        )
        for x, b, raw in zip(xs + offset, bars, vals):
            _annotate(ax, x, b.get_height(), raw)
    ax.set_xticks(xs)
    ax.set_xticklabels(schedulers_k, rotation=20, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="best", framealpha=0.9)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _row_value(row: pd.Series, cols: dict[tuple[str, str], dict[str, str]],
               scenario: str, scheduler: str, metric: str) -> float | None:
    col = cols.get((scenario, scheduler), {}).get(metric)
    if col is None or col not in row.index:
        return None
    v = row[col]
    if pd.isna(v):
        return None
    return float(v)


def _emit_for_slice(
    plots_dir: pathlib.Path,
    mbs: int,
    nr: int,
    row: pd.Series,
    cols: dict[tuple[str, str], dict[str, str]],
) -> None:
    scenarios = sorted({sc for sc, _ in cols})
    schedulers = _ordered_schedulers(sorted({sch for _, sch in cols}))

    # --- Plot 1: hom throughput (fast tier only — single bar per scheduler).
    if HOM_SCENARIO in scenarios:
        vals = [_row_value(row, cols, HOM_SCENARIO, s, "thr_fast") for s in schedulers]
        _plot_single_bar(
            plots_dir / f"throughput__{HOM_SCENARIO}__nr={nr}_mbs={mbs}.png",
            schedulers, vals,
            title=f"hom throughput  (num_robots={nr}, max_batch_size={mbs})",
            ylabel="Throughput  (successes / sec)",
            color="#1f77b4",
        )

    # --- Plots 2, 3: 1f9s / 5f5s throughput (fast + slow grouped bars).
    for scenario in TWO_TIER_SCENARIOS:
        if scenario not in scenarios:
            continue
        fast = [_row_value(row, cols, scenario, s, "thr_fast") for s in schedulers]
        slow = [_row_value(row, cols, scenario, s, "thr_slow") for s in schedulers]
        _plot_grouped_bars(
            plots_dir / f"throughput__{scenario}__nr={nr}_mbs={mbs}.png",
            schedulers,
            series=[
                ("fast tier", fast, "#1f77b4"),
                ("slow tier", slow, "#ff7f0e"),
            ],
            title=f"{scenario} throughput  (num_robots={nr}, max_batch_size={mbs})",
            ylabel="Throughput  (successes / sec)",
        )

    # --- Plots 4–6: per scenario, avg starvation + worst-robot starvation.
    for scenario in (HOM_SCENARIO, *TWO_TIER_SCENARIOS):
        if scenario not in scenarios:
            continue
        avg = [_row_value(row, cols, scenario, s, "starv") for s in schedulers]
        worst = [_row_value(row, cols, scenario, s, "worst") for s in schedulers]
        _plot_grouped_bars(
            plots_dir / f"starvation__{scenario}__nr={nr}_mbs={mbs}.png",
            schedulers,
            series=[
                ("avg starvation", avg, "#2ca02c"),
                ("worst-robot starvation", worst, "#d62728"),
            ],
            title=f"{scenario} starvation  (num_robots={nr}, max_batch_size={mbs})",
            ylabel="Starvation rate",
        )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("summary_dir", type=pathlib.Path,
                   help="Directory containing results_max_batch_size=*.csv (output of summarize_sweep_runs.py).")
    p.add_argument("--plots-subdir", default="plots",
                   help="Subdir under summary_dir to write PNGs into (default: plots).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    summary_dir = args.summary_dir.resolve()
    if not summary_dir.is_dir():
        raise SystemExit(f"Not a directory: {summary_dir}")
    plots_dir = summary_dir / args.plots_subdir
    plots_dir.mkdir(parents=True, exist_ok=True)

    by_mbs = _load_summary(summary_dir)
    n_written = 0
    for mbs, df in by_mbs.items():
        cols = _parse_columns(df)
        for nr, row in df.iterrows():
            before = len(list(plots_dir.iterdir()))
            _emit_for_slice(plots_dir, mbs, int(nr), row, cols)
            after = len(list(plots_dir.iterdir()))
            n_written += after - before
    print(f"Wrote {n_written} PNGs to {plots_dir}")


if __name__ == "__main__":
    main()
