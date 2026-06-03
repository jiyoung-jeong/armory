"""Combine network-ablation panels into one appendix figure per ablation mode.

Lays out a 3-row x 2-col grid: rows are scenarios (All Fast / One Fast /
Half Fast), columns are System throughput and Average starvation. The x-axis
(latency median in ms, or jitter sigma) and per-column y-axis are shared so
scenarios read on the same scale; a single scheduler legend sits on top.

All panels in one figure must share the same ablation mode (median or
variance) — the mode is auto-detected from each run dir and they must agree.

Run (one figure per mode):
    uv run python scripts/interactive/plot_network_ablation_combined.py \\
        --out experiments/sweeps/interactive/net_ablation_plots/median_combined.png \\
        --panel "All Fast:experiments/.../net_ablation_medians_hom/<stamp>" \\
        --panel "One Fast:experiments/.../net_ablations_medians_1f9s" \\
        --panel "Half Fast:experiments/.../net_ablations_medians_5f5s"
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

_HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from plot_network_ablation import (  # noqa: E402
    SCHEDULER_COLORS,
    SCHEDULER_MARKERS,
    _collect,
    _scheduler_display,
    _scheduler_sort_key,
)


def _series(data, sch, metric):
    pts = sorted((x, vals) for (x, s), vals in data.items() if s == sch)
    xs = [x for x, _ in pts]
    ys = [float(np.mean(v[metric])) for _, v in pts]
    return xs, ys


def _draw(ax, data, metric, *, y_scale, log_x):
    schedulers = sorted({s for _, s in data}, key=_scheduler_sort_key)
    handles = []
    yvals: list[float] = []
    for sch in schedulers:
        xs, ys = _series(data, sch, metric)
        if not xs:
            continue
        ys = [y * y_scale for y in ys]
        yvals.extend(ys)
        (line,) = ax.plot(
            xs, ys,
            marker=SCHEDULER_MARKERS.get(sch, "o"),
            color=SCHEDULER_COLORS.get(sch, "#444444"),
            linewidth=2.0, markersize=6.5,
            label=_scheduler_display(sch),
        )
        handles.append((sch, line))
    if log_x:
        ax.set_xscale("log")
        all_x = sorted({x for x, _ in data})
        ax.set_xticks(all_x)
        ax.set_xticklabels([f"{x:g}" for x in all_x])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_axisbelow(True)
    ax.grid(axis="y", linestyle="-", color="#dddddd", linewidth=0.8)
    ax.tick_params(axis="both", labelsize=11)
    return handles, yvals


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--panel", action="append", required=True,
                   help='"<row label>:<run dir>" — repeat once per scenario row.')
    p.add_argument("--out", type=pathlib.Path, required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    panels = []
    modes = set()
    for spec in args.panel:
        label, _, run = spec.partition(":")
        run_dir = pathlib.Path(run).expanduser().resolve()
        if not run_dir.is_dir():
            raise SystemExit(f"Not a directory: {run_dir}")
        data, mode = _collect(run_dir)
        if not data:
            raise SystemExit(f"No ok cases parsed under {run_dir}")
        panels.append((label.strip(), data))
        modes.add(mode)
    if len(modes) != 1:
        raise SystemExit(f"Panels mix ablation modes: {modes}")
    mode = modes.pop()

    if mode == "median":
        xlabel = "Network latency median (ms)"
        log_x = True
    else:
        xlabel = "Network latency jitter ($\\sigma$, log-space)"
        log_x = False

    n = len(panels)
    fig, axes = plt.subplots(
        n, 2, figsize=(8.6, 2.7 * n + 0.6),
        sharex="col", sharey=False, squeeze=False,
    )
    fig.set_facecolor("white")

    legend_handles = None
    col_yvals: list[list[float]] = [[], []]  # accumulate per-column data extents
    for r, (label, data) in enumerate(panels):
        h_thr, y_thr = _draw(axes[r][0], data, "thr", y_scale=1.0, log_x=log_x)
        _, y_st = _draw(axes[r][1], data, "starv", y_scale=100.0, log_x=log_x)
        col_yvals[0].extend(y_thr)
        col_yvals[1].extend(y_st)
        axes[r][0].set_ylabel(label, fontsize=13, fontweight="bold")
        if legend_handles is None:
            legend_handles = h_thr

    # Give every subplot in a column the SAME min/max, padded by 12% of the
    # column's data span so no line hugs the top/bottom edge.
    for c in range(2):
        ys = col_yvals[c]
        if not ys:
            continue
        lo, hi = min(ys), max(ys)
        pad = 0.12 * (hi - lo or 1.0)
        ylo, yhi = lo - pad, hi + pad
        for r in range(n):
            axes[r][c].set_ylim(ylo, yhi)

    axes[0][0].set_title("System throughput (successes / min)", fontsize=12)
    axes[0][1].set_title("Average starvation rate (%)", fontsize=12)
    for c in range(2):
        axes[n - 1][c].set_xlabel(xlabel, fontsize=12)

    if legend_handles:
        handles = [ln for _, ln in legend_handles]
        labels = [_scheduler_display(s) for s, _ in legend_handles]
        fig.legend(
            handles, labels, loc="upper center", ncol=len(handles),
            fontsize=11, frameon=False, bbox_to_anchor=(0.5, 1.0),
        )

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150, facecolor="white")
    fig.savefig(args.out.with_suffix(".pdf"), facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {args.out} (+ .pdf)  [mode={mode}, {n} scenarios]")


if __name__ == "__main__":
    main()
