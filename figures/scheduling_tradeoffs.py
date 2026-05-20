"""Throughput vs. execution horizon and starvation rate."""

from __future__ import annotations

import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import style

OUTPUT_DIR = pathlib.Path(__file__).parent

# starvation_rate (fraction), dynamic trial successes, static trial successes
# Each trial entry is out of 30 runs.
_STARVATION_DATA = [
    (0.0, [7, 7, 6], [6, 5, 6]),
    (0.1, [6, 7, 7], [6, 5, 6]),
    (0.3, [4, 4, 5], [5, 6, 5]),
    (0.5, [3, 2, 2], [4, 5, 4]),
    (0.7, [1, 2, 0], [3, 4, 3]),
    (0.9, [0, 0, 0], [2, 3, 2]),
]

# min_execution_horizon (steps), dynamic trial successes, static trial successes
# Each trial entry is out of 30 runs.
_HORIZON_DATA = [
    (5, [7, 8, 6], [7, 6, 5]),
    (10, [7, 8, 8], [6, 6, 7]),
    (20, [6, 4, 5], [7, 7, 5]),
    (30, [4, 2, 3], [5, 6, 6]),
    (40, [4, 2, 3], [5, 5, 6]),
    (50, [0, 1, 2], [5, 6, 6]),
    (60, [1, 0, 1], [6, 6, 6]),
]
_TRIALS_PER_GROUP = 30


def plot_horizon(
    ax: plt.Axes | None = None, *, output_path: pathlib.Path | None = None
) -> plt.Axes:
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(3.5, 2.8))

    horizons = np.array([d[0] for d in _HORIZON_DATA])
    dyn = np.array([d[1] for d in _HORIZON_DATA], dtype=float) / _TRIALS_PER_GROUP
    sta = np.array([d[2] for d in _HORIZON_DATA], dtype=float) / _TRIALS_PER_GROUP

    for label, tpr, color_idx, marker in [
        ("Static task", sta, 0, "s"),
        ("Dynamic task", dyn, 1, "o"),
    ]:
        mean = tpr.mean(axis=1)
        std = tpr.std(axis=1)
        c = style.color(color_idx)
        ax.plot(horizons, mean, marker=marker, color=c, label=label, zorder=3)
        ax.fill_between(horizons, mean - std, mean + std, color=c, alpha=0.18, linewidth=0)

    ax.set_xlabel("Min. execution horizon (steps)")
    ax.set_ylabel("Throughput (succ/s)")
    ax.set_xlim(0, 65)
    ax.set_ylim(0, 0.40)
    ax.set_xticks(horizons)
    style.yonly_grid(ax)
    ax.legend()

    if standalone:
        fig.tight_layout()
        out = output_path or (OUTPUT_DIR / "throughput_vs_horizon.pdf")
        fig.savefig(out)
        plt.close(fig)
        print(f"Saved {out}")

    return ax


def plot_starvation(
    ax: plt.Axes | None = None, *, output_path: pathlib.Path | None = None
) -> plt.Axes:
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(3.5, 2.8))

    rates = np.array([d[0] for d in _STARVATION_DATA]) * 100  # fraction → %
    dyn = np.array([d[1] for d in _STARVATION_DATA], dtype=float) / _TRIALS_PER_GROUP
    sta = np.array([d[2] for d in _STARVATION_DATA], dtype=float) / _TRIALS_PER_GROUP

    for label, tpr, color_idx, marker in [
        ("Static task", sta, 0, "s"),
        ("Dynamic task", dyn, 1, "o"),
    ]:
        mean = tpr.mean(axis=1)
        std = tpr.std(axis=1)
        c = style.color(color_idx)
        ax.plot(rates, mean, marker=marker, color=c, label=label, zorder=3)
        ax.fill_between(rates, mean - std, mean + std, color=c, alpha=0.18, linewidth=0)

    ax.set_xlabel("Starvation rate (%)")
    ax.set_ylabel("Throughput (succ/s)")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 0.40)
    ax.set_xticks(rates)
    style.yonly_grid(ax)
    ax.legend()

    if standalone:
        fig.tight_layout()
        out = output_path or (OUTPUT_DIR / "throughput_vs_starvation.pdf")
        fig.savefig(out)
        plt.close(fig)
        print(f"Saved {out}")

    return ax


def plot_combined(*, output_path: pathlib.Path | None = None) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8))
    plot_horizon(axes[0])
    plot_starvation(axes[1])
    fig.tight_layout()
    out = output_path or (OUTPUT_DIR / "scheduling_tradeoffs.pdf")
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved {out}")
    return fig


def main() -> None:
    style.apply()
    plot_horizon()
    plot_starvation()
    plot_combined()


if __name__ == "__main__":
    main()
