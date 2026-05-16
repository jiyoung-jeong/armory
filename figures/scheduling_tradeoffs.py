"""Throughput vs. execution horizon and starvation rate (skeleton — axes only)."""

from __future__ import annotations

import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import style
from matplotlib.lines import Line2D

OUTPUT_DIR = pathlib.Path(__file__).parent


def plot_horizon(
    ax: plt.Axes | None = None, *, output_path: pathlib.Path | None = None
) -> plt.Axes:
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(3.5, 2.8))

    ax.set_xlabel("Execution horizon (ms)")
    ax.set_ylabel("Throughput (succ/s)")
    ax.set_xlim(0, 1000)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels([])
    style.yonly_grid(ax)
    ax.legend(
        handles=[
            Line2D([], [], color=style.color(0), label="Static task"),
            Line2D([], [], color=style.color(1), label="Dynamic task"),
        ]
    )

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

    ax.set_xlabel("Starvation rate (%)")
    ax.set_ylabel("Throughput (succ/s)")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels([])
    style.yonly_grid(ax)
    ax.legend(
        handles=[
            Line2D([], [], color=style.color(0), label="Static task"),
            Line2D([], [], color=style.color(1), label="Dynamic task"),
        ]
    )

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
