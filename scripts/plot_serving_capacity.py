"""Export serving-capacity curves from per-epoch CSV (means and sample SD)."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = pd.read_csv(args.csv)
    args.output.mkdir(parents=True, exist_ok=True)
    keys = ["robots", "control_hz", "request_hz", "max_batch_size", "algorithm"]
    metrics = [
        "completion_rps",
        "slo_percent",
        "starvation_percent",
        "response_p99_ms",
        "queue_wait_p95_ms",
        "actual_batch_mean",
        "unanswered",
        "discarded_prefix_mean",
        "tick_lateness_p99_ms",
        "executed_actions_per_s",
    ]
    grouped = data.groupby(keys)[metrics].agg(["mean", "std"])
    grouped.columns = ["_".join(column) for column in grouped.columns]
    grouped.to_csv(args.output / "aggregate.csv")
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.3), constrained_layout=True)
    baseline = data[data.control_hz == 20]
    repeat_counts = baseline.groupby(keys).size()
    repeat_label = (
        f"{repeat_counts.min()}–{repeat_counts.max()} epochs per condition"
        if repeat_counts.min() != repeat_counts.max()
        else f"{repeat_counts.min()} epoch(s) per condition"
    )
    labels = [
        ("completion_rps", "Completed chunks / second"),
        ("response_p99_ms", "P99 response latency (ms; responded only)"),
        ("slo_percent", "156 ms attainment (% of ALL sent requests)"),
        ("starvation_percent", "Control ticks with no action (%)"),
    ]
    for ax, (metric, label) in zip(axes.flat, labels, strict=True):
        for batch, frame in baseline.groupby("max_batch_size"):
            group = frame.groupby("robots")[metric].agg(["mean", "std"])
            ax.errorbar(
                group.index,
                group["mean"],
                yerr=group["std"].fillna(0),
                marker="o",
                capsize=3,
                label=f"max batch {batch}",
            )
        ax.set(xlabel="Robots", ylabel=label)
        ax.grid(alpha=0.2)
    demand = baseline.groupby("robots").offered_rps.mean()
    axes[0, 0].plot(demand.index, demand.values, "--", color="gray", label="Offered requests")
    axes[0, 0].legend()
    axes[0, 1].axhline(156, ls="--", color="gray", label="PI0 reference = 156 ms")
    axes[0, 1].legend(fontsize=9)
    axes[1, 0].axhline(98, ls="--", color="gray")
    axes[1, 0].set_ylim(-3, 103)
    axes[1, 1].set_ylim(bottom=0)
    fig.suptitle(
        "PI05 / RTX A6000 / loopback / LIBERO snapshots / 2 Hz per robot\n"
        f"20 Hz control, horizon 10; {repeat_label}; mean ± sample SD"
    )
    for ext in ["png", "svg"]:
        fig.savefig(args.output / f"capacity.{ext}", dpi=170)
    plt.close(fig)
    probes = data[data.robots.isin([1, 4])]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.3), constrained_layout=True)
    for ax, metric, label in zip(
        axes,
        ["slo_percent", "starvation_percent"],
        ["156 ms attainment (%)", "Control ticks with no action (%)"],
        strict=True,
    ):
        for robots, frame in probes.groupby("robots"):
            group = frame.groupby("control_hz")[metric].agg(["mean", "std"])
            ax.errorbar(
                group.index,
                group["mean"],
                yerr=group["std"].fillna(0),
                marker="o",
                capsize=3,
                label=f"{robots} robot(s)",
            )
        ax.set(
            xlabel="Action consumption / control frequency (Hz)",
            ylabel=label,
            xticks=[10, 20, 40],
            ylim=(-3, 103),
        )
        ax.grid(alpha=0.2)
        ax.legend()
    axes[0].axhline(98, ls="--", color="gray")
    fig.suptitle(
        "Request frequency stays 2 Hz; horizon stays 10\n"
        "Measured request SLO and action availability are distinct"
    )
    for ext in ["png", "svg"]:
        fig.savefig(args.output / f"slo_vs_actions.{ext}", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    main()
