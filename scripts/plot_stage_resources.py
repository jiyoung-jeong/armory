"""Plot measured stage activity counters and an aligned GPU timeline."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection

COLORS = {"vlm_embed": "#4e79a7", "vlm_prefill": "#59a14f", "action": "#e15759"}


def plots(root):
    folder = root / "nsys/analysis"
    summary = pd.read_csv(folder / "resource_summary.csv")
    labels = ["Image/text embedding", "Prefix prefill", "VLM total", "Action (10 iterations)"]
    stages = ["vlm_embed", "vlm_prefill", "vlm_total", "action"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), layout="constrained")
    for ax, metric, title in zip(
        axes,
        ["tensor_active_pct", "sm_active_pct"],
        ["Tensor pipe active", "SM active"],
        strict=True,
    ):
        for i, b in enumerate([1, 5]):
            data = summary[summary.batch_size.eq(b)].set_index("stage").loc[stages]
            bars = ax.bar(
                np.arange(4) + (i - 0.5) * 0.36,
                data[metric],
                0.36,
                label=f"Batch {b}",
                color=["#648fff", "#dc267f"][i],
            )
            ax.bar_label(bars, fmt="%.1f", fontsize=8, padding=3)
        ax.set(
            xticks=np.arange(4), xticklabels=labels, ylim=(0, 108), ylabel="Percent", title=title
        )
        ax.tick_params(axis="x", labelsize=8)
        ax.grid(axis="y", alpha=0.2)
        ax.legend()
    fig.suptitle(
        "Live GPU counters over attributed stage activity; 10 calls per batch; 10 kHz sampling (CPU work and GPU gaps excluded)",
        fontsize=10,
    )
    for ext in ["png", "pdf"]:
        fig.savefig(folder / f"resource_comparison.{ext}", dpi=180)
    plt.close(fig)
    samples = pd.read_csv(folder / "resource_samples.csv")
    calls = pd.read_csv(folder / "calls.csv")
    events = pd.read_csv(
        folder / "gpu_activities.csv",
        usecols=["batch_size", "phase", "index", "start", "end", "stage"],
    )
    fig, axes = plt.subplots(2, 1, figsize=(13, 7), layout="constrained")
    for ax, b in zip(axes, [1, 5], strict=True):
        call = calls[
            calls.batch_size.eq(b) & calls.phase.eq("profile") & calls["index"].eq(5)
        ].iloc[0]
        data = samples[samples.timestamp_ns.between(call.start, call.end)]
        x = (data.timestamp_ns - call.start) / 1e6
        ax.plot(x, data.sm_active_pct, color="#999999", lw=0.6, alpha=0.75, label="SM active")
        ax.plot(x, data.tensor_active_pct, color="#2040ae", lw=0.9, label="Tensor active")
        selected = events[
            events.batch_size.eq(b) & events.phase.eq("profile") & events["index"].eq(5)
        ].copy()
        selected["stage"] = selected.stage.replace({"vlm_mixed": "vlm_prefill"})
        for stage, y in zip(COLORS, [-8, -15, -22], strict=True):
            group = selected[selected.stage.eq(stage)]
            segments = [
                [((r.start - call.start) / 1e6, y), ((r.end - call.start) / 1e6, y)]
                for r in group.itertuples()
            ]
            ax.add_collection(
                LineCollection(segments, color=COLORS[stage], linewidth=4, label=stage)
            )
        ax.set(
            xlim=(0, (call.end - call.start) / 1e6),
            ylim=(-29, 105),
            yticks=[0, 25, 50, 75, 100],
            ylabel="Percent",
            xlabel="Milliseconds from infer_batch start",
            title=f"Batch {b}: profile call 5 (one observed call)",
        )
        ax.legend(loc="upper left", fontsize=8, ncol=5)
        ax.grid(alpha=0.2)
    for ext in ["png", "pdf"]:
        fig.savefig(folder / f"resource_timeline.{ext}", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    plots(parser.parse_args().root)
