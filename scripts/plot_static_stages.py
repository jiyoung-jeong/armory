"""Create shareable figures from validated per-stage Nsight analysis."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection

COLORS = {"vlm_embed": "#4e79a7", "vlm_prefill": "#59a14f", "action": "#e15759", "other": "#999999"}
LABELS = {
    "vlm_embed": "VLM: image/text embeddings",
    "vlm_prefill": "VLM: prefill / shared VLM",
    "action": "Action generation (10 iterations)",
    "other": "Other / mixed",
}


def plots(folder):
    dest = folder / "analysis"
    summary = pd.read_csv(dest / "summary.csv")
    steps = pd.read_csv(dest / "action_steps.csv")
    calls = pd.read_csv(dest / "calls.csv")
    calls = calls[calls.phase.eq("profile")]
    events = pd.read_csv(
        dest / "gpu_activities.csv",
        usecols=[
            "start",
            "end",
            "stream",
            "kind",
            "phase",
            "batch_size",
            "index",
            "stage",
            "action_step",
            "bytes",
            "copy_kind",
        ],
    )
    events = events[events.phase.eq("profile")]
    fig, ax = plt.subplots(figsize=(9, 5), layout="constrained")
    bottom = np.zeros(len(summary))
    for stage in ["vlm_embed", "vlm_prefill", "action", "other"]:
        if stage == "vlm_prefill":
            values = summary.vlm_prefill_activity_ms + summary.vlm_mixed_activity_ms
        elif stage == "other":
            values = summary.mixed_activity_ms + summary.unattributed_activity_ms
        else:
            values = summary[stage + "_activity_ms"]
        ax.bar(summary.batch_size, values, bottom=bottom, color=COLORS[stage], label=LABELS[stage])
        bottom += values.to_numpy()
    ax.plot(
        summary.batch_size, summary.full_ms, "ko--", label="Full infer_batch wall time (profiled)"
    )
    ax.set(
        xlabel="Actual batch size",
        ylabel="Milliseconds per batch",
        xticks=summary.batch_size,
        title="GPU activity by logical stage: kernels + copies + memsets\nMean of 10 profiled calls per B; gaps contain no recorded GPU activity",
    )
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.2)
    fig.savefig(dest / "stage_cost.png", dpi=180)
    fig.savefig(dest / "stage_cost.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4), layout="constrained")
    for size, group in steps.groupby("batch_size"):
        by_step = group.groupby("action_step").activity_union_ms.agg(["mean", "std"])
        x = by_step.index.to_numpy() + 1
        ax.plot(x, by_step["mean"], "o-", label=f"B{size}")
        ax.fill_between(
            x, by_step["mean"] - by_step["std"], by_step["mean"] + by_step["std"], alpha=0.1
        )
    ax.set(
        xlabel="Denoising iteration (updates the entire chunk)",
        ylabel="GPU activity union (ms)",
        xticks=range(1, 11),
        title="Action loop: per-iteration GPU cost (mean ± sample SD; 10 calls per B)",
    )
    ax.legend(ncol=5)
    ax.grid(alpha=0.2)
    fig.savefig(dest / "action_iterations.png", dpi=180)
    fig.savefig(dest / "action_iterations.pdf")
    plt.close(fig)

    extremes = sorted(set([summary.batch_size.min(), summary.batch_size.max()]))
    fig, axes = plt.subplots(
        len(extremes), 1, figsize=(13, 4 * len(extremes)), layout="constrained", squeeze=False
    )
    for size, axis in zip(extremes, axes[:, 0], strict=True):
        index = 5 if len(calls[calls.batch_size.eq(size)]) > 5 else 0
        call = calls[calls.batch_size.eq(size) & calls["index"].eq(index)].iloc[0]
        selected = events[events.batch_size.eq(size) & events["index"].eq(index)].copy()
        selected["lane"] = selected.stage.replace(
            {"vlm_mixed": "vlm_prefill", "mixed": "other", "unattributed": "other"}
        )
        for stage, y in [("vlm_embed", 3), ("vlm_prefill", 2), ("action", 1), ("other", 0)]:
            for kernel, offset, width, color in [
                (True, 0.09, 7, COLORS[stage]),
                (False, -0.09, 4, "#333333"),
            ]:
                group = selected[
                    selected.lane.eq(stage)
                    & (selected.kind.eq("KERNEL") if kernel else ~selected.kind.eq("KERNEL"))
                ]
                segments = [
                    [
                        (float(r.start - call.start) / 1e6, y + offset),
                        (float(r.end - call.start) / 1e6, y + offset),
                    ]
                    for r in group.itertuples()
                ]
                axis.add_collection(LineCollection(segments, colors=color, linewidths=width))
        for step, group in selected[selected.action_step.ge(0)].groupby("action_step"):
            start = (group.start.min() - call.start) / 1e6
            end = (group.end.max() - call.start) / 1e6
            axis.text((start + end) / 2, 1.31, str(int(step) + 1), ha="center", fontsize=8)
        axis.axvline(call.full_ms, color="black", linestyle="--", linewidth=1)
        axis.set(
            xlim=(-2, call.full_ms + 3),
            ylim=(-0.5, 3.6),
            yticks=[3, 2, 1, 0],
            yticklabels=["VLM embedding", "VLM prefill", "Action", "Other"],
            xlabel="Milliseconds from infer_batch start",
            title=f"B{size}, profiled call index {index}: colored = kernels, dark = copy/memset; action iterations numbered",
        )
        axis.grid(axis="x", alpha=0.2)
    fig.savefig(dest / "gpu_timeline.png", dpi=170)
    fig.savefig(dest / "gpu_timeline.pdf")
    plt.close(fig)

    copy_rows = []
    for (batch, kind), group in events[events.kind.eq("MEMCPY")].groupby(
        ["batch_size", "copy_kind"]
    ):
        copy_rows.append(
            dict(
                batch_size=batch,
                copy_kind=kind,
                bytes_per_call=group.bytes.sum() / len(calls[calls.batch_size.eq(batch)]),
            )
        )
    pd.DataFrame(copy_rows).to_csv(dest / "copy_directions.csv", index=False)
    print("Saved stage_cost, action_iterations, gpu_timeline (PNG + PDF)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    plots(parser.parse_args().folder)
