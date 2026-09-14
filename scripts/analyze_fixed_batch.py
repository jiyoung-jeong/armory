"""Summarize synchronized fixed-input and prewarmed dynamic-shape timings."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from scripts.analyze_followup_events import stats


def analyze(path):
    manifest = json.loads((path / "manifest.json").read_text())
    assert manifest["status"] == "complete"
    calls = pd.read_json(path / "calls.jsonl", lines=True)
    assert len(calls) == manifest["calls"]
    fixed = calls[calls.phase == "fixed"]
    expected = manifest["repeats"] * manifest["samples_per_size_per_repeat"]
    assert fixed.groupby("batch_size").size().to_dict() == {size: expected for size in range(1, 5)}
    baseline = fixed[fixed.batch_size == 1].duration_ms.mean()
    rows = []
    for (phase, size), g in calls.groupby(["phase", "batch_size"]):
        s = stats(g.duration_ms)
        rows.append(
            dict(
                phase=phase,
                batch_size=int(size),
                **s,
                ms_per_request=s["mean"] / size,
                requests_per_second=1000 * size / s["mean"],
                throughput_speedup_vs_fixed_b1=size * baseline / s["mean"],
            )
        )
    summary = pd.DataFrame(rows)
    summary.to_csv(path / "summary.csv", index=False)
    by_repeat = (
        fixed.groupby(["repeat", "batch_size"])
        .duration_ms.agg(["count", "mean", "std", "min", "max"])
        .reset_index()
    )
    by_repeat.to_csv(path / "repeats.csv", index=False)
    dynamic = calls[calls.phase == "dynamic"]
    transitions = (
        dynamic.groupby(["previous_batch", "batch_size"])
        .duration_ms.agg(["count", "mean", "max"])
        .reset_index()
    )
    transitions.to_csv(path / "transitions.csv", index=False)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), layout="constrained")
    for phase, g in summary.groupby("phase"):
        axes[0].plot(g.batch_size, g["mean"], "o-", label=phase)
        axes[1].plot(g.batch_size, g.ms_per_request, "o-", label=phase)
        axes[2].plot(g.batch_size, g.requests_per_second, "o-", label=phase)
    for size, g in by_repeat.groupby("batch_size"):
        axes[0].scatter([size] * len(g), g["mean"], s=16, alpha=0.5, color="black")
    for ax, ylabel in zip(
        axes,
        ["Full infer_batch (ms)", "Milliseconds per request", "Requests / second"],
        strict=True,
    ):
        ax.set(xlabel="Actual batch size", ylabel=ylabel, xticks=[1, 2, 3, 4])
        ax.grid(alpha=0.2)
    axes[0].legend()
    fig.suptitle(
        "Fixed real LIBERO inputs; pi05, 10 denoising steps, JAX; no renderer\nDynamic sequence 1-2-3-2 after all shapes were warmed; black points = fixed block means"
    )
    fig.savefig(path / "batch_cost.png", dpi=170)
    fig.savefig(path / "batch_cost.pdf")
    plt.close(fig)
    print(summary.to_string(index=False))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    analyze(args.path)


if __name__ == "__main__":
    main()
