"""Summarize matched scheduler trials and timing-model sensitivity separately."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from scripts.analyze_followup_events import analyze_trial
from scripts.analyze_local_batch_sweep import summarize_trial
from scripts.analyze_mirror_provenance import analyze as analyze_provenance


def summarize(root):
    rows = []
    robot_rows = []
    audits = []
    for p in sorted(root.glob("*/run_*/manifest.json")):
        m = json.loads(p.read_text())
        if m["status"] != "complete":
            continue
        run = p.parent
        if not (run / "event_analysis/summary.json").exists():
            analyze_trial(run)
        if not (run / "mirror_audit/summary.json").exists():
            analyze_provenance(run)
        row, per_robot = summarize_trial(run)
        robot_rows.extend(dict(r, algorithm=m["algorithm"]) for r in per_robot)
        row["algorithm"] = m["algorithm"]
        row["condition"] = f"{m['algorithm']} / B{m['max_batch_size']}"
        rows.append(row)
        audits.append(json.loads((run / "mirror_audit/summary.json").read_text()))
    if not rows:
        return
    f = pd.DataFrame(rows)
    f.to_csv(root / "trials.csv", index=False)
    pd.DataFrame(robot_rows).to_csv(root / "robots.csv", index=False)
    g = (
        f.groupby(["algorithm", "max_batch"])
        .agg(
            repeats=("repeat", "count"),
            success_mean=("successes_per_min", "mean"),
            success_sd=("successes_per_min", "std"),
            starvation_mean=("post_first_starvation_rate", "mean"),
            starvation_sd=("post_first_starvation_rate", "std"),
            chunk_p95_ms=("chunk_latency_p95_ms", "mean"),
            actual_batch=("actual_batch_mean", "mean"),
            stored_chunks=("stored_chunks", "sum"),
            processed=("processed_observations", "sum"),
        )
        .reset_index()
    )
    g.to_csv(root / "conditions.csv", index=False)
    (root / "mirror_audits.json").write_text(json.dumps(audits, indent=2))
    order = ["lookahead-actions / B2", "lookahead-actions / B3", "round-robin / B2"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, metric, label, scale in zip(
        axes,
        ["successes_per_min", "post_first_starvation_rate", "chunk_latency_p95_ms"],
        ["Successful tasks / min", "Post-first starvation (%)", "Chunk latency p95 (ms)"],
        [1, 100, 1],
        strict=True,
    ):
        for x, key in enumerate(order):
            values = f[f.condition == key][metric] * scale
            if len(values):
                ax.bar(x, values.mean(), color=["#2878b5", "#f28e2b", "#59a14f"][x], alpha=0.65)
                ax.scatter([x] * len(values), values, color="black", s=20, zorder=3)
        ax.set_xticks(range(3), ["Lookahead B2", "Lookahead B3", "Round Robin B2"], rotation=15)
        ax.set_ylabel(label)
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle(
        f"4 LIBERO robots; mirror-aligned code; {len(f)} completed 180 s runs; black points = trials"
    )
    fig.tight_layout()
    fig.savefig(root / "comparison.png", dpi=170)
    fig.savefig(root / "comparison.pdf")
    plt.close(fig)
    print(g.to_string(index=False), flush=True)


def plot_model(root):
    f = pd.read_csv(root / "conditions.csv")
    colors = dict(
        zip(
            ["lookahead-actions", "round-robin", "max-batch", "greedy-deadline"],
            ["#2878b5", "#59a14f", "#af7aa1", "#f28e2b"],
            strict=True,
        )
    )
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for ax, cost in zip(axes, ["fixed_isolated", "observed_corun"], strict=True):
        for alg, color in colors.items():
            selected = f[(f.cost_source == cost) & (f.gap_ms == 5) & (f.algorithm == alg)]
            g = selected.groupby("max_batch").starvation_rate.agg(["mean", "min", "max"]) * 100
            ax.plot(g.index, g["mean"], marker="o", label=alg, color=color)
            ax.fill_between(g.index, g["min"], g["max"], color=color, alpha=0.15)
        ax.set(title=cost, xlabel="Maximum batch size", xticks=[1, 2, 3, 4])
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("Starved control ticks (%)")
    axes[1].legend(fontsize=8)
    fig.suptitle(
        "CPU supply model, not task success; 5 ms gap; band = synchronous/staggered tick phases"
    )
    fig.tight_layout()
    fig.savefig(root / "scheduler_supply.png", dpi=170)
    fig.savefig(root / "scheduler_supply.pdf")
    plt.close(fig)
    sensitivity = root / "latency_sensitivity.csv"
    if sensitivity.exists():
        f = pd.read_csv(sensitivity)
        fig, ax = plt.subplots(figsize=(8, 4))
        for (alg, cap), g in f.groupby(["algorithm", "max_batch"]):
            g = g.sort_values("latency_scale")
            ax.plot(g.latency_scale, g.starvation_rate * 100, marker="o", label=f"{alg} B{cap}")
        ax.set(
            xlabel="Inference cost multiplier (all batch sizes)",
            ylabel="Starved ticks (%)",
            title="Timing-model sensitivity; 4 robots, staggered 20 Hz, 5 ms gap",
        )
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(root / "latency_sensitivity.png", dpi=170)
        fig.savefig(root / "latency_sensitivity.pdf")
        plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    p.add_argument("--model", type=Path)
    args = p.parse_args()
    summarize(args.root)
    if args.model:
        plot_model(args.model)


if __name__ == "__main__":
    main()
