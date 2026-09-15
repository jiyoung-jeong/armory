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


def summarize_context(root):
    """Aggregate logged provenance and active-episode slack, separately from scores."""
    rows = []
    suffix_rows = []
    for path in sorted(root.glob("*/run_*/mirror_audit/provenance_and_slack.csv")):
        run = path.parent.parent
        manifest = json.loads((run / "manifest.json").read_text())
        if manifest["status"] != "complete":
            continue
        frame = pd.read_csv(path)
        frame["algorithm"] = manifest["algorithm"]
        frame["max_batch"] = manifest["max_batch_size"]
        rows.append(frame)
        stages = pd.read_csv(run / "event_analysis/response_lifecycle.csv")
        joined = frame.merge(
            stages,
            left_on="processed_request_id",
            right_on="request_id",
            validate="one_to_one",
            suffixes=("", "_stage"),
        )
        # Isolate arrival/queue prediction from changed or future observation origins.
        same = joined[
            (~joined.forecast_is_future)
            & joined.same_predicted_observation
            & joined.received
            & joined.estimated_slack_deficit_ms.notna()
        ]
        suffix_error = same.actual_new_chunk_actions - same.predicted_new_chunk_actions
        suffix_rows.append(
            dict(
                algorithm=manifest["algorithm"],
                max_batch=manifest["max_batch_size"],
                repeat=manifest["repeat"],
                compared=len(same),
                exact_suffix=int(suffix_error.eq(0).sum()),
                suffix_mae=suffix_error.abs().mean(),
                suffix_error_distribution=suffix_error.value_counts().sort_index().to_dict(),
            )
        )
    if not rows:
        return
    (root / "suffix_prediction_trials.json").write_text(json.dumps(suffix_rows, indent=2))
    f = pd.concat(rows, ignore_index=True)
    provenance, slack = [], []
    for (algorithm, cap), g in f.groupby(["algorithm", "max_batch"]):
        known = g.loc[~g.forecast_is_future, "same_observation_origin_error"].dropna()
        future = g.loc[g.forecast_is_future, "same_observation_origin_error"].dropna()
        provenance.append(
            dict(
                algorithm=algorithm,
                max_batch=cap,
                processed=len(g),
                slot_updates=int(g.slot_updated.sum()),
                known_observations=len(known),
                known_errors=int(known.ne(0).sum()),
                future_forecasts=int(g.forecast_is_future.sum()),
                future_observations_recorded=len(future),
                future_errors=int(future.ne(0).sum()),
                missing_forecast_observations=int((~g.forecast_observation_recorded).sum()),
                known_error_distribution=known.value_counts().sort_index().to_dict(),
                future_error_distribution=future.value_counts().sort_index().to_dict(),
            )
        )
        admitted = g[g.estimated_slack_deficit_ms.notna()]
        slack.append(
            dict(
                algorithm=algorithm,
                max_batch=cap,
                admitted=len(admitted),
                queue_at_infer_mean=admitted.queue_at_infer.mean(),
                estimated_slack_mean_ms=admitted.estimated_existing_queue_slack_ms.mean(),
                infer_to_admission_mean_ms=admitted.actual_infer_to_admission_ms.mean(),
                estimated_deficit_mean_ms=admitted.estimated_slack_deficit_ms.mean(),
                estimated_deficit_p95_ms=admitted.estimated_slack_deficit_ms.quantile(0.95),
                positive_deficit_fraction=admitted.estimated_slack_deficit_ms.gt(0).mean(),
            )
        )
    (root / "provenance_conditions.json").write_text(json.dumps(provenance, indent=2))
    pd.DataFrame(slack).to_csv(root / "slack_conditions.csv", index=False)
    robots = pd.read_csv(root / "robots.csv")
    robots.groupby(["algorithm", "max_batch", "robot", "tasks"]).agg(
        repeats=("repeat", "count"),
        success=("success", "sum"),
        starvation_mean=("post_first_starvation_rate", "mean"),
        maximum_streak_steps=("max_streak_steps", "max"),
    ).reset_index().to_csv(root / "robot_conditions.csv", index=False)

    telemetry_path = root / "gpu_telemetry.jsonl"
    if not telemetry_path.exists():
        return
    samples = []
    for line in telemetry_path.read_text().splitlines():
        entry = json.loads(line)
        fields = [value.strip() for value in entry["values"].split(",")]
        if len(fields) != 6:
            raise ValueError(f"Unexpected GPU telemetry: {entry}")
        samples.append(
            dict(
                time=entry["time"],
                sm_mhz=float(fields[0].split()[0]),
                watts=float(fields[1].split()[0]),
                temperature_c=float(fields[2]),
                sw_thermal=fields[4] == "Active",
                hw_thermal=fields[5] == "Active",
            )
        )
    samples = pd.DataFrame(samples)
    thermal = []
    for trial in pd.read_csv(root / "trials.csv").itertuples():
        active = samples[samples.time.between(trial.first_step, trial.last_step)]
        if active.empty:
            continue
        thermal.append(
            dict(
                algorithm=trial.algorithm,
                max_batch=trial.max_batch,
                repeat=trial.repeat,
                samples=len(active),
                first_sample_delay_s=active.time.min() - trial.first_step,
                last_sample_before_end_s=trial.last_step - active.time.max(),
                sm_mhz_mean=active.sm_mhz.mean(),
                sm_mhz_min=active.sm_mhz.min(),
                temperature_c_mean=active.temperature_c.mean(),
                temperature_c_max=active.temperature_c.max(),
                watts_mean=active.watts.mean(),
                sw_thermal_samples=int(active.sw_thermal.sum()),
                hw_thermal_samples=int(active.hw_thermal.sum()),
            )
        )
    pd.DataFrame(thermal).to_csv(root / "thermal_trials.csv", index=False)


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
    summarize_context(args.root)
    if args.model:
        plot_model(args.model)


if __name__ == "__main__":
    main()
