"""Validate and summarize ordinary fixed-batch timing separately from host-stage calls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def read_jsonl(path):
    return pd.DataFrame([json.loads(line) for line in path.read_text().splitlines() if line])


def load_gpu_telemetry(path, blocks):
    stream = path / "gpu_stream.jsonl"
    if not stream.exists():
        return read_jsonl(path / "gpu_telemetry.jsonl"), "per-query"
    gpu = read_jsonl(stream)
    gpu["phase"] = "other"
    gpu["repeat"] = np.nan
    gpu["batch_size"] = np.nan
    gpu["context_changed"] = False
    for block in blocks.itertuples():
        start, end = block.start_epoch, block.start_epoch + block.wall_seconds
        inside = gpu.epoch.between(start, end, inclusive="neither")
        stable = gpu.epoch.between(start + 1, end - 1, inclusive="neither")
        gpu.loc[inside & ~stable, "context_changed"] = True
        gpu.loc[stable, ["phase", "repeat", "batch_size"]] = [
            "fixed",
            block.repeat,
            block.batch_size,
        ]
    return gpu, "persistent nvidia-smi; first/last second of each fixed block excluded"


def summarize(path, *, partial=False):
    manifest = json.loads((path / "manifest.json").read_text())
    if not partial:
        assert manifest["status"] == "complete" and manifest["validated_outputs"]
    calls = read_jsonl(path / "calls.jsonl")
    blocks = read_jsonl(path / "blocks.jsonl")
    gpu, telemetry_source = load_gpu_telemetry(path, blocks)
    if calls.empty or blocks.empty:
        raise ValueError("No complete measurement block yet")
    assert list(calls["index"]) == list(range(len(calls)))
    assert calls.duration_ms.gt(0).all()
    # During a live run, only fully completed fixed blocks enter partial reports.
    done = set(zip(blocks.repeat, blocks.batch_size, strict=True))
    fixed = calls[calls.phase.eq("fixed")].copy()
    fixed = fixed[[key in done for key in zip(fixed.repeat, fixed.batch_size, strict=True)]]
    components = calls[calls.phase.eq("components")].copy()
    fields = [
        "prepare_inputs_ms",
        "sample_dispatch_ms",
        "materialize_ms",
        "output_transform_ms",
        "outside_ranges_ms",
    ]
    if len(components):
        np.testing.assert_allclose(
            components[fields].sum(axis=1), components.duration_ms, atol=1e-8
        )
        assert components[fields].ge(0).all().all()
    if not partial:
        assert len(calls) == manifest["calls"]
        expected = manifest["repeats"] * manifest["samples_per_batch_per_repeat"]
        assert fixed.groupby("batch_size").size().to_dict() == {
            b: expected for b in manifest["batch_sizes"]
        }
        expected_components = manifest["repeats"] * manifest["component_samples_per_block"]
        assert components.groupby("batch_size").size().to_dict() == {
            b: expected_components for b in manifest["batch_sizes"]
        }
        assert len(blocks) == manifest["repeats"] * len(manifest["batch_sizes"])
    by_block = []
    for (rep, size), g in fixed.groupby(["repeat", "batch_size"]):
        block = blocks[(blocks.repeat == rep) & (blocks.batch_size == size)].iloc[0]
        assert len(g) == manifest["samples_per_batch_per_repeat"]
        assert g.duration_ms.sum() / 1000 < block.wall_seconds
        by_block.append(
            dict(
                repeat=rep,
                batch_size=size,
                count=len(g),
                mean_ms=g.duration_ms.mean(),
                p50_ms=g.duration_ms.quantile(0.5),
                p95_ms=g.duration_ms.quantile(0.95),
                p99_ms=g.duration_ms.quantile(0.99),
                min_ms=g.duration_ms.min(),
                max_ms=g.duration_ms.max(),
                timed_service_requests_per_s=1000 * size / g.duration_ms.mean(),
                observed_block_requests_per_s=size * len(g) / block.wall_seconds,
                start_epoch=block.start_epoch,
                wall_seconds=block.wall_seconds,
            )
        )
    by_block = pd.DataFrame(by_block)
    by_block.to_csv(path / "block_summary.csv", index=False)
    usable_gpu = gpu[gpu.phase.eq("fixed") & ~gpu.context_changed].copy()
    usable_gpu = usable_gpu[
        [key in done for key in zip(usable_gpu["repeat"], usable_gpu.batch_size, strict=True)]
    ]
    baseline = fixed.loc[fixed.batch_size == 1, "duration_ms"].mean()
    rows = []
    for size, g in fixed.groupby("batch_size"):
        t = usable_gpu[usable_gpu.batch_size == size]
        b = by_block[by_block.batch_size == size]
        mean = g.duration_ms.mean()
        rows.append(
            dict(
                batch_size=int(size),
                calls=len(g),
                blocks=len(b),
                mean_ms=mean,
                p50_ms=g.duration_ms.quantile(0.5),
                p95_ms=g.duration_ms.quantile(0.95),
                p99_ms=g.duration_ms.quantile(0.99),
                min_ms=g.duration_ms.min(),
                max_ms=g.duration_ms.max(),
                ms_per_request=mean / size,
                timed_service_requests_per_s=1000 * size / mean,
                throughput_speedup_vs_b1=size * baseline / mean,
                observed_block_requests_per_s=size * len(g) / b.wall_seconds.sum(),
                block_mean_std_ms=b.mean_ms.std(),
                gpu_samples=len(t),
                gpu_util_mean_percent=t.gpu_util_percent.mean(),
                gpu_util_p95_percent=t.gpu_util_percent.quantile(0.95),
                power_mean_w=t.power_w.mean(),
                power_max_w=t.power_w.max(),
                sampled_memory_peak_mib=t.memory_used_mib.max(),
                temperature_mean_c=t.temperature_c.mean(),
                temperature_max_c=t.temperature_c.max(),
                sm_clock_mean_mhz=t.sm_clock_mhz.mean(),
                sm_clock_min_mhz=t.sm_clock_mhz.min(),
                sm_clock_max_mhz=t.sm_clock_mhz.max(),
                sw_thermal_samples=int(t.sw_thermal.sum()),
                hw_thermal_samples=int(t.hw_thermal.sum()),
            )
        )
    summary = pd.DataFrame(rows)
    summary.to_csv(path / "summary.csv", index=False)
    if len(components):
        stages = components.groupby("batch_size")[["duration_ms", *fields]].mean().reset_index()
        stages["sample_and_materialize_ms"] = stages.sample_dispatch_ms + stages.materialize_ms
        stages["calls"] = stages.batch_size.map(components.groupby("batch_size").size())
        stages.to_csv(path / "component_summary.csv", index=False)
        fig, ax = plt.subplots(figsize=(8, 4), layout="constrained")
        bottom = np.zeros(len(stages))
        for field, label, color in [
            ("prepare_inputs_ms", "Input preparation (host wall)", "#59a14f"),
            ("sample_and_materialize_ms", "Model call + materialization (host wall)", "#2878b5"),
            ("output_transform_ms", "Output transform", "#f28e2b"),
            ("outside_ranges_ms", "Outside ranges", "#888888"),
        ]:
            ax.bar(stages.batch_size, stages[field], bottom=bottom, label=label, color=color)
            bottom += stages[field].to_numpy()
        ax.set(
            xlabel="Actual batch size",
            ylabel="Milliseconds",
            xticks=manifest["batch_sizes"],
            title="Separate component calls; no extra GPU synchronization; not kernel durations",
        )
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.2)
        fig.savefig(path / "host_components.png", dpi=170)
        fig.savefig(path / "host_components.pdf")
        plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), layout="constrained")
    for field, label in [("p50_ms", "p50"), ("p95_ms", "p95"), ("p99_ms", "p99")]:
        axes[0].plot(summary.batch_size, summary[field], "o-", label=label)
    axes[0].set_ylabel("Full infer_batch latency (ms)")
    axes[0].legend()
    axes[1].plot(
        summary.batch_size, summary.timed_service_requests_per_s, "o-", label="Timed calls"
    )
    axes[1].plot(
        summary.batch_size, summary.observed_block_requests_per_s, "s--", label="Whole block"
    )
    axes[1].set_ylabel("Requests / second")
    axes[1].legend()
    axes[2].plot(summary.batch_size, summary.ms_per_request, "o-")
    axes[2].set_ylabel("Amortized milliseconds / request")
    for ax in axes:
        ax.set(xlabel="Actual batch size", xticks=manifest["batch_sizes"])
        ax.grid(alpha=0.2)
    fig.suptitle(
        f"pi05_libero, A6000, 10 denoising steps; {len(fixed):,} ordinary calls; no simulator"
    )
    fig.savefig(path / "batch_cost.png", dpi=170)
    fig.savefig(path / "batch_cost.pdf")
    plt.close(fig)
    fig, axes = plt.subplots(3, 1, figsize=(11, 7), sharex=True, layout="constrained")
    start = gpu.epoch.min()
    colors = {b: plt.get_cmap("tab10")(i) for i, b in enumerate(manifest["batch_sizes"])}
    for ax, field, label in zip(
        axes,
        ["temperature_c", "sm_clock_mhz", "power_w"],
        ["Temperature (C)", "SM clock (MHz)", "GPU power (W)"],
        strict=True,
    ):
        ax.plot(
            (gpu.epoch - start) / 60, gpu[field], color="#cccccc", linewidth=0.8, label="All phases"
        )
        for size, t in usable_gpu.groupby("batch_size"):
            ax.scatter(
                (t.epoch - start) / 60, t[field], s=7, color=colors[size], label=f"B{int(size)}"
            )
        ax.set_ylabel(label)
        ax.grid(alpha=0.2)
    axes[0].legend(ncol=6, fontsize=8)
    axes[-1].set_xlabel("Minutes from first telemetry sample")
    fig.suptitle("Read-only GPU telemetry; colored points = ordinary fixed blocks")
    fig.savefig(path / "gpu_state.png", dpi=170)
    fig.savefig(path / "gpu_state.pdf")
    plt.close(fig)
    validation = dict(
        status="partial" if partial else "complete",
        ordinary_calls=len(fixed),
        component_calls=len(components),
        completed_blocks=len(blocks),
        outputs_validated_against_fixed_seed_reference=manifest.get("validated_outputs", False),
        component_time_partition_conserved=True,
        ordinary_gpu_samples=len(usable_gpu),
        telemetry_source=telemetry_source,
        gpu_sample_interval_median_s=float(gpu.epoch.diff().median()),
        gpu_sample_interval_p95_s=float(gpu.epoch.diff().quantile(0.95)),
        transition_gpu_samples_excluded=int(gpu.context_changed.sum()),
        caveats=[
            "Percentiles pool fixed calls across repeated blocks; no claim of independent samples.",
            "Five input snapshots remain fixed; this is not a task-success or traffic experiment.",
            "Host phase times include asynchronous dispatch/wait and are not pure GPU kernel times.",
            "Memory is sampled total GPU framebuffer usage after all batch shapes are warmed.",
            "Power and utilization are sampled device readings, not kernel counter utilization.",
            "Clock, power, and cooling are not controlled; their observed values are reported.",
        ],
    )
    (path / "analysis_validation.json").write_text(json.dumps(validation, indent=2))
    print(summary.to_string(index=False))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    parser.add_argument("--partial", action="store_true")
    args = parser.parse_args()
    summarize(args.path, partial=args.partial)


if __name__ == "__main__":
    main()
