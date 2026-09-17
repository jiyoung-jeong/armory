"""Summarize bounded GEMM samples; never treat replay timings as serving latency."""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

METRICS = {
    "kernel_ms": ("gpu__time_duration.sum", "ns", 1e-6),
    "sm_pct": ("sm__throughput.avg.pct_of_peak_sustained_elapsed", "%", 1),
    "tensor_pct": (
        "sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_elapsed",
        "%",
        1,
    ),
    "dram_pct": ("gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed", "%", 1),
    "dram_gbps": ("dram__bytes.sum.per_second", "byte/s", 1e-9),
    "l2_pct": ("lts__throughput.avg.pct_of_peak_sustained_elapsed", "%", 1),
    "l2_hit_pct": ("lts__t_sector_hit_rate.pct", "%", 1),
    "occupancy_pct": ("sm__warps_active.avg.pct_of_peak_sustained_active", "%", 1),
    "theoretical_occupancy_pct": ("sm__maximum_warps_per_active_cycle_pct", "%", 1),
    "registers_per_thread": ("launch__registers_per_thread", "register/thread", 1),
    "shared_bytes_per_block": ("launch__shared_mem_per_block", "byte/block", 1),
    "replay_passes": ("profiler__replayer_passes", "pass", 1),
}


def family(name):
    match = re.search(r"(cutlass_80_[^>]+|ampere_[\w]+)", name)
    if not match:
        raise ValueError(f"Unrecognized selected kernel: {name}")
    return match[1]


def analyze(root, reference):
    dest = root / "analysis"
    dest.mkdir(exist_ok=True)
    nsys = pd.read_csv(reference / "analysis/kernels.csv")
    nsys = nsys[~nsys.kernel.isin(["MEMCPY", "MEMSET"])].copy()
    frames = []
    checks = {}
    for batch in [1, 5]:
        folder = root / f"b{batch}_gemm"
        manifest = json.loads((folder / "run/manifest.json").read_text())
        assert manifest["status"] == "complete" and manifest["outputs_validated"]
        assert manifest["stablehlo_checks"][str(batch)]["identical"]
        original = (folder / f"run/b{batch}.original.stablehlo").read_bytes()
        assert original == (reference / f"run/b{batch}.original.stablehlo").read_bytes()
        # NCU 2026.1 --page raw: wide table with a units row immediately after the header.
        raw = pd.read_csv(folder / "metrics.csv", dtype=str)
        units = raw.iloc[0]
        assert pd.isna(units["ID"])
        data = raw.iloc[1:].copy()
        assert len(data) == 6 and data.ID.nunique() == 6
        result = pd.DataFrame(
            dict(
                batch_size=batch,
                id=data.ID.astype(int),
                kernel=data["Kernel Name"],
                grid=data["Grid Size"],
                block=data["Block Size"],
            )
        )
        result["family"] = result.kernel.map(family)
        for label, (metric, unit, scale) in METRICS.items():
            assert units[metric] == unit, (metric, units[metric], unit)
            result[label] = pd.to_numeric(data[metric].str.replace(",", "", regex=False)) * scale
            assert np.isfinite(result[label]).all()
        assert result.replay_passes.ge(1).all()
        assignments = {}
        for name in result.family.unique():
            matches = nsys[nsys.batch_size.eq(batch) & nsys.kernel.str.contains(name, regex=False)]
            assert set(matches.stage) == {"vlm_prefill"}, (batch, name, matches.stage.tolist())
            assignments[name] = dict(
                reference_stage="vlm_prefill", nsys_ms_per_infer=float(matches.total_ms.sum() / 10)
            )
        checks[batch] = dict(
            captured_kernels=len(result),
            outputs_validated=True,
            stablehlo_matches_nsys=True,
            kernel_family_reference=assignments,
            runtime_commit=manifest["git_commit"],
        )
        frames.append(result)
    per_launch = pd.concat(frames, ignore_index=True)
    per_launch.to_csv(dest / "kernels.csv", index=False)
    grouped = per_launch.groupby(["batch_size", "family", "grid", "block"])
    summary = grouped[list(METRICS)].mean().reset_index()
    summary["sample_count"] = grouped.size().to_numpy()
    summary.to_csv(dest / "summary.csv", index=False)
    validation = dict(
        status="complete",
        checks=checks,
        stage_assignment="Kernel family unique to prefill in prior NVTX/HLO-resolved Nsight reference; identical StableHLO. This NCU collection did not record live NVTX stage ranges.",
        timing_scope="Individual kernel replay samples; not full prefill cost or serving latency",
        warmup_excluded=True,
        clock_control="none",
        cache_control="all",
        replay_mode="kernel",
        metric_definitions={k: dict(name=n, unit=u, scale=s) for k, (n, u, s) in METRICS.items()},
    )
    (dest / "validation.json").write_text(json.dumps(validation, indent=2))
    print(summary.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument(
        "--reference", type=Path, default=Path("output/static_stages_20260917/main_graph_exit")
    )
    args = parser.parse_args()
    analyze(args.root, args.reference)
