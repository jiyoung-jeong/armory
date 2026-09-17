"""Match NCU launch-configuration samples to a separately traced identical graph.

The weighted tables are kernel-mixture diagnostics, not live stage counters.
Ambiguous cross-stage configurations are kept separately and not attributed.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
from scripts.analyze_ncu_prefill import METRICS as BASE_METRICS
from scripts.analyze_static_stages import HloStages, stage_name

METRICS = {
    **BASE_METRICS,
    "l2_gbps": ("lts__t_bytes.sum.per_second", "byte/s", 1e-9),
    "l1_hit_pct": ("l1tex__t_sector_hit_rate.pct", "%", 1),
    "active_warps_per_scheduler": ("smsp__warps_active.avg.per_cycle_active", "warp", 1),
    "eligible_warps_per_scheduler": ("smsp__warps_eligible.avg.per_cycle_active", "warp", 1),
    "issued_warps_per_scheduler": ("smsp__issue_active.avg.per_cycle_active", None, 1),
    "no_eligible_pct": ("smsp__issue_inst0.avg.pct_of_peak_sustained_active", "%", 1),
    "issue_active_pct": ("smsp__issue_active.avg.pct_of_peak_sustained_active", "%", 1),
}
KEYS = ["batch_size", "kernel_key", "grid", "block", "static_shared", "dynamic_shared"]


def kernel_key(name):
    # NCU's default pretty-printer shortens C++ namespaces/template argument types.
    # CUTLASS's concrete instantiation remains unique; preserve every tile/dtype flag.
    match = re.search(r"cutlass_80_[^>]+", name)
    return match[0] if match else name


def reference_configs(reference):
    activities = pd.read_csv(reference / "analysis/gpu_activities.csv")
    activities = activities[activities.phase.eq("profile") & activities.kind.eq("KERNEL")]
    con = sqlite3.connect(f"file:{reference / 'timeline.sqlite'}?mode=ro", uri=True)
    launches = pd.read_sql_query(
        "select start,end,gridX,gridY,gridZ,blockX,blockY,blockZ,staticSharedMemory,dynamicSharedMemory from CUPTI_ACTIVITY_KIND_KERNEL",
        con,
    )
    matched = activities.merge(launches, on=["start", "end"], validate="one_to_one")
    matched["kernel_key"] = matched.kernel.map(kernel_key)
    for label, prefix in [("grid", "grid"), ("block", "block")]:
        # Convert numpy scalars to plain Python integers for stable tuple formatting.
        matched[label] = [
            str(tuple(map(int, v))) for v in matched[[prefix + x for x in "XYZ"]].to_numpy()
        ]
    matched = matched.rename(
        columns={"staticSharedMemory": "static_shared", "dynamicSharedMemory": "dynamic_shared"}
    )
    grouped = matched.groupby(KEYS)
    configs = grouped.agg(
        stage=("stage", lambda x: ",".join(sorted(set(x)))),
        reference_total_ms=("duration_ms", "sum"),
        reference_count=("start", "size"),
    ).reset_index()
    stage_costs = matched.groupby(["batch_size", "stage"]).duration_ms.sum()
    return configs, stage_costs


def analyze(root, reference, batches=(1, 5)):
    dest = root / "analysis"
    dest.mkdir(exist_ok=True)
    configs, stage_costs = reference_configs(reference)
    configs.to_csv(dest / "reference_configs.csv", index=False)
    frames, checks = [], {}
    units_record = {}
    for batch in batches:
        folder = root / f"ncu_b{batch}"
        manifest = json.loads((folder / "run/manifest.json").read_text())
        assert manifest["status"] == "complete" and manifest["outputs_validated"]
        assert manifest["stablehlo_checks"][str(batch)]["identical"]
        assert (folder / f"run/b{batch}.original.stablehlo").read_bytes() == (
            reference / f"run/b{batch}.original.stablehlo"
        ).read_bytes()
        hlo = HloStages((folder / f"run/b{batch}.optimized_hlo.txt").read_text())
        hlo_names = {n.replace(".", "_"): n for n in hlo.instructions}
        raw = pd.read_csv(folder / "metrics.csv", dtype=str)
        units, data = raw.iloc[0], raw.iloc[1:].copy()
        assert data.ID.nunique() == len(data)
        result = pd.DataFrame(
            dict(batch_size=batch, id=data.ID.astype(int), kernel=data["Kernel Name"])
        )
        result["kernel_key"] = result.kernel.map(kernel_key)

        def compiled_stage(name):
            instruction = hlo_names.get(name)
            if instruction is None:
                return "unknown_library_kernel"
            direct = hlo.instructions[instruction]["tags"]
            return stage_name(direct if direct else hlo.instruction_tags(instruction))

        result["own_hlo_stage"] = result.kernel.map(compiled_stage)
        for label, name in [("grid", "Grid Size"), ("block", "Block Size")]:
            result[label] = data[name].map(lambda x: str(tuple(ast.literal_eval(x))))
        for label in ["static", "dynamic"]:
            result[label + "_shared"] = pd.to_numeric(
                data["launch__shared_mem_per_block_" + label].str.replace(",", "", regex=False)
            ).astype(int)
        for label, (metric, unit, scale) in METRICS.items():
            assert pd.isna(units[metric]) if unit is None else units[metric] == unit, (
                metric,
                units[metric],
                unit,
            )
            result[label] = (
                pd.to_numeric(data[metric].str.replace(",", "", regex=False), errors="coerce")
                * scale
            )
            assert not np.isinf(result[label]).any(), label
            # Empty hit-rate denominators are allowed, but never silently become zero.
            assert np.isfinite(result[label]).all() or label in {"l1_hit_pct", "l2_hit_pct"}, label
            units_record[label] = dict(metric=metric, unit=unit, scale=scale)
        stall_metrics = [
            x
            for x in data
            if re.fullmatch(r"smsp__average_warps_issue_stalled_.+_per_issue_active.ratio", x)
        ]
        assert stall_metrics
        for metric in stall_metrics:
            reason = metric.removeprefix("smsp__average_warps_issue_stalled_").removesuffix(
                "_per_issue_active.ratio"
            )
            label = "warp_cycles_" + reason
            result[label] = pd.to_numeric(
                data[metric].str.replace(",", "", regex=False), errors="coerce"
            )
            units_record[label] = dict(metric=metric, unit=str(units[metric]))
        result = result.merge(configs, on=KEYS, how="left", validate="many_to_one")
        stage_names = ["action", "vlm_embed", "vlm_prefill"]
        conflicts = result[
            result.stage.isin(stage_names)
            & result.own_hlo_stage.isin(stage_names)
            & result.stage.ne(result.own_hlo_stage)
        ]
        # A reused fusion can retain another caller's metadata, or autotuning
        # can renumber generated names. Do not force a cross-run attribution.
        result["reference_stage"] = result.stage
        result["stage_match_conflict"] = result.index.isin(conflicts.index)
        result.loc[result.stage_match_conflict, "stage"] = pd.NA
        checks[batch] = dict(
            samples=len(result),
            independent_hlo_stage_conflicts=len(conflicts),
            excluded_conflicts=conflicts[["kernel", "stage", "own_hlo_stage"]].to_dict("records"),
            unmatched=int(result.stage.isna().sum()),
            ambiguous=int(result.stage.fillna("").str.contains(",").sum()),
            stablehlo_matches_reference=True,
            optimized_hlo_byte_identical=(folder / f"run/b{batch}.optimized_hlo.txt").read_bytes()
            == (reference / f"run/b{batch}.optimized_hlo.txt").read_bytes(),
            outputs_validated=True,
        )
        frames.append(result)
    samples = pd.concat(frames, ignore_index=True)
    samples.to_csv(dest / "ncu_kernels.csv", index=False)
    numeric = [*METRICS, *[c for c in samples if c.startswith("warp_cycles_")]]
    grouped = samples.groupby(KEYS, dropna=False)
    representative = (
        grouped[numeric]
        .mean()
        .reset_index()
        .merge(configs, on=KEYS, how="left", validate="one_to_one")
    )
    representative["samples"] = grouped.size().to_numpy()
    rejected = samples.loc[samples.stage_match_conflict, KEYS].drop_duplicates()
    representative = representative.merge(
        rejected.assign(stage_rejected=True), on=KEYS, how="left", validate="one_to_one"
    )
    representative.loc[representative.stage_rejected.eq(True), "stage"] = pd.NA
    representative.to_csv(dest / "ncu_configs.csv", index=False)
    rows = []
    for (batch, stage), group in representative.dropna(subset=["stage"]).groupby(
        ["batch_size", "stage"]
    ):
        if "," in stage:
            continue
        weights = group.reference_total_ms
        total = float(stage_costs.loc[(batch, stage)])
        row = dict(
            batch_size=int(batch),
            stage=stage,
            configurations=len(group),
            sampled_kernel_time_coverage_pct=100 * weights.sum() / total,
            reference_kernel_ms_per_call=total / 10,
        )
        for metric in numeric:
            valid = group[metric].notna()
            row[metric] = (
                float(np.average(group.loc[valid, metric], weights=weights[valid]))
                if valid.any()
                else None
            )
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary.to_csv(dest / "ncu_weighted_diagnostics.csv", index=False)
    top = representative[representative.stage.isin(["vlm_embed", "vlm_prefill", "action"])].copy()
    top["stage_kernel_time_share_pct"] = [
        100 * r.reference_total_ms / stage_costs.loc[(r.batch_size, r.stage)]
        for r in top.itertuples()
    ]
    top = (
        top.sort_values("reference_total_ms", ascending=False)
        .groupby(["batch_size", "stage"])
        .head(6)
    )
    top.to_csv(dest / "ncu_top_kernels.csv", index=False)
    validation = dict(
        status="complete",
        batches=checks,
        metrics=units_record,
        denominator="Weighted diagnostic estimate over mapped unique-stage kernel configurations. Weights are reference Nsight kernel durations; excludes copies/gaps, ambiguous configurations and cross-run stage conflicts. Not direct stage-range counters or warm-cache occupancy.",
        stage_assignment="Cross-run kernel + grid + block + static/dynamic shared-memory match against HLO-resolved Nsight trace with identical StableHLO; no live NCU stage NVTX. XLA autotuning selected some different compiled kernels/configurations. own_hlo_stage is supplemental metadata from the NCU run optimized HLO, with call-site tags taking priority over shared fusion child metadata; it is not used to assign reference time weights to unmatched kernels.",
        cache_control="all",
        clock_control="none",
        replay_mode="kernel",
        filter_mode="per-launch-config",
        launch_count=1,
        stall_definition="Average warp cycles in each state per issued instruction; not percentage of elapsed inference time. Not-selected reflects available parallelism, not necessarily a bottleneck.",
    )
    (dest / "ncu_validation.json").write_text(json.dumps(validation, indent=2))
    print(
        summary[
            [
                "batch_size",
                "stage",
                "sampled_kernel_time_coverage_pct",
                "tensor_pct",
                "dram_pct",
                "l2_pct",
                "eligible_warps_per_scheduler",
                "no_eligible_pct",
            ]
        ]
        .round(3)
        .to_string(index=False)
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 5])
    args = parser.parse_args()
    analyze(args.root, args.root / "nsys", args.batches)
