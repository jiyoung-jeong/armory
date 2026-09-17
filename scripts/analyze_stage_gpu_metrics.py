"""Integrate sampled GPU counters over HLO-attributed GPU activity intervals.

These are sampled estimates, not NCU range counters or model FLOP utilization.
Keep a one-sample alignment sensitivity and the exact denominator in outputs.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

METRICS = {
    "tensor_active_pct": "Tensor Active [Throughput %]",
    "sm_active_pct": "SMs Active [Throughput %]",
    "sm_issue_pct": "SM Issue [Throughput %]",
    "compute_warps_pct": "Compute Warps in Flight [Throughput %]",
    "dram_read_pct": "DRAM Read Bandwidth [Throughput %]",
    "dram_write_pct": "DRAM Write Bandwidth [Throughput %]",
}
STAGES = {
    "vlm_embed": {"vlm_embed"},
    "vlm_prefill": {"vlm_prefill", "vlm_mixed"},
    "vlm_total": {"vlm_embed", "vlm_prefill", "vlm_mixed"},
    "action": {"action"},
}


def merge_intervals(intervals):
    merged = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(end, merged[-1][1])
        else:
            merged.append([start, end])
    return np.asarray(merged, dtype=np.int64).reshape(-1, 2)


def integrate(timestamps, values, intervals, shift=0):
    """Treat sample at t[i] as the mean over (t[i-1], t[i]].

    Shift the bins by one period in either direction to quantify timestamp
    alignment sensitivity, without claiming sub-sample temporal resolution.
    """
    ts = timestamps + shift
    widths = np.diff(ts)
    cumulative = np.r_[0.0, np.cumsum(widths * values[1:])]
    points = intervals.ravel()
    assert points.min() >= ts[0] and points.max() <= ts[-1]
    idx = np.searchsorted(ts, points, side="right") - 1
    idx = np.minimum(idx, len(ts) - 2)
    area = cumulative[idx] + (points - ts[idx]) * values[idx + 1]
    area = area.reshape(-1, 2)
    duration = np.diff(intervals, axis=1).sum()
    return float(np.diff(area, axis=1).sum() / duration)


def analyze(folder):
    dest = folder / "analysis"
    manifest = json.loads((folder / "run/manifest.json").read_text())
    assert manifest["status"] == "complete" and manifest["outputs_validated"]
    assert all(x["identical"] for x in manifest["stablehlo_checks"].values())
    con = sqlite3.connect(f"file:{folder / 'timeline.sqlite'}?mode=ro", uri=True)
    warnings = con.execute("select severity,text from DIAGNOSTIC_EVENT where severity>1").fetchall()
    assert not warnings, warnings
    info = pd.read_sql_query("select * from TARGET_INFO_GPU_METRICS", con)
    assert info.typeId.nunique() == 1, "More than one device's metric stream"
    raw = pd.read_sql_query("select timestamp,metricId,value from GPU_METRICS", con)
    samples = raw.pivot(index="timestamp", columns="metricId", values="value").sort_index()
    assert samples.notna().all().all()
    ts = samples.index.to_numpy(np.int64)
    widths = np.diff(ts)
    period = int(np.median(widths))
    assert widths.min() > 0 and widths.max() <= 2 * period, "Missing metric samples"
    data = {}
    for label, name in METRICS.items():
        metric_id = info.loc[info.metricName.eq(name), "metricId"].item()
        values = samples[metric_id].to_numpy(float)
        assert np.isfinite(values).all() and ((0 <= values) & (values <= 100)).all()
        data[label] = values
    activities = pd.read_csv(dest / "gpu_activities.csv")
    activities = activities[activities.phase.eq("profile")]
    rows = []
    for (batch, index), group in activities.groupby(["batch_size", "index"]):
        for stage, labels in STAGES.items():
            selected = group[group.stage.isin(labels)]
            intervals = merge_intervals(
                selected[["start", "end"]].itertuples(index=False, name=None)
            )
            duration = int(np.diff(intervals, axis=1).sum())
            row = dict(
                batch_size=int(batch),
                index=int(index),
                stage=stage,
                activity_ms=duration / 1e6,
                interval_count=len(intervals),
            )
            for metric, values in data.items():
                row[metric] = integrate(ts, values, intervals)
                alternatives = [
                    integrate(ts, values, intervals, shift) for shift in [-period, period]
                ]
                row[metric + "_alignment_delta_pp"] = max(
                    abs(v - row[metric]) for v in alternatives
                )
            rows.append(row)
    per_call = pd.DataFrame(rows)
    per_call.to_csv(dest / "resource_calls.csv", index=False)
    keys = ["batch_size", "stage"]
    summary = per_call.groupby(keys)[["activity_ms", *METRICS]].mean().reset_index()
    for metric in METRICS:
        grouped = per_call.groupby(keys)[metric]
        summary[metric + "_std"] = grouped.std(ddof=1).to_numpy()
        summary[metric + "_alignment_max_pp"] = (
            per_call.groupby(keys)[metric + "_alignment_delta_pp"].max().to_numpy()
        )
    summary["calls"] = per_call.groupby(keys).size().to_numpy()
    summary.to_csv(dest / "resource_summary.csv", index=False)
    sampled = pd.DataFrame({"timestamp_ns": ts, **data})
    sampled.to_csv(dest / "resource_samples.csv", index=False)
    validation = dict(
        status="complete",
        diagnostics=warnings,
        samples=len(ts),
        sample_period_ns=dict(median=period, minimum=int(widths.min()), maximum=int(widths.max())),
        counter_stream_type_id=int(info.typeId.iloc[0]),
        denominator="Union of all stage-attributed GPU kernel, memcpy and memset intervals; excludes gaps/CPU work. VLM total includes embedding, prefill and VLM-shared fusion.",
        estimator="Time-integral of sampled metrics over union intervals, sample at t represents preceding interval. One-period shifts both directions quantify alignment sensitivity; samples cannot resolve individual short kernels.",
        does_not_measure="Model FLOPS utilization; whole CPU stage span; NCU SM throughput; warm-cache kernel-specific counters",
        metric_names=METRICS,
        stage_labels={k: sorted(v) for k, v in STAGES.items()},
        outputs_validated=True,
    )
    (dest / "resource_validation.json").write_text(json.dumps(validation, indent=2))
    print(summary[[*keys, "activity_ms", *METRICS]].round(3).to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    analyze(parser.parse_args().folder)
