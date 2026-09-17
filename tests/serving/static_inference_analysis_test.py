"""Keep diagnostic calls and warmup GPU samples out of ordinary benchmark scores."""

import json

import pandas as pd
import pytest
from scripts.analyze_static_inference import summarize


@pytest.mark.parametrize("continuous", [False, True])
def test_benchmark_excludes_component_latencies_and_transition_telemetry(tmp_path, continuous):
    manifest = dict(
        status="complete",
        validated_outputs=True,
        calls=6,
        batch_sizes=[1, 2],
        repeats=1,
        samples_per_batch_per_repeat=2,
        component_samples_per_block=1,
    )
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    rows = []
    blocks = []
    for b, latencies, wall in [(1, [10.0, 20.0], 6.0), (2, [30.0, 50.0], 8.0)]:
        for latency in latencies:
            rows.append(
                dict(index=len(rows), phase="fixed", repeat=1, batch_size=b, duration_ms=latency)
            )
        rows.append(
            dict(
                index=len(rows),
                phase="components",
                repeat=1,
                batch_size=b,
                duration_ms=1000.0,
                prepare_inputs_ms=100.0,
                sample_dispatch_ms=800.0,
                materialize_ms=80.0,
                output_transform_ms=10.0,
                outside_ranges_ms=10.0,
            )
        )
        blocks.append(dict(repeat=1, batch_size=b, wall_seconds=wall, start_epoch=float(b) * 10))
    gpu = []
    for phase, b, memory, changed in [
        ("shape_warmup", 1, 90000, False),
        ("fixed", 1, 1000, False),
        ("fixed", 2, 2000, False),
        ("fixed", 2, 80000, True),
    ]:
        gpu.append(
            dict(
                phase=phase,
                batch_size=b,
                repeat=1,
                context_changed=changed,
                epoch=[10.5, 12.0, 22.0, 27.5][len(gpu)],
                gpu_util_percent=50.0,
                power_w=100.0,
                memory_used_mib=memory,
                temperature_c=80.0,
                sm_clock_mhz=1700.0,
                sw_thermal=False,
                hw_thermal=False,
            )
        )
    for filename, items in [
        ("calls.jsonl", rows),
        ("blocks.jsonl", blocks),
        ("gpu_telemetry.jsonl", gpu),
    ]:
        (tmp_path / filename).write_text("".join(json.dumps(row) + "\n" for row in items))
    if continuous:
        (tmp_path / "gpu_stream.jsonl").write_text("".join(json.dumps(row) + "\n" for row in gpu))
    result = summarize(tmp_path).set_index("batch_size")
    assert result.loc[1, "calls"] == 2
    assert result.loc[1, "p99_ms"] == pytest.approx(19.9)
    assert result.loc[1, "timed_service_requests_per_s"] == pytest.approx(1000 / 15)
    assert result.loc[1, "observed_block_requests_per_s"] == pytest.approx(2 / 6.0)
    assert result.loc[1, "sampled_memory_peak_mib"] == 1000
    assert result.loc[2, "sampled_memory_peak_mib"] == 2000
    stages = pd.read_csv(tmp_path / "component_summary.csv")
    assert stages.duration_ms.eq(1000).all()
    validation = json.loads((tmp_path / "analysis_validation.json").read_text())
    assert validation["ordinary_calls"] == 4
    assert validation["component_calls"] == 2
    assert validation["ordinary_gpu_samples"] == 2
    assert validation["transition_gpu_samples_excluded"] == (2 if continuous else 1)
