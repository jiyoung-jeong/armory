"""Join completed epochs to empty-batch, GPU and request-cycle diagnostics."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scripts.analyze_serving_capacity import rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summaries = []
    diagnostics = []
    examples = {}
    sensitivity = []
    for root in args.roots:
        batches = rows(root / "policy/server/batches.jsonl")
        telemetry = rows(root / "gpu_stream.jsonl")
        for path in sorted(root.glob("r*/summary.json")):
            summary = json.loads(path.read_text())
            summary["root"] = root.name
            summaries.append(summary)
            case = json.loads((path.parent / "case.json").read_text())
            if case["status"] != "complete":
                raise ValueError(f"Incomplete case: {path}")
            start, end = case["measure_start"], case["measure_end"]
            selected = [b for b in batches if start <= b["inference_start_time"] < end]
            gpu = [r for r in telemetry if start <= r["epoch"] < end]
            measured = pd.read_csv(path.parent / "requests.csv")
            # Offline sensitivity only: the experiment's primary SLO remains 156 ms.
            for deadline in [156, 200, 250, 500, 1000]:
                sensitivity.append(
                    dict(
                        root=root.name,
                        case=path.parent.name,
                        robots=case["robots"],
                        control_hz=case["control_hz"],
                        repeat=case["repeat"],
                        deadline_ms=deadline,
                        attainment_percent=100
                        * (measured.latency_ms <= deadline).sum()
                        / len(measured),
                    )
                )
            for name, passed, starved in [
                ("pass_starved", True, True),
                ("miss_available", False, False),
            ]:
                match = measured[
                    measured.responded
                    & (measured.slo_pass == passed)
                    & (measured.cycle_starved == starved)
                ]
                if name in examples or match.empty:
                    continue
                row = match.iloc[0]
                events = rows(path.parent / f"broker_{int(row.robot)}.jsonl")
                source = next(
                    e for e in events if e.get("request_timestamp") == row.request_timestamp
                )
                null_offsets = [
                    (e["time"] - row.request_timestamp) * 1000
                    for e in events
                    if e["kind"] == "step"
                    and e["local_chunk_index"] is None
                    and row.request_timestamp <= e["time"] < row.cycle_end
                ]
                examples[name] = dict(
                    root=root.name,
                    case=path.parent.name,
                    robot=int(row.robot),
                    request_id=int(row.request_id),
                    latency_ms=float(row.latency_ms),
                    control_hz=case["control_hz"],
                    queue_after_request_tick=source["queue_after"],
                    starvation_offsets_after_request_ms=null_offsets,
                )
            d = {key: summary[key] for key in ["root", "case", "robots", "control_hz", "repeat"]}
            d.update(
                empty_batches=sum(b["batch_size"] == 0 for b in selected),
                inference_batches=sum(b["batch_size"] > 0 for b in selected),
                gpu_samples=len(gpu),
                sw_thermal_samples=sum(r["sw_thermal"] for r in gpu),
                hw_thermal_samples=sum(r["hw_thermal"] for r in gpu),
                all_requests=summary["sent"],
                unanswered=summary["unanswered"],
                inference_p99_ms=measured.inference_ms.quantile(0.99),
                received_slo_percent=100 * measured.slo_pass.sum() / measured.responded.sum()
                if measured.responded.sum()
                else np.nan,
            )
            for key in [
                "temperature_c",
                "sm_clock_mhz",
                "power_w",
                "memory_used_mib",
                "gpu_util_percent",
            ]:
                values = [r[key] for r in gpu]
                d[key + "_mean"] = np.mean(values) if values else np.nan
                d[key + "_max"] = max(values) if values else np.nan
            needed = int(np.ceil(0.98 * len(measured)))
            observed = sorted(measured.latency_ms.dropna())
            d["empirical_deadline_for_98_ms"] = (
                observed[needed - 1] if len(observed) >= needed else np.nan
            )
            d["coverage_can_reach_98"] = len(observed) >= needed
            diagnostics.append(d)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "mismatch_examples.json").write_text(json.dumps(examples, indent=2))
    pd.DataFrame(summaries).to_csv(args.output / "epochs.csv", index=False)
    pd.DataFrame(sensitivity).to_csv(args.output / "deadline_sensitivity.csv", index=False)
    pd.DataFrame(diagnostics).to_csv(args.output / "diagnostics.csv", index=False)
    data = pd.DataFrame(summaries)
    columns = ["pass_available", "pass_starved", "miss_available", "miss_starved"]
    data.groupby(["robots", "control_hz", "max_batch_size"])[columns].sum().to_csv(
        args.output / "request_cycle_counts.csv"
    )
    print(
        data.groupby(["robots", "control_hz", "max_batch_size"])[
            ["completion_rps", "slo_percent", "starvation_percent", "response_p99_ms"]
        ]
        .mean()
        .round(2)
    )


if __name__ == "__main__":
    main()
