"""Account for all sent requests, including those with no observed response.

156 ms is a borrowed PI0/Thor reference, not a measured PI05 application SLO.
Starvation is a null action at an actual control tick after fixed warmup. A
request-cycle cross-tab describes association, not a causal attribution.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np


def rows(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def percentile(values, q):
    return float(np.percentile(values, q)) if values else None


def account(sent, chunks, steps, start, end, slo_ms):
    """First matching response wins; unanswered requests remain in denominator."""
    by_request = {}
    duplicates = 0
    for chunk in sorted(chunks, key=lambda c: c["response_timestamp"]):
        key = chunk["request_timestamp"]
        if key in by_request:
            duplicates += 1
        else:
            by_request[key] = chunk
    all_sent = sorted(sent, key=lambda r: r["request_timestamp"])
    step_times = [s["time"] for s in steps]
    starved_times = sorted(s["time"] for s in steps if s["local_chunk_index"] is None)
    request_rows = []
    for i, request in enumerate(all_sent):
        ts = request["request_timestamp"]
        if not start <= ts < end:
            continue
        chunk = by_request.get(ts)
        latency = (chunk["response_timestamp"] - ts) * 1000 if chunk else None
        if latency is not None and latency < 0:
            raise ValueError("Negative response latency")
        cycle_end = all_sent[i + 1]["request_timestamp"] if i + 1 < len(all_sent) else None
        # Only complete request cycles fully contained in the measurement window.
        cycle_starved = None
        if cycle_end is not None and cycle_end <= end:
            a, b = (
                bisect.bisect_left(starved_times, ts),
                bisect.bisect_left(starved_times, cycle_end),
            )
            cycle_starved = b > a
        request_rows.append(
            dict(
                request_timestamp=ts,
                observation_step=request["observation_step"],
                request_id=chunk["request_id"] if chunk else None,
                chunk_id=chunk["chunk_id"] if chunk else None,
                latency_ms=latency,
                responded=chunk is not None,
                slo_pass=latency is not None and latency <= slo_ms,
                cycle_end=cycle_end,
                cycle_starved=cycle_starved,
                payload_bytes=request.get("payload_bytes"),
            )
        )
    measured_steps = [s for s in steps if start <= s["time"] < end]
    if not request_rows or not measured_steps or not step_times:
        raise ValueError("Empty request or control measurement window")
    return request_rows, measured_steps, duplicates


def analyze_case(dest, server):
    config = json.loads((dest / "case.json").read_text())
    if config["status"] != "complete":
        raise ValueError("Refusing incomplete trial")
    start, end = config["measure_start"], config["measure_end"]
    requests, steps, chunks, late, robot_summary = [], [], [], [], []
    duplicates = 0
    for robot in range(config["robots"]):
        events = rows(dest / f"broker_{robot}.jsonl")
        robot_steps = [e for e in events if e["kind"] == "step"]
        sent = [e for e in robot_steps if "request_timestamp" in e]
        robot_chunks = rows(dest / f"chunks_{robot}.jsonl")
        req, measured, dup = account(sent, robot_chunks, robot_steps, start, end, config["slo_ms"])
        expected = config["request_hz"] * config["seconds"]
        if abs(len(req) - expected) > 1:
            raise ValueError(
                f"Robot {robot} failed to produce requested open-loop load: {len(req)}"
            )
        for r in req:
            r["robot"] = robot
        requests.extend(req)
        steps.extend(measured)
        duplicates += dup
        chunks.extend(c for c in robot_chunks if start <= c["response_timestamp"] < end)
        robot_summary.append(
            dict(
                robot=robot,
                sent=len(req),
                responded=sum(r["responded"] for r in req),
                slo_percent=100 * sum(r["slo_pass"] for r in req) / len(req),
                starvation_percent=100
                * sum(s["local_chunk_index"] is None for s in measured)
                / len(measured),
            )
        )
        late.extend(t["lateness_ms"] for t in rows(dest / f"ticks_{robot}.jsonl"))
    arrivals = {
        (e["robot_id"], e["observation_step"], e["request_timestamp"]): e
        for e in rows(server / "events.jsonl")
        if e["kind"] == "request"
    }
    batches = rows(server / "batches.jsonl")
    processed = {}
    for batch in batches:
        for reqid in batch.get("processed_request_ids", []):
            processed[reqid] = batch
    wait_ms, inference_ms, ingress_ms = [], [], []
    for req in requests:
        arrival = arrivals.get(
            (f"robot_{req['robot']}", req["observation_step"], req["request_timestamp"])
        )
        req["arrived"] = arrival is not None
        batch = processed.get(arrival["request_id"]) if arrival else None
        req["processed"] = batch is not None
        req["server_request_id"] = arrival["request_id"] if arrival else None
        if arrival:
            ingress_ms.append((arrival["arrival_time"] - req["request_timestamp"]) * 1000)
        if batch:
            wait = (batch["inference_start_time"] - arrival["arrival_time"]) * 1000
            wait_ms.append(wait)
            inference_ms.append(batch["inference_duration"] * 1000)
            req.update(
                wait_ms=wait,
                inference_ms=batch["inference_duration"] * 1000,
                actual_batch=batch["batch_size"],
            )
    active_batches = [
        b for b in batches if b["batch_size"] > 0 and start <= b["inference_start_time"] < end
    ]
    received_events = [
        e
        for robot in range(config["robots"])
        for e in rows(dest / f"broker_{robot}.jsonl")
        if e["kind"] == "chunk_received" and start <= e["time"] < end
    ]
    prefix_discard = [
        min(e["max_execution_horizon"], max(0, e["next_action_index"] - e["action_index_start"]))
        for e in received_events
    ]
    latencies = [r["latency_ms"] for r in requests if r["responded"]]
    cross = Counter(
        f"{'pass' if r['slo_pass'] else 'miss'}_{'starved' if r['cycle_starved'] else 'available'}"
        for r in requests
        if r["cycle_starved"] is not None
    )
    summary = {
        k: config[k]
        for k in [
            "robots",
            "control_hz",
            "request_hz",
            "max_batch_size",
            "repeat",
            "algorithm",
            "seconds",
            "slo_ms",
        ]
    }
    summary.update(
        case=dest.name,
        sent=len(requests),
        responded=sum(r["responded"] for r in requests),
        arrived=sum(r["arrived"] for r in requests),
        processed=sum(r["processed"] for r in requests),
        unanswered=sum(not r["responded"] for r in requests),
        duplicate_responses=duplicates,
        offered_rps=len(requests) / (end - start),
        completion_rps=len(chunks) / (end - start),
        slo_percent=100 * sum(r["slo_pass"] for r in requests) / len(requests),
        response_p50_ms=percentile(latencies, 50),
        response_p95_ms=percentile(latencies, 95),
        response_p99_ms=percentile(latencies, 99),
        queue_wait_p95_ms=percentile(wait_ms, 95),
        inference_p50_ms=percentile(inference_ms, 50),
        ingress_p95_ms=percentile(ingress_ms, 95),
        starvation_percent=100 * sum(s["local_chunk_index"] is None for s in steps) / len(steps),
        control_ticks=len(steps),
        executed_actions=sum(s["local_chunk_index"] is not None for s in steps),
        executed_actions_per_s=sum(s["local_chunk_index"] is not None for s in steps)
        / (end - start),
        actual_control_hz_per_robot=len(steps) / (end - start) / config["robots"],
        actual_batch_mean=float(np.mean([b["batch_size"] for b in active_batches]))
        if active_batches
        else None,
        discarded_prefix_mean=float(np.mean(prefix_discard)) if prefix_discard else None,
        tick_lateness_p99_ms=percentile(late, 99),
        tick_lateness_max_ms=max(late),
        **{
            key: cross[key]
            for key in ["pass_available", "pass_starved", "miss_available", "miss_starved"]
        },
    )
    with (dest / "requests.csv").open("w") as log:
        writer = csv.DictWriter(log, fieldnames=list(dict.fromkeys(k for r in requests for k in r)))
        writer.writeheader()
        writer.writerows(requests)
    (dest / "robots_summary.json").write_text(json.dumps(robot_summary, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summaries = []
    for root in args.roots:
        for path in sorted(root.glob("r*/case.json")):
            summary = analyze_case(path.parent, root / "policy/server")
            summary["root"] = root.name
            summaries.append(summary)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as log:
        writer = csv.DictWriter(log, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)


if __name__ == "__main__":
    main()
