"""Validate episode isolation and link broker starvation to server processing stages."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def stats(values):
    a = np.asarray(values, dtype=float)
    a = a[np.isfinite(a)]
    return (
        dict(
            count=len(a),
            mean=float(a.mean()),
            p50=float(np.percentile(a, 50)),
            p95=float(np.percentile(a, 95)),
            max=float(a.max()),
        )
        if len(a)
        else dict(count=0)
    )


def analyze_trial(run):
    manifest = json.loads((run / "manifest.json").read_text())
    assert manifest["status"] == "complete"
    output = run / "event_analysis"
    output.mkdir(exist_ok=True)
    events = read_jsonl(run / "policy/server/events.jsonl")
    batches = read_jsonl(run / "policy/server/batches.jsonl")
    decisions = {
        row["batch_id"]: row for row in read_jsonl(run / "policy/server/scheduler_decisions.jsonl")
    }
    requests = {e["request_id"]: e for e in events if e["kind"] == "request"}
    sent = {e["request_id"]: e for e in events if e["kind"] == "response_sent"}
    server_dropped = {e["request_id"]: e for e in events if e["kind"] == "response_discarded"}
    acks = {e["request_id"]: e for e in events if e["kind"] == "ack"}
    client_dropped, received, outgoing, steps = {}, {}, {}, {}
    step_count = saved_count = 0
    packet_stats = []
    for path in sorted((run / "client").glob("broker_events_*.jsonl")):
        robot = "robot_" + path.stem.rsplit("_", 1)[1]
        current = None
        for e in read_jsonl(path):
            e["robot_id"] = robot
            if e["kind"] == "reset":
                current = e["episode_id"]
            elif e["kind"] == "chunk_received":
                assert current == e["episode_id"], "Cross-episode broker application"
                received[e["request_id"]] = e
            elif e["kind"] == "response_discarded":
                client_dropped[e["request_id"]] = e
            elif e["kind"] == "step":
                assert current == e["episode_id"]
                key = (robot, e["episode_id"], e["observation_step"])
                assert key not in outgoing
                outgoing[key] = e
                packet_stats.append(
                    dict(
                        robot_id=robot,
                        time=e["time"],
                        payload_bytes=e["payload_bytes"],
                        serialize_ms=(e["serialize_end"] - e["request_timestamp"]) * 1000,
                        websocket_send_ms=(e["send_end"] - e["serialize_end"]) * 1000,
                    )
                )
    saved = {}
    episode_meta = {}
    for path in sorted((run / "client").glob("*/*/metadata.json")):
        meta = json.loads(path.read_text())
        rid, eid = f"robot_{meta['robot_idx']}", meta["episode_id"]
        assert eid
        episode_meta[(rid, eid)] = meta
        recorded = pd.read_parquet(path.parent / "steps.parquet")
        rows = []
        for s in recorded.itertuples():
            e = outgoing[(rid, eid, s.env_step)]
            assert (
                pd.isna(s.action_chunk_index) and e["local_chunk_index"] is None
            ) or s.action_chunk_index == e["local_chunk_index"]
            assert s.actions_left == e["queue_before"]
            rows.append(e)
            step_count += 1
        steps[(rid, eid)] = rows
        chunk_path = path.parent / "action_chunks.parquet"
        if chunk_path.exists():
            chunks = pd.read_parquet(chunk_path)
            assert chunks.episode_id.eq(eid).all(), "Saved chunk belongs to a different episode"
            for c in chunks.itertuples():
                assert c.request_id not in saved
                assert requests[c.request_id]["episode_id"] == eid
                assert received[c.request_id]["episode_id"] == eid
                saved[c.request_id] = c
                saved_count += 1
    stages = []
    for b in batches:
        for rid, req, chunk, eid in zip(
            b["processed_robot_ids"],
            b["processed_request_ids"],
            b["chunk_ids"],
            b["processed_episode_ids"],
            strict=True,
        ):
            source = requests[req]
            assert source["episode_id"] == eid
            packet = outgoing[(rid, eid, source["observation_step"])]
            assert abs(packet["request_timestamp"] - source["request_timestamp"]) < 1e-6
            r, s = received.get(req), sent.get(req)
            d = decisions[b["batch_id"]]
            t0, t1 = b["inference_start_time"], b["inference_start_time"] + b["inference_duration"]
            row = dict(
                robot_id=rid,
                episode_id=eid,
                request_id=req,
                chunk_id=chunk,
                batch_id=b["batch_id"],
                batch_size=b["batch_size"],
                request_time=source["request_timestamp"],
                arrival_time=source["arrival_time"],
                scheduled_at=d["started_at"],
                scheduler_ms=d["duration"] * 1000,
                predicted_deadline=d["deadlines"].get(rid),
                infer_start=t0,
                infer_end=t1,
                request_to_arrival_ms=(source["arrival_time"] - source["request_timestamp"]) * 1000,
                arrival_to_infer_ms=(t0 - source["arrival_time"]) * 1000,
                inference_ms=b["inference_duration"] * 1000,
                sent=s is not None,
                server_discarded=req in server_dropped,
                client_discarded=req in client_dropped,
                received=r is not None,
                acked=req in acks,
                saved=req in saved,
            )
            prediction = d["notes"].get("prediction")
            if prediction:
                pc = next(c for c in prediction["chunks"] if c["robot_id"] == rid)
                row.update(
                    predicted_batch_size=len(prediction["chunks"]),
                    predicted_action_index_start=pc["action_index_start"],
                    predicted_inference_ms=prediction["inference_duration"] * 1000,
                    inference_prediction_error_ms=(
                        b["inference_duration"] - prediction["inference_duration"]
                    )
                    * 1000,
                    predicted_completion=prediction["completion_time"],
                    completion_prediction_error_ms=(t1 - prediction["completion_time"]) * 1000,
                    predicted_arrival=pc["arrival_time"],
                    predicted_first_executed_index=pc["first_executed_index"],
                    predicted_new_chunk_actions=max(
                        0, pc["max_execution_horizon"] - pc["first_executed_index"]
                    ),
                )
                if r:
                    actual_skip = max(0, r["next_action_index"] - r["action_index_start"])
                    row.update(
                        arrival_prediction_error_ms=(r["time"] - pc["arrival_time"]) * 1000,
                        actual_action_index_start=r["action_index_start"],
                        action_start_prediction_error=r["action_index_start"]
                        - pc["action_index_start"],
                        actual_queue_net_actions=r["queue_after"] - r["queue_before"],
                        actual_first_executed_index=actual_skip,
                        actual_new_chunk_actions=max(0, r["max_execution_horizon"] - actual_skip),
                    )
            if s:
                row.update(
                    send_start=s["send_start"],
                    send_complete=s["send_complete"],
                    infer_to_send_ms=(s["send_start"] - t1) * 1000,
                    response_send_ms=(s["send_complete"] - s["send_start"]) * 1000,
                )
            if r:
                assert r["episode_id"] == eid
                row.update(
                    receive_end=r["receive_end"],
                    queue_apply=r["time"],
                    queue_before=r["queue_before"],
                    queue_after=r["queue_after"],
                    receive_to_apply_ms=(r["time"] - r["receive_end"]) * 1000,
                    request_to_apply_ms=(r["time"] - source["request_timestamp"]) * 1000,
                )
                if s:
                    row["send_to_receive_ms"] = (r["receive_end"] - s["send_complete"]) * 1000
            stages.append(row)
    stage_map = {row["request_id"]: row for row in stages}
    assert len(stage_map) == len(stages)
    assert set(saved) <= set(stage_map) and set(acks) <= set(stage_map)
    incidents = []
    for (rid, eid), rows in steps.items():
        rs = sorted(
            (r for r in received.values() if r["robot_id"] == rid and r["episode_id"] == eid),
            key=lambda r: r["time"],
        )
        has_action = False
        i = 0
        while i < len(rows):
            if rows[i]["local_chunk_index"] is not None:
                has_action = True
                i += 1
                continue
            j = i + 1
            while j < len(rows) and rows[j]["local_chunk_index"] is None:
                j += 1
            if has_action:
                start = rows[i]["time"]
                end = rows[j]["time"] if j < len(rows) else None
                next_receipt = next(
                    (r for r in rs if r["time"] >= start and r["queue_after"] > 0), None
                )
                previous = next((r for r in reversed(rs) if r["time"] < start), None)
                incident = dict(
                    robot_id=rid,
                    episode_id=eid,
                    start=start,
                    end=end,
                    first_step=rows[i]["observation_step"],
                    length_steps=j - i,
                    observed_empty_ms=(end - start) * 1000 if end else None,
                    previous_request=previous["request_id"] if previous else None,
                    recovery_request=next_receipt["request_id"] if next_receipt else None,
                    recovery_stage="no_recovery_recorded",
                )
                if next_receipt:
                    stage = stage_map[next_receipt["request_id"]]
                    state = (
                        "before_inference"
                        if start < stage["infer_start"]
                        else "during_inference"
                        if start < stage["infer_end"]
                        else "after_inference"
                    )
                    incident.update(
                        recovery_stage=state,
                        recovery_batch=stage["batch_id"],
                        recovery_batch_size=stage["batch_size"],
                        recovery_scheduled_before_empty=stage["scheduled_at"] <= start,
                        recovery_infer_start=stage["infer_start"],
                        recovery_infer_end=stage["infer_end"],
                        recovery_apply=stage["queue_apply"],
                        recover_minus_predicted_deadline_ms=(
                            stage["queue_apply"] - stage["predicted_deadline"]
                        )
                        * 1000
                        if stage["predicted_deadline"]
                        else None,
                    )
                incidents.append(incident)
            i = j
    frame = pd.DataFrame(stages)
    frame.to_csv(output / "response_lifecycle.csv", index=False)
    pd.DataFrame(incidents).to_csv(output / "starvation_incidents.csv", index=False)
    packets = pd.DataFrame(packet_stats)
    packets.to_csv(output / "packet_timings.csv", index=False)
    summary = dict(
        run=run.name,
        profiling=manifest["profiling"],
        max_batch=manifest["max_batch_size"],
        episodes=len(episode_meta),
        validated_steps=step_count,
        validated_saved_chunks=saved_count,
        cross_episode_applied=0,
        cross_episode_saved=0,
        processed=len(stages),
        stages={
            k: int(frame[k].sum())
            for k in ["sent", "received", "acked", "saved", "server_discarded", "client_discarded"]
        },
        unobserved_delivery=int((~frame.sent & ~frame.server_discarded).sum()),
        stage_timings={k: stats(frame[k]) for k in frame.columns if k.endswith("_ms")},
        payload_bytes=stats(packets.payload_bytes),
        serialization_ms=stats(packets.serialize_ms),
        websocket_send_ms=stats(packets.websocket_send_ms),
        starvation_incidents=len(incidents),
        recovery_stages=pd.Series([i["recovery_stage"] for i in incidents])
        .value_counts()
        .to_dict(),
    )
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path)
    args = parser.parse_args()
    for run in args.runs:
        analyze_trial(run)


if __name__ == "__main__":
    main()
