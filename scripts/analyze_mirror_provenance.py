"""Join forecast, selected request and processed snapshot to actual broker state."""

from __future__ import annotations

import argparse
import bisect
import json
from pathlib import Path

import pandas as pd
from scripts.analyze_followup_events import read_jsonl, stats


def analyze(run):
    dest = run / "mirror_audit"
    dest.mkdir(exist_ok=True)
    manifest = json.loads((run / "manifest.json").read_text())
    assert manifest["status"] == "complete"
    requests = {
        e["request_id"]: e
        for e in read_jsonl(run / "policy/server/events.jsonl")
        if e["kind"] == "request"
    }
    decisions = {
        d["batch_id"]: d for d in read_jsonl(run / "policy/server/scheduler_decisions.jsonl")
    }
    timeline = {}
    steps = {}
    for r in range(4):
        next_action = 0
        for e in read_jsonl(run / f"client/broker_events_{r}.jsonl"):
            eid = e["episode_id"]
            key = (f"robot_{r}", eid)
            if e["kind"] == "reset":
                next_action = 0
            elif e["kind"] == "step":
                next_action += e["local_chunk_index"] is not None
                e["post_action_index"] = next_action
                steps[(*key, e["observation_step"])] = e
            if e["kind"] in ("step", "chunk_received"):
                timeline.setdefault(key, []).append(e)
    for ev in timeline.values():
        ev.sort(key=lambda e: e["time"])
    episode_last_tick = {
        key: max(e["time"] for e in ev if e["kind"] == "step") for key, ev in timeline.items()
    }
    stage = pd.read_csv(run / "event_analysis/response_lifecycle.csv").set_index("request_id")
    rows = []
    for b in read_jsonl(run / "policy/server/batches.jsonl"):
        prediction = decisions[b["batch_id"]]["notes"].get("prediction")
        if not prediction:
            continue
        by_robot = {p["robot_id"]: p for p in prediction["chunks"]}
        for rid, processed_id in zip(
            b["processed_robot_ids"], b["processed_request_ids"], strict=True
        ):
            p = by_robot[rid]
            if "selected_request_id" not in p:
                continue
            selected = requests[p["selected_request_id"]]
            processed = requests[processed_id]
            key = (rid, selected["episode_id"])
            assert selected["episode_id"] == processed["episode_id"]
            selected_step = steps[(*key, selected["observation_step"])]
            processed_step = steps[(*key, processed["observation_step"])]
            assert selected_step["post_action_index"] == p["selected_action_index_start"]
            forecast_step = steps.get((*key, p["observation_step"]))
            t0 = b["inference_start_time"]
            events = timeline[key]
            times = [e["time"] for e in events]
            i = bisect.bisect_right(times, t0) - 1
            state = events[i] if i >= 0 else None
            prior_steps = [e for e in events[: i + 1] if e["kind"] == "step"]
            last_tick = prior_steps[-1] if prior_steps else None
            response = stage.loc[processed_id]
            row = dict(
                batch_id=b["batch_id"],
                batch_size=b["batch_size"],
                robot_id=rid,
                selected_request_id=p["selected_request_id"],
                processed_request_id=processed_id,
                slot_updated=processed_id != p["selected_request_id"],
                predicted_observation_step=p["observation_step"],
                selected_observation_step=selected["observation_step"],
                processed_observation_step=processed["observation_step"],
                predicted_start=p["action_index_start"],
                selected_start=p["selected_action_index_start"],
                processed_start=processed_step["post_action_index"],
                same_predicted_observation=p["observation_step"] == processed["observation_step"],
                forecast_is_future=p["observation_step"] > selected["observation_step"],
                selected_to_processed_action_shift=processed_step["post_action_index"]
                - p["selected_action_index_start"],
                total_start_error=processed_step["post_action_index"] - p["action_index_start"],
                inference_ms=b["inference_duration"] * 1000,
                received=bool(response["received"]),
                episode_active_at_infer=t0 < episode_last_tick[key],
            )
            if forecast_step:
                truth = forecast_step["post_action_index"]
                row.update(
                    forecast_observation_recorded=True,
                    same_observation_origin_error=truth - p["action_index_start"],
                    forecast_to_selected_shift=p["selected_action_index_start"] - truth,
                )
                assert (
                    row["total_start_error"]
                    == row["same_observation_origin_error"]
                    + row["forecast_to_selected_shift"]
                    + row["selected_to_processed_action_shift"]
                )
            else:
                row["forecast_observation_recorded"] = False
            if state and last_tick and row["episode_active_at_infer"]:
                q = state["queue_after"]
                # Tick-based estimate; not a counterfactual measured exhaustion time.
                next_tick = max(t0, last_tick["time"] + 0.05)
                row.update(
                    queue_at_infer=q,
                    estimated_existing_queue_slack_ms=(next_tick + q * 0.05 - t0) * 1000,
                )
                if response["received"] and response["queue_apply"] < episode_last_tick[key]:
                    required = (response["queue_apply"] - t0) * 1000
                    row.update(
                        actual_infer_to_admission_ms=required,
                        estimated_slack_deficit_ms=required
                        - row["estimated_existing_queue_slack_ms"],
                    )
            rows.append(row)
    f = pd.DataFrame(rows)
    assert not f.empty
    f.to_csv(dest / "provenance_and_slack.csv", index=False)
    same = f[f.same_predicted_observation]
    summary = dict(
        run=run.name,
        algorithm=manifest.get("algorithm"),
        max_batch=manifest["max_batch_size"],
        processed=len(f),
        slot_updates=int(f.slot_updated.sum()),
        total_start_error=f.total_start_error.value_counts().sort_index().to_dict(),
        known_observation_origin_error=f.loc[~f.forecast_is_future, "same_observation_origin_error"]
        .dropna()
        .value_counts()
        .sort_index()
        .to_dict(),
        future_observation_origin_error=f.loc[f.forecast_is_future, "same_observation_origin_error"]
        .dropna()
        .value_counts()
        .sort_index()
        .to_dict(),
        future_forecasts=int(f.forecast_is_future.sum()),
        same_observation_origin_error=f.same_observation_origin_error.dropna()
        .value_counts()
        .sort_index()
        .to_dict(),
        forecast_observation_coverage=int(f.forecast_observation_recorded.sum()),
        same_processed_observation_count=len(same),
        same_processed_observation_error=same.total_start_error.value_counts()
        .sort_index()
        .to_dict(),
        queue_at_infer=stats(f.queue_at_infer),
        estimated_slack_ms=stats(f.estimated_existing_queue_slack_ms),
        estimated_slack_deficit_ms=stats(f.estimated_slack_deficit_ms.dropna()),
        positive_estimated_deficits=int((f.estimated_slack_deficit_ms > 0).sum()),
        admitted_with_slack_estimate=int(f.estimated_slack_deficit_ms.notna().sum()),
        slack_scope="Inference and admission before the episode's last recorded control tick",
        caveat="Queue uses timestamp-ordered broker events; 20 Hz slack extrapolation is an estimate, not observed counterfactual exhaustion.",
    )
    (dest / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("runs", nargs="+", type=Path)
    for run in p.parse_args().runs:
        analyze(run)


if __name__ == "__main__":
    main()
