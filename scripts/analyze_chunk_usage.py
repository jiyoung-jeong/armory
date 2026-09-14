"""Audit saved action use, reconstruct broker queues, and reconcile server outputs.

Run against completed local_batch_sweep trials. This reads existing logs only;
it neither starts a server nor changes the broker or scheduling behavior.
"""

from __future__ import annotations

import argparse
import json
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd


def replay_episode(steps: pd.DataFrame, chunks: pd.DataFrame):
    """Reconstruct queue updates using receipt execution_start_step, not wall time.

    The agent lock serializes get_action and receive_response. A receipt whose
    execution_start_step is N followed get_action(N-1) and precedes get_action(N).
    Chunk row order breaks ties between receipts at the same control step.
    Every reconstructed pop and pre-pop depth must match the recorded step.
    A missing tail step is rejected instead of inventing its action selection.
    """
    if steps.empty or list(steps.env_step) != list(range(len(steps))):
        raise ValueError("Expected a nonempty episode with contiguous steps starting at zero")
    chunks = chunks.reset_index(drop=True)
    if len(chunks) and (
        not chunks.execution_start_step.is_monotonic_increasing
        or (chunks.execution_start_step < 0).any()
        or (chunks.execution_start_step > len(steps)).any()
    ):
        raise ValueError("Receipt order is inconsistent or includes an unrecorded control step")
    records = []
    for i, chunk in chunks.iterrows():
        horizon = int(chunk.max_execution_horizon)
        action_count = len(chunk.actions)
        if not 0 <= horizon <= action_count:
            raise ValueError("Execution horizon exceeds saved action array")
        records.append(
            dict(
                local_chunk_index=i,
                request_id=int(chunk.request_id),
                chunk_id=int(chunk.chunk_id),
                observation_step=int(chunk.observation_step),
                action_index_start=int(chunk.action_index_start),
                execution_start_step=int(chunk.execution_start_step),
                request_timestamp=float(chunk.request_timestamp),
                response_timestamp=float(chunk.response_timestamp),
                generated_actions=action_count,
                execution_horizon=horizon,
                outside_execution_horizon=action_count - horizon,
                skipped_past_prefix=0,
                replaced_before_execution=0,
                executed_actions=0,
                remaining_at_episode_snapshot=0,
                first_executed_index=None,
            )
        )
    queue = deque()
    next_action = 0
    cursor = 0
    executed = []

    def receive(index):
        row = records[index]
        row["queue_before"] = len(queue)
        row["next_action_index_at_receipt"] = next_action
        removed = 0
        start = row["action_index_start"]
        while queue and queue[-1][2] >= start:
            old_chunk, _, _ = queue.pop()
            records[old_chunk]["replaced_before_execution"] += 1
            removed += 1
        for offset in range(row["execution_horizon"]):
            if start + offset < next_action:
                row["skipped_past_prefix"] += 1
            else:
                queue.append((index, offset, start + offset))
        row["enqueued_actions"] = row["execution_horizon"] - row["skipped_past_prefix"]
        row["replaced_existing_actions_on_receipt"] = removed
        row["queue_after"] = len(queue)
        row["net_queue_addition"] = len(queue) - row["queue_before"]

    for step in steps.itertuples():
        while cursor < len(records) and records[cursor]["execution_start_step"] <= step.env_step:
            receive(cursor)
            cursor += 1
        depth = len(queue)
        action = queue.popleft() if queue else None
        expected = (
            None
            if pd.isna(step.action_chunk_index)
            else (int(step.action_chunk_index), int(step.action_index))
        )
        if (None if action is None else action[:2]) != expected or depth != step.actions_left:
            raise ValueError(f"Broker replay mismatch at step {step.env_step}")
        if action is not None:
            index, offset, _ = action
            row = records[index]
            row["executed_actions"] += 1
            if row["first_executed_index"] is None:
                row["first_executed_index"] = offset
            executed.append(
                dict(
                    env_step=int(step.env_step),
                    local_chunk_index=index,
                    action_index=offset,
                    step_record_timestamp=float(step.timestamp),
                    request_to_step_record_ms=(step.timestamp - row["request_timestamp"]) * 1000,
                )
            )
            next_action += 1
    while cursor < len(records):
        receive(cursor)
        cursor += 1
    for index, _, _ in queue:
        records[index]["remaining_at_episode_snapshot"] += 1
    for row in records:
        if row["execution_horizon"] != sum(
            row[name]
            for name in [
                "skipped_past_prefix",
                "executed_actions",
                "replaced_before_execution",
                "remaining_at_episode_snapshot",
            ]
        ):
            raise ValueError("Action accounting does not conserve the execution horizon")
        if row["executed_actions"]:
            row["usage_category"] = "executed"
        elif not row["enqueued_actions"]:
            row["usage_category"] = "no_usable_prefix_remaining"
        elif not row["remaining_at_episode_snapshot"]:
            row["usage_category"] = "fully_replaced_before_execution"
        elif not row["replaced_before_execution"]:
            row["usage_category"] = "remaining_at_episode_snapshot"
        else:
            row["usage_category"] = "partly_replaced_partly_remaining"

    starved = steps.action_chunk_index.isna().to_numpy()
    served = np.flatnonzero(~starved)
    first = int(served[0]) if len(served) else len(steps)
    post = starved & (np.arange(len(steps)) >= first)
    starts = np.flatnonzero(np.diff(np.r_[False, post].astype(int)) == 1)
    ends = np.flatnonzero(np.diff(np.r_[post, False].astype(int)) == -1) + 1
    streaks = []
    for a, z in zip(starts, ends, strict=True):
        streaks.append(
            dict(
                first_starved_step=int(a),
                end_step_exclusive=int(z),
                length_steps=int(z - a),
                next_recorded_action_exists=bool(z < len(steps)),
                first_starved_step_timestamp=float(steps.timestamp.iloc[a]),
                next_action_step_timestamp=float(steps.timestamp.iloc[z])
                if z < len(steps)
                else None,
            )
        )
    return records, executed, streaks


def reconcile_trial(run: Path, saved: pd.DataFrame, episode_bounds: dict):
    """Outer-join processed, ACKed and saved results without assuming all were sent."""
    batches = pd.read_json(run / "policy/server/batches.jsonl", lines=True, convert_dates=False)
    events = pd.read_json(run / "policy/server/events.jsonl", lines=True, convert_dates=False)
    requests = {}
    for rid, group in events[events.kind == "request"].groupby("robot_id"):
        episode = -1
        for event in group.itertuples():
            if event.observation_step == 0:
                episode += 1
            key = (rid, int(event.request_id))
            if key in requests:
                raise ValueError("Duplicate received request ID")
            requests[key] = (event, episode)
    acked = {}
    for row in events[events.kind == "ack"].itertuples():
        key = (row.robot_id, int(row.request_id), int(row.chunk_id))
        if key in acked:
            raise ValueError("Duplicate ACK")
        acked[key] = row
    saved_lookup = {}
    for row in saved.itertuples():
        key = (row.robot_id, int(row.request_id), int(row.chunk_id))
        if key in saved_lookup:
            raise ValueError("Saved chunk key occurs in multiple episodes")
        saved_lookup[key] = row
    rows = []
    processed = set()
    for batch in batches.itertuples():
        for rid, req, chunk in zip(
            batch.processed_robot_ids, batch.processed_request_ids, batch.chunk_ids, strict=True
        ):
            key = (rid, int(req), int(chunk))
            if key in processed:
                raise ValueError("Duplicate processed output key")
            processed.add(key)
            request, source_episode = requests[(rid, int(req))]
            bound = episode_bounds[(rid, source_episode)]
            ack = acked.get(key)
            record = saved_lookup.get(key)
            if record is not None:
                if not np.isclose(
                    record.request_timestamp, request.request_timestamp, rtol=0, atol=1e-6
                ):
                    raise ValueError("Request timestamp join mismatch")
            finish = float(batch.inference_start_time + batch.inference_duration)
            evidence_time = float(ack.receive_time) if ack is not None else finish
            stage = (
                "saved" if record is not None else "acked_not_saved" if ack else "no_recorded_ack"
            )
            rows.append(
                dict(
                    robot_id=rid,
                    request_id=int(req),
                    chunk_id=int(chunk),
                    batch_id=int(batch.batch_id),
                    actual_batch_size=int(batch.batch_size),
                    source_episode=source_episode,
                    request_observation_step=int(request.observation_step),
                    source_request_timestamp=float(request.request_timestamp),
                    saved_episode=int(record.episode) if record is not None else None,
                    saved_in_source_episode=record.episode == source_episode
                    if record is not None
                    else None,
                    source_episode_truncated=bound["truncated"],
                    source_last_step_timestamp=bound["last"],
                    inference_end_timestamp=finish,
                    server_send_timestamp=float(ack.server_send_time) if ack else None,
                    ack_receive_timestamp=float(ack.receive_time) if ack else None,
                    acked=ack is not None,
                    saved=record is not None,
                    recorded_executed_actions=record.executed_actions if record is not None else 0,
                    stage=stage,
                    after_source_last_recorded_step=evidence_time > bound["last"],
                    ms_after_source_last_recorded_step=(evidence_time - bound["last"]) * 1000,
                )
            )
    if not set(saved_lookup) <= processed or not set(acked) <= processed:
        raise ValueError("Saved/ACKed output has no matching processed output")
    return rows


def analyze(root: Path, output: Path):
    output.mkdir(parents=True, exist_ok=True)
    chunk_rows, action_rows, streak_rows, stage_rows, trial_rows = [], [], [], [], []
    total_steps = total_episodes = 0
    for manifest_path in sorted(root.glob("run_*/manifest.json")):
        m = json.loads(manifest_path.read_text())
        if m["status"] != "complete" or m["profiling"]:
            continue
        run = manifest_path.parent
        common = dict(
            run=m["name"], robots=m["robots"], max_batch=m["max_batch_size"], repeat=m["repeat"]
        )
        bounds = {}
        local_chunks, local_actions, local_streaks = [], [], []
        step_count = 0
        for path in sorted((run / "client").glob("*/*/steps.parquet")):
            meta = json.loads((path.parent / "metadata.json").read_text())
            steps = pd.read_parquet(path)
            chunk_path = path.parent / "action_chunks.parquet"
            chunks = pd.read_parquet(chunk_path) if chunk_path.exists() else pd.DataFrame()
            episode = dict(
                **common,
                robot_id=f"robot_{meta['robot_idx']}",
                episode=meta["episode_idx"],
                task_id=meta["task_id"],
                episode_truncated=meta["truncated"],
                episode_success=meta["success"],
            )
            cr, ar, sr = replay_episode(steps, chunks)
            local_chunks.extend(dict(**episode, **row) for row in cr)
            local_actions.extend(dict(**episode, **row) for row in ar)
            local_streaks.extend(dict(**episode, **row) for row in sr)
            bounds[(episode["robot_id"], episode["episode"])] = dict(
                first=float(steps.timestamp.iloc[0]),
                last=float(steps.timestamp.iloc[-1]),
                truncated=meta["truncated"],
            )
            total_episodes += 1
            step_count += len(steps)
        df = pd.DataFrame(local_chunks)
        if df.empty:
            raise ValueError(f"No saved chunks in {run}")
        reconciled = reconcile_trial(run, df, bounds)
        stages = pd.DataFrame(reconciled)
        trial_rows.append(
            dict(
                **common,
                processed=len(stages),
                acked=int(stages.acked.sum()),
                saved=len(df),
                used_chunks=int((df.executed_actions > 0).sum()),
                unused_chunk_fraction=float((df.executed_actions == 0).mean()),
                executed_actions=int(df.executed_actions.sum()),
                executed_actions_per_chunk=float(df.executed_actions.mean()),
                net_queue_addition_per_chunk=float(df.net_queue_addition.mean()),
                skipped_prefix_per_chunk=float(df.skipped_past_prefix.mean()),
                replacement_per_chunk=float(df.replaced_existing_actions_on_receipt.mean()),
                remaining_at_snapshot=int(df.remaining_at_episode_snapshot.sum()),
                steps=step_count,
                starvation_streaks=len(local_streaks),
                starvation_streak_p95_steps=float(
                    np.percentile([s["length_steps"] for s in local_streaks], 95)
                )
                if local_streaks
                else 0.0,
                maximum_starvation_streak_steps=max(
                    (s["length_steps"] for s in local_streaks), default=0
                ),
                request_to_step_record_p95_ms=float(
                    np.percentile([a["request_to_step_record_ms"] for a in local_actions], 95)
                ),
            )
        )
        total_steps += step_count
        chunk_rows.extend(local_chunks)
        action_rows.extend(local_actions)
        streak_rows.extend(local_streaks)
        stage_rows.extend(dict(**common, **row) for row in reconciled)
        print(f"Validated {m['name']}: {step_count} steps, {len(df)} saved chunks", flush=True)
    if not trial_rows:
        raise ValueError("No completed unprofiled trials")
    frames = {
        "chunks": pd.DataFrame(chunk_rows),
        "executed_actions": pd.DataFrame(action_rows),
        "starvation_streaks": pd.DataFrame(streak_rows),
        "response_stages": pd.DataFrame(stage_rows),
        "trials": pd.DataFrame(trial_rows),
    }
    join_keys = ["run", "robot_id", "request_id", "chunk_id"]
    frames["chunks"] = frames["chunks"].merge(
        frames["response_stages"][join_keys + ["source_episode", "saved_in_source_episode"]],
        on=join_keys,
        how="left",
        validate="one_to_one",
    )
    frames["chunks"][frames["chunks"].saved_in_source_episode.eq(False)].to_csv(
        output / "cross_episode_chunks.csv", index=False
    )
    for name, frame in frames.items():
        frame.to_csv(output / f"{name}.csv", index=False)
    t = frames["trials"]
    conditions = (
        t.groupby(["robots", "max_batch"])
        .agg(
            repeats=("run", "count"),
            executed_actions_per_chunk_mean=("executed_actions_per_chunk", "mean"),
            executed_actions_per_chunk_std=("executed_actions_per_chunk", "std"),
            net_queue_addition_per_chunk_mean=("net_queue_addition_per_chunk", "mean"),
            skipped_prefix_per_chunk_mean=("skipped_prefix_per_chunk", "mean"),
            replacement_per_chunk_mean=("replacement_per_chunk", "mean"),
            unused_chunk_fraction_mean=("unused_chunk_fraction", "mean"),
            request_to_step_record_p95_ms_mean=("request_to_step_record_p95_ms", "mean"),
            maximum_starvation_streak_steps=("maximum_starvation_streak_steps", "max"),
        )
        .reset_index()
    )
    conditions.to_csv(output / "conditions.csv", index=False)
    stages = frames["response_stages"]
    missing = stages[~stages.saved]
    validation = dict(
        trials=len(t),
        episodes=total_episodes,
        reconstructed_steps=total_steps,
        every_recorded_action_and_queue_depth_matched=True,
        per_chunk_action_partition_conserved=True,
        processed=len(stages),
        acked=int(stages.acked.sum()),
        saved=len(chunk_rows),
        acked_not_saved=int((~stages.saved & stages.acked).sum()),
        no_recorded_ack=int((~stages.acked).sum()),
        saved_without_ack=int((stages.saved & ~stages.acked).sum()),
        saved_in_other_episode=int((stages.saved & stages.saved_in_source_episode.eq(False)).sum()),
        missing_after_source_last_recorded_step=int(missing.after_source_last_recorded_step.sum()),
        missing_acked_from_completed_episode=int(
            (~stages.saved & stages.acked & ~stages.source_episode_truncated).sum()
        ),
        missing_acked_from_truncated_episode=int(
            (~stages.saved & stages.acked & stages.source_episode_truncated).sum()
        ),
        missing_no_ack_from_truncated_episode=int(
            (~stages.acked & stages.source_episode_truncated).sum()
        ),
        cross_episode_recorded_actions=int(
            stages.loc[stages.saved_in_source_episode.eq(False), "recorded_executed_actions"].sum()
        ),
        limitations=[
            "Queue depths are reconstructed, not newly measured receipt events.",
            "Independent send-complete events and exact snapshot/reset timestamps were not logged.",
            "No recorded ACK does not establish whether sending or receiving occurred.",
            "request_to_step_record_ms excludes observation creation and ends after apply_action.",
            "Unsaved outputs have no recorded action use; actual use outside recorded steps is not observed.",
        ],
    )
    (output / "validation.json").write_text(json.dumps(validation, indent=2))
    plot_summary(frames["chunks"], frames["starvation_streaks"], output)
    print(conditions.to_string(index=False))
    print(json.dumps(validation, indent=2))


def plot_summary(chunks, streaks, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    chunks = chunks[chunks.robots == 4]
    streaks = streaks[streaks.robots == 4]
    if chunks.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), layout="constrained")
    means = chunks.groupby("max_batch")
    x = np.arange(len(means))
    bottom = np.zeros(len(x))
    for column, label in [
        ("executed_actions", "Executed"),
        ("skipped_past_prefix", "Past prefix at receipt"),
        ("replaced_before_execution", "Replaced by later chunk"),
        ("remaining_at_episode_snapshot", "Remaining at episode end"),
    ]:
        values = means[column].mean().to_numpy()
        axes[0].bar(x, values, bottom=bottom, label=label)
        bottom += values
    axes[0].set(
        xticks=x,
        xticklabels=list(means.groups),
        xlabel="Maximum batch size",
        ylabel="Actions per saved chunk",
        title="Reconstructed action accounting",
    )
    axes[0].legend(fontsize=8)
    for i, (batch, group) in enumerate(streaks.groupby("max_batch")):
        freq = group.length_steps.value_counts(normalize=True).reindex(range(1, 15), fill_value=0)
        axes[1].bar(
            np.arange(1, 15) + (i - 1) * 0.25,
            freq.values * 100,
            width=0.25,
            label=f"Max batch {batch}",
        )
    axes[1].set(
        xlabel="Consecutive starved steps",
        ylabel="Share of starvation events (%)",
        title="After first executed model action",
        xticks=range(1, 15),
    )
    axes[1].legend(fontsize=8)
    fig.suptitle(
        "4 robots; pooled 3 x 180 s trials, seed 7; includes recorded cross-episode actions"
    )
    fig.savefig(output / "chunk_use_comparison.png", dpi=170)
    fig.savefig(output / "chunk_use_comparison.pdf")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    analyze(args.root, args.output or args.root / "chunk_usage")


if __name__ == "__main__":
    main()
