"""Summarize completed local batch trials without mixing profiler runs into comparisons."""

from __future__ import annotations

import argparse
import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def percentile(values, p):
    return float(np.percentile(values, p)) if len(values) else float("nan")


def summarize_trial(path):
    manifest = json.loads((path / "manifest.json").read_text())
    if manifest["status"] != "complete":
        return None, []
    client = path / "client"
    episodes = pd.read_csv(client / "results.csv")
    robot_rows = []
    intervals = []
    chunks = []
    chunk_gaps = []
    starts = []
    ends = []
    streak_lengths = []
    for rid, robot_episodes in episodes.groupby("robot_idx"):
        robot_intervals = []
        robot_chunk_gaps = []
        robot_streaks = []
        for p in sorted((client / str(rid)).glob("*/steps.parquet")):
            s = pd.read_parquet(p)
            if s.empty:
                continue
            t = s.timestamp.to_numpy()
            starts.append(float(t[0]))
            ends.append(float(t[-1]))
            robot_intervals.extend(np.diff(t) * 1000)
            starved = s.action_chunk_index.isna().to_numpy()
            served = np.flatnonzero(~starved)
            first = int(served[0]) if len(served) else len(s)
            post = np.where(np.arange(len(s)) >= first, starved, False)
            a = np.flatnonzero(np.diff(np.r_[False, post].astype(int)) == 1)
            z = np.flatnonzero(np.diff(np.r_[post, False].astype(int)) == -1) + 1
            robot_streaks.extend((z - a).tolist())
            chunk_path = p.parent / "action_chunks.parquet"
            if not chunk_path.exists():
                if s.action_chunk_index.notna().any():
                    raise ValueError(f"Missing chunks for executed actions: {p.parent}")
                # save_action_chunks omits the file if no response arrived.
                # The episode's steps and starvation still count above.
                continue
            c = pd.read_parquet(
                chunk_path,
                columns=["request_id", "chunk_id", "request_timestamp", "response_timestamp"],
            )
            c["robot_id"] = f"robot_{rid}"
            chunks.append(c)
            robot_chunk_gaps.extend(np.diff(c.response_timestamp.to_numpy()) * 1000)
        intervals.extend(robot_intervals)
        chunk_gaps.extend(robot_chunk_gaps)
        streak_lengths.extend(robot_streaks)
        observed = int(robot_episodes.observed_steps.sum())
        starved = int(robot_episodes.starvation_steps.sum())
        post_steps = int(robot_episodes.post_first_starvation_steps.sum())
        post_observed = int(robot_episodes.post_first_observed_steps.sum())
        robot_rows.append(
            dict(
                run=manifest["name"],
                robot=int(rid),
                robots=manifest["robots"],
                max_batch=manifest["max_batch_size"],
                repeat=manifest["repeat"],
                profiling=manifest["profiling"],
                tasks=",".join(str(x) for x in sorted(robot_episodes.task_id.unique())),
                success=int(robot_episodes.success.sum()),
                truncated=int(robot_episodes.truncated.sum()),
                steps=observed,
                starvation_steps=starved,
                post_first_starvation_steps=post_steps,
                post_first_starvation_rate=post_steps / post_observed
                if post_observed
                else float("nan"),
                starvation_streaks=len(robot_streaks),
                max_streak_steps=max(robot_streaks, default=0),
                step_interval_p95_ms=percentile(robot_intervals, 95),
                chunk_gap_mean_ms=float(np.mean(robot_chunk_gaps))
                if robot_chunk_gaps
                else float("nan"),
                chunk_gap_p95_ms=percentile(robot_chunk_gaps, 95),
            )
        )
    chunks = pd.concat(chunks, ignore_index=True)
    batches = pd.read_json(path / "policy/server/batches.jsonl", lines=True, convert_dates=False)
    infer = batches[batches.batch_size > 0].copy()
    events = pd.read_json(path / "policy/server/events.jsonl", lines=True, convert_dates=False)
    req = events[events.kind == "request"]
    ack = events[events.kind == "ack"]
    processed = (
        infer[["processed_robot_ids", "processed_request_ids", "chunk_ids", "batch_id"]]
        .explode(["processed_robot_ids", "processed_request_ids", "chunk_ids"])
        .rename(
            columns={
                "processed_robot_ids": "robot_id",
                "processed_request_ids": "request_id",
                "chunk_ids": "chunk_id",
            }
        )
    )
    joined = chunks.merge(
        processed, on=["robot_id", "request_id", "chunk_id"], how="left", validate="one_to_one"
    )
    if joined.batch_id.isna().any():
        raise ValueError(f"Unmatched saved chunks in {path}")
    mismatches = sum(
        sum(a != b for a, b in zip(row.request_ids, row.processed_request_ids, strict=True))
        for row in infer.itertuples()
    )
    observed = int(episodes.observed_steps.sum())
    post_observed = int(episodes.post_first_observed_steps.sum())
    completed = episodes[~episodes.truncated]
    latency = (chunks.response_timestamp - chunks.request_timestamp) * 1000
    samples = [
        json.loads(x)["value"].split(",")
        for x in (path / "gpu_samples.jsonl").read_text().splitlines()
    ]
    row = dict(
        run=manifest["name"],
        robots=manifest["robots"],
        max_batch=manifest["max_batch_size"],
        repeat=manifest["repeat"],
        profiling=manifest["profiling"],
        seconds=manifest["seconds"],
        success=int(episodes.success.sum()),
        completed=len(completed),
        failed_completed=int((~completed.success).sum()),
        truncated=int(episodes.truncated.sum()),
        successes_per_min=float(episodes.success.sum()) / manifest["seconds"] * 60,
        steps=observed,
        starvation_steps=int(episodes.starvation_steps.sum()),
        starvation_rate=float(episodes.starvation_steps.sum()) / observed,
        post_first_starvation_steps=int(episodes.post_first_starvation_steps.sum()),
        post_first_starvation_rate=float(episodes.post_first_starvation_steps.sum())
        / post_observed,
        starvation_streaks=len(streak_lengths),
        max_streak_steps=max(streak_lengths, default=0),
        step_interval_mean_ms=float(np.mean(intervals)),
        step_interval_p95_ms=percentile(intervals, 95),
        step_interval_p99_ms=percentile(intervals, 99),
        step_interval_max_ms=max(intervals),
        chunk_latency_mean_ms=float(latency.mean()),
        chunk_latency_p95_ms=percentile(latency, 95),
        chunk_latency_p99_ms=percentile(latency, 99),
        chunk_gap_mean_ms=float(np.mean(chunk_gaps)) if chunk_gaps else float("nan"),
        chunk_gap_p95_ms=percentile(chunk_gaps, 95),
        stored_chunks=len(chunks),
        received_observations=len(req),
        processed_observations=len(processed),
        ack_count=len(ack),
        selected_id_mismatches=int(mismatches),
        actual_batch_mean=float(infer.batch_size.mean()),
        inference_batches=len(infer),
        sampled_peak_gpu_mib=max(float(x[0]) for x in samples) if samples else float("nan"),
        first_step=min(starts),
        last_step=max(ends),
    )
    for size in [1, 2, 3, 4]:
        q = infer[infer.batch_size == size]
        row[f"batch_{size}_count"] = len(q)
        row[f"batch_{size}_mean_ms"] = float(q.inference_duration.mean() * 1000)
        row[f"batch_{size}_p95_ms"] = percentile(q.inference_duration.to_numpy() * 1000, 95)
    return row, robot_rows


def analyze(root):
    rows = []
    robots = []
    for p in sorted(root.glob("*/manifest.json")):
        row, rr = summarize_trial(p.parent)
        if row is not None:
            rows.append(row)
            robots.extend(rr)
    if not rows:
        raise ValueError("No completed trials")
    df = pd.DataFrame(rows)
    df.to_csv(root / "trials.csv", index=False)
    pd.DataFrame(robots).to_csv(root / "robots.csv", index=False)
    normal = df[~df.profiling]
    if normal.empty:
        print(df.to_string(index=False))
        return
    groups = normal.groupby(["robots", "max_batch"])
    stats = groups.agg(
        repeats=("run", "count"),
        successes_per_min_mean=("successes_per_min", "mean"),
        successes_per_min_std=("successes_per_min", "std"),
        starvation_rate_mean=("starvation_rate", "mean"),
        post_first_starvation_rate_mean=("post_first_starvation_rate", "mean"),
        post_first_starvation_rate_std=("post_first_starvation_rate", "std"),
        chunk_latency_p95_ms_mean=("chunk_latency_p95_ms", "mean"),
        step_interval_p95_ms_mean=("step_interval_p95_ms", "mean"),
        actual_batch_mean=("actual_batch_mean", "mean"),
        chunk_gap_mean_ms=("chunk_gap_mean_ms", "mean"),
        chunk_gap_p95_ms_mean=("chunk_gap_p95_ms", "mean"),
        sampled_peak_gpu_mib=("sampled_peak_gpu_mib", "max"),
    ).reset_index()
    stats.to_csv(root / "conditions.csv", index=False)
    fig, axes_grid = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    axes = axes_grid.ravel()
    for n, g in normal.groupby("robots"):
        for ax, metric, label, scale in [
            (axes[0], "successes_per_min", "Successful tasks / min", 1),
            (axes[1], "post_first_starvation_rate", "Post-first-action starvation (%)", 100),
            (axes[2], "chunk_latency_p95_ms", "Stored chunk latency p95 (ms)", 1),
            (axes[3], "chunk_gap_mean_ms", "Within-episode chunk receipt gap mean (ms)", 1),
        ]:
            means = g.groupby("max_batch")[metric].mean() * scale
            ax.plot(means.index, means.values, marker="o", label=f"{n} robots")
            ax.scatter(g.max_batch, g[metric] * scale, alpha=0.5, s=20)
            ax.set(xlabel="Maximum batch size", ylabel=label, xticks=[1, 2, 4])
            ax.grid(alpha=0.2)
    axes[0].legend()
    durations = ", ".join(f"{v:g}" for v in sorted(normal.seconds.unique()))
    fig.suptitle(f"Local LIBERO: {durations} s trials, seed 7, 20 Hz; no profiler")
    fig.savefig(root / "comparison.png", dpi=160)
    fig.savefig(root / "comparison.pdf")
    plt.close(fig)
    print(stats.to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=pathlib.Path)
    analyze(parser.parse_args().output)


if __name__ == "__main__":
    main()
