"""Join complete Nsight inference ranges to server batches and broker queue events."""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scripts.analyze_followup_events import read_jsonl, stats


def merge_intervals(intervals):
    merged = []
    for start, end in sorted(intervals):
        if end <= start:
            continue
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    return merged


def union_length(intervals):
    return sum(b - a for a, b in merge_intervals(intervals))


def clipped(intervals, start, end):
    return [(max(a, start), min(b, end)) for a, b in intervals if a < end and b > start]


def analyze(run):
    dest = run / "nsight_analysis"
    dest.mkdir(exist_ok=True)
    con = sqlite3.connect(f"file:{(run / 'timeline.sqlite').resolve()}?mode=ro", uri=True)
    epoch = con.execute("select utcEpochNs from TARGET_INFO_SESSION_START_TIME").fetchone()[0] / 1e9
    duration_ns = (
        int(
            con.execute(
                "select value from META_DATA_CAPTURE where name='RUN_DURATION_MS'"
            ).fetchone()[0]
        )
        * 1000000
    )
    mode_row = con.execute(
        "select value from META_DATA_CAPTURE where name='CUDA_GRAPH_TRACE_OPTIONS:MODE'"
    ).fetchone()
    graph_mode = str(mode_row[0]).lower() if mode_row else "unknown"
    tables = {row[0] for row in con.execute("select name from sqlite_master where type='table'")}
    nvtx = con.execute(
        "select start,end,text,globalTid from NVTX_EVENTS where text like 'armory.infer%'"
    ).fetchall()
    ranges = []
    excluded = 0
    for start, end, label, tid in nvtx:
        # StartEnd NVTX ranges open at capture stop can have invalid huge end values.
        if end is None or not 0 <= start < end <= duration_ns:
            excluded += 1
            continue
        match = re.fullmatch(r"armory.infer batch=(\d+) size=(\d+)", label)
        assert match
        ranges.append(
            dict(
                batch_id=int(match[1]),
                batch_size=int(match[2]),
                nvtx_start_ns=start,
                nvtx_end_ns=end,
                duration_ms=(end - start) / 1e6,
                nvtx_start_epoch=epoch + start / 1e9,
                nvtx_end_epoch=epoch + end / 1e9,
                global_pid=tid & ~0xFFFFFF,
            )
        )
    frame = pd.DataFrame(ranges).sort_values("nvtx_start_ns")
    assert not frame.empty and frame.global_pid.nunique() == 1
    worker = int(frame.global_pid.iloc[0])
    batches = pd.DataFrame(read_jsonl(run / "policy/server/batches.jsonl"))
    frame = frame.merge(batches, on=["batch_id", "batch_size"], validate="one_to_one")
    frame["start_error_ms"] = (frame.nvtx_start_epoch - frame.inference_start_time) * 1000
    frame["end_error_ms"] = (
        frame.nvtx_end_epoch - frame.inference_start_time - frame.inference_duration
    ) * 1000
    assert frame.start_error_ms.abs().max() < 1 and frame.end_error_ms.abs().max() < 1
    first, last = int(frame.nvtx_start_ns.min()), int(frame.nvtx_end_ns.max())
    activity = {}
    activity_stats = {}
    all_ranges = []
    for kind in ["KERNEL", "GRAPH_TRACE", "MEMCPY", "MEMSET"]:
        table = "CUPTI_ACTIVITY_KIND_" + kind
        if table not in tables:
            continue
        intervals = con.execute(
            f"select start,end from {table} where globalPid=? and end>start", (worker,)
        ).fetchall()
        intervals = clipped(intervals, 0, duration_ns)
        activity[kind] = merge_intervals(intervals)
        activity_stats[kind] = dict(
            count=len(intervals),
            duration_sum_ms=sum(b - a for a, b in intervals) / 1e6,
            union_ms=union_length(intervals) / 1e6,
        )
        all_ranges.extend(intervals)
    merged = merge_intervals(all_ranges)
    for kind, intervals in {**activity, "ALL_RECORDED": merged}.items():
        frame[f"{kind.lower()}_union_ms"] = [
            union_length(clipped(intervals, a, b)) / 1e6
            for a, b in zip(frame.nvtx_start_ns, frame.nvtx_end_ns, strict=True)
        ]
    frame.to_csv(dest / "linked_batches.csv", index=False)
    runtime = pd.read_sql_query(
        "select s.value as name,count(*) as calls,sum(r.end-r.start)/1e6 as elapsed_sum_ms from CUPTI_ACTIVITY_KIND_RUNTIME r join StringIds s on r.nameId=s.id where (r.globalTid & ~16777215)=? and r.end>r.start and r.start>=0 and r.end<=? group by s.value order by elapsed_sum_ms desc",
        con,
        params=(worker, duration_ns),
    )
    runtime.to_csv(dest / "cuda_api_elapsed.csv", index=False)
    copies = pd.read_sql_query(
        "select copyKind,count(*) as count,sum(end-start)/1e6 as duration_sum_ms,sum(bytes) as bytes from CUPTI_ACTIVITY_KIND_MEMCPY where globalPid=? group by copyKind",
        con,
        params=(worker,),
    )
    copies.to_csv(dest / "copies.csv", index=False)
    sync_by_thread = {}
    for start, end, tid in con.execute(
        "select r.start,r.end,r.globalTid from CUPTI_ACTIVITY_KIND_RUNTIME r "
        "join StringIds s on r.nameId=s.id where s.value='cuStreamSynchronize'"
    ):
        sync_by_thread.setdefault(tid, []).append((start, end))
    phase_rows = []
    starts = frame.nvtx_start_ns.to_numpy()
    for start, end, label, tid in con.execute(
        "select start,end,text,globalTid from NVTX_EVENTS where text like 'armory.openpi.%'"
    ):
        if end is None or (tid & ~0xFFFFFF) != worker or end <= start:
            continue
        index = int(np.searchsorted(starts, start, side="right") - 1)
        if index < 0:
            continue
        parent = frame.iloc[index]
        if end > parent.nvtx_end_ns:
            continue
        sync = clipped(sync_by_thread.get(tid, []), start, end)
        device = clipped(merged, start, end)
        phase_rows.append(
            dict(
                batch_id=int(parent.batch_id),
                batch_size=int(parent.batch_size),
                phase=label.rsplit(".", 1)[1],
                duration_ms=(end - start) / 1e6,
                host_stream_sync_ms=union_length(sync) / 1e6,
                recorded_cuda_union_ms=union_length(device) / 1e6,
                sync_overlapping_cuda_ms=sum(
                    union_length(clipped(device, a, b)) for a, b in merge_intervals(sync)
                )
                / 1e6,
            )
        )
    phase_summary = {}
    if phase_rows:
        phase_frame = pd.DataFrame(phase_rows)
        assert phase_frame.groupby("batch_id").size().eq(4).all()
        assert phase_frame.batch_id.nunique() == len(frame)
        phase_frame.to_csv(dest / "adapter_phases.csv", index=False)
        for (size, phase), group in phase_frame.groupby(["batch_size", "phase"]):
            phase_summary.setdefault(int(size), {})[phase] = dict(
                **stats(group.duration_ms),
                host_stream_sync_mean_ms=float(group.host_stream_sync_ms.mean()),
                recorded_cuda_union_mean_ms=float(group.recorded_cuda_union_ms.mean()),
                sync_overlapping_cuda_mean_ms=float(group.sync_overlapping_cuda_ms.mean()),
            )
        order = ["prepare_inputs", "sample_dispatch", "materialize", "output_transform"]
        means = phase_frame.groupby(["batch_size", "phase"]).duration_ms.mean().unstack()[order]
        ax = means.plot.bar(stacked=True, figsize=(8, 4), rot=0)
        ax.set(
            xlabel="Actual batch size",
            ylabel="Host wall time (ms)",
            title="OpenPI host call phases; sample_dispatch includes synchronization",
        )
        ax.figure.tight_layout()
        ax.figure.savefig(dest / "adapter_phases.png", dpi=170)
        ax.figure.savefig(dest / "adapter_phases.pdf")
        plt.close(ax.figure)
    gaps = (frame.nvtx_start_ns.to_numpy()[1:] - frame.nvtx_end_ns.to_numpy()[:-1]) / 1e6
    incident_path = run / "event_analysis/starvation_incidents.csv"
    has_events = incident_path.exists()
    incidents = (
        pd.read_csv(incident_path)
        if has_events
        else pd.DataFrame(columns=["start", "recovery_stage"])
    )
    in_capture = incidents[
        (incidents.start >= epoch + first / 1e9) & (incidents.start < epoch + last / 1e9)
    ].copy()
    in_capture.to_csv(dest / "starvation_in_capture.csv", index=False)
    by_size = {int(size): stats(group.duration_ms) for size, group in frame.groupby("batch_size")}
    prediction_summary = {}
    lifecycle_path = run / "event_analysis/response_lifecycle.csv"
    if lifecycle_path.exists():
        lifecycle = pd.read_csv(lifecycle_path)
        if "predicted_inference_ms" in lifecycle:
            selected = lifecycle[lifecycle.batch_id.isin(frame.batch_id)].copy()
            comparable = selected[selected.batch_size.eq(selected.predicted_batch_size)].copy()
            unique = comparable.drop_duplicates("batch_id")
            unique.to_csv(dest / "inference_predictions.csv", index=False)
            comparable.to_csv(dest / "response_predictions.csv", index=False)
            prediction_summary = dict(
                scope="Complete inferences inside capture, matching predicted/actual batch size",
                mismatched_batch_requests=int(len(selected) - len(comparable)),
                inferences_by_batch={
                    int(size): {
                        key: stats(group[key].dropna())
                        for key in [
                            "predicted_inference_ms",
                            "inference_ms",
                            "inference_prediction_error_ms",
                            "completion_prediction_error_ms",
                        ]
                    }
                    for size, group in unique.groupby("batch_size")
                },
                received_responses_by_batch={
                    int(size): {
                        key: stats(group[key].dropna())
                        for key in [
                            "arrival_prediction_error_ms",
                            "predicted_new_chunk_actions",
                            "actual_new_chunk_actions",
                            "actual_queue_net_actions",
                            "action_start_prediction_error",
                        ]
                    }
                    for size, group in comparable[comparable.received].groupby("batch_size")
                },
            )
    summary = dict(
        graph_trace=graph_mode,
        decision_time_predictions=prediction_summary,
        run=run.name,
        worker_pid=(worker >> 24) & 0xFFFFFF,
        epoch=epoch,
        capture_duration_s=duration_ns / 1e9,
        complete_inferences=len(frame),
        excluded_nvtx_ranges=excluded,
        start_alignment_max_abs_ms=float(frame.start_error_ms.abs().max()),
        end_alignment_max_abs_ms=float(frame.end_error_ms.abs().max()),
        by_batch=by_size,
        complete_inference_span_s=(last - first) / 1e9,
        inference_wall_ms=float(frame.duration_ms.sum()),
        inference_wall_fraction=float(frame.duration_ms.sum() / ((last - first) / 1e6)),
        between_inference_gap_ms=stats(gaps),
        adapter_phases_by_batch=phase_summary,
        cuda_activity=activity_stats,
        recorded_activity_union_s=union_length(merged) / 1e9,
        starvation_in_capture=len(in_capture) if has_events else None,
        recovery_stages_in_capture=in_capture.recovery_stage.value_counts().to_dict(),
        caveats=[
            "CUDA Graph intervals may include internal gaps. Their union is not GPU/SM utilization.",
            "CUDA API elapsed overlaps device work and other threads; do not add to GPU durations.",
            "Separate LIBERO rendering processes are not traced.",
            "Profiler runs are diagnostic only and excluded from baseline throughput comparisons.",
        ],
    )
    (dest / "summary.json").write_text(json.dumps(summary, indent=2))
    if has_events:
        plot(run, dest, epoch, first / 1e9, last / 1e9, frame, activity, in_capture, graph_mode)
    con.close()
    print(json.dumps(summary), flush=True)
    return summary


def plot(run, dest, epoch, first, last, batches, activity, incidents, graph_mode):
    # Longest observed starvation onset with room around it, ties by earliest time.
    eligible = incidents[
        (incidents.start > epoch + first + 1) & (incidents.start < epoch + last - 1)
    ]
    chosen = (
        eligible.sort_values(["length_steps", "start"], ascending=[False, True]).iloc[0]
        if len(eligible)
        else None
    )
    center = float(chosen.start) - epoch if chosen is not None else (first + last) / 2
    for name, lo, hi in [
        ("capture", first, last),
        ("starvation_detail", max(first, center - 1), min(last, center + 2)),
    ]:
        fig, axes = plt.subplots(
            6,
            1,
            figsize=(14, 10),
            sharex=True,
            gridspec_kw={"height_ratios": [1, 1, 1.2, 1.2, 1.2, 1.2]},
        )
        for b in batches.itertuples():
            a, z = b.nvtx_start_ns / 1e9, b.nvtx_end_ns / 1e9
            if a >= hi or z <= lo:
                continue
            axes[0].broken_barh([(a, z - a)], (0, 1), facecolors=plt.cm.tab10(b.batch_size))
            if name == "starvation_detail":
                axes[0].text(
                    (a + z) / 2,
                    0.5,
                    f"B{b.batch_size}\n#{b.batch_id}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    clip_on=True,
                )
        axes[0].set_ylabel("infer_batch")
        axes[0].set_yticks([])
        kinds = (
            ["KERNEL", "MEMCPY", "MEMSET"]
            if graph_mode == "node"
            else ["KERNEL", "GRAPH_TRACE", "MEMCPY"]
        )
        labels = (
            ["kernel incl. nodes", "copy", "memset"]
            if graph_mode == "node"
            else ["kernel", "graph span", "copy"]
        )
        for j, kind in enumerate(kinds):
            spans = clipped(activity.get(kind, []), int(lo * 1e9), int(hi * 1e9))
            axes[1].broken_barh(
                [(a / 1e9, (b - a) / 1e9) for a, b in spans], (j, 0.8), facecolors=plt.cm.tab10(j)
            )
        axes[1].set_yticks([0.4, 1.4, 2.4], labels, fontsize=8)
        for robot in range(4):
            ax = axes[robot + 2]
            ev = read_jsonl(run / f"client/broker_events_{robot}.jsonl")
            queue = []
            for e in ev:
                if e["kind"] == "reset":
                    queue.append((e["time"] - epoch, 0))
                elif e["kind"] in ("step", "chunk_received"):
                    queue.append((e["time"] - epoch, e["queue_after"]))
            queue.sort()
            if queue:
                ax.step(*zip(*queue, strict=True), where="post", color="#286da8", linewidth=1)
            for e in ev:
                if e["kind"] == "chunk_received" and lo <= e["time"] - epoch <= hi:
                    ax.plot(e["time"] - epoch, e["queue_after"], "v", color="#238b45", markersize=4)
            subset = incidents[incidents.robot_id == f"robot_{robot}"]
            for s in subset.itertuples():
                if np.isfinite(s.end):
                    ax.axvspan(s.start - epoch, s.end - epoch, color="#d73027", alpha=0.2)
            ax.set_ylabel(f"robot {robot}\nqueued actions")
            ax.set_ylim(-0.5, 11)
            ax.grid(alpha=0.2)
        axes[-1].set_xlim(lo, hi)
        axes[-1].set_xlabel(
            "Seconds since Nsight capture start; red = observed action starvation, green = chunk receipt"
        )
        fig.suptitle(
            f"{run.name}: aligned inference and action queues\nCUDA {graph_mode} trace; server process tree only, not GPU utilization"
        )
        fig.tight_layout()
        fig.savefig(dest / f"{name}.png", dpi=160)
        fig.savefig(dest / f"{name}.pdf")
        plt.close(fig)
    if chosen is not None:
        (dest / "selected_starvation.json").write_text(chosen.to_json(indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path)
    args = parser.parse_args()
    for run in args.runs:
        analyze(run)


if __name__ == "__main__":
    main()
