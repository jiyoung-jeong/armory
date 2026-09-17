"""Attribute Nsight GPU activities using CUDA correlation IDs and enclosing XLA thunks."""

from __future__ import annotations

import argparse
import bisect
import collections
import heapq
import json
import re
import sqlite3
from pathlib import Path

import pandas as pd
from scripts.analyze_nsight_followup import union_length

STAGES = ["vlm_embed", "vlm_prefill", "action"]
TAG = re.compile(r"armory_stage_(vlm_embed|vlm_prefill|action)")
CALL = re.compile(r"armory.stage_call phase=(\w+) b=(\d+) i=(\d+)")


class HloStages:
    def __init__(self, text):
        self.instructions = {}
        self.computations = collections.defaultdict(list)
        current = None
        for line in text.splitlines():
            match = re.match(r"^(?:ENTRY )?%([\w.-]+)\s*\(", line)
            if match:
                current = match[1]
            match = re.match(r"^\s+(?:ROOT )?%([\w.-]+) = ", line)
            if match:
                name = match[1]
                assert name not in self.instructions, name
                self.instructions[name] = dict(
                    tags=set(TAG.findall(line)),
                    refs=re.findall(r"(?:calls|body|condition|to_apply)=%([\w.-]+)", line),
                    computation=current,
                    line=line,
                )
                self.computations[current].append(name)
        self.cache = {}

    def computation_tags(self, name):
        if name not in self.cache:
            tags = set()
            self.cache[name] = tags
            for instruction in self.computations.get(name, []):
                tags.update(self.instruction_tags(instruction))
        return self.cache[name]

    def instruction_tags(self, name):
        row = self.instructions.get(name)
        if row is None:
            return set()
        tags = set(row["tags"])
        for ref in row["refs"]:
            tags.update(self.computation_tags(ref))
        return tags

    def label_tags(self, label):
        tags = set(TAG.findall(label))
        op = re.search(r"hlo_op=([^,#]+)", label)
        if op:
            name = op[1]
            if name.endswith("_body") or name.endswith("_condition"):
                name = name.rsplit("_", 1)[0]
            tags.update(self.instruction_tags(name))
        return tags

    def loop_context_tags(self, label):
        op = re.search(r"hlo_op=([^,#]+)_(?:body|condition)(?:[,#])", label)
        if op and op[1] in self.instructions:
            return self.instructions[op[1]]["tags"]
        return set()

    def action_loop_names(self):
        return [
            name
            for name, row in self.instructions.items()
            if " while(" in row["line"] and re.search(r'armory_stage_action/while"', row["line"])
        ]


def stage_name(tags):
    if len(tags) == 1:
        return next(iter(tags))
    if tags and tags <= {"vlm_embed", "vlm_prefill"}:
        return "vlm_mixed"
    return "mixed" if tags else "unattributed"


def correlate_apis(apis, nvtx, resolver):
    """Match CPU API intervals on their own thread, never GPU-time containment."""
    ranges = collections.defaultdict(list)
    for start, end, label, tid in nvtx:
        if end is not None and end > start and label and label.startswith("Thunk:"):
            ranges[tid].append((start, end, label))
    grouped = collections.defaultdict(list)
    for api in apis:
        grouped[api["tid"]].append(api)
    result = {}
    for tid, group in grouped.items():
        regions = sorted(ranges[tid], key=lambda row: row[0])
        pointer = 0
        active = {}
        ends = []
        for api in sorted(group, key=lambda x: x["start"]):
            while pointer < len(regions) and regions[pointer][0] <= api["start"]:
                row = regions[pointer]
                active[pointer] = row
                heapq.heappush(ends, (row[1], pointer))
                pointer += 1
            while ends and ends[0][0] < api["start"]:
                _, index = heapq.heappop(ends)
                active.pop(index, None)
            enclosing = sorted(
                (row for row in active.values() if row[1] >= api["end"]),
                key=lambda row: row[1] - row[0],
            )
            selected = ""
            tags = set()
            # A shared fused computation may retain a different caller's source
            # metadata. The executing loop's call-site is authoritative here.
            for _, _, label in enclosing:
                found = resolver.loop_context_tags(label)
                if len(found) == 1:
                    selected = label
                    tags = found
                    break
            if not tags:
                for _, _, label in enclosing:
                    found = resolver.label_tags(label)
                    if found:
                        selected = label
                        tags = found
                        break
            key = (api["pid"], api["correlation"])
            assert key not in result, key
            result[key] = dict(stage=stage_name(tags), label=selected, api=api)
    return result


def analyze(folder):
    run = folder / "run"
    dest = folder / "analysis"
    dest.mkdir(exist_ok=True)
    manifest = json.loads((run / "manifest.json").read_text())
    assert manifest["status"] == "complete" and manifest["outputs_validated"]
    con = sqlite3.connect(f"file:{folder / 'timeline.sqlite'}?mode=ro", uri=True)
    nvtx = con.execute(
        "select n.start,n.end,coalesce(n.text,s.value),n.globalTid from NVTX_EVENTS n left join StringIds s on s.id=n.textId"
    ).fetchall()
    calls = []
    for start, end, label, tid in nvtx:
        match = CALL.fullmatch(label or "")
        if match and end and end > start:
            calls.append(
                dict(
                    phase=match[1],
                    batch_size=int(match[2]),
                    index=int(match[3]),
                    start=start,
                    end=end,
                    pid=tid & ~0xFFFFFF,
                )
            )
    calls.sort(key=lambda row: row["start"])
    assert calls and len({c["pid"] for c in calls}) == 1
    worker = calls[0]["pid"]
    starts = [c["start"] for c in calls]
    for left, right in zip(calls, calls[1:]):
        assert left["end"] <= right["start"]

    def owner(start, end):
        index = bisect.bisect_right(starts, start) - 1
        if index >= 0 and end <= calls[index]["end"]:
            return index
        return None

    # Each call uses its own optimized HLO module: names can repeat between B shapes.
    resolvers = {
        b: HloStages((run / f"b{b}.optimized_hlo.txt").read_text()) for b in manifest["batch_sizes"]
    }

    loop_regions = {}
    for index, call in enumerate(calls):
        names = resolvers[call["batch_size"]].action_loop_names()
        loop_regions[index] = sorted(
            (start, end)
            for start, end, label, tid in nvtx
            if end
            and (tid & ~0xFFFFFF) == worker
            and call["start"] <= start < end <= call["end"]
            and any(
                re.search(r"hlo_op=" + re.escape(name) + r"_body(?:[,#])", label or "")
                for name in names
            )
        )
    runtime = []
    for start, end, tid, corr, name in con.execute(
        "select a.start,a.end,a.globalTid,a.correlationId,s.value from CUPTI_ACTIVITY_KIND_RUNTIME a join StringIds s on s.id=a.nameId where (a.globalTid & ~16777215)=?",
        (worker,),
    ):
        index = owner(start, end)
        if index is not None:
            runtime.append(
                dict(
                    start=start,
                    end=end,
                    tid=tid,
                    pid=worker,
                    correlation=corr,
                    name=name,
                    call=index,
                )
            )
    activities = []
    tables = {row[0] for row in con.execute("select name from sqlite_master")}
    for kind in ["KERNEL", "MEMCPY", "MEMSET"]:
        table = "CUPTI_ACTIVITY_KIND_" + kind
        if table not in tables:
            continue
        query = (
            f"select a.start,a.end,a.correlationId,a.streamId,s.value,a.graphNodeId,0,0 from {table} a join StringIds s on s.id=a.demangledName where a.globalPid=?"
            if kind == "KERNEL"
            else f"select start,end,correlationId,streamId,'{kind}',graphNodeId,bytes,"
            + ("copyKind" if kind == "MEMCPY" else "0")
            + f" from {table} where globalPid=?"
        )
        for start, end, corr, stream, name, node, byte_count, copy_kind in con.execute(
            query, (worker,)
        ):
            if end <= start:
                continue
            index = owner(start, end)
            if index is None:
                continue
            activities.append(
                dict(
                    start=start,
                    end=end,
                    correlation=corr,
                    stream=stream,
                    kernel=name,
                    graph_node=node,
                    bytes=byte_count,
                    copy_kind=copy_kind,
                    kind=kind,
                    call=index,
                )
            )
    needed = {(worker, a["correlation"]) for a in activities}
    associations = {}
    for b, resolver in resolvers.items():
        selected = [
            a
            for a in runtime
            if calls[a["call"]]["batch_size"] == b and (worker, a["correlation"]) in needed
        ]
        associations.update(correlate_apis(selected, nvtx, resolver))
    for event in activities:
        match = associations.get((worker, event["correlation"]))
        event["stage"] = match["stage"] if match else "unattributed"
        event["thunk"] = match["label"] if match else ""
        event["api_name"] = match["api"]["name"] if match else ""
        event["api_start"] = match["api"]["start"] if match else None
        event["correlation_matched"] = bool(match)
        event["action_step"] = -1
        if match:
            for step, (start, end) in enumerate(loop_regions[event["call"]]):
                if start <= match["api"]["start"] and match["api"]["end"] <= end:
                    event["action_step"] = step
                    break
        event.update(
            batch_size=calls[event["call"]]["batch_size"],
            phase=calls[event["call"]]["phase"],
            index=calls[event["call"]]["index"],
            duration_ms=(event["end"] - event["start"]) / 1e6,
        )
        if match:
            assert match["api"]["call"] == event["call"]
    frame = pd.DataFrame(activities)
    frame.to_csv(dest / "gpu_activities.csv", index=False)
    call_rows = []
    stage_rows = []
    for i, call in enumerate(calls):
        own = frame[frame.call == i]
        kernels = own[own.kind.eq("KERNEL")]

        def intervals(df):
            return list(zip(df.start, df.end, strict=True))

        kernel_union = union_length(intervals(kernels)) / 1e6
        activity_union = union_length(intervals(own)) / 1e6
        full = (call["end"] - call["start"]) / 1e6
        row = dict(
            **call,
            full_ms=full,
            kernel_union_ms=kernel_union,
            kernel_sum_ms=kernels.duration_ms.sum(),
            activity_union_ms=activity_union,
            no_recorded_cuda_ms=full - activity_union,
            kernel_count=len(kernels),
            streams=own.stream.nunique(),
        )
        assert row["no_recorded_cuda_ms"] >= -1e-5
        for stage in [*STAGES, "vlm_mixed", "mixed", "unattributed"]:
            selected = kernels[kernels.stage.eq(stage)]
            duration = union_length(intervals(selected)) / 1e6
            row[stage + "_kernel_ms"] = duration
            all_stage = own[own.stage.eq(stage)]
            stage_copies = all_stage[all_stage.kind.eq("MEMCPY")]
            row[stage + "_activity_ms"] = union_length(intervals(all_stage)) / 1e6
            row[stage + "_copy_ms"] = union_length(intervals(stage_copies)) / 1e6
            stage_rows.append(
                dict(
                    batch_size=call["batch_size"],
                    phase=call["phase"],
                    index=call["index"],
                    stage=stage,
                    kernel_count=len(selected),
                    kernel_sum_ms=selected.duration_ms.sum(),
                    kernel_union_ms=duration,
                    activity_union_ms=row[stage + "_activity_ms"],
                    copy_union_ms=row[stage + "_copy_ms"],
                    copy_bytes=int(stage_copies.bytes.sum()),
                    span_ms=(selected.end.max() - selected.start.min()) / 1e6
                    if len(selected)
                    else 0,
                )
            )
        copies = own[own.kind.eq("MEMCPY")]
        row["copy_union_ms"] = union_length(intervals(copies)) / 1e6
        row["copy_bytes"] = int(copies.bytes.sum())
        for kind, label in [(1, "h2d"), (2, "d2h"), (8, "d2d")]:
            subset = copies[copies.copy_kind.eq(kind)]
            row[label + "_copy_ms"] = union_length(intervals(subset)) / 1e6
            row[label + "_bytes"] = int(subset.bytes.sum())
        row["cross_stage_overlap_ms"] = (
            sum(
                row[stage + "_activity_ms"]
                for stage in [*STAGES, "vlm_mixed", "mixed", "unattributed"]
            )
            - activity_union
        )
        sync = [
            (a["start"], a["end"]) for a in runtime if a["call"] == i and "Synchronize" in a["name"]
        ]
        row["host_sync_union_ms"] = union_length(sync) / 1e6
        row["action_loop_body_count"] = len(loop_regions[i])
        call_rows.append(row)
    per_call = pd.DataFrame(call_rows)
    per_call.to_csv(dest / "calls.csv", index=False)
    pd.DataFrame(stage_rows).to_csv(dest / "stages.csv", index=False)
    measured = per_call[per_call.phase.eq("profile")]
    assert measured.groupby("batch_size").size().to_dict() == {
        b: manifest["samples"] for b in manifest["batch_sizes"]
    }
    assert measured.action_loop_body_count.eq(manifest["num_steps"]).all()
    columns = [
        "cross_stage_overlap_ms",
        *[
            s + suffix
            for s in [*STAGES, "vlm_mixed", "mixed", "unattributed"]
            for suffix in ["_activity_ms", "_copy_ms"]
        ],
        *[kind + suffix for kind in ["h2d", "d2h", "d2d"] for suffix in ["_copy_ms", "_bytes"]],
        "full_ms",
        "kernel_union_ms",
        "activity_union_ms",
        "no_recorded_cuda_ms",
        "host_sync_union_ms",
        "copy_union_ms",
        "copy_bytes",
        "kernel_count",
        *[s + "_kernel_ms" for s in [*STAGES, "vlm_mixed", "mixed", "unattributed"]],
    ]
    summary = measured.groupby("batch_size")[columns].mean().reset_index()
    summary["vlm_kernel_ms"] = (
        summary.vlm_embed_kernel_ms + summary.vlm_prefill_kernel_ms + summary.vlm_mixed_kernel_ms
    )
    summary["known_stage_fraction"] = (
        1 - (summary.mixed_kernel_ms + summary.unattributed_kernel_ms) / summary.kernel_union_ms
    )
    summary["vlm_activity_ms"] = (
        summary.vlm_embed_activity_ms
        + summary.vlm_prefill_activity_ms
        + summary.vlm_mixed_activity_ms
    )
    summary["activity_stage_coverage"] = (
        1
        - (summary.mixed_activity_ms + summary.unattributed_activity_ms) / summary.activity_union_ms
    )
    host_calls = pd.read_json(run / "calls.jsonl", lines=True)
    host_controls = host_calls.groupby(["batch_size", "phase"]).duration_ms.mean().unstack()
    for phase in ["before", "profile", "after"]:
        summary[phase + "_wall_ms"] = (
            summary.batch_size.map(host_controls[phase]) if phase in host_controls else float("nan")
        )
    summary["capture_off_mean_ms"] = summary[["before_wall_ms", "after_wall_ms"]].mean(axis=1)
    summary["control_basis"] = (
        "before_only" if manifest.get("capture_until_exit") else "before_and_after"
    )
    summary["capture_overhead_percent"] = 100 * (
        summary.profile_wall_ms / summary.capture_off_mean_ms - 1
    )
    paired = measured.merge(
        host_calls[host_calls.phase.eq("profile")],
        on=["phase", "batch_size", "index"],
        validate="one_to_one",
    )
    assert (paired.full_ms - paired.duration_ms).abs().max() < 1
    summary.to_csv(dest / "summary.csv", index=False)
    relevant = frame[frame.phase.eq("profile")]
    relevant.groupby(["batch_size", "stage", "kernel"]).agg(
        count=("duration_ms", "size"),
        total_ms=("duration_ms", "sum"),
        mean_ms=("duration_ms", "mean"),
    ).reset_index().sort_values(["batch_size", "total_ms"], ascending=[True, False]).to_csv(
        dest / "kernels.csv", index=False
    )
    step_rows = []
    for (batch, index, step), events in relevant[relevant.action_step.ge(0)].groupby(
        ["batch_size", "index", "action_step"]
    ):
        kernels = events[events.kind.eq("KERNEL")]
        copies = events[events.kind.eq("MEMCPY")]
        step_rows.append(
            dict(
                batch_size=batch,
                index=index,
                action_step=step,
                kernel_union_ms=union_length(list(zip(kernels.start, kernels.end, strict=True)))
                / 1e6,
                activity_union_ms=union_length(list(zip(events.start, events.end, strict=True)))
                / 1e6,
                copy_union_ms=union_length(list(zip(copies.start, copies.end, strict=True))) / 1e6,
                kernel_count=len(kernels),
            )
        )
    steps = pd.DataFrame(step_rows)
    assert (
        steps.groupby(["batch_size", "index"])
        .action_step.apply(list)
        .apply(lambda values: values == list(range(manifest["num_steps"])))
        .all()
    )
    steps.to_csv(dest / "action_steps.csv", index=False)
    diagnostic_rows = con.execute(
        "select timestamp,severity,text,globalPid,timestampType,source from DIAGNOSTIC_EVENT where severity in (2,3)"
    ).fetchall()
    diagnostics = [
        dict(
            timestamp=r[0],
            severity=r[1],
            message=r[2],
            global_pid=r[3],
            inference_worker=r[3] == worker,
            timestamp_type=r[4],
            source=r[5],
        )
        for r in diagnostic_rows
    ]
    (dest / "trace_diagnostics.json").write_text(json.dumps(diagnostics, indent=2))
    worker_warnings = [r for r in diagnostics if r["inference_worker"]]
    print("Inference worker diagnostic warnings:", worker_warnings)
    validation = dict(
        status="complete",
        trace_quality="worker_warnings_present" if worker_warnings else "no_worker_warnings",
        worker_warnings=worker_warnings,
        kernel_counts_per_batch=measured.groupby("batch_size").kernel_count.apply(list).to_dict(),
        calls=len(measured),
        stablehlo_equal=all(x["identical"] for x in manifest["stablehlo_checks"].values()),
        gpu_events=len(relevant),
        correlation_matched_events=int(relevant.correlation_matched.sum()),
        gpu_correlation_coverage=float(relevant.correlation_matched.mean()),
        action_loop_names={b: r.action_loop_names() for b, r in resolvers.items()},
        action_loop_body_counts=measured.groupby("batch_size")
        .action_loop_body_count.apply(list)
        .to_dict(),
        stage_coverage=summary.set_index("batch_size").known_stage_fraction.to_dict(),
        activity_stage_coverage=summary.set_index("batch_size").activity_stage_coverage.to_dict(),
        nvtx_host_timer_max_difference_ms=float((paired.full_ms - paired.duration_ms).abs().max()),
        max_cross_stage_overlap_ms=float(measured.cross_stage_overlap_ms.max()),
        kernel_sum_is_not_wall_time=True,
        host_sync_overlaps_gpu=True,
    )
    (dest / "validation.json").write_text(json.dumps(validation, indent=2))
    print(summary.to_string(index=False))
    print(json.dumps(validation, indent=2))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    analyze(parser.parse_args().folder)
