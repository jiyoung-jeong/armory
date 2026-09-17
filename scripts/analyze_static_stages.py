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
            stage_rows.append(
                dict(
                    batch_size=call["batch_size"],
                    phase=call["phase"],
                    index=call["index"],
                    stage=stage,
                    kernel_count=len(selected),
                    kernel_sum_ms=selected.duration_ms.sum(),
                    kernel_union_ms=duration,
                    span_ms=(selected.end.max() - selected.start.min()) / 1e6
                    if len(selected)
                    else 0,
                )
            )
        copies = own[own.kind.eq("MEMCPY")]
        row["copy_union_ms"] = union_length(intervals(copies)) / 1e6
        row["copy_bytes"] = int(copies.bytes.sum())
        sync = [
            (a["start"], a["end"]) for a in runtime if a["call"] == i and "Synchronize" in a["name"]
        ]
        row["host_sync_union_ms"] = union_length(sync) / 1e6
        loops = resolvers[call["batch_size"]].action_loop_names()
        row["action_loop_body_count"] = sum(
            bool(re.search(r"hlo_op=" + re.escape(name) + r"_body(?:[,#])", label or ""))
            for start, end, label, tid in nvtx
            if end and call["start"] <= start < end <= call["end"]
            for name in loops
        )
        call_rows.append(row)
    per_call = pd.DataFrame(call_rows)
    per_call.to_csv(dest / "calls.csv", index=False)
    pd.DataFrame(stage_rows).to_csv(dest / "stages.csv", index=False)
    measured = per_call[per_call.phase.eq("profile")]
    assert measured.groupby("batch_size").size().to_dict() == {
        b: manifest["samples"] for b in manifest["batch_sizes"]
    }
    columns = [
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
    summary.to_csv(dest / "summary.csv", index=False)
    relevant = frame[frame.phase.eq("profile")]
    relevant.groupby(["batch_size", "stage", "kernel"]).agg(
        count=("duration_ms", "size"),
        total_ms=("duration_ms", "sum"),
        mean_ms=("duration_ms", "mean"),
    ).reset_index().sort_values(["batch_size", "total_ms"], ascending=[True, False]).to_csv(
        dest / "kernels.csv", index=False
    )
    validation = dict(
        status="complete",
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
