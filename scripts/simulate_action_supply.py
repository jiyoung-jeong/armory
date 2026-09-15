"""CPU event simulation using the actual schedulers and client ActionChunkBroker.

This measures action availability under a timing model, not task success or an
optimal scheduling bound. No dynamics/model/network execution is simulated.
"""

from __future__ import annotations

import argparse
import heapq
import itertools
import json
import logging
from pathlib import Path
from queue import Queue
from unittest.mock import patch

import numpy as np
import pandas as pd

from armory.scheduling.baselines import (
    GreedyDeadlineScheduler,
    MaxBatchScheduler,
    RoundRobinScheduler,
)
from armory.scheduling.latency import LatencyTracker
from armory.scheduling.lookahead_actions import LookaheadActionsScheduler
from armory.serving.protocol import SchedulerConfig
from armory.serving.rtc import InferType
from armory.serving.schemas import AckNotification, ResponseBatch, SlotRequest
from armory_client.action_chunk_broker import ActionChunkBroker
from armory_client.messages import InferResponse, ResponseAck

CLASSES = {
    "round-robin": RoundRobinScheduler,
    "max-batch": MaxBatchScheduler,
    "greedy-deadline": GreedyDeadlineScheduler,
    "lookahead-actions": LookaheadActionsScheduler,
}


class FixedLatency(LatencyTracker):
    def __init__(self, costs):
        super().__init__()
        self.costs = costs

    def observation_latency(self, robot_id):
        return 0.002

    def action_latency(self, robot_id):
        return 0.001

    def infer_latency(self, batch_size):
        return self.costs[batch_size]

    def _update_measurement(self, *args):
        pass


def simulate(algorithm, cap, costs, *, seconds=60, gap_ms=5, stagger=False):
    period = 50_000  # integer microseconds avoid accumulated control-clock drift
    now = 0
    base = 1_000_000.0
    queue = Queue()
    scheduler = CLASSES[algorithm](SchedulerConfig(scheduling_algorithm=algorithm), queue, cap)
    scheduler.latency_tracker = FixedLatency(costs)
    scheduler.mirror.latency_tracker = scheduler.latency_tracker
    brokers = [ActionChunkBroker(1, 10) for _ in range(4)]
    latest = {}
    last_served = {}
    filtered_requests = 0
    events = []
    serial = itertools.count()
    request_ids = itertools.count(1)
    busy = False
    schedule_pending = False
    end = round(seconds * 1e6)
    measured = supplied = chunks = queue_net = 0
    durations = []
    actual_sizes = []

    def add(at, kind, data=None):
        heapq.heappush(events, (at, next(serial), kind, data))

    def attempt(at):
        nonlocal schedule_pending
        if not busy and not schedule_pending:
            schedule_pending = True
            add(at, "schedule")

    for r in range(4):
        add(r * period // 4 if stagger else 0, "tick", (r, 0))
    with patch("time.time", lambda: base + now / 1e6):
        while events:
            now, _, kind, data = heapq.heappop(events)
            if now >= end:
                break
            if kind == "tick":
                r, tick = data
                broker = brokers[r]
                action = broker.get_action(tick)
                if now >= 5_000_000:  # identical fixed warm interval for every robot/policy
                    measured += 1
                    supplied += action is not None
                req = SlotRequest(
                    slot_index=r,
                    robot_id=f"robot_{r}",
                    request_id=next(request_ids),
                    arrival_timestamp=base + (now + 2000) / 1e6,
                    observation_step=tick,
                    action_index_start=broker.next_action_step,
                    request_timestamp=base + now / 1e6,
                    deadline=0,
                    min_execution_horizon=1,
                    max_execution_horizon=10,
                    infer_type=InferType.SYNC,
                    params=None,
                    noise=None,
                    control_hz=20,
                )
                add(now + 2000, "request", req)
                add(now + period, "tick", (r, tick + 1))
            elif kind == "request":
                latest[data.robot_id] = data
                scheduler.update(data)
                attempt(now)
            elif kind == "schedule":
                schedule_pending = False
                if busy:
                    continue
                scheduler.schedule()
                if queue.empty():
                    continue
                batch = queue.get_nowait()
                assert queue.empty() and batch.idle_duration == 0
                # As in the GPU worker, read the latest atomic request per robot.
                kept = []
                for selected, cid in zip(batch.requests, batch.chunk_ids, strict=True):
                    req = latest[selected.robot_id]
                    previous = last_served.get(req.robot_id)
                    if previous is None or selected.can_serve(previous, req.action_index_start):
                        kept.append((req, cid))
                        last_served[req.robot_id] = req.action_index_start
                    else:
                        filtered_requests += 1
                duration = costs[len(kept)] if kept else 0.0
                busy = True
                add(now + round(duration * 1e6), "complete", (batch, kept, now, duration))
            elif kind == "complete":
                batch, kept, started, duration = data
                responses = [
                    InferResponse(
                        robot_id=r.robot_id,
                        request_id=r.request_id,
                        chunk_id=cid,
                        observation_step=r.observation_step,
                        action_index_start=r.action_index_start,
                        request_timestamp=r.request_timestamp,
                        actions=np.zeros((10, 7)),
                        min_execution_horizon=1,
                        max_execution_horizon=10,
                    )
                    for r, cid in kept
                ]
                scheduler.on_batch_completed(
                    ResponseBatch(
                        responses=responses,
                        batch_id=batch.batch_id,
                        batch_size=len(responses),
                        inference_start_time=base + started / 1e6,
                        inference_duration=duration,
                    )
                )
                for response in responses:
                    add(now + 1000, "response", response)
                if now >= 5_000_000:
                    actual_sizes.append(len(responses))
                    durations.append(duration)
                # Reserve the measured inter-inference gap; request events cannot bypass it.
                add(now + round(gap_ms * 1000), "ready")
            elif kind == "ready":
                busy = False
                attempt(now)
            elif kind == "response":
                r = int(data.robot_id.rsplit("_", 1)[1])
                broker = brokers[r]
                before = broker.num_actions_available
                chunk = broker.receive_response(data)
                if now >= 5_000_000:
                    chunks += 1
                    queue_net += broker.num_actions_available - before
                scheduler.update_ack(
                    AckNotification(
                        robot_id=data.robot_id,
                        server_send_time=base + (now - 1000) / 1e6,
                        ack=ResponseAck(
                            request_id=data.request_id,
                            chunk_id=data.chunk_id,
                            observation_step=data.observation_step,
                            action_index_start=data.action_index_start,
                            min_execution_horizon=1,
                            max_execution_horizon=10,
                            receive_time=base + now / 1e6,
                            execution_start_step=chunk.execution_start_step,
                            first_executed_index=max(
                                0, broker.next_action_step - data.action_index_start
                            ),
                        ),
                    )
                )
    return dict(
        algorithm=algorithm,
        max_batch=cap,
        gap_ms=gap_ms,
        stagger=stagger,
        seconds=seconds,
        measured_steps=measured,
        supplied_steps=supplied,
        starvation_rate=1 - supplied / measured,
        supplied_actions_per_s=supplied / (seconds - 5),
        chunks=chunks,
        queue_net_per_chunk=queue_net / chunks,
        worker_filtered_requests=filtered_requests,
        actual_batch_mean=float(np.mean(actual_sizes)),
        infer_time_sum_s=sum(durations),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=60)
    parser.add_argument("--latency-sensitivity", action="store_true")
    args = parser.parse_args()
    if args.seconds <= 5:
        parser.error("seconds must exceed the 5-second warm interval")
    args.output.mkdir(parents=True, exist_ok=False)
    logging.disable(logging.CRITICAL)
    fixed = pd.read_csv("output/fixed_batch_20260915/summary.csv")
    costs_fixed = {
        int(r.batch_size): r.mean / 1000 for r in fixed[fixed.phase == "fixed"].itertuples()
    }
    frames = []
    for p in Path("output/local_followup_20260915").glob("run_r4_*/policy/server/batches.jsonl"):
        frames.append(pd.read_json(p, lines=True))
    live = pd.concat(frames)
    costs_live = live.groupby("batch_size").inference_duration.mean().to_dict()
    costs = {"fixed_isolated": costs_fixed, "observed_corun": costs_live}
    (args.output / "costs.json").write_text(json.dumps(costs, indent=2))
    rows = []
    for cost_name, table in costs.items():
        for gap in (0, 5):
            for stagger in (False, True):
                for alg in CLASSES:
                    for cap in (1, 2, 3, 4):
                        row = simulate(
                            alg, cap, table, seconds=args.seconds, gap_ms=gap, stagger=stagger
                        )
                        row["cost_source"] = cost_name
                        rows.append(row)
                        print(json.dumps(row), flush=True)
                        pd.DataFrame(rows).to_csv(args.output / "conditions.csv", index=False)
    sensitivity_rows = []
    if args.latency_sensitivity:
        for scale in [1, 0.9, 0.8, 0.7, 0.5]:
            for algorithm in ["lookahead-actions", "round-robin"]:
                for cap in [2, 3]:
                    row = simulate(
                        algorithm,
                        cap,
                        {k: v * scale for k, v in costs_live.items()},
                        seconds=args.seconds,
                        stagger=True,
                    )
                    row["latency_scale"] = scale
                    sensitivity_rows.append(row)
        pd.DataFrame(sensitivity_rows).to_csv(args.output / "latency_sensitivity.csv", index=False)
    (args.output / "manifest.json").write_text(
        json.dumps(
            dict(
                status="complete",
                conditions=len(rows),
                latency_sensitivity_conditions=len(sensitivity_rows),
                seconds=args.seconds,
                description=__doc__,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
