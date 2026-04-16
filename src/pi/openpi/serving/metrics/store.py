from __future__ import annotations

import bisect
from dataclasses import dataclass
from dataclasses import field
import itertools
import logging
import threading
import time

import numpy as np
from openpi_client.messages import EpisodeEnd
from openpi_client.messages import EpisodeStart
from openpi_client.messages import InferResponse
from openpi_client.messages import ResponseAck
from openpi_client.schemas import JSONDataclass

from openpi.serving.metrics.schemas import BatchSummary
from openpi.serving.metrics.schemas import Episode
from openpi.serving.metrics.schemas import RequestRecord
from openpi.serving.metrics.schemas import ResponseRecord
from openpi.serving.metrics.schemas import Robot
from openpi.serving.metrics.schemas import RobotID
from openpi.serving.metrics.schemas import window_filter
from openpi.serving.schemas import SchedulerDecision
from openpi.serving.schemas import SlotRequest

logger = logging.getLogger(__name__)

# TODO: make sure nans are nans and not 0s
# TODO: make sure s, ms, and ns are consistent
# TODO: figure out if locking is necessary, if we can get away without it
# temporary hack to not serialize the lock in metrics store
lock: threading.RLock = threading.RLock()


@dataclass
class Snapshot:
    """Windowed metrics snapshot. All stats, SLA, and starvation computed here."""

    start_time: float
    end_time: float
    sla_pct: float
    # Per-robot windowed (times_relative_to_server_start, actions_left_values) with nan separators between episodes.
    robot_actions_left: dict[RobotID, tuple[np.ndarray, np.ndarray]]
    per_robot: dict[RobotID, dict]
    requests: list[RequestRecord]
    responses: list[ResponseRecord]
    batches: list[BatchSummary]
    completed_episodes: list[tuple[RobotID, Episode]]
    sla_capacity_curve: list[dict]
    healthy_robots_over_time: list[dict]
    # Chart data (formerly in history())
    batch_history: list[dict]
    outbound_delays_ms: dict[str, list[float]]
    scheduler_timing_ms: dict[str, list[float]]
    replan_markers: list[dict]
    kickoff_markers: list[dict]
    task_events: list[dict]
    task_progress: list[dict]
    scheduling_decisions: list[dict]

    @property
    def uptime_s(self) -> float:
        uptime_s = self.end_time - self.start_time
        if uptime_s < 0:
            logger.warning("Uptime is negative: %s", uptime_s)
            return 0.0
        return uptime_s

    @property
    def total_batches(self) -> int:
        return len(self.batches)

    @property
    def total_requests(self) -> int:
        return len(self.requests)

    @property
    def total_responses(self) -> int:
        return len(self.responses)

    @property
    def gpu_times_ms(self) -> list[float]:
        return [b.gpu_time_ms for b in self.batches]

    @property
    def queue_delays_ms(self) -> list[float]:
        return [r.queue_delay_ms for r in self.responses]

    @property
    def requests_per_second(self) -> float:
        return len(self.requests) / self.uptime_s if self.uptime_s > 0 else 0.0

    @property
    def responses_per_second(self) -> float:
        return len(self.responses) / self.uptime_s if self.uptime_s > 0 else 0.0

    @property
    def avg_gpu_time_ms(self) -> float:
        t = self.gpu_times_ms
        return float(np.mean(t)) if t else 0.0

    @property
    def avg_queue_delay_ms(self) -> float:
        d = self.queue_delays_ms
        return float(np.mean(d)) if d else 0.0

    @property
    def p50_latency_ms(self) -> float:
        lats = [r.total_latency_ms for r in self.responses]
        return float(np.percentile(lats, 50)) if lats else 0.0

    @property
    def p99_latency_ms(self) -> float:
        lats = [r.total_latency_ms for r in self.responses]
        return float(np.percentile(lats, 99)) if lats else 0.0

    @property
    def task_success_rate_pct(self) -> float:
        if not self.completed_episodes:
            return 0.0
        return sum(1 for _, ep in self.completed_episodes if ep.success) / len(self.completed_episodes) * 100

    @property
    def tp_suc_per_sec_all(self) -> float:
        return sum(1 for _, ep in self.completed_episodes if ep.success) / self.uptime_s if self.uptime_s > 0 else 0.0

    @property
    def replan_times_s(self) -> list[float]:
        """Backward-compat shim for older dashboard code paths."""
        return [float(marker.get("t", 0.0)) for marker in self.replan_markers]


@dataclass
class MetricsStore(JSONDataclass):
    """Single-call-site metrics store. All updates go through record_batch / record_ack."""

    start_time: float = float("inf")
    end_time: float = 0.0
    robots: dict[RobotID, Robot] = field(default_factory=dict)
    batches: list[BatchSummary] = field(default_factory=list)
    scheduler_decisions: list[SchedulerDecision] = field(default_factory=list)

    def __post_init__(self):
        self.batches = [BatchSummary.from_json(b) for b in self.batches]
        self.robots = {
            robot_id: v
            if isinstance(v, Robot)
            else Robot(**v)
            if isinstance(v, dict)
            else Robot(robot_id=robot_id, episodes=[])
            for robot_id, v in self.robots.items()
        }
        self.scheduler_decisions = [SchedulerDecision.from_json(s) for s in self.scheduler_decisions]

    def record_batch(self, responses: list[InferResponse]) -> None:
        """Called once per batch by _router_task."""
        with lock:
            self.batches.append(
                BatchSummary(
                    batch_id=len(self.batches),
                    robot_ids=[r.robot_id for r in responses],
                    request_ids=[r.request_id for r in responses],
                    inference_start_time=responses[0].inference_start_time,
                    inference_end_time=responses[0].inference_end_time,
                )
            )

    def record_scheduler_decisions(self, samples: list[SchedulerDecision]) -> None:
        """Called from the server process when the scheduler publishes timing samples."""
        with lock:
            self.scheduler_decisions.extend(samples)

    # request/response lifecycle

    def record_request(self, robot_id: str, request: SlotRequest) -> None:
        """Called when client sends InferRequest."""
        with lock:
            record = RequestRecord(
                robot_id=robot_id,
                request_id=request.request_id,
                observation_step=request.observation_step,
                action_start_step=request.action_start_step,
                execution_horizon=request.execution_horizon,
                request_timestamp=request.request_timestamp,
                server_arrival_time=request.arrival_timestamp,  # FIXME: make timestamp/arrival time naming convention consistent
            )
            self.robots[robot_id].add_request(record)

            self.start_time = min(self.start_time, request.request_timestamp)

    def record_response(
        self,
        robot_id: str,
        response: InferResponse,
        ack: ResponseAck,
    ) -> None:
        """Called when client sends ResponseAck."""
        with lock:
            request_record = self.robots[robot_id].get_request(ack.request_id)
            batch = next(b for b in reversed(self.batches) if ack.request_id in b.request_ids)
            self.robots[robot_id].add_response(
                ResponseRecord(
                    request=request_record,
                    batch_id=batch.batch_id,
                    execution_horizon=response.execution_horizon,
                    inference_start_time=batch.inference_start_time,
                    inference_end_time=batch.inference_end_time,
                    server_send_time=response.server_send_time,
                    receive_time=ack.receive_time,
                    execution_start_step=ack.execution_start_step,
                    first_executed_index=ack.first_executed_index,
                )
            )
            self.end_time = max(self.end_time, time.time())

    def record_episode_start(
        self,
        robot_id: str,
        episode_start: EpisodeStart,
    ) -> None:
        """Called when client streams an in-progress task step count."""
        with lock:
            if robot_id not in self.robots:
                self.robots[robot_id] = Robot(robot_id=robot_id, episodes=[])
            self.robots[robot_id].start_episode(episode_start)

    def record_episode_step(self, robot_id: str, timestamp: float) -> None:
        with lock:
            if robot_id in self.robots and self.robots[robot_id].episodes:
                self.robots[robot_id].add_step(timestamp)
                self.end_time = max(self.end_time, timestamp)

    def record_episode_end(
        self,
        robot_id: str,
        episode_end: EpisodeEnd,
    ) -> None:
        """Called when client streams an in-progress task step count."""
        with lock:
            self.robots[robot_id].end_episode(episode_end)
            self.end_time = max(self.end_time, time.time())

    def snapshot(self, window_s: float | None = None, *, sla_pct: float = 10.0) -> Snapshot:
        """Single source of truth for the dashboard. Computes all stats, SLA, and chart data."""
        from collections import deque

        with lock:
            start_timestamp = self.end_time - window_s if window_s is not None else self.start_time
            t0 = start_timestamp

            batches = window_filter(
                self.batches,
                lambda b: b.inference_end_time,
                (start_timestamp, self.end_time),
            )
            requests = list(
                itertools.chain.from_iterable(
                    robot.get_requests(start_timestamp, self.end_time) for robot in self.robots.values()
                )
            )
            responses = list(
                itertools.chain.from_iterable(
                    robot.get_responses(start_timestamp, self.end_time) for robot in self.robots.values()
                )
            )

            robot_actions_left = {}
            for robot_id, robot in self.robots.items():
                times, values = robot.get_actions_left_timed(start_timestamp, self.end_time)
                robot_actions_left[robot_id] = (
                    times - t0 if len(times) > 0 else times,
                    values,
                )

            completed_episodes: list[tuple[RobotID, Episode]] = [
                (robot_id, episode)
                for robot_id, robot in self.robots.items()
                for episode in robot.episodes
                if episode.success is not None
                and episode.requests
                and start_timestamp <= episode.requests[-1].request_timestamp < self.end_time
            ]

            # ---- per-robot stats ----
            responses_by_robot: dict[str, list[ResponseRecord]] = {}
            for resp in responses:
                responses_by_robot.setdefault(resp.request.robot_id, []).append(resp)

            eps_by_robot: dict[str, list[Episode]] = {}
            for robot_id, ep in completed_episodes:
                eps_by_robot.setdefault(robot_id, []).append(ep)

            duration_s = self.end_time - start_timestamp
            active_rates: list[float] = []
            per_robot: dict[str, dict] = {}

            for robot_id, (_times, alh) in robot_actions_left.items():
                valid = alh[~np.isnan(alh)]
                observed_steps = len(valid)
                starved_steps = int(np.sum(valid == 0))
                starvation_rate_pct = starved_steps / observed_steps * 100 if observed_steps > 0 else 0.0

                robot_eps = eps_by_robot.get(robot_id, [])
                tp = sum(1 for ep in robot_eps if ep.success) / duration_s if duration_s > 0 else 0.0

                robot_resps = responses_by_robot.get(robot_id, [])
                net_delays = [r.outbound_ms for r in robot_resps if r.receive_time > 0]
                avg_net_delay = float(np.mean(net_delays)) if net_delays else 0.0

                if observed_steps > 0:
                    active_rates.append(starvation_rate_pct)

                per_robot[robot_id] = {
                    "observed_steps": observed_steps,
                    "starved_steps": starved_steps,
                    "starvation_rate_pct": starvation_rate_pct,
                    "healthy": starvation_rate_pct <= sla_pct,
                    "tp_suc_per_sec_robot": tp,
                    "avg_network_delay_ms": avg_net_delay,
                }

            # ---- SLA capacity curve ----
            n_active = len(active_rates)
            sla_capacity_curve = [
                {
                    "sla_pct": float(thr),
                    "healthy_robot_count": (n_healthy := sum(1 for r in active_rates if r <= thr)),
                    "active_robot_count": n_active,
                    "healthy_robot_ratio_pct": n_healthy / n_active * 100 if n_active > 0 else 0.0,
                }
                for thr in range(101)
            ]

            # ---- healthy robots over time: rolling K-step window per robot ----
            # At each step (sorted by timestamp), starvation rate = zeros in last K values.
            k = 20
            _windows: dict[str, deque[float]] = {}
            all_steps: list[tuple[float, str, float]] = sorted(
                (
                    (ts, robot_id, al)
                    for robot_id, robot in self.robots.items()
                    for episode in robot.episodes
                    for ts, al in episode.get_windowed_steps(start_timestamp, self.end_time)
                ),
                key=lambda x: x[0],
            )
            healthy_robots_over_time = []
            for ts, robot_id, al in all_steps:
                w = _windows.setdefault(robot_id, deque(maxlen=k))
                w.append(al)
                active_count = len(_windows)
                healthy_count = sum(1 for w in _windows.values() if sum(v == 0 for v in w) / len(w) * 100 <= sla_pct)
                healthy_robots_over_time.append(
                    {
                        "t": round(ts - t0, 3),
                        "healthy_robot_count": healthy_count,
                        "active_robot_count": active_count,
                    }
                )

            # ---- batch history for charts ----
            plan_activation_abs = sorted(
                sample.recorded_at
                for sample in self.scheduler_decisions
                if sample.metric_name == "ilp_plan_activated" and sample.recorded_at <= self.end_time
            )
            kickoff_abs = sorted(
                sample.recorded_at
                for sample in self.scheduler_decisions
                if sample.metric_name == "ilp_replan_kickoff" and sample.recorded_at <= self.end_time
            )
            replan_markers = [
                {
                    "t": round(ts - t0, 3),
                    "plan_index": idx,
                }
                for idx, ts in enumerate(plan_activation_abs)
                if start_timestamp <= ts < self.end_time
            ]
            kickoff_markers = []
            kickoff_cursor = 0
            for plan_index, activation_ts in enumerate(plan_activation_abs):
                while kickoff_cursor + 1 < len(kickoff_abs) and kickoff_abs[kickoff_cursor + 1] <= activation_ts:
                    kickoff_cursor += 1
                if kickoff_cursor < len(kickoff_abs) and kickoff_abs[kickoff_cursor] <= activation_ts:
                    kickoff_ts = kickoff_abs[kickoff_cursor]
                    if start_timestamp <= kickoff_ts < self.end_time:
                        kickoff_markers.append(
                            {
                                "t": round(kickoff_ts - t0, 3),
                                "plan_index": plan_index,
                            }
                        )

            response_by_id: dict[int, ResponseRecord] = {
                resp.request.request_id: resp
                for robot in self.robots.values()
                for episode in robot.episodes
                for resp in episode.responses
            }
            batch_history = []
            for i, b in enumerate(batches):
                plan_index = None
                if plan_activation_abs:
                    idx = bisect.bisect_right(plan_activation_abs, b.inference_start_time) - 1
                    if idx >= 0:
                        plan_index = idx
                per_req = []
                for rid, req_id in zip(b.robot_ids, b.request_ids, strict=True):
                    resp = response_by_id.get(req_id)
                    if resp is not None:
                        inbound_ms = round(
                            (resp.request.server_arrival_time - resp.request.request_timestamp) * 1000,
                            2,
                        )
                        queue_ms = round(
                            (b.inference_start_time - resp.request.server_arrival_time) * 1000,
                            2,
                        )
                    else:
                        inbound_ms = queue_ms = 0.0
                    per_req.append(
                        {
                            "robot_id": rid,
                            "inbound_ms": inbound_ms,
                            "queue_ms": queue_ms,
                            "infer_ms": round(b.gpu_time_ms, 2),
                        }
                    )
                batch_history.append(
                    {
                        "t": round(b.inference_end_time - t0, 3),
                        "batch_size": len(b.robot_ids),
                        "gpu_time_ms": round(b.gpu_time_ms, 2),
                        "idle_before_ms": round(
                            (b.inference_start_time - batches[i - 1].inference_end_time) * 1000,
                            2,
                        )
                        if i > 0
                        else 0.0,
                        "inference_start_t": round(b.inference_start_time - t0, 3),
                        "inference_end_t": round(b.inference_end_time - t0, 3),
                        "robot_ids": b.robot_ids,
                        "plan_index": plan_index,
                        "per_request": per_req,
                    }
                )

            # ---- outbound delays ----
            outbound_delays_ms: dict[str, list[float]] = {}
            for robot_id, robot in self.robots.items():
                delays = [
                    round(resp.outbound_ms, 2)
                    for episode in robot.episodes
                    for resp in episode.responses
                    if resp.receive_time > 0 and resp.receive_time >= start_timestamp
                ]
                if delays:
                    outbound_delays_ms[robot_id] = delays

            # ---- scheduler timings + scheduling decisions ----
            scheduler_timing_ms: dict[str, list[float]] = {}
            scheduling_decisions: list[dict] = []
            for sample in self.scheduler_decisions:
                if sample.metric_name == "batch_scheduled":
                    scheduling_decisions.append(
                        {
                            "t": round(sample.recorded_at - t0, 3),
                            "candidates": sample.candidates,
                            "scheduled": sample.scheduled,
                        }
                    )
                else:
                    scheduler_timing_ms.setdefault(f"{sample.scheduler_name}.{sample.metric_name}", []).append(
                        round(sample.duration_ms, 3)
                    )

            # ---- task events (completed episodes in window) ----
            task_events = []
            for robot_id, robot in self.robots.items():
                for ep_idx, episode in enumerate(robot.episodes):
                    if episode.success is None or not episode.requests:
                        continue
                    event_time = episode.requests[-1].request_timestamp
                    if event_time < start_timestamp:
                        continue
                    task_events.append(
                        {
                            "t": round(event_time - t0, 3),
                            "robot_id": robot_id,
                            "taskkey": f"{episode.task_suite_name}/{episode.task_id}",
                            "task_suite_name": episode.task_suite_name,
                            "task_id": episode.task_id,
                            "task_language": episode.task_language,
                            "episode_idx": ep_idx,
                            "success": episode.success,
                            "duration_s": round(
                                episode.requests[-1].request_timestamp - episode.start_time,
                                3,
                            ),
                            "steps_taken": episode.num_steps,
                            "total_episodes": len(robot.episodes),
                            "max_episode_steps": episode.max_episode_steps,
                            "max_duration_s": 0.0,
                        }
                    )

            # ---- task progress (in-progress episodes) ----
            task_progress = []
            for robot_id, robot in self.robots.items():
                if not robot.episodes:
                    continue
                episode = robot.episodes[-1]
                if episode.success is not None or not episode.requests:
                    continue
                update_time = episode.requests[-1].request_timestamp
                if update_time < start_timestamp:
                    continue
                ep_idx = len(robot.episodes) - 1
                task_progress.append(
                    {
                        "t": round(update_time - t0, 3),
                        "robot_id": robot_id,
                        "taskkey": f"{episode.task_suite_name}/{episode.task_id}",
                        "task_suite_name": episode.task_suite_name,
                        "task_id": episode.task_id,
                        "task_language": episode.task_language,
                        "episode_idx": ep_idx,
                        "current_step": episode.num_steps,
                        "max_episode_steps": episode.max_episode_steps,
                        "total_episodes": len(robot.episodes),
                    }
                )

            return Snapshot(
                start_time=t0,
                end_time=self.end_time if self.end_time > t0 else time.time(),
                sla_pct=sla_pct,
                robot_actions_left=robot_actions_left,
                per_robot=per_robot,
                requests=requests,
                responses=responses,
                batches=batches,
                completed_episodes=completed_episodes,
                sla_capacity_curve=sla_capacity_curve,
                healthy_robots_over_time=healthy_robots_over_time,
                batch_history=batch_history,
                outbound_delays_ms=outbound_delays_ms,
                scheduler_timing_ms=scheduler_timing_ms,
                replan_markers=replan_markers,
                kickoff_markers=kickoff_markers,
                task_events=task_events,
                task_progress=task_progress,
                scheduling_decisions=scheduling_decisions,
            )

    def reset(self) -> None:
        """Clear all accumulated metrics and reset counters."""
        with lock:
            self.batches.clear()
            self.scheduler_decisions.clear()
            self.robots.clear()
