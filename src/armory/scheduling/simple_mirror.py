"""Simplified mirror for testing baseline schedulers.

The full ``Mirror`` in ``mirror.py`` tracks per-step observation state,
simulates control ticks, and supports search-side checkpoint/restore.
``SimpleMirror`` strips that bookkeeping out and tracks only what the
baseline schedulers in ``baselines.py`` actually consult:

- one ``deadline`` per robot — the wall time at which the robot is predicted
  to run out of queued actions,
- a queue of in-flight inference batches, used by ``in_flight_batches_count``
  and ``next_time_server_available``.

Each ``queue_batch`` advances every batched robot's deadline by one chunk
length (``max_execution_horizon / control_hz``), starting from
``max(current_deadline, chunk_arrival_time)``. ``update_batch_completion``
pops the head batch and decrements the per-robot in-flight counter;
``confirm_chunk`` is a no-op (acks don't influence deadline estimates here).

Search-side methods (``checkpoint``, ``restore``, ``fast_forward``) are not
implemented; ``SimpleMirror`` is intended for non-search baselines only.
"""

from __future__ import annotations

import itertools
import logging
import time
from collections import deque
from dataclasses import dataclass

from armory.scheduling.latency import LatencyTracker
from armory.serving.schemas import (
    AckNotification,
    ActionChunk,
    ResponseBatch,
    RobotID,
    SlotRequest,
)

logger = logging.getLogger(__name__)


@dataclass
class _Robot:
    control_hz: float
    max_execution_horizon: int
    deadline: float
    pending_chunks: int = 0


@dataclass
class _Batch:
    batch_id: int
    robot_ids: list[RobotID]
    chunk_ids: list[int]
    completion_time: float


class SimpleMirror:
    def __init__(self, latency_tracker: LatencyTracker | None = None):
        self.robots: dict[RobotID, _Robot] = {}
        self.latency_tracker = latency_tracker
        self.in_flight_batches: deque[_Batch] = deque()
        self.last_batch_completed_time: float = 0.0
        self.chunk_id_counter = itertools.count(1)

    @property
    def in_flight_batches_count(self) -> int:
        return len(self.in_flight_batches)

    def reset_robot(self, robot_id: RobotID) -> None:
        self.robots.pop(robot_id, None)

    def clear_all(self) -> None:
        self.robots.clear()
        self.in_flight_batches.clear()
        self.last_batch_completed_time = 0.0
        self.chunk_id_counter = itertools.count(1)

    def receive_request(self, request: SlotRequest) -> bool:
        robot = self.robots.get(request.robot_id)
        if robot is None:
            self.robots[request.robot_id] = _Robot(
                control_hz=request.control_hz,
                max_execution_horizon=request.max_execution_horizon,
                deadline=request.request_timestamp,
            )
        else:
            robot.max_execution_horizon = request.max_execution_horizon
        return True

    def next_time_server_available(self) -> float:
        if not self.in_flight_batches:
            return max(time.time(), self.last_batch_completed_time)
        return self.in_flight_batches[-1].completion_time

    def deadlines(self) -> dict[RobotID, float]:
        return {rid: r.deadline for rid, r in self.robots.items()}

    def schedulable_robot_ids(
        self,
    ) -> list[RobotID]:
        return [rid for rid in self.robots.keys() if self.robots[rid].pending_chunks == 0]

    def queue_batch(
        self,
        batch: list[RobotID],
        batch_id: int,
        *,
        origin: str = "queued",
        chunk_ids: list[int] | None = None,
    ) -> list[ActionChunk]:
        assert self.latency_tracker is not None
        dispatch_time = self.next_time_server_available()
        infer_lat = self.latency_tracker.infer_latency(len(batch))
        completion_time = dispatch_time + infer_lat

        chunks: list[ActionChunk] = []
        for i, rid in enumerate(batch):
            robot = self.robots[rid]
            chunk_id = chunk_ids[i] if chunk_ids is not None else next(self.chunk_id_counter)
            arrival = completion_time + self.latency_tracker.action_latency(rid)
            chunk = ActionChunk(
                chunk_id=chunk_id,
                observation_step=0,
                action_index_start=0,
                min_execution_horizon=robot.max_execution_horizon,
                max_execution_horizon=robot.max_execution_horizon,
                arrival_time=arrival,
                origin=origin,
            )
            chunks.append(chunk)
            robot.pending_chunks += 1
            chunk_duration = robot.max_execution_horizon / robot.control_hz
            robot.deadline = max(robot.deadline, arrival) + chunk_duration

        self.in_flight_batches.append(
            _Batch(
                batch_id=batch_id,
                robot_ids=list(batch),
                chunk_ids=[c.chunk_id for c in chunks],
                completion_time=completion_time,
            )
        )
        return chunks

    def update_batch_completion(self, batch: ResponseBatch) -> None:
        if not self.in_flight_batches or self.in_flight_batches[0].batch_id != batch.batch_id:
            logger.debug(
                "SimpleMirror ignoring stale ResponseBatch %s (head=%s)",
                batch.batch_id,
                self.in_flight_batches[0].batch_id if self.in_flight_batches else None,
            )
            return
        in_flight = self.in_flight_batches.popleft()
        self.last_batch_completed_time = batch.inference_start_time + batch.inference_duration
        for rid in in_flight.robot_ids:
            robot = self.robots.get(rid)
            if robot is not None:
                robot.pending_chunks = max(0, robot.pending_chunks - 1)

    def confirm_chunk(self, ack: AckNotification) -> None:
        return

    def to_dict(self) -> dict:
        now = time.time()
        return {
            "robots": {
                rid: {
                    "control_hz": r.control_hz,
                    "max_execution_horizon": r.max_execution_horizon,
                    "deadline_rel": r.deadline - now,
                    "pending_chunks": r.pending_chunks,
                }
                for rid, r in sorted(self.robots.items())
            },
            "in_flight_batches": [
                {
                    "batch_id": b.batch_id,
                    "robot_ids": b.robot_ids,
                    "chunk_ids": b.chunk_ids,
                    "completion_time_rel": b.completion_time - now,
                }
                for b in self.in_flight_batches
            ],
            "last_batch_completed_time_rel": (
                self.last_batch_completed_time - now if self.last_batch_completed_time > 0 else None
            ),
        }
