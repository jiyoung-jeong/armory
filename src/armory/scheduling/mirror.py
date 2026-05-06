"""
Mirror of robot state on the scheduler side.

Two parallel sequences track a robot's progress:

- Control steps are the robot's discrete clock ticks at ``control_hz``. Each
  ``ControlStep`` records the wall time of the tick, the observation captured
  at it (``observation_step``, monotonically increasing), and which action
  index — if any — was executed at that tick (``action_step``).
- Action indexes are positions in the global, monotonically increasing
  sequence of actions produced by inference. Each ``ActionChunk`` covers
  ``[action_index_start, action_index_start + execution_horizon)``.

The two sequences are decoupled: a control step may execute no action (when
the next action index is not yet available on the robot), and a single chunk
spans many control steps. ``next_action_step`` on a control step is the
action index the robot will try to execute on its next tick.

Search and production both mutate Mirror through the same primitives:

- ``get_chunks`` builds (without queueing) anticipated chunks for a batch
  dispatched at ``dispatch_time``.
- ``fast_forward(time, robot_ids, chunks)`` queues those chunks and advances
  every robot's clock to ``time``.

Because both ``Robot.steps`` and ``Robot.chunks`` are append-only during
simulation, search uses ``checkpoint`` / ``restore`` to revert mutations
instead of deep-copying the mirror per DFS frame.
"""

from __future__ import annotations

import itertools
import logging
import time
from collections import deque
from dataclasses import dataclass, replace

from armory.scheduling.latency import LatencyTracker
from armory.serving.schemas import (
    AckNotification,
    ActionChunk,
    ResponseBatch,
    RobotID,
    SlotRequest,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


@dataclass
class ControlStep:
    time: float
    observation_step: int
    action_step: int | None  # which action index was executed at this step
    next_action_step: int


# TODO: another source of inconsistency is that we do not directly verify chunks against control steps on acks, we may need to just track which observation step it arrived before
class Robot:
    """Mirror of a single robot's control steps and action chunks.

    Invariant: once constructed, callers must seed the robot with an initial
    control step (``observation_step=0``, ``action_index_start=0``) via
    ``step()`` before invoking any other method. ``Mirror.receive_request``
    enforces this by calling ``step()`` immediately after construction.
    """

    def __init__(self, control_hz: float, execution_horizon: int):
        self.control_hz = control_hz
        self.execution_horizon = execution_horizon

        # Both lists are sorted increasing by time by assertion.
        self.steps: list[ControlStep] = []
        # Includes chunks that are in-transit.
        self.chunks: list[ActionChunk] = []

    def step(self, request: SlotRequest) -> None:
        if not self.steps:
            assert request.action_index_start == 0
            control_step = ControlStep(
                time=request.request_timestamp,
                observation_step=request.observation_step,
                action_step=None,
                next_action_step=0,
            )
        else:
            executed_action_on_step = request.action_index_start == self.steps[-1].next_action_step
            control_step = ControlStep(
                time=request.request_timestamp,
                observation_step=request.observation_step,
                action_step=request.action_index_start if executed_action_on_step else None,
                next_action_step=request.action_index_start + 1
                if executed_action_on_step
                else request.action_index_start,
            )
            assert control_step.time > self.steps[-1].time
            assert control_step.observation_step == self.steps[-1].observation_step + 1
            assert (
                control_step.action_step is None
                or control_step.action_step == self.steps[-1].next_action_step
            ), (
                f"action_step {control_step.action_step} is not the next action step {self.steps[-1].next_action_step}, steps: {self.steps}"
            )
            assert (
                control_step.action_step is None
                or control_step.next_action_step == control_step.action_step + 1
            )

        self.steps.append(control_step)

    def queue_chunk(self, chunk: ActionChunk) -> None:
        self.chunks.append(chunk)

    def update_chunk_arrival_time(
        self, chunk_id: int, refined_arrival_time: float, *, arrived: bool = False
    ) -> None:
        for i, chunk in enumerate(self.chunks):
            if chunk.chunk_id == chunk_id:
                self.chunks[i] = replace(chunk, arrival_time=refined_arrival_time, arrived=arrived)
                return

    @property
    def max_overall_action_step(self) -> int:
        return self.chunks[-1].action_index_start + self.chunks[-1].execution_horizon - 1

    def get_latest_control_step_before(self, time: float) -> ControlStep | None:
        for step in reversed(self.steps):
            if step.time < time:
                return step
        return None

    def actions_executed(self) -> int:
        steps = [s for s in self.steps if s.action_step is not None]
        if not steps:
            return 0
        return steps[-1].action_step - steps[0].action_step + 1

    def action_is_available(self, action_step: int, time: float) -> bool:
        for chunk in self.chunks:
            if (
                chunk.action_index_start
                <= action_step
                <= chunk.action_index_start + chunk.execution_horizon - 1
            ) and chunk.arrival_time <= time:
                return True
        return False

    def advance_step(self, prev_step: ControlStep) -> ControlStep:
        next_time = prev_step.time + 1 / self.control_hz
        action_step = (
            prev_step.next_action_step
            if self.action_is_available(prev_step.next_action_step, next_time)
            else None
        )
        return ControlStep(
            time=next_time,
            observation_step=prev_step.observation_step + 1,
            action_step=action_step,
            next_action_step=action_step + 1
            if action_step is not None
            else prev_step.next_action_step,
        )

    def deadline(self) -> float:
        if not self.chunks:
            return self.steps[-1].time

        step = self.steps[-1]

        while step.next_action_step <= self.max_overall_action_step:
            step = self.advance_step(step)

        return step.time

    def step_forward(self, time: float) -> None:
        while self.steps[-1].time < time:
            self.steps.append(self.advance_step(self.steps[-1]))

    def next_chunk_start(self, cs: ControlStep) -> int:
        """Action index the next queued chunk should start at, given control step ``cs``."""
        return cs.action_step if cs.action_step is not None else cs.next_action_step

    def remove_chunk(self, chunk_id: int) -> None:
        for i, chunk in enumerate(self.chunks):
            if chunk.chunk_id == chunk_id:
                del self.chunks[i]
                return
        raise KeyError(f"chunk_id {chunk_id} not found")


@dataclass
class Batch:
    batch_id: int
    robot_ids: list[RobotID]
    chunk_ids: list[int]

    @property
    def size(self) -> int:
        return len(self.robot_ids)


@dataclass(frozen=True)
class Checkpoint:
    """Snapshot of a Mirror's append-only state. Restore truncates lists back."""

    lengths: dict[RobotID, tuple[int, int]]  # (n_steps, n_chunks) per robot
    queued_batches: int


class Mirror:
    def __init__(self, latency_tracker: LatencyTracker | None = None):
        self.robots: dict[RobotID, Robot] = {}
        # Optional so existing tests can construct a Mirror without a tracker;
        # methods that need it (get_chunks, update_completion) assert non-None.
        self.latency_tracker = latency_tracker
        self.in_flight_batches: deque[Batch] = deque()
        self.last_batch_completed_time: float = 0.0
        self.chunk_id_counter = itertools.count(1)

    @property
    def in_flight_batches_count(self) -> int:
        return len(self.in_flight_batches)

    def reset_robot(self, robot_id: RobotID) -> None:
        self.robots.pop(robot_id, None)

    def receive_request(self, request: SlotRequest, control_hz: float) -> None:
        if request.robot_id not in self.robots:
            # NOTE: for now, assume control_hz and execution_horizon are fixed for a robot's lifetime
            self.robots[request.robot_id] = Robot(control_hz, request.execution_horizon)
        self.robots[request.robot_id].step(request)

    def _next_chunk_context(self, rid: RobotID, dispatch_time: float) -> tuple[ControlStep, int]:
        robot = self.robots[rid]
        obs_cutoff = dispatch_time - self.latency_tracker.observation_latency(rid)
        cs = robot.get_latest_control_step_before(obs_cutoff)
        assert cs is not None, f"robot {rid} has no control step before {obs_cutoff}"
        return cs, robot.next_chunk_start(cs)

    def queue_batch(self, batch: list[RobotID], batch_id: int) -> list[ActionChunk]:
        assert self.latency_tracker is not None
        dispatch_time = self.next_time_server_available()
        infer_lat = self.latency_tracker.infer_latency(len(batch))
        chunks: list[ActionChunk] = []
        for rid in batch:
            robot = self.robots[rid]
            cs, action_index_start = self._next_chunk_context(rid, dispatch_time)
            chunk = ActionChunk(
                chunk_id=next(self.chunk_id_counter),
                observation_step=cs.observation_step,
                arrival_time=dispatch_time + infer_lat + self.latency_tracker.action_latency(rid),
                action_index_start=action_index_start,
                execution_horizon=robot.execution_horizon,
            )
            chunks.append(chunk)
            robot.queue_chunk(chunk)
        self.in_flight_batches.append(
            Batch(batch_id=batch_id, robot_ids=batch, chunk_ids=[c.chunk_id for c in chunks])
        )
        return chunks

    def update_batch_completion(self, batch: ResponseBatch) -> None:
        """Refine each chunk's arrival_time once GPU inference has completed."""
        assert self.latency_tracker is not None
        in_flight = self.in_flight_batches.popleft()
        assert in_flight.batch_id == batch.batch_id

        actual_completion = batch.inference_start_time + batch.inference_duration
        self.last_batch_completed_time = actual_completion

        served_chunk_ids = set([response.chunk_id for response in batch.responses])
        for robot_id, chunk_id in zip(in_flight.robot_ids, in_flight.chunk_ids):
            robot = self.robots.get(robot_id)
            if robot is None:
                logger.debug("Ignoring completion for unknown robot: %s", robot_id)
                continue
            if chunk_id not in served_chunk_ids:
                robot.remove_chunk(chunk_id)
            else:
                robot.update_chunk_arrival_time(
                    chunk_id, actual_completion + self.latency_tracker.action_latency(robot_id)
                )

    def confirm_chunk(self, ack: AckNotification) -> None:
        robot = self.robots.get(ack.robot_id)
        if robot is None:
            logger.debug("Ignoring ack for unknown robot: %s", ack.robot_id)
            return
        robot.update_chunk_arrival_time(ack.chunk_id, ack.receive_time, arrived=True)

    def next_time_server_available(self) -> float:
        if not self.in_flight_batches:
            return time.time()
        assert self.last_batch_completed_time is not None
        return self.last_batch_completed_time + sum(
            self.latency_tracker.infer_latency(b.size) for b in self.in_flight_batches
        )

    def schedulable_requests(self, requests: dict[RobotID, SlotRequest]) -> list[SlotRequest]:
        schedulable_requests: list[SlotRequest] = []

        dispatch_time = self.next_time_server_available()
        for robot_id, request in requests.items():
            robot = self.robots[robot_id]
            _, action_index_start = self._next_chunk_context(robot_id, dispatch_time)
            if len(robot.chunks) == 0 or action_index_start > robot.chunks[-1].action_index_start:
                schedulable_requests.append(request)
            # else:
            #     logger.debug("Request %s is not schedulable", request.robot_id)
            #     logger.debug("Action index start: %d", action_index_start)
            #     logger.debug("Request action index start: %d", request.action_index_start)
            #     logger.debug("Dispatch time: %f", dispatch_time)
            #     logger.debug("Request: %s", request)
            #     logger.debug("Robot: %s", robot_id)
            #     logger.debug("Robot steps: %s", self.robots[robot_id].steps)
            #     logger.debug("Robot chunks: %s", self.robots[robot_id].chunks)
        return schedulable_requests

    # below are methods only used by search
    def fast_forward(
        self,
        time: float,
    ) -> None:
        """Advance every robot's clock to ``time``."""
        for robot in self.robots.values():
            robot.step_forward(time)

    def checkpoint(self) -> Checkpoint:
        return Checkpoint(
            lengths={rid: (len(r.steps), len(r.chunks)) for rid, r in self.robots.items()},
            queued_batches=len(self.in_flight_batches),
        )

    def restore(self, ckpt: Checkpoint) -> None:
        for rid in list(self.robots.keys()):
            if rid not in ckpt.lengths:
                del self.robots[rid]
                continue
            n_steps, n_chunks = ckpt.lengths[rid]
            r = self.robots[rid]
            del r.steps[n_steps:]
            del r.chunks[n_chunks:]
        while len(self.in_flight_batches) > ckpt.queued_batches:
            self.in_flight_batches.pop()

    def deadlines(self) -> dict[RobotID, float]:
        return {rid: robot.deadline() for rid, robot in self.robots.items()}
