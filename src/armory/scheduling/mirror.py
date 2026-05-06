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

import logging
from dataclasses import dataclass, replace

from armory.scheduling.latency import LatencyTracker
from armory.serving.schemas import (
    AckNotification,
    CompletionNotification,
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


@dataclass(frozen=True)
class ActionChunk:
    request_id: int  # identity that flows through engine completion + robot ack
    observation_step: int  # step when observation was captured
    arrival_time: float  # estimated/actual time the chunk lands on the robot
    action_index_start: int  # action index of the first action in the chunk
    execution_horizon: int
    arrived: bool = False


@dataclass(frozen=True)
class Checkpoint:
    """Snapshot of a Mirror's append-only state. Restore truncates lists back."""

    lengths: dict[RobotID, tuple[int, int]]  # (n_steps, n_chunks) per robot


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
        assert not self.chunks or self.chunks[-1].observation_step < chunk.observation_step
        self.chunks.append(chunk)

    def confirm_chunk(self, ack: AckNotification) -> None:
        for i, chunk in enumerate(self.chunks):
            if chunk.request_id == ack.request_id:
                self.chunks[i] = replace(chunk, arrival_time=ack.receive_time, arrived=True)
                return

    def update_completion(self, request_id: int, refined_arrival_time: float) -> None:
        for i, chunk in enumerate(self.chunks):
            if chunk.request_id == request_id:
                self.chunks[i] = replace(chunk, arrival_time=refined_arrival_time)
                return

    @property
    def max_arrived_action_step(self) -> int:
        """Last action index available from any arrived chunk, or -1 if none."""
        for chunk in reversed(self.chunks):
            if chunk.arrived:
                return chunk.action_index_start + chunk.execution_horizon - 1
        return -1

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


class Mirror:
    def __init__(self, latency_tracker: LatencyTracker | None = None):
        self.robots: dict[RobotID, Robot] = {}
        # Optional so existing tests can construct a Mirror without a tracker;
        # methods that need it (get_chunks, update_completion) assert non-None.
        self.latency_tracker = latency_tracker

    def reset_robot(self, robot_id: RobotID) -> None:
        self.robots.pop(robot_id, None)

    def receive_request(self, request: SlotRequest, control_hz: float) -> None:
        if request.robot_id not in self.robots:
            # NOTE: for now, assume control_hz and execution_horizon are fixed for a robot's lifetime
            self.robots[request.robot_id] = Robot(control_hz, request.execution_horizon)
        self.robots[request.robot_id].step(request)

    def get_chunks(
        self,
        batch: list[RobotID],
        request_ids: list[int],
        dispatch_time: float,
    ) -> list[ActionChunk]:
        """Build (without queueing) anticipated chunks for a batch dispatched at ``dispatch_time``.

        Caller is expected to follow up with ``fast_forward(post_dispatch_time, batch, chunks)``.
        """
        assert self.latency_tracker is not None
        infer_lat = self.latency_tracker.infer_latency(len(batch))
        chunks: list[ActionChunk] = []
        for rid, req_id in zip(batch, request_ids, strict=True):
            robot = self.robots[rid]
            obs_cutoff = dispatch_time - self.latency_tracker.observation_latency(rid)
            cs = robot.get_latest_control_step_before(obs_cutoff)
            assert cs is not None, f"robot {rid} has no control step before {obs_cutoff}"
            chunks.append(
                ActionChunk(
                    request_id=req_id,
                    observation_step=cs.observation_step,
                    arrival_time=dispatch_time
                    + infer_lat
                    + self.latency_tracker.action_latency(rid),
                    action_index_start=robot.next_chunk_start(cs),
                    execution_horizon=robot.execution_horizon,
                    arrived=False,
                )
            )
        return chunks

    def fast_forward(
        self,
        time: float,
        robot_ids: list[RobotID],
        chunks: list[ActionChunk],
    ) -> None:
        """Queue chunks for ``robot_ids`` then advance every robot's clock to ``time``."""
        for rid, chunk in zip(robot_ids, chunks, strict=True):
            self.robots[rid].queue_chunk(chunk)
        for robot in self.robots.values():
            robot.step_forward(time)

    def update_completion(self, notification: CompletionNotification, now: float) -> None:
        """Refine a chunk's arrival_time once GPU inference has completed."""
        assert self.latency_tracker is not None
        robot = self.robots.get(notification.robot_id)
        if robot is None:
            logger.debug("Ignoring completion for unknown robot: %s", notification.robot_id)
            return
        refined = now + self.latency_tracker.action_latency(notification.robot_id)
        robot.update_completion(notification.request_id, refined)

    def confirm_chunk(self, ack: AckNotification) -> None:
        robot = self.robots.get(ack.robot_id)
        if robot is None:
            logger.debug("Ignoring ack for unknown robot: %s", ack.robot_id)
            return
        robot.confirm_chunk(ack)

    def checkpoint(self) -> Checkpoint:
        return Checkpoint(
            lengths={rid: (len(r.steps), len(r.chunks)) for rid, r in self.robots.items()},
        )

    def restore(self, ckpt: Checkpoint) -> None:
        for rid in list(self.robots.keys()):
            if rid not in ckpt.lengths:
                # Robot added after the checkpoint; drop it.
                del self.robots[rid]
                continue
            n_steps, n_chunks = ckpt.lengths[rid]
            r = self.robots[rid]
            del r.steps[n_steps:]
            del r.chunks[n_chunks:]

    def deadlines(self) -> dict[RobotID, float]:
        return {rid: robot.deadline() for rid, robot in self.robots.items()}
