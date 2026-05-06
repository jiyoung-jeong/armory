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
"""

from __future__ import annotations

import copy as deepcopy
import logging
import time
from dataclasses import dataclass, replace

from armory.scheduling.latency import LatencyTracker
from armory.serving.schemas import AckNotification, RobotID, SlotRequest

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


@dataclass
class ControlStep:
    time: float
    observation_step: int
    action_step: int | None  # which action index was executed at this step
    next_action_step: int


# TODO: will improve speed later, focus on correctness for now
@dataclass(frozen=True)
class ActionChunk:
    observation_step: int  # step when observation was captured
    arrival_time: float  # time when the chunk becomes available on the robot
    action_index_start: int  # action index of the first action in the chunk
    execution_horizon: int
    arrived: bool = False


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
        """Updates a chunk's actual arrival time."""
        for i, chunk in enumerate(self.chunks):
            if chunk.observation_step == ack.observation_step:
                self.chunks[i] = replace(chunk, arrival_time=ack.receive_time, arrived=True)
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


class Mirror:
    def __init__(self):
        self.robots: dict[RobotID, Robot] = {}
        self.next_time_server_is_available: float = 0

    def reset_robot(self, RobotID: str) -> None:
        self.robots.pop(RobotID, None)

    def receive_request(self, request: SlotRequest, control_hz: float) -> None:
        if request.RobotID not in self.robots:
            # NOTE: for now, assume control_hz and execution_horizon are fixed for a robot's lifetime
            self.robots[request.RobotID] = Robot(control_hz, request.execution_horizon)
        self.robots[request.RobotID].step(request)

    def queue_batch(self, batch: list[RobotID], latency_tracker: LatencyTracker) -> None:
        """Estimate the chunks that will be queued if we start an inference for the given batch at the given time."""
        # FIXME: really shouldn't use this copy pattern
        future = deepcopy.copy(self).fast_forward(self.next_time_server_is_available)

        for request in batch:
            # TODO: maybe we can hide this logic inside Robot, or maybe we have to keep it here since we pass latency_tracker
            control_step = future.robots[request.RobotID].get_latest_control_step_before(
                self.next_time_server_is_available
                - latency_tracker.observation_latency(request.RobotID)
            )

            # TODO: might need an ID on this, can incrementally update predictions based on gpu completion, and then ack
            self.robots[request.RobotID].queue_chunk(
                ActionChunk(
                    observation_step=control_step.observation_step,
                    arrival_time=self.next_time_server_is_available
                    + latency_tracker.infer_latency(len(batch))
                    + latency_tracker.action_latency(request.RobotID),
                    action_index_start=control_step.action_step
                    or control_step.next_action_step,  # FIXME: this is too ugly
                    execution_horizon=request.execution_horizon,
                    arrived=False,
                )
            )

        # TODO: read this line carefully
        self.next_time_server_is_available = max(
            self.next_time_server_is_available, time.time()
        ) + latency_tracker.infer_latency(len(batch))

    def confirm_chunk(self, ack: AckNotification) -> None:
        robot = self.robots.get(ack.RobotID)
        if robot is None:
            logger.debug("Ignoring ack for unknown robot: %s", ack.RobotID)
            return
        robot.confirm_chunk(ack)

    def fast_forward(
        self,
        time: float,
    ) -> None:
        """Simulates time forward to the given time"""
        for robot in self.robots.values():
            robot.step_forward(time)

    def deadlines(self) -> dict[RobotID, float]:
        return {rid: robot.deadline() for rid, robot in self.robots.items()}
