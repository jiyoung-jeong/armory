"""
Mirror of robot state on the scheduler side.

Two parallel sequences track a robot's progress:

- Control steps are the robot's discrete clock ticks at ``control_hz``. Each
  ``ControlStep`` records the wall time of the tick, the observation captured
  at it (``observation_step``, monotonically increasing), and which action
  index — if any — was executed at that tick (``action_step``).
- Action indexes are positions in the global, monotonically increasing
  sequence of actions produced by inference. Each ``ActionChunk`` covers
  ``[action_start_step, action_start_step + execution_horizon)``.

The two sequences are decoupled: a control step may execute no action (when
the next action index is not yet available on the robot), and a single chunk
spans many control steps. ``next_action_step`` on a control step is the
action index the robot will try to execute on its next tick.
"""

from __future__ import annotations

import logging
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
    action_start_step: int  # action index of the first action in the chunk
    execution_horizon: int
    arrived: bool = False


class Robot:
    """Mirror of a single robot's control steps and action chunks.

    Invariant: once constructed, callers must seed the robot with an initial
    control step (``observation_step=0``, ``action_start_step=0``) via
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
            assert request.action_start_step == 0
            control_step = ControlStep(
                time=request.request_timestamp,
                observation_step=request.observation_step,
                action_step=None,
                next_action_step=0,
            )
        else:
            executed_action_on_step = request.action_start_step == self.steps[-1].next_action_step
            control_step = ControlStep(
                time=request.request_timestamp,
                observation_step=request.observation_step,
                action_step=request.action_start_step if executed_action_on_step else None,
                next_action_step=request.action_start_step + 1
                if executed_action_on_step
                else request.action_start_step,
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

    def send_response(self, chunk: ActionChunk) -> None:
        assert not self.chunks or self.chunks[-1].observation_step < chunk.observation_step
        self.chunks.append(chunk)

    def receive_response(self, ack: AckNotification) -> None:
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
                return chunk.action_start_step + chunk.execution_horizon - 1
        return -1

    @property
    def max_overall_action_step(self) -> int:
        return self.chunks[-1].action_start_step + self.chunks[-1].execution_horizon - 1

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
                chunk.action_start_step
                <= action_step
                <= chunk.action_start_step + chunk.execution_horizon - 1
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

    def reset_robot(self, RobotID: str) -> None:
        self.robots.pop(RobotID, None)

    def receive_request(self, request: SlotRequest, control_hz: float) -> None:
        if request.RobotID not in self.robots:
            # NOTE: for now, assume control_hz and execution_horizon are fixed for a robot's lifetime
            self.robots[request.RobotID] = Robot(control_hz, request.execution_horizon)
        self.robots[request.RobotID].step(request)

    def schedule_pending_chunk(self, RobotID: str, chunk: ActionChunk) -> None:
        self.robots[RobotID].send_response(chunk)

    def receive_response(self, ack: AckNotification) -> None:
        robot = self.robots.get(ack.RobotID)
        if robot is None:
            logger.debug("Ignoring ack for unknown robot: %s", ack.RobotID)
            return
        robot.receive_response(ack)

    def get_chunks(
        self, RobotIDs: list[RobotID], latency_tracker: LatencyTracker, time: float
    ) -> list[ActionChunk]:
        """
        Returns a list of ActionChunks that would be queued if we started an inference for the RobotIDs at the given time.
        """
        control_steps = []
        for rid in RobotIDs:
            obs_time = time - latency_tracker.observation_latency(rid)
            control_steps.append(self.robots[rid].get_latest_control_step_before(obs_time))

        inference_latency = latency_tracker.infer_latency(len(RobotIDs))
        chunks = []
        for control_step, rid in zip(control_steps, RobotIDs):
            robot = self.robots[rid]
            if robot.chunks:
                observation_step = max(
                    control_step.observation_step,
                    robot.chunks[-1].observation_step + 1,
                )
                action_start_step = max(
                    control_step.next_action_step,
                    robot.max_overall_action_step + 1,
                )
            else:
                observation_step = control_step.observation_step
                action_start_step = control_step.next_action_step
            chunks.append(
                ActionChunk(
                    observation_step=observation_step,
                    arrival_time=time + inference_latency + latency_tracker.action_latency(rid),
                    action_start_step=action_start_step,
                    execution_horizon=robot.execution_horizon,
                    arrived=True,
                )
            )
        return chunks

    def fast_forward(
        self,
        time: float,
        RobotIDs: list[RobotID],
        chunks: list[ActionChunk],
    ) -> None:
        """Simulates time forward to the given time, sending responses and advancing robot steps."""
        # NOTE: we send responses here so they are available while stepping
        # it doesn't matter thaot they are "sent" before the actual sending time
        # because arrival_time handles the timing around this
        for rid, chunk in zip(RobotIDs, chunks):
            self.robots[rid].send_response(chunk)

        for robot in self.robots.values():
            robot.step_forward(time)

    def deadlines(self) -> dict[RobotID, float]:
        return {rid: robot.deadline() for rid, robot in self.robots.items()}

    def anticipate_request(
        self,
        base: SlotRequest,
        chunk: ActionChunk,
        dispatch_time: float,
        request_id: int,
    ) -> SlotRequest:
        """Project ``base`` onto a planned future chunk.

        Returns a synthetic SlotRequest with corrected observation_step,
        action_start_step, and timestamps so the GPU treats it as a fresh
        request riding on whatever observation bytes are in ``base.slot_index``.
        """
        return replace(
            base,
            request_id=request_id,
            observation_step=chunk.observation_step,
            action_start_step=chunk.action_start_step,
            request_timestamp=dispatch_time,
            arrival_timestamp=dispatch_time,
            deadline=dispatch_time + chunk.execution_horizon / base.control_hz,
        )
