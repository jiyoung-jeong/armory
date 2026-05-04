"""
TODO: docs

action indexes vs. control steps
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from typing import TypeAlias

from armory.scheduling.latency import LatencyTracker
from armory.serving.schemas import AckNotification, SlotRequest

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


robot_id: TypeAlias = str


@dataclass
class ControlStep:
    time: float
    observation_step: int
    action_step: int | None
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
    # TODO: a robot will always have at least one control step, need to make this clear
    def __init__(self, control_hz: float, execution_horizon: int):
        self.control_hz = control_hz
        self.execution_horizon = execution_horizon

        # Both lists will be sorted increasing by time by assertion
        self.steps: list[ControlStep] = []
        # includes chunks that are in-transit
        self.chunks: list[ActionChunk] = []

    def step(self, control_step: ControlStep) -> None:
        # TODO: pass more info from client and directly assert/test here
        assert not self.steps or self.steps[-1].time < control_step.time
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
        self.robots: dict[robot_id, Robot] = {}

    def reset_robot(self, robot_id: str) -> None:
        self.robots.pop(robot_id, None)

    def receive_request(self, request: SlotRequest, control_hz: float) -> None:
        if request.robot_id not in self.robots:
            # NOTE: for now, assume control_hz and execution_horizon are fixed for a robot's lifetime
            self.robots[request.robot_id] = Robot(control_hz, request.execution_horizon)
        self.robots[request.robot_id].step(
            ControlStep(
                time=request.request_timestamp,
                observation_step=request.observation_step,
                action_step=request.action_start_step,
                next_action_step=request.action_start_step + 1,
            )
        )

    def schedule_pending_chunk(self, robot_id: str, chunk: ActionChunk) -> None:
        self.robots[robot_id].send_response(chunk)

    def receive_response(self, ack: AckNotification) -> None:
        robot = self.robots.get(ack.robot_id)
        if robot is None:
            logger.debug("Ignoring ack for unknown robot: %s", ack.robot_id)
            return
        robot.receive_response(ack)

    def get_chunks(
        self, robot_ids: list[robot_id], latency_tracker: LatencyTracker, time: float
    ) -> list[ActionChunk]:
        """
        Returns a list of ActionChunks that would be queued if we started an inference for the robot_ids at the given time.
        """
        control_steps = []
        for rid in robot_ids:
            obs_time = time - latency_tracker.observation_latency(rid)
            control_steps.append(self.robots[rid].get_latest_control_step_before(obs_time))

        inference_latency = latency_tracker.infer_latency(len(robot_ids))
        return [
            ActionChunk(
                observation_step=control_step.observation_step,
                arrival_time=time + inference_latency + latency_tracker.action_latency(rid),
                action_start_step=control_step.next_action_step,
                execution_horizon=self.robots[rid].execution_horizon,
                arrived=True,
            )
            for i, (control_step, rid) in enumerate(zip(control_steps, robot_ids))
        ]

    def fast_forward(
        self,
        time: float,
        robot_ids: list[robot_id],
        chunks: list[ActionChunk],
    ) -> None:
        """Simulates time forward to the given time, sending responses and advancing robot steps."""
        # NOTE: we send responses here so they are available while stepping
        # it doesn't matter that they are "sent" before the actual sending time
        # because arrival_time handles the timing around this
        for rid, chunk in zip(robot_ids, chunks):
            self.robots[rid].send_response(chunk)

        for robot in self.robots.values():
            robot.step_forward(time)

    def deadlines(self) -> dict[robot_id, float]:
        return {rid: robot.deadline() for rid, robot in self.robots.items()}
