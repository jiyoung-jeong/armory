"""
TODO: docs

action indexes vs. control steps
"""

from __future__ import annotations
from dataclasses import dataclass, replace
import itertools
import logging
from typing import TypeAlias
from collections import deque
import copy

from armory.scheduling.latency import LatencyTracker
from armory.serving.schemas import AckNotification, SlotRequest

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


robot_id: TypeAlias = str


@dataclass
class Event:
    time: float


@dataclass
class ControlStep(Event):
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
        assert (
            not self.chunks or self.chunks[-1].observation_step < chunk.observation_step
        )
        self.chunks.append(chunk)

    def receive_response(self, ack: AckNotification) -> None:
        """Updates a chunk's actual arrival time."""
        for i, chunk in enumerate(self.chunks):
            if chunk.observation_step == ack.observation_step:
                self.chunks[i] = replace(
                    chunk, arrival_time=ack.receive_time, arrived=True
                )
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
        step = self.steps[-1]

        while step.action_step <= self.max_overall_action_step:
            step = self.advance_step(step)

        return step.time

    def step_forward(self, time: float) -> None:
        while self.steps[-1].time < time:
            self.steps.append(self.advance_step(self.steps[-1]))


class Mirror:
    def __init__(self):
        self.robots: dict[robot_id, Robot] = {}

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
        self.robots[ack.robot_id].receive_response(ack)

    def get_chunks(
        self, robot_ids: list[robot_id], latency_tracker: LatencyTracker, time: float
    ) -> list[ActionChunk]:
        control_steps = []
        for rid in robot_ids:
            obs_time = time - latency_tracker.observation_latency(rid)
            control_steps.append(
                self.robots[rid].get_latest_control_step_before(obs_time)
            )

        inference_latency = latency_tracker.infer_latency(len(robot_ids))
        return [
            ActionChunk(
                observation_step=control_step.observation_step,
                arrival_time=time
                + inference_latency
                + latency_tracker.action_latency(rid),
                action_start_step=control_step.action_step,
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
        """Jumps to next time the GPU is available."""
        # NOTE: we send responses here so they are available while stepping
        # it doesn't matter that they are "sent" before the actual sending time
        # because arrival_time handles the timing around this
        for rid, chunk in zip(robot_ids, chunks):
            self.robots[rid].send_response(chunk)

        for robot in self.robots.values():
            robot.step_forward(time)

    def total_actions(self) -> int:
        return sum(robot.actions_executed() for robot in self.robots.values())


# TODO: can be modified to support different search strategies/pruning/stopping criteria
def search(
    initial_mirror: Mirror,
    latency_tracker: LatencyTracker,
    start_time: float,
    horizon: float,
) -> list[tuple[robot_id, ...]]:
    """Search through all schedules that keep the GPU busy until time + horizon."""

    best_objective = -float("inf")
    best_schedule: list[tuple[robot_id, ...]] = []
    end_time = start_time + horizon
    # FIXME: don't access private
    max_batch_size = max(latency_tracker._infer_latency.keys())
    nodes_visited = 0
    branches_pruned_time = 0
    leaves = 0

    logger.debug(
        "search start: t=%.4f horizon=%.4f end_time=%.4f robots=%d max_batch=%d",
        start_time,
        horizon,
        end_time,
        len(initial_mirror.robots),
        max_batch_size,
    )

    def objective(mirror: Mirror, time: float) -> float:
        gpu_time = time - start_time
        if gpu_time <= 0:
            return -float("inf")
        new_actions = mirror.total_actions() - initial_mirror.total_actions()
        return new_actions / gpu_time

    def generate_candidates(mirror: Mirror):
        # TODO: for now just return combinations
        return itertools.chain.from_iterable(
            itertools.combinations(mirror.robots.keys(), i)
            for i in range(1, max_batch_size + 1)
        )

    def dfs(
        mirror: Mirror,
        time: float,
        schedule: list[tuple[robot_id, ...]],
        max_depth: int = 1,
    ) -> None:
        nonlocal \
            best_objective, \
            best_schedule, \
            nodes_visited, \
            branches_pruned_time, \
            leaves
        nodes_visited += 1
        objective_value = objective(mirror, time)
        if objective_value > best_objective:
            best_objective = objective_value
            best_schedule = schedule
            logger.debug(
                "new best: depth=%d objective=%.4f schedule=%s",
                len(schedule),
                objective_value,
                schedule,
            )

        if len(schedule) == max_depth:
            return

        candidates = list(generate_candidates(mirror))
        if not candidates:
            leaves += 1
            logger.debug(
                "leaf (no candidates): depth=%d t=%.4f robots=%d",
                len(schedule),
                time,
                len(mirror.robots),
            )
            return

        expanded = 0
        for batch in candidates:
            next_time = time + latency_tracker.infer_latency(len(batch))
            if next_time > end_time:
                branches_pruned_time += 1
                continue
            expanded += 1

            next_state = copy.deepcopy(mirror)
            chunks = next_state.get_chunks(list(batch), latency_tracker, time)
            next_state.fast_forward(next_time, list(batch), chunks)
            dfs(next_state, next_time, schedule + [batch])

        if expanded == 0:
            leaves += 1
            logger.debug(
                "leaf (all branches past end_time): depth=%d t=%.4f candidates=%d",
                len(schedule),
                time,
                len(candidates),
            )

    dfs(initial_mirror, start_time, [])
    logger.debug(
        "search done: nodes=%d leaves=%d pruned_time=%d best_objective=%.4f best_len=%d",
        nodes_visited,
        leaves,
        branches_pruned_time,
        best_objective,
        len(best_schedule),
    )
    return best_schedule
