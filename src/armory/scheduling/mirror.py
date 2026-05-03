"""
TODO: docs

action indexes vs. control steps
"""

from __future__ import annotations
from dataclasses import dataclass, replace
import itertools
from typing import TypeAlias
from collections import deque
import copy

from armory_client.messages import InferRequest, InferResponse, ResponseAck
from armory.scheduling.latency import LatencyTracker


robot_id: TypeAlias = str


@dataclass
class Event:
    time: float


@dataclass
class ControlStep(Event):
    observation_step: int
    action_step: int | None


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

        # Both lists should be sorted increasing by time
        self.steps: list[ControlStep] = []
        # includes chunks that are in-transit
        self.chunks: deque[ActionChunk] = deque()

    def step(self, control_step: ControlStep) -> None:
        # TODO: pass more info from client and directly assert/test here
        assert self.steps == [] or self.steps[-1].time < control_step.time
        self.steps.append(control_step)

    def send_response(self, response: InferResponse) -> None:
        self.chunks.append(
            ActionChunk(
                observation_step=response.observation_step,
                arrival_time=response.server_arrival_time,
                action_start_step=response.action_start_step,
                execution_horizon=response.execution_horizon,
                arrived=False,
            )
        )

    def receive_response(self, response: ResponseAck) -> None:
        """Updates a chunk's actual arrival time."""
        for i, chunk in enumerate(self.chunks):
            # NOTE: should be okay to assume that observation_step unique per request
            if chunk.observation_step == response.observation_step:
                self.chunks[i] = replace(
                    chunk, arrival_time=response.receive_time, arrived=True
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

    def step_forward(self, time: float) -> None:
        if not self.steps:
            return
        while self.steps[-1].time < time:
            prev_step = self.steps[-1]
            next_action_step = (
                (prev_step.action_step + 1) if prev_step.action_step is not None else 0
            )
            max_arrived = self.max_arrived_action_step
            self.steps.append(
                ControlStep(
                    time=prev_step.time + 1 / self.control_hz,
                    observation_step=prev_step.observation_step + 1,
                    action_step=(
                        next_action_step
                        if next_action_step <= max_arrived
                        else (max_arrived if max_arrived >= 0 else None)
                    ),
                )
            )

    def actions_executed(self) -> int:
        steps = [s for s in self.steps if s.action_step is not None]
        if not steps:
            return 0
        return steps[-1].action_step - steps[0].action_step + 1


class Mirror:
    def __init__(self):
        self.time: float = 0
        self.robots: dict[robot_id, Robot] = {}

    def receive_request(self, request: InferRequest, control_hz: float) -> None:
        if request.robot_id not in self.robots:
            # NOTE: for now, assume control_hz and execution_horizon are fixed for a robot's lifetime
            self.robots[request.robot_id] = Robot(control_hz, request.execution_horizon)
        self.robots[request.robot_id].step(
            ControlStep(
                time=request.request_timestamp,
                observation_step=request.observation_step,
                action_step=request.action_start_step,
            )
        )

    def send_response(self, response: InferResponse) -> None:
        self.robots[response.robot_id].send_response(response)

    def receive_response(self, response: ResponseAck) -> None:
        robot = self.robots[self._request_to_robot[response.request_id]]
        robot.receive_response(response)

    def get_chunks(
        self, robot_ids: list[robot_id], latency_tracker: LatencyTracker
    ) -> list[ActionChunk]:
        control_steps = []
        for rid in robot_ids:
            obs_time = self.time - latency_tracker.observation_latency(rid)
            control_steps.append(
                self.robots[rid].get_latest_control_step_before(obs_time)
            )

        inference_latency = latency_tracker.infer_latency(len(robot_ids))
        # synthesized chunks use a synthetic, monotonically-decreasing request_id
        # so they don't collide with real ones from the wire
        return [
            ActionChunk(
                request_id=-(i + 1),
                observation_step=control_step.observation_step,
                arrival_time=self.time
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

        self.time = time

    def total_actions(self) -> int:
        return sum(robot.actions_executed() for robot in self.robots.values())


# TODO: can be modified to support different search strategies/pruning/stopping criteria
def search(
    initial_mirror: Mirror, latency_tracker: LatencyTracker, horizon: float
) -> list[tuple[robot_id, ...]]:
    """Search through all schedules that keep the GPU busy until time + horizon."""

    best_objective = -float("inf")
    best_schedule: list[tuple[robot_id, ...]] = []
    end_time = initial_mirror.time + horizon
    # FIXME: don't access private
    max_batch_size = max(latency_tracker._infer_latency.keys())

    def objective(mirror: Mirror) -> float:
        gpu_time = mirror.time - initial_mirror.time
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

    def dfs(mirror: Mirror, schedule: list[tuple[robot_id, ...]]) -> None:
        nonlocal best_objective, best_schedule
        objective_value = objective(mirror)
        if objective_value > best_objective:
            best_objective = objective_value
            best_schedule = schedule

        for batch in generate_candidates(mirror):
            next_time = mirror.time + latency_tracker.infer_latency(len(batch))
            if next_time > end_time:
                continue

            next_state = copy.deepcopy(mirror)
            chunks = next_state.get_chunks(list(batch), latency_tracker)
            next_state.fast_forward(next_time, list(batch), chunks)
            dfs(next_state, schedule + [batch])

    dfs(initial_mirror, [])
    return best_schedule


"""
simulating the future:
- i could have api that gets next action index at any time,
    - i need this to know which chunks to queue
- if i don't maintain any sort of curren state and calculate everything on the fly, i can just queue all events that I know
"""
