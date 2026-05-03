"""
TODO: docs

action indexes vs. control steps
"""

from __future__ import annotations
from dataclasses import dataclass
import itertools
from typing import TypeAlias
from sortedcontainers import SortedList
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
    arrival_time: float  # step when the chunk becomes available
    action_start_index: int  # action index of the first action in the chunk
    horizon: int


class Robot:
    def __init__(self, control_hz: float, horizon: int):
        self.control_hz = control_hz
        self.horizon = horizon

        self.steps: SortedList[ControlStep] = (
            SortedList()
        )  # steps that we have collected for sure
        self.arrived_chunks: SortedList[ActionChunk] = (
            SortedList()
        )  # steps that we have arrived but not processed yet
        # we keep these separate so we can make best-effort estimates of when chunks will be available
        self.pending_chunks: SortedList[ActionChunk] = (
            SortedList()
        )  # this is separate because we add to pending chunks only after we ack from robot

    def step(self, control_step: ControlStep) -> None:
        # TODO: pass more info from client and directly assert/test here
        self.steps.add(control_step)

    def send_response(self, response: InferResponse) -> None:
        self.pending_chunks.add(
            ActionChunk(
                response.server_arrival_time,
                response.observation_step,
                response.action_start_index,
                response.horizon,
            )
        )

    def receive_response(self, response: ResponseAck) -> None:
        # TODO: remove from pending chunks
        self.arrived_chunks.add(
            ActionChunk(
                response.observation_step,
                response.arrival_step,
                response.action_start_index,
                response.horizon,
            )
        )

    @property
    def max_arrived_action_step(self) -> int:
        return (
            self.arrived_chunks[-1].action_start_index
            + self.arrived_chunks[-1].horizon
            - 1
        )

    @property
    def max_overall_action_step(self) -> int:
        max_overall_action_step = self.max_arrived_action_step
        if self.pending_chunks:
            max_overall_action_step = max(
                max_overall_action_step,
                self.pending_chunks[-1].action_start_index
                + self.pending_chunks[-1].horizon
                - 1,
            )
        return max_overall_action_step

    def get_latest_control_step_before(self, time: float) -> ControlStep | None:
        for step in reversed(self.steps):
            if step.time < time:
                return step
        return None

    def step_forward(self, time: float) -> None:
        while self.steps[-1].time < time:
            prev_step = self.steps[-1]
            next_action_step = prev_step.action_step
            self.steps.append(
                ControlStep(
                    prev_step.time + 1 / self.control_hz,
                    prev_step.observation_step + 1,
                    next_action_step
                    if next_action_step <= self.max_arrived_action_step
                    else self.max_arrived_action_step(),
                )
            )

        self.steps.pop()


class Mirror:
    def __init__(self):
        self.time: float = 0
        self.robots: dict[robot_id, Robot] = {}

    def receive_request(self, request: InferRequest) -> None:
        if request.robot_id not in self.robots:
            # NOTE: for now, assume control_hz and execution_horizon are fixed for a robot's lifetime
            self.robots[request.robot_id] = Robot(
                request.control_hz, request.execution_horizon
            )
        self.robots[request.robot_id].step(
            ControlStep(request.observation_step, request.action_start_step)
        )

    def send_response(self, response: InferResponse) -> None:
        self.robots[response.robot_id].send_response(response)

    def receive_response(self, response: ResponseAck) -> None:
        self.robots[response.robot_id].receive_response(response)

    def get_chunks(
        self, robot_ids: list[robot_id], latency_tracker: LatencyTracker
    ) -> list[ActionChunk]:
        control_steps = []
        for robot_id in robot_ids:
            time = self.time - latency_tracker.observation_latency(robot_id)
            control_steps.append(
                self.robots[robot_id].get_latest_control_step_before(time)
            )

        inference_latency = latency_tracker.infer_latency(len(robot_ids))
        return [
            ActionChunk(
                control_step.observation_step,
                time + inference_latency + latency_tracker.action_latency(robot_id),
                control_step.action_step,
                self.robots[robot_id].horizon,
            )
            for control_step, robot_id in zip(control_steps, robot_ids)
        ]

    def fast_forward(self, time: float, chunks: list[ActionChunk]) -> None:
        """Jumps to next time the GPU is available"""
        # process chunks
        for chunk in chunks:
            self.robots[chunk.robot_id].receive_response(chunk)

        # step forward in time
        for robot in self.robots.valuess():
            robot.step_forward(time)

        self.time = time

    def total_actions(self) -> int:
        return sum(robot.actions_executed for robot in self.robots.values())


# TODO: can be modified to support different search strategies/pruning/stopping criteria
def search(
    initial_mirror: Mirror, latency_tracker: LatencyTracker, horizon: float
) -> list[Robot]:
    """Search through all schedules that keep the GPU busy until time + horizon"""

    best_objective = -float("inf")
    best_schedule = []
    end_time = initial_mirror.time + horizon
    # FIXME: don't access private
    max_batch_size = max(latency_tracker._infer_latency.keys())

    def objective(mirror: Mirror) -> float:
        gpu_time = mirror.time - initial_mirror.time
        new_actions = mirror.total_actions() - initial_mirror.total_actions()
        return new_actions / gpu_time

    def generate_candidates(mirror: Mirror) -> list[Robot]:
        # TODO: for now just return combinations
        return itertools.chain.from_iterable(
            [
                itertools.combinations(mirror.robots, i)
                for i in range(1, max_batch_size + 1)
            ]
        )

    def dfs(mirror: Mirror, schedule: list[list[robot_id]]):
        nonlocal best_objective, best_schedule
        objective_value = objective(mirror)
        if objective_value > best_objective:
            best_objective = objective_value
            best_schedule = schedule

        candidates = generate_candidates(mirror)
        for batch in candidates:
            next_time = mirror.time + latency_tracker.infer_latency(batch)
            if next_time > end_time:
                continue

            next_state = copy.deepcopy(mirror)
            chunks = next_state.get_chunks(batch, latency_tracker)
            next_state.fast_forward(next_time, chunks)
            dfs(next_state, schedule + [batch])

    dfs(initial_mirror, [])
    return best_schedule


"""
simulating the future:
- i could have api that gets next action index at any time, 
    - i need this to know which chunks to queue
- if i don't maintain any sort of curren state and calculate everything on the fly, i can just queue all events that I know
"""
