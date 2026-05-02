"""
TODO: docs

action indexes vs. control steps
"""
from __future__ import annotations
from collections.abc import Callable
from dataclasses import dataclass
import itertools
from typing import TypeAlias
from sortedcontainers import SortedList

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
    observation_step: int # step when observation was captured
    arrival_step: int # step when the chunk becomes available
    action_start_index: int # action index of the first action in the chunk
    horizon: int

class Robot:
    def __init__(self, robot_id: int):
        self.robot_id = robot_id
        self.steps: SortedList[ControlStep] = SortedList() # steps that we have collected for sure
        self.chunks: SortedList[ActionChunk] = [] # includes both pending and arrived chunks

    def step(self, control_step: ControlStep) -> None:
        # TODO: pass more info from client and directly assert/test here
        self.steps.add(control_step)

    def receive_request(self, request: InferRequest) -> None:


        pass

    def send_response(self, response: InferResponse) -> None:
        pass

    def step_info(self) -> tuple[int, int]:
        """action steps, total steps"""
        return len(self.steps), len(self.steps)

    def steps_left() -> int:
        pass

    def time_left(self) -> float:
        pass

    def next_action_index_at_time(self, time: float) -> int:
        # two cases: either it's 

        pass

# TODO: wire classes from __init__.py
class Mirror:
    def __init__(self):
        self.time: float = 0
        self.robots: list[Robot] = []

    def step(robot: Robot, ControlStep: ControlStep):
        pass

    def receive_request(request: InferRequest):
        pass

    def send_response(response: InferResponse):
        pass
    
    def receive_response(response: ResponseAck):
        pass

    def fast_forward(time: float):
        # queue known events

        # TODO: save a checkpoint
        pass

    def rollback():
        # TODO: restore from checkpoint
        pass

# TODO: could search on horizon of gpu end times
def search(mirror: Mirror, latency_tracker: LatencyTracker, horizon: float, objective: Callable[None, float]) -> list[Robot]:
    """Search through all schedules that keep the GPU busy until time + horizon"""

    best_objective = -float('inf')
    best_schedule = []
    end_time = mirror.time + horizon

    def generate_candidates(mirror: Mirror) -> list[Robot]:
        # TODO: for now just return combinations
        return itertools.combinations(mirror.robots, len(mirror.robots))

    def dfs(schedule: list[list[robot_id]]):
        nonlocal best_objective, best_schedule
        if objective() > best_objective:
            best_objective = objective()
            best_schedule = schedule

        candidates = generate_candidates(mirror)
        for batch in candidates:
            next_time = mirror.time + latency_tracker.infer_latency(batch)
            if next_time > end_time:
                continue

            # TODO: what do I need to queue here?
            mirror.fast_forward(next_time)
            dfs([schedule + [batch]])
            mirror.rollback()
    
    dfs([])
    return best_schedule


