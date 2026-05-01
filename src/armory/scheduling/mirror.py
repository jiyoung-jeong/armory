"""
TODO: docs

action indexes vs. control steps
"""
from __future__ import annotations
from collections.abc import Callable
from dataclasses import dataclass
from sortedcontainers import SortedList

from armory_client.messages import InferRequest, InferResponse, ResponseAck


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
        self.steps: SortedList[ControlStep] = SortedList()
        self.arrived_chunks: SortedList[ActionChunk] = []
        self.pending_chunks: SortedList[ActionChunk] = []

    def step():
        # TODO: pass more info from client and directly assert/test here
        pass

    def receive_request():
        pass

    def send_response():
        pass

    def step_info() -> tuple[int, int]:
        """action steps, total steps"""
        return len(self.steps), len(self.steps)

    def steps_left() -> int:
        pass

    def time_left() -> float:
        pass


class Mirror:
    def __init__(self):
        self.time: float = 0
        self.robots: list[Robot] = []
        self.events: SortedList[Event] = SortedList()

    def step(robot: Robot, ControlStep: ControlStep):
        pass

    def receive_request(request: InferRequest):
        pass

    def send_response(response: InferResponse):
        pass
    
    def receive_response(response: ResponseAck):
        pass

    def fast_forward(time: float):
        # TODO: save a checkpoint
        pass

    def rollback():
        # TODO: restore from checkpoint
        pass

def search(mirror: Mirror, latency_tracker: LatencyTracker, horizon: float, objective: Callable[None, float]) -> list[Robot]:
    """Search through all schedules that keep the GPU busy until time + horizon"""

    best_objective = -float('inf')
    schedule = []

    def dfs(batch: list[Robot]):
        if objective() > best_objective:
            best_objective = objective()
            best_schedule = schedule

        candidates = generate_candidates(batch)
        for candidate in candidates:
            batch_time = TODO
            next_time = self.time + batch_time
            if next_time > horizon:
                continue

            self.fast_forward(next_time)
            dfs(batch + [candidate])
            self.rollback()
    
    dfs(0, self.robots)
    return best_schedule


