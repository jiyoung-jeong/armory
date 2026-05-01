from __future__ import annotations
from bisect import bisect_right
from dataclasses import dataclass, field
from typing import TypeAlias

sim_time: TypeAlias = int       # milliseconds
control_step: TypeAlias = int   # index into control periods
CONTROL_PERIOD: sim_time = 50   # milliseconds per control step


def time_to_step(time: sim_time) -> control_step:
    """Convert sim_time to control step assuming uniform grid. Use SimulatorParameters methods for per-robot conversion."""
    return time // CONTROL_PERIOD


def step_to_time(step: control_step) -> sim_time:
    """Convert control step to sim_time assuming uniform grid. Use SimulatorParameters methods for per-robot conversion."""
    return step * CONTROL_PERIOD


def generate_control_step_times(
    num_robots: int,
    end_time: sim_time,
    control_period: sim_time = CONTROL_PERIOD,
    jitter_std: float = 1.0,
    rng: "np.random.Generator | None" = None,
) -> list[list[sim_time]]:
    """Pre-sample per-robot control step times with random offset and jitter.

    Each robot's first step is at a uniform random offset in [0, control_period).
    Subsequent steps are spaced by control_period + N(0, jitter_std) ms,
    clamped to be strictly increasing.
    """
    import numpy as np
    if rng is None:
        rng = np.random.default_rng()

    result = []
    for _ in range(num_robots):
        offset = int(rng.integers(0, control_period))
        times: list[sim_time] = [offset]
        # Generate enough steps to cover well past end_time
        while times[-1] < end_time + control_period * 20:
            jitter = round(float(rng.normal(0, jitter_std)))
            next_time = times[-1] + control_period + jitter
            next_time = max(next_time, times[-1] + 1)  # ensure strictly increasing
            times.append(next_time)
        result.append(times)
    return result


@dataclass
class SimulatorParameters:
    end_time: sim_time = 0
    d_infer: list[sim_time] = field(default_factory=list)  # for each batch size

    num_robots: int = 0
    d_send: list[sim_time] = field(default_factory=list)  # for each robot
    d_recv: list[sim_time] = field(default_factory=list)  # for each robot
    execution_horizon: list[control_step] = field(default_factory=list)  # for each robot
    step_based_coverage: bool = False  # chunks valid over fixed step window, ignoring starvation

    # Per-robot control step times. If empty, falls back to uniform grid.
    # control_step_times[robot][step] = sim_time of that control step.
    control_step_times: list[list[sim_time]] = field(default_factory=list)

    def time_to_step_robot(self, time: sim_time, robot: int) -> control_step:
        """Convert sim_time to control step for a specific robot."""
        if not self.control_step_times:
            return time // CONTROL_PERIOD
        return bisect_right(self.control_step_times[robot], time) - 1

    def step_to_time_robot(self, step: control_step, robot: int) -> sim_time:
        """Convert control step to sim_time for a specific robot."""
        if not self.control_step_times:
            return step * CONTROL_PERIOD
        return self.control_step_times[robot][step]

    def num_steps_robot(self, robot: int) -> int:
        """Number of control steps for this robot up to end_time (inclusive)."""
        if not self.control_step_times:
            return self.end_time // CONTROL_PERIOD
        return bisect_right(self.control_step_times[robot], self.end_time)


@dataclass(frozen=True)
class ActionChunk:
    """A contiguous range of actions [start_action, start_action + horizon)
    that becomes available at arrival_step."""
    start_action: int
    horizon: control_step
    arrival_step: control_step
    observation_step: control_step = 0  # step when observation was captured

    @property
    def end_action(self) -> int:
        """Last action index covered (inclusive)."""
        return self.start_action + self.horizon - 1

    def covers(self, action: int) -> bool:
        return self.start_action <= action <= self.end_action

    @property
    def last_step(self) -> control_step:
        """Last control step this chunk covers (step-based mode)."""
        return self.observation_step + self.horizon - 1

    def covers_step(self, step: control_step) -> bool:
        """Does this chunk cover the given step? (step-based mode)"""
        return self.arrival_step <= step <= self.last_step

    def actions_left_at(self, step: control_step) -> int:
        """How many actions remain in this chunk at the given control step."""
        return self.end_action - step


@dataclass
class RobotState:
    """Tracks which action a robot executes at each control step.

    action_index[s] = the action the robot executes at control step s.

    Two coverage modes:
    - Action-based (default): advance if a chunk covers the next action index.
      Starvation causes the robot to fall behind, and later chunks "catch up".
    - Step-based (step_based_coverage=True): advance if the current step falls
      within a chunk's fixed time window [arrival_step, observation_step + horizon - 1].
      Coverage is independent of the robot's action progress. Matches the ILP model.
    """
    action_index: list[int] = field(default_factory=list)
    chunks: list[ActionChunk] = field(default_factory=list)
    step_based_coverage: bool = False

    def get_action(self, step: control_step) -> int:
        """Which action is the robot executing at this control step?"""
        return self.action_index[step]

    def advance_to(self, step: control_step) -> None:
        """Simulate forward to the given control step."""
        while len(self.action_index) <= step:
            current_step = len(self.action_index)
            prev = self.action_index[-1] if self.action_index else -1
            want = prev + 1

            if self.step_based_coverage:
                can_advance = any(chunk.covers_step(current_step) for chunk in self.chunks)
            else:
                can_advance = any(
                    chunk.arrival_step <= current_step and chunk.covers(want)
                    for chunk in self.chunks
                )
            self.action_index.append(want if can_advance else max(prev, 0))

    def deadline(self) -> control_step:
        """Control step at which this robot runs out of actions."""
        current_step = max(len(self.action_index) - 1, 0)

        if not self.chunks:
            return current_step

        if self.step_based_coverage:
            # Find furthest step reachable through contiguous chunk windows
            reachable = current_step
            changed = True
            while changed:
                changed = False
                for chunk in self.chunks:
                    if chunk.arrival_step <= reachable + 1 and chunk.last_step > reachable:
                        reachable = chunk.last_step
                        changed = True
            return reachable
        else:
            current_action = self.action_index[-1] if self.action_index else -1
            # Find furthest contiguous action reachable
            reachable = current_action
            changed = True
            while changed:
                changed = False
                for chunk in self.chunks:
                    if chunk.start_action <= reachable + 1 and chunk.end_action > reachable:
                        reachable = chunk.end_action
                        changed = True
            return current_step + (reachable - current_action)

    @property
    def steps_executed(self) -> int:
        return self.action_index[-1]

    @property
    def total_steps(self) -> int:
        return len(self.action_index)

    def actions_left(self) -> list[int]:
        """For each control step, how many buffered actions remained."""
        result = []
        for step, action in enumerate(self.action_index):
            chunk = next(
                (c for c in reversed(self.chunks) if c.arrival_step <= step),
                None
            )
            if chunk is None:
                result.append(0)
            elif self.step_based_coverage:
                result.append(max(0, chunk.last_step - step))
            else:
                result.append(chunk.end_action - action)
        return result


@dataclass
class BatchRecord:
    robot_indices: list[int]
    start_time: sim_time
    duration: sim_time


@dataclass
class SimulationState:
    time: sim_time = 0
    server_available_at: sim_time = 0
    robots: list[RobotState] = field(default_factory=list)


@dataclass
class Event: 
    time: sim_time
    

'''
robot control step
robot request
robot response send
robot response receive

'''