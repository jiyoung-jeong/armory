from __future__ import annotations
import numpy as np
from classes import (
    SimulatorParameters,
    sim_time,
    control_step,
    ActionChunk,
    RobotState,
    SimulationState,
    BatchRecord
)
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)

logging.basicConfig(level=logging.INFO)

from abc import ABC, abstractmethod

class Scheduler(ABC):
    """Abstract base class for scheduling algorithms."""
    def __init__(self, simulator_parameters: SimulatorParameters):
        self.simulator_parameters = simulator_parameters
        pass


    @abstractmethod
    def select_robots(self, simulator: Simulator) -> list[int]:
        """
        Select which robots to schedule at this step.
        Returns list of robot IDs to include in the batch.
        Timing calculations are handled by the simulator.
        """
        pass

class Simulator:
    def __init__(self, simulator_parameters: SimulatorParameters):
        self.params = simulator_parameters
        self.state = SimulationState(
            time=0,
            robots=[
                RobotState(step_based_coverage=simulator_parameters.step_based_coverage)
                for _ in range(simulator_parameters.num_robots)
            ],
        )

        self.batch_history: list[BatchRecord] = []
        self._server_available_stack: list[sim_time] = []
        self._time_stack: list[sim_time] = []
        self._action_index_lengths_stack: list[list[int]] = []

    # FIXME: weird coupling between simulator and scheduler
    def run(self, scheduler: Scheduler) -> None:
        with tqdm(total=self.params.end_time) as pbar:
            prev = 0
            while self.state.time < self.params.end_time:
                robot_indices = scheduler.select_robots(self)
                self.step(robot_indices)
                pbar.update(self.state.time - prev)
                prev = self.state.time
        # Fill action_index to the end of the simulation window
        for i, robot in enumerate(self.state.robots):
            step = self.params.time_to_step_robot(self.params.end_time - 1, i)
            if step >= 0:
                robot.advance_to(step)

    def step(self, robot_indices: list[int]) -> None:
        # Save state for undo BEFORE any mutations
        self._time_stack.append(self.state.time)
        self._action_index_lengths_stack.append([len(r.action_index) for r in self.state.robots])

        # Advance all robots up to current time
        for i, robot in enumerate(self.state.robots):
            step = self.params.time_to_step_robot(self.state.time, i)
            if step >= 0:
                robot.advance_to(step)

        batch_size = len(robot_indices)
        d_infer = self.params.d_infer[batch_size]

        for robot_index in robot_indices:
            robot = self.state.robots[robot_index]
            obs_step = self.params.time_to_step_robot(self.state.time - self.params.d_send[robot_index], robot_index)
            # If observation is before the robot's first step, use action 0
            if obs_step >= 0 and robot.action_index:
                start_action = robot.get_action(obs_step)
            else:
                start_action = 0
            arrival_step = self.params.time_to_step_robot(self.state.time + d_infer + self.params.d_recv[robot_index], robot_index)
            chunk = ActionChunk(
                start_action=start_action,
                horizon=self.params.execution_horizon[robot_index],
                arrival_step=max(0, arrival_step),
                observation_step=max(0, obs_step),
            )
            robot.chunks.append(chunk)

        batch = BatchRecord(robot_indices=robot_indices, start_time=self.state.time, duration=d_infer)
        self._server_available_stack.append(self.state.server_available_at)
        self.batch_history.append(batch)

        self.state.server_available_at = max(self.state.server_available_at, batch.start_time + batch.duration)
        self.state.time = max(self.state.time + 1, self.state.server_available_at)
        logger.debug(
            f"Time {self.state.time}: Scheduled batch {robot_indices}, "
            f"d_infer={d_infer}, server available at {self.state.server_available_at}"
        )

    def undo(self) -> None:
        batch = self.batch_history.pop()
        self.state.server_available_at = self._server_available_stack.pop()
        self.state.time = self._time_stack.pop()
        saved_lengths = self._action_index_lengths_stack.pop()

        for index in batch.robot_indices:
            self.state.robots[index].chunks.pop()

        # Restore ALL robots' action_index to pre-step lengths
        for i, robot in enumerate(self.state.robots):
            del robot.action_index[saved_lengths[i]:]

    def calculate_starvation(self) -> int:
        total_steps_executed = sum(robot.steps_executed for robot in self.state.robots)
        total_steps = sum(robot.total_steps for robot in self.state.robots)
        return total_steps - total_steps_executed

    def actions_left(self) -> np.ndarray:
        arrays = [np.array(robot.actions_left()) for robot in self.state.robots]
        if not arrays:
            return np.array([])
        max_len = max(len(a) for a in arrays)
        # Pad shorter arrays with 0 so np.stack works with varying step counts
        padded = [np.pad(a, (0, max_len - len(a))) for a in arrays]
        return np.stack(padded)
