from __future__ import annotations

import logging
import time

from evaluation.runtime import agent as _agent
from evaluation.runtime import environment as _environment
from evaluation.runtime import subscriber as _subscriber

# How long before the step deadline to switch from time.sleep to spinning.
_SPIN_WINDOW_S = 0.002


class Runtime:
    def __init__(
        self,
        environment: _environment.Environment,
        agent: _agent.Agent,
        subscribers: list[_subscriber.Subscriber],
        step_rate: float = 0,
        deadline: float = 0,
    ) -> None:
        """
        Initialize the runtime loop for a single agent/environment rollout.

        Args:
            environment (_environment.Environment): The environment instance in which the agent will be rolled out.
            agent (_agent.Agent): The agent controlling the environment.
            subscribers (list[_subscriber.Subscriber]): List of subscriber objects that receive episode and step events.
            step_rate (float, optional): Desired control rate in Hz (steps per second). If <= 0, runs as fast as possible. Defaults to 0.
            deadline (float, optional): Absolute time (seconds since epoch or perf_counter) after which execution will stop. If 0, continues indefinitely. Defaults to 0.
        """

        self._environment = environment
        self._agent = agent
        self._subscribers = subscribers
        self._step_rate = step_rate
        self._deadline = deadline

        self._in_episode = False

    def run(self) -> None:
        while self.time() < self._deadline:  # TODO: pull out into function
            self._run_episode()

        # Final reset, this is important for real environments to move the robot to its home position.
        self._environment.reset()

    def mark_episode_complete(self) -> None:
        self._in_episode = False

    def _run_episode(self) -> None:
        logging.info("Starting episode...")
        self._environment.reset()
        self._agent.reset()
        for subscriber in self._subscribers:
            subscriber.on_episode_start()

        self._in_episode = True
        step_time = 1 / self._step_rate if self._step_rate > 0 else 0
        last_step_time = time.perf_counter()

        while self._in_episode and time.time < self._deadline:
            self._step()

            next_step_time = last_step_time + step_time
            # Hybrid pacing: OS sleep granularity can overshoot by ~1ms, so
            # sleep only until _SPIN_WINDOW_S before the deadline and spin the
            # remainder to hold control_hz without burning the whole period.
            while True:
                remaining = next_step_time - time.perf_counter()
                if remaining <= _SPIN_WINDOW_S:
                    break
                time.sleep(remaining - _SPIN_WINDOW_S)
            while time.perf_counter() < next_step_time:
                pass
            last_step_time = time.perf_counter()

        logging.info("Episode completed.")
        for subscriber in self._subscribers:
            subscriber.on_episode_end()
        # TODO: currently a hack, reset agent after subscribers processed the episode data. overall increases calls to reset() to twice
        self._agent.reset()

    def _step(self) -> None:
        observation = self._environment.get_observation()
        action = self._agent.get_action(observation)
        self._environment.apply_action(action)

        for subscriber in self._subscribers:
            subscriber.on_step(observation, action)

        if self._environment.is_episode_complete():
            self.mark_episode_complete()

    def close(self) -> None:
        self._environment.close()
        for subscriber in self._subscribers:
            subscriber.close()
