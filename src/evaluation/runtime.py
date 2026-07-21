from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass

import numpy as np

from armory_client.schemas import Action, Observation
from evaluation.recording import Timestamp
from evaluation.runtime import agent as _agent
from evaluation.runtime import environment as _environment

logger = logging.getLogger(__name__)

# How long before the step deadline to switch from time.sleep to spinning.
_SPIN_WINDOW_S = 0.002


@dataclass
class Rollout:
    """The product of running one episode: everything needed to log/save it.

    Per-step data is captured live (``observations`` and ``timestamps``); the
    outcome is read from the env at episode end. Policy-internal data (action
    chunks, queue depth) is *not* here — the driver snapshots that from the
    broker, so the Runtime stays agnostic to how the agent decides.
    """

    observations: list[Observation]
    timestamps: list[Timestamp]
    success: bool
    initial_state: np.ndarray | None


class Runtime:
    """Runs a single episode: the env<->agent control loop, paced to ``control_hz``."""

    def __init__(
        self,
        environment: _environment.Environment,
        agent: _agent.Agent,
        control_hz: float = 0.0,
    ) -> None:
        self._environment = environment
        self._agent = agent
        self._control_hz = control_hz

    def run_episode(self, deadline: float = math.inf) -> Rollout:
        """Run one episode and return its Rollout.

        Args:
            deadline: absolute ``time.monotonic()`` after which the loop stops
                even if the episode has not finished. Defaults to no limit.
        """
        logger.info("Starting episode...")
        self._environment.reset()
        self._agent.reset()

        observations: list[Observation] = []
        timestamps: list[Timestamp] = []

        step_time = 1 / self._control_hz if self._control_hz > 0 else 0.0
        last_step_time = time.perf_counter()

        while not self._environment.is_episode_complete() and time.monotonic() < deadline:
            observation, action = self._step()
            observations.append(observation)
            timestamps.append(
                Timestamp(
                    # Wall clock (not perf_counter) so it lines up with the
                    # broker's request_timestamp when deriving per-step cost.
                    timestamp=time.time(),
                    env_step=observation.step,
                    action_chunk_index=action.action_chunk_index,
                    action_index=action.index_in_chunk,
                )
            )
            last_step_time = self._pace(last_step_time, step_time)

        logger.info("Episode completed.")
        return Rollout(
            observations=observations,
            timestamps=timestamps,
            success=self._environment.current_success,
            initial_state=self._environment.current_initial_state,
        )

    def _step(self) -> tuple[Observation, Action]:
        observation = self._environment.get_observation()
        action = self._agent.get_action(observation)
        self._environment.apply_action(action)
        return observation, action

    def _pace(self, last_step_time: float, step_time: float) -> float:
        """Hold ``control_hz`` by sleeping then spinning the last ~1ms.

        OS sleep granularity can overshoot by ~1ms, so we sleep only until
        ``_SPIN_WINDOW_S`` before the target and spin the remainder.
        """
        if step_time <= 0:
            return time.perf_counter()
        next_step_time = last_step_time + step_time
        while True:
            remaining = next_step_time - time.perf_counter()
            if remaining <= _SPIN_WINDOW_S:
                break
            time.sleep(remaining - _SPIN_WINDOW_S)
        while time.perf_counter() < next_step_time:
            pass
        return time.perf_counter()
