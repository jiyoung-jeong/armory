from __future__ import annotations

import logging
import math
import time
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np

from armory_client.schemas import Action, ActionChunk, Observation
from evaluation.agents.base import Agent
from evaluation.envs.base import Environment
from evaluation.recording import Timestamp

logger = logging.getLogger(__name__)

# How long before the step deadline to switch from time.sleep to spinning.
_SPIN_WINDOW_S = 0.002


@dataclass(frozen=True)
class Rollout:
    """The product of running one episode: everything needed to log/save it.

    Per-step data is captured live (``observations`` and ``timestamps``); the
    outcome is read from the env at episode end. Agent diagnostics are captured
    at the same boundary, before the next episode's reset clears them.
    """

    observations: tuple[Observation, ...]
    timestamps: tuple[Timestamp, ...]
    success: bool
    initial_state: np.ndarray | None
    action_chunks: tuple[ActionChunk, ...]
    actions_left: tuple[int, ...]


class Runtime:
    """Runs the env<->agent control loop, optionally persisting episodes in the background."""

    def __init__(
        self,
        environment: Environment,
        agent: Agent,
        control_hz: float = 0.0,
        episode_sink: Callable[[Rollout], None] | None = None,
    ) -> None:
        self._environment = environment
        self._agent = agent
        self._control_hz = control_hz
        self._episode_sink = episode_sink
        # A single worker preserves episode order for sinks that allocate output
        # names sequentially, while allowing the next rollout to begin during IO.
        self._save_executor = (
            ThreadPoolExecutor(max_workers=1, thread_name_prefix="episode-save")
            if episode_sink is not None
            else None
        )
        self._save_futures: list[Future[None]] = []

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

        episode_data = self._agent.snapshot_episode_data()
        logger.info("Episode completed.")
        rollout = Rollout(
            observations=tuple(observations),
            timestamps=tuple(timestamps),
            success=self._environment.current_success,
            initial_state=self._environment.current_initial_state,
            action_chunks=tuple(episode_data.action_chunks),
            actions_left=tuple(episode_data.actions_left),
        )
        if self._episode_sink is not None:
            assert self._save_executor is not None
            self._save_futures.append(self._save_executor.submit(self._episode_sink, rollout))
        return rollout

    def close(self) -> None:
        """Release the environment and agent resources owned by this runtime."""
        try:
            try:
                self._environment.close()
            finally:
                self._agent.close()
        finally:
            if self._save_executor is not None:
                # Do not return until every accepted rollout is durable.
                self._save_executor.shutdown(wait=True)
                # ``shutdown`` waits but does not re-raise worker exceptions.
                for save_future in self._save_futures:
                    save_future.result()

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
