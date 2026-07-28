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
from evaluation.types import Timestamp

logger = logging.getLogger(__name__)

_SPIN_WINDOW_S = 0.002


def _has_time_for_step(deadline: float, step_time: float) -> bool:
    """Whether a step started now would land before ``deadline``.

    Checked ahead of the step rather than trimmed after it: taking it would send
    another inference request, changing the batch the server assembles for the
    robots still inside the window.
    """
    return time.monotonic() + step_time < deadline


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
    truncated: bool
    initial_state: np.ndarray | None
    action_chunks: tuple[ActionChunk, ...]
    actions_left: tuple[int, ...]


# TODO: episode sink pattern is weird, just return rollout and send function call to ThreadPoolExecutor
# also don't have runtime own threadpoolexecutor, since close might be called before. when it comes time
# we can discuss how to manage lifecycle. Maybe the caller should make sure ThreadPoolExecutor finishes.
class Runtime:
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

    def run_episode(self, deadline: float = math.inf) -> Rollout | None:
        """Run one episode and return its Rollout, or None if time has run out.

        No step is started or recorded past ``deadline``, so the trace only
        covers the window in which the whole fleet was running: steps taken
        after peers exit see a lighter load than the experiment is measuring.

        Args:
            deadline: absolute ``time.monotonic()`` bounding the episode.

        Returns:
            The episode's Rollout, or ``None`` when the deadline left no room to
            record a step — the caller should stop looping.
        """
        step_time = 1 / self._control_hz if self._control_hz > 0 else 0.0
        if not _has_time_for_step(deadline, step_time):
            return None

        logger.info("Starting episode...")
        self._environment.reset()
        self._agent.reset()

        if not _has_time_for_step(deadline, step_time):
            return None

        observations: list[Observation] = []
        timestamps: list[Timestamp] = []

        last_step_time = time.perf_counter()

        while not self._environment.is_episode_complete() and _has_time_for_step(
            deadline, step_time
        ):
            observation, action = self._step()
            # Wall clock (not perf_counter) so it lines up with the broker's
            # request_timestamp when deriving per-step cost.
            step_timestamp = time.time()
            if time.monotonic() > deadline:
                break
            observations.append(observation)
            timestamps.append(
                Timestamp(
                    timestamp=step_timestamp,
                    env_step=observation.step,
                    action_chunk_index=action.action_chunk_index,
                    action_index=action.index_in_chunk,
                )
            )
            last_step_time = self._pace(last_step_time, step_time)

        if not timestamps:
            return None

        episode_data = self._agent.snapshot_episode_data()
        truncated = not self._environment.is_episode_complete()
        logger.info("Episode truncated by deadline." if truncated else "Episode completed.")
        rollout = Rollout(
            observations=tuple(observations),
            timestamps=tuple(timestamps),
            success=self._environment.current_success,
            truncated=truncated,
            initial_state=self._environment.current_initial_state,
            action_chunks=tuple(episode_data.action_chunks),
            # The agent records one entry per get_action call, including a step
            # dropped above for landing past the deadline.
            actions_left=tuple(episode_data.actions_left[: len(timestamps)]),
        )
        if self._episode_sink is not None:
            assert self._save_executor is not None
            self._save_futures.append(self._save_executor.submit(self._episode_sink, rollout))
        return rollout

    def close(self) -> None:
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
        """Hold ``control_hz`` by sleeping then spinning the last _SPIN_WINDOW_S."""
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
