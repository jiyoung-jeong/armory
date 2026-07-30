from __future__ import annotations

import dataclasses
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
from evaluation.types import StepRecord

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

    Per-step data is captured live (``observations`` and ``steps``); the
    outcome is read from the env at episode end. Agent diagnostics are captured
    at the same boundary, before the next episode's reset clears them.
    """

    observations: tuple[Observation, ...]
    steps: tuple[StepRecord, ...]
    success: bool
    truncated: bool
    initial_state: np.ndarray | None
    action_chunks: tuple[ActionChunk, ...]


def _with_actions_left(
    steps: list[StepRecord], actions_left: tuple[int, ...]
) -> tuple[StepRecord, ...]:
    """Fold the agent's queue-depth history into the per-step records.

    The agent records one entry per ``get_action`` call, so it can be longer
    than the steps recorded here; steps with no entry keep ``None``.
    """
    return tuple(
        dataclasses.replace(step, actions_left=actions_left[i]) if i < len(actions_left) else step
        for i, step in enumerate(steps)
    )


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
        self._step_time = 1 / control_hz if control_hz > 0 else 0.0
        self._episode_sink = episode_sink
        # A single worker preserves episode order for sinks that allocate output
        # names sequentially, while allowing the next rollout to begin during IO.
        self._save_executor = (
            ThreadPoolExecutor(max_workers=1, thread_name_prefix="episode-save")
            if episode_sink is not None
            else None
        )
        self._save_futures: list[Future[None]] = []

    def has_time_to_step(self, deadline: float) -> bool:
        """Whether a step started now would land before ``deadline``.

        Lets a caller skip preparing an episode it has no time to step, which
        for a sim environment costs far more than the check.
        """
        return _has_time_for_step(deadline, self._step_time)

    def start_episode(self) -> None:
        """Prepare the next episode so ``run_episode`` can step immediately.

        Split from the stepping so a fleet can hold every robot here, at the
        point where the environments are reset but no requests have been sent.
        Resets are uneven — a LIBERO one settles the scene over several sim
        steps — so robots released before it start stepping seconds apart.
        """
        logger.info("Starting episode...")
        self._environment.reset()
        self._agent.reset()

    def run_episode(self, deadline: float = math.inf) -> Rollout | None:
        """Step the episode prepared by ``start_episode`` and return its Rollout.

        No step is started or recorded past ``deadline``, so the trace only
        covers the window in which the whole fleet was running: steps taken
        after peers exit see a lighter load than the experiment is measuring.

        Args:
            deadline: absolute ``time.monotonic()`` bounding the episode.

        Returns:
            The episode's Rollout, or ``None`` when the deadline left no room to
            record a step — the caller should stop looping.
        """
        observations: list[Observation] = []
        steps: list[StepRecord] = []

        last_step_time = time.perf_counter()

        while not self._environment.is_episode_complete() and self.has_time_to_step(deadline):
            observation, action = self._step()
            # Wall clock (not perf_counter) so it lines up with the broker's
            # request_timestamp when deriving per-step cost.
            step_timestamp = time.time()
            if time.monotonic() > deadline:
                break
            observations.append(observation)
            steps.append(
                StepRecord(
                    timestamp=step_timestamp,
                    env_step=observation.step,
                    action_chunk_index=action.action_chunk_index,
                    action_index=action.index_in_chunk,
                )
            )
            last_step_time = self._pace(last_step_time, self._step_time)

        if not steps:
            return None

        episode_data = self._agent.snapshot_episode_data()
        truncated = not self._environment.is_episode_complete()
        logger.info("Episode truncated by deadline." if truncated else "Episode completed.")
        rollout = Rollout(
            observations=tuple(observations),
            steps=_with_actions_left(steps, tuple(episode_data.actions_left)),
            success=self._environment.current_success,
            truncated=truncated,
            initial_state=self._environment.current_initial_state,
            action_chunks=tuple(episode_data.action_chunks),
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
