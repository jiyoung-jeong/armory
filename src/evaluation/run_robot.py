"""Single-robot rollout driver, shared by ``scripts/run.py`` and (once
simplified) ``scripts/run_all.py``.

Owns the episode loop and per-episode save; delegates the control loop to
``Runtime`` and the writing to ``save.save_episode``. Construction and teardown
of the env / agent / websocket belong to the caller (so a multiprocessing worker
can build them inside its own process).
"""

from __future__ import annotations

import logging
import math
import time

from armory_client.action_chunkers.action_chunk_broker import ActionChunkBroker
from evaluation.runtime import agent as _agent
from evaluation.runtime import environment as _environment
from evaluation.runtime.runtime import Runtime
from evaluation.save import SaveMeta, build_episode_save_data, save_episode

logger = logging.getLogger(__name__)


def run_robot(
    *,
    environment: _environment.Environment,
    agent: _agent.Agent,
    meta: SaveMeta,
    broker: ActionChunkBroker | None = None,
    num_episodes: int = 1,
    time_limit: float = 0.0,
) -> None:
    """Run up to ``num_episodes`` episodes (or until ``time_limit`` elapses), saving each.

    Args:
        broker: the agent's action-chunk broker, if any. Snapshotted after each
            episode for logging; ``None`` for agents that don't use one.
    """
    runtime = Runtime(environment, agent, control_hz=meta.control_hz)
    deadline = time.monotonic() + time_limit if time_limit > 0 else math.inf

    episode = 0
    while episode < num_episodes and time.monotonic() < deadline:
        rollout = runtime.run_episode(deadline)

        # Snapshot the broker's decision trace now, before the next episode's
        # reset() clears it. Empty for brokerless agents.
        action_chunks = list(broker.action_chunks) if broker is not None else []
        actions_left = list(broker.actions_left_history) if broker is not None else []

        data = build_episode_save_data(rollout, action_chunks, actions_left)
        save_episode(data, meta)
        episode += 1

    logger.info("robot %d: ran %d episode(s)", meta.robot_idx, episode)
