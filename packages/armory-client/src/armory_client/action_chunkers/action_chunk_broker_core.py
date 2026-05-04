"""Synchronous, time-injected core of ActionChunkBroker.

Holds all queue/chunk state. The threaded ActionChunkBroker wrapper passes
``time.time()`` into ``on_observation`` / ``on_infer_response``; tests pass a
deterministic ``now`` to drive the same logic without threads or websockets.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from armory_client.messages import InferResponse
from armory_client.schemas import Action, ActionChunk, Observation


@dataclass(frozen=True)
class ObservationEvent:
    """The payload the broker would have sent over the wire for this observation."""

    observation_step: int
    action_start_step: int
    execution_horizon: int
    deadline: float
    request_timestamp: float


@dataclass(frozen=True)
class AckEvent:
    """The payload the broker would have sent back to the server for this chunk."""

    request_id: int
    response_timestamp: float
    execution_start_step: int
    first_executed_index: int


class ActionChunkBrokerCore:
    def __init__(self, control_hz: float, execution_horizon: int) -> None:
        self._step_duration = 1 / control_hz
        self.execution_horizon = execution_horizon

        self._action_queue: deque[Action] = deque()
        self._action_chunks: list[ActionChunk] = []
        self._next_observation_step: int = 0
        self._next_action_step: int = 0
        self._actions_left_history: list[int] = []
        self._prev_action: Action = self._create_null_action(-1)

    def on_observation(self, obs: Observation, *, now: float) -> tuple[Action, ObservationEvent]:
        self._actions_left_history.append(len(self._action_queue))

        self._next_observation_step = obs.step + 1
        if self._action_queue:
            action = self._action_queue.popleft()
            self._next_action_step += 1
        else:
            action = self._create_null_action(obs.step)

        self._prev_action = action

        event = ObservationEvent(
            observation_step=obs.step,
            action_start_step=self._next_action_step,
            execution_horizon=self.execution_horizon,
            deadline=self.deadline_at(now),
            request_timestamp=now,
        )
        return action, event

    def on_infer_response(self, infer_response: InferResponse, *, now: float) -> AckEvent:
        action_chunk = ActionChunk(
            observation_step=infer_response.observation_step,
            action_start_step=infer_response.action_start_step,
            execution_start_step=self._next_observation_step,
            actions=infer_response.actions,
            execution_horizon=infer_response.execution_horizon,
            request_timestamp=infer_response.request_timestamp,
            response_timestamp=now,
            request_id=infer_response.request_id,
            noise=infer_response.noise,
        )

        self._action_chunks.append(action_chunk)
        self._update_action_queue(action_chunk)
        first_executed_index = max(0, self._next_action_step - action_chunk.action_start_step)

        return AckEvent(
            request_id=action_chunk.request_id,
            response_timestamp=action_chunk.response_timestamp,
            execution_start_step=action_chunk.execution_start_step,
            first_executed_index=first_executed_index,
        )

    def reset(self) -> None:
        self._action_queue.clear()
        self._action_chunks = []
        self._next_observation_step = 0
        self._next_action_step = 0
        self._actions_left_history = []

    def _update_action_queue(self, action_chunk: ActionChunk) -> None:
        while self._action_queue and self._action_queue[-1].step >= action_chunk.action_start_step:
            self._action_queue.pop()

        # assumes that pausing is preferable to executing actions past the execution horizon
        self._action_queue.extend(
            Action(
                step=action_chunk.action_start_step + i,
                action=action_chunk.get_action(i),
                action_chunk_index=len(self._action_chunks) - 1,
                index_in_chunk=i,
            )
            for i in range(action_chunk.execution_horizon)
            if action_chunk.action_start_step + i >= self._next_action_step
        )

    def _create_null_action(self, observation_step: int) -> Action:
        # FIXME: hardcoded, should move this outside of this class
        action = np.zeros(7)
        action[-1] = (
            self.current_action_chunk.get_action(-1)[-1]
            if self.current_action_chunk is not None
            else 0.0
        )
        return Action(
            step=observation_step,
            action=action,
            action_chunk_index=None,
            index_in_chunk=None,
        )

    def deadline_at(self, now: float) -> float:
        return now + len(self._action_queue) * self._step_duration

    @property
    def next_action_step(self) -> int:
        return self._next_action_step

    @property
    def next_observation_step(self) -> int:
        return self._next_observation_step

    @property
    def action_chunks(self) -> list[ActionChunk]:
        return self._action_chunks

    @property
    def current_action_chunk(self) -> ActionChunk | None:
        return self._action_chunks[-1] if self._action_chunks else None

    @property
    def actions_left_history(self) -> list[int]:
        return list(self._actions_left_history)

    @property
    def prev_action(self) -> Action:
        return self._prev_action
