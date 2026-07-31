import dataclasses
from collections import deque

from armory_client.messages import InferResponse
from armory_client.schemas import Action, ActionChunk


class ActionChunkBroker:
    def __init__(
        self,
        min_execution_horizon: int = 0,
        max_execution_horizon: int = 0,
    ) -> None:
        self.min_execution_horizon = min_execution_horizon
        self.max_execution_horizon = max_execution_horizon
        self._action_queue: deque[Action] = deque()
        self._action_chunks: list[ActionChunk] = []
        self._next_observation_step: int = 0  # next observation step to see
        self._next_action_step: int = 0  # next action step to execute

    def reset(self) -> None:
        self._next_observation_step = 0
        self._next_action_step = 0
        self._action_queue.clear()
        self._action_chunks = []

    def get_action(self, observation_step: int) -> Action | None:
        self._next_observation_step = observation_step + 1
        if not self._action_queue:
            return None
        self._next_action_step += 1
        # Depth before the pop, so a starved step is the one that records 0.
        actions_left = len(self._action_queue)
        return dataclasses.replace(self._action_queue.popleft(), actions_left=actions_left)

    def receive_response(self, infer_response: InferResponse) -> ActionChunk:
        action_chunk = ActionChunk.from_infer_response(
            infer_response=infer_response,
            execution_start_step=self._next_observation_step,
        )

        self._action_chunks.append(action_chunk)
        self._update_action_queue(action_chunk)
        return action_chunk

    def _update_action_queue(self, action_chunk: ActionChunk) -> None:
        while self._action_queue and self._action_queue[-1].step >= action_chunk.action_index_start:
            self._action_queue.pop()

        # assumes that pausing is preferable to executing actions past the execution horizon
        self._action_queue.extend(
            Action(
                step=action_chunk.action_index_start + i,
                action=action_chunk.get_action(i),
                action_chunk_index=len(self._action_chunks) - 1,
                index_in_chunk=i,
            )
            for i in range(action_chunk.max_execution_horizon)
            if action_chunk.action_index_start + i >= self._next_action_step
        )

    @property
    def action_chunks(self) -> list[ActionChunk]:
        return list(self._action_chunks)

    @property
    def current_action_chunk(self) -> ActionChunk | None:
        return self._action_chunks[-1] if self._action_chunks else None

    @property
    def next_action_step(self) -> int:
        return self._next_action_step

    @property
    def num_actions_available(self) -> int:
        return len(self._action_queue)
