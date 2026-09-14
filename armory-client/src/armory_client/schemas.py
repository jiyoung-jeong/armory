import time
from dataclasses import dataclass

import numpy as np

from armory_client import messages


@dataclass(frozen=True)
class ActionChunk:
    chunk_id: int
    observation_step: int
    action_index_start: int
    execution_start_step: int
    actions: np.ndarray
    min_execution_horizon: int
    max_execution_horizon: int
    request_timestamp: float
    response_timestamp: float
    request_id: int = -1
    noise: np.ndarray | None = None
    episode_id: str = ""

    @classmethod
    def from_infer_response(
        cls,
        infer_response: messages.InferResponse,
        execution_start_step: int,
    ) -> "ActionChunk":
        # NOTE: copy attributes instead of composition to make it easier to serialize
        return ActionChunk(
            chunk_id=infer_response.chunk_id,
            observation_step=infer_response.observation_step,
            action_index_start=infer_response.action_index_start,
            execution_start_step=execution_start_step,
            actions=infer_response.actions,
            min_execution_horizon=infer_response.min_execution_horizon,
            max_execution_horizon=infer_response.max_execution_horizon,
            request_timestamp=infer_response.request_timestamp,
            response_timestamp=time.time(),
            request_id=infer_response.request_id,
            noise=infer_response.noise,
            episode_id=infer_response.episode_id,
        )

    @property
    def latency(self) -> float:
        return self.response_timestamp - self.request_timestamp

    def get_action(self, index: int) -> np.ndarray:
        return self.actions[index]


@dataclass(frozen=True)
class Action:
    """The action_chunk_index and index_in_chunk will be None for the null action."""

    step: int
    action: np.ndarray
    action_chunk_index: int | None
    index_in_chunk: int | None
    # Queue depth when this action was taken; None for agents with no queue.
    actions_left: int | None = None


@dataclass
class Observation:
    step: int
