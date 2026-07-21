from dataclasses import dataclass

import numpy as np
from typing_extensions import override

from armory_client.schemas import Action, Observation
from evaluation.runtime import environment as _environment

IMAGE_SIZE = 224


@dataclass
class MockObservation(Observation):
    state: np.ndarray
    image: np.ndarray
    wrist_image: np.ndarray
    prompt: str = ""


class MockEnvironment(_environment.Environment):
    """Dummy environment that returns zero observations and counts steps.

    Still emits full-size images so it exercises the real wire payload for
    networking/scheduling experiments driven by a ``PolicyAgent``.
    """

    def __init__(self, *, max_episode_steps: int = 100, state_dim: int = 8) -> None:
        self._max_episode_steps = max_episode_steps
        self._state_dim = state_dim

        self._step = 0
        self._done = True

    @override
    def reset(self) -> None:
        self._step = 0
        self._done = False

    @override
    def is_episode_complete(self) -> bool:
        return self._done

    @override
    def get_observation(self) -> MockObservation:
        img = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
        return MockObservation(
            step=self._step,
            state=np.zeros(self._state_dim, dtype=np.float32),
            image=img,
            wrist_image=img,
            prompt="mock task",
        )

    @override
    def apply_action(self, action: Action) -> None:
        self._step += 1
        if self._step >= self._max_episode_steps:
            self._done = True

    @override
    def close(self) -> None:
        pass

    @property
    @override
    def task_language(self) -> str:
        return "mock task"
