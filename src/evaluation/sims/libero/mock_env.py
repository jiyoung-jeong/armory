import time
from dataclasses import dataclass

import numpy as np

from armory_client.schemas import Action, Observation

IMAGE_SIZE = 224


@dataclass
class MockObservation(Observation):
    prompt: str = ""


class MockEnvironment:
    """Dummy environment that returns zero observations and counts steps."""

    def __init__(
        self,
        *,
        max_episode_steps: int = 100,
        state_dim: int = 8,
        deadline_monotonic: float | None = None,
    ) -> None:
        self._max_episode_steps = max_episode_steps
        self._state_dim = state_dim
        self._deadline_monotonic = deadline_monotonic

        self._step = 0
        self._done = True
        self._current_success = False

    def reset(self) -> None:
        self._step = 0
        self._done = False
        self._current_success = False

    def is_episode_complete(self) -> bool:
        if (
            not self._done
            and self._deadline_monotonic is not None
            and time.monotonic() >= self._deadline_monotonic
        ):
            self._done = True
        return self._done

    def get_observation(self) -> MockObservation:
        # TODO: does mock need to return images? might be helpful for networking experiments?
        img = np.zeros((self, IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
        return MockObservation(
            step=self._step,
            state=np.zeros(self._state_dim, dtype=np.float32),
            image=img,
            wrist_image=img,
            prompt="mock task",
        )

    def apply_action(self, action: Action) -> None:
        self._step += 1
        if self._step >= self._max_episode_steps:
            self._done = True

    def close(self) -> None:
        pass

    @property
    def max_episode_steps(self) -> int:
        return self._max_episode_steps

    @property
    def current_success(self) -> bool:
        return self._current_success

    @property
    def current_initial_state(self) -> np.ndarray:
        return np.zeros(1, dtype=np.float32)
