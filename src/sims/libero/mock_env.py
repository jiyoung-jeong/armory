from dataclasses import dataclass
from armory_client.schemas import Observation
from armory_client.schemas import Action
import numpy as np


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
        image_size: int = 224,
        control_hz: float = 20.0,
        task_id: int = 0,
        episode_idx: int = 0,
    ) -> None:
        self._max_episode_steps = max_episode_steps
        self._state_dim = state_dim
        self._image_size = image_size
        self._control_hz = control_hz
        self._task_id = task_id
        self._episode_idx = episode_idx

        self._step = 0
        self._done = True
        self._current_success = False

    def reset(self) -> None:
        self._step = 0
        self._done = False
        self._current_success = False

    def is_episode_complete(self) -> bool:
        return self._done

    def get_observation(self) -> MockObservation:
        img = np.zeros((self._image_size, self._image_size, 3), dtype=np.uint8)
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

    # Properties used by TaskMetricsPublisher
    @property
    def control_hz(self) -> float:
        return self._control_hz

    @property
    def max_episode_steps(self) -> int:
        return self._max_episode_steps

    @property
    def current_success(self) -> bool:
        return self._current_success

    @property
    def episode_idx(self) -> int:
        return self._episode_idx

    @property
    def current_initial_state(self) -> np.ndarray:
        return np.zeros(1, dtype=np.float32)

