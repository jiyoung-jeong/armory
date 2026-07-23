import abc

import numpy as np

from armory_client.schemas import Action, ActionChunk, Observation


class Environment(abc.ABC):
    @abc.abstractmethod
    def reset(self) -> None:
        """Called once before each episode."""

    @abc.abstractmethod
    def is_episode_complete(self) -> bool:
        pass

    @abc.abstractmethod
    def get_observation(self) -> Observation:
        pass

    @abc.abstractmethod
    def apply_action(self, action: Action) -> None:
        pass

    @abc.abstractmethod
    def create_null_action(
        self, observation: Observation, current_action_chunk: ActionChunk | None
    ) -> Action:
        """Create the environment-specific action used when no chunk is available."""
        pass

    @abc.abstractmethod
    def close(self) -> None:
        pass

    # --- Episode outcome / metadata read at episode end for logging. ---
    # Concrete defaults so envs that have no notion of these don't need to
    # override them.

    @property
    def current_success(self) -> bool:
        """Whether the most recent episode achieved its task."""
        return False

    @property
    def current_initial_state(self) -> np.ndarray | None:
        """The initial state the current episode was reset to, if any."""
        return None

    @property
    def task_language(self) -> str:
        """Natural-language description of the task, for saved metadata."""
        return ""
