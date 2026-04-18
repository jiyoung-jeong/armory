import abc

from openpi_client.schemas import Observation, Action


class Subscriber(abc.ABC):
    """Subscribes to events in the runtime.

    Subscribers can be used to save data, visualize, etc.
    """

    @abc.abstractmethod
    def on_episode_start(self) -> None:
        """Called when an episode starts."""

    @abc.abstractmethod
    def on_step(self, observation: Observation, action: Action) -> None:
        """Append a step to the episode."""

    @abc.abstractmethod
    def on_episode_end(self) -> None:
        """Called when an episode ends."""

    def close(self) -> None:
        """Called when the runtime is closing."""
        pass
