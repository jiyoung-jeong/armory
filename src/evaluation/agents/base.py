import abc

from armory_client.schemas import Action, ActionChunk, Observation


class Agent(abc.ABC):
    @abc.abstractmethod
    def get_action(self, observation: Observation) -> Action:
        pass

    @abc.abstractmethod
    def reset(self) -> None:
        pass

    @abc.abstractmethod
    def close(self) -> None:
        """Release resources owned by this agent."""

    @property
    @abc.abstractmethod
    def action_chunks(self) -> tuple[ActionChunk, ...]:
        """Chunks received since the last ``reset``, read before the next one."""
