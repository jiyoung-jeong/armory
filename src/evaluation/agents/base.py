import abc
from dataclasses import dataclass

from armory_client.schemas import Action, ActionChunk, Observation


@dataclass(frozen=True)
class AgentEpisodeData:
    """Agent-owned diagnostics captured with a completed episode."""

    action_chunks: tuple[ActionChunk, ...] = ()
    actions_left: tuple[int, ...] = ()


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

    @abc.abstractmethod
    def snapshot_episode_data(self) -> AgentEpisodeData:
        """Return a stable snapshot of diagnostics for the current episode."""
