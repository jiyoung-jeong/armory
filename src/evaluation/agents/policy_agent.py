from typing_extensions import override

from armory_client.action_chunkers.action_chunk_broker import ActionChunkBroker
from armory_client.schemas import Action, Observation
from evaluation.agents.base import Agent, AgentEpisodeData


class PolicyAgent(Agent):
    """An agent that queries a remote policy server, mediated by an action-chunk broker.

    The broker owns the transport (websocket) and turns the server's stream of
    action chunks into a single action per control step.
    """

    def __init__(self, broker: ActionChunkBroker) -> None:
        self._broker = broker

    @property
    def broker(self) -> ActionChunkBroker:
        return self._broker

    @override
    def get_action(self, observation: Observation) -> Action:
        return self._broker.infer(observation)

    @override
    def reset(self) -> None:
        self._broker.reset()

    @override
    def close(self) -> None:
        self._broker.close()

    @override
    def snapshot_episode_data(self) -> AgentEpisodeData:
        action_chunks, actions_left = self._broker.snapshot_episode_data()
        return AgentEpisodeData(
            action_chunks=tuple(action_chunks), actions_left=tuple(actions_left)
        )
