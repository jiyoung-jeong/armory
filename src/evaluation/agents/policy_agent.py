from typing_extensions import override

from armory_client.action_chunkers.action_chunk_broker import ActionChunkBroker
from armory_client.schemas import Action, Observation
from evaluation.runtime import agent as _agent


class PolicyAgent(_agent.Agent):
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
