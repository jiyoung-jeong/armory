from typing_extensions import override

from openpi_client import action_chunkers as _action_chunkers
from openpi_client.schemas import Observation, Action
from openpi_client.runtime import agent as _agent


# TODO: fix typing on broker, it was base_policy
class PolicyAgent(_agent.Agent):
    """An agent that uses a policy to determine actions."""

    def __init__(self, broker: _action_chunkers.ActionChunkBroker) -> None:
        self._broker = broker

    @override
    def get_action(self, observation: Observation) -> Action:
        return self._broker.infer(observation)

    def reset(self) -> None:
        self._broker.reset()
