import numpy as np
from typing_extensions import override

from armory_client.schemas import Action, Observation
from evaluation.runtime import agent as _agent

# 6-DoF arm + gripper.
_ACTION_DIM = 7


class MockAgent(_agent.Agent):
    """An agent that returns null actions without contacting any server.

    Unlike ``PolicyAgent``, this needs neither a websocket nor a broker — it lets
    you exercise the runtime/env/save loop offline (no GPU policy server). It is
    the reason the transport lives behind the ``Agent`` interface rather than
    being baked into every agent.
    """

    def __init__(self, action_dim: int = _ACTION_DIM) -> None:
        self._action_dim = action_dim

    @override
    def get_action(self, observation: Observation) -> Action:
        return Action(
            step=observation.step,
            action=np.zeros(self._action_dim, dtype=np.float32),
            action_chunk_index=None,
            index_in_chunk=None,
        )

    @override
    def reset(self) -> None:
        pass
