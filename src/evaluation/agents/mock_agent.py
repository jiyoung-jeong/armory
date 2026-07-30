from collections.abc import Callable

from typing_extensions import override

from armory_client.schemas import Action, ActionChunk, Observation
from evaluation.agents.base import Agent


class MockAgent(Agent):
    """An agent that returns null actions without contacting any server.

    Unlike ``PolicyAgent``, this needs neither a websocket nor a broker — it lets
    you exercise the runtime/env/save loop offline (no GPU policy server). It is
    the reason the transport lives behind the ``Agent`` interface rather than
    being baked into every agent.
    """

    def __init__(
        self, create_null_action: Callable[[Observation, ActionChunk | None], Action]
    ) -> None:
        self._create_null_action = create_null_action

    @override
    def get_action(self, observation: Observation) -> Action:
        return self._create_null_action(observation, None)

    @override
    def reset(self) -> None:
        pass

    @override
    def close(self) -> None:
        pass

    @property
    @override
    def action_chunks(self) -> tuple[ActionChunk, ...]:
        return ()
