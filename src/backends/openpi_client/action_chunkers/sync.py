from openpi_client.schemas import Observation
from openpi_client.action_chunkers.action_chunk_broker import ActionChunkBroker
from openpi_client.client import BidirectionalWebsocket
from typing_extensions import override


class SyncBroker(ActionChunkBroker):
    """Streams observations continuously but gates re-inference until current chunk is exhausted.

    The server will not re-infer until the previous chunk has been fully executed, equivalent to
    the original synchronous one-at-a-time behavior but without blocking the client.
    """

    def __init__(
        self, ws_client: BidirectionalWebsocket, control_hz: int, realtime: bool = True, execution_horizon: int = 0
    ):
        server_action_horizon = ws_client.server_metadata.action_horizon
        resolved_execution_horizon = server_action_horizon if execution_horizon <= 0 else int(execution_horizon)
        assert 1 <= resolved_execution_horizon <= server_action_horizon
        super().__init__(
            ws_client=ws_client,
            control_hz=control_hz,
            realtime=realtime,
            execution_horizon=resolved_execution_horizon,
        )

    @override
    def _infer(self, obs: Observation) -> None:
        if len(self._action_queue) > 0:
            return

        self._ws_client.send(obs, self.deadline, self._next_action_step, execution_horizon=self.execution_horizon)
