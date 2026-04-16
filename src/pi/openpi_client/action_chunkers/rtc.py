from openpi_client.schemas import Observation
from openpi_client.action_chunkers.action_chunk_broker import ActionChunkBroker
from openpi_client.client import BidirectionalWebsocket
from typing_extensions import override
from openpi_client import messages


class InferenceTimeRTCBroker(ActionChunkBroker):
    def __init__(
        self, ws_client: BidirectionalWebsocket, control_hz: int, realtime: bool = True, execution_horizon: int = 0
    ):
        """
        Args:
            ws_client: the websocket client to use for inference
            control_hz: the control frequency of the environment
            realtime: whether to run in realtime mode, setting this False essentially means inference latency is 0
            execution_horizon: how many steps in the predicted chunk the robot is willing to execute
        """
        super().__init__(
            ws_client=ws_client, control_hz=control_hz, realtime=realtime, execution_horizon=execution_horizon
        )

    @override
    def _infer(self, obs: Observation) -> None:
        self._ws_client.send(
            obs,
            self.deadline,
            self._next_action_step,
            infer_type=messages.InferType.INFERENCE_TIME_RTC,
            execution_horizon=self.execution_horizon,
        )
