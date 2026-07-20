from typing_extensions import override

from armory_client import messages
from armory_client.action_chunkers.action_chunk_broker import ActionChunkBroker
from armory_client.client import BidirectionalWebsocket
from armory_client.schemas import Observation

# TODO: delete this file, inference strategy should be decided on connection initiation


class InferenceTimeRTCBroker(ActionChunkBroker):
    def __init__(
        self,
        ws_client: BidirectionalWebsocket,
        control_hz: int,
        realtime: bool = True,
        min_execution_horizon: int = 0,
        max_execution_horizon: int = 0,
        real: bool = False,
    ):
        """
        Args:
            ws_client: the websocket client to use for inference
            control_hz: the control frequency of the environment
            realtime: whether to run in realtime mode, setting this False essentially means inference latency is 0
            max_execution_horizon: how many steps in the predicted chunk the robot is willing to execute
            real: whether null actions should hold the observed robot state
        """
        super().__init__(
            ws_client=ws_client,
            control_hz=control_hz,
            realtime=realtime,
            min_execution_horizon=min_execution_horizon,
            max_execution_horizon=max_execution_horizon,
            real=real,
        )

    @override
    def _infer(self, obs: Observation) -> None:
        self._ws_client.send(
            obs,
            self.deadline,
            self._next_action_step,
            infer_type=messages.InferType.INFERENCE_TIME_RTC,
            min_execution_horizon=self.min_execution_horizon,
            max_execution_horizon=self.max_execution_horizon,
        )
