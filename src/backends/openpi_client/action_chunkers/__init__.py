from dataclasses import dataclass
from enum import Enum
from typing import Type

from openpi_client.client import BidirectionalWebsocket
from openpi_client.action_chunkers.action_chunk_broker import ActionChunkBroker
from openpi_client.action_chunkers.sync import SyncBroker
from openpi_client.action_chunkers.naive_async import NaiveAsyncBroker
from openpi_client.action_chunkers.rtc import InferenceTimeRTCBroker
from openpi_client.action_chunkers.temporal_ensembling import TemporalEnsemblingBroker


@dataclass
class BrokerConfig:
    """Configuration for action chunk brokers."""

    ws_client: BidirectionalWebsocket
    control_hz: int
    execution_horizon: int
    # Optional parameters for specific brokers
    m_param: float = 1.0  # For temporal_ensembling: exponential decay rate


# Mappings outside the enum to avoid conflicts
_CLASS_MAPPING = {
    "sync": SyncBroker,
    "rtc": InferenceTimeRTCBroker,
    "naive_async": NaiveAsyncBroker,
    "temporal_ensembling": TemporalEnsemblingBroker,
    # "vlash": VLashBroker,
}


class ActionChunkBrokerType(Enum):
    SYNC = "sync"
    NAIVE_ASYNC = "naive_async"
    RTC = "rtc"
    TEMPORAL_ENSEMBLING = "temporal_ensembling"
    # TODO: vlash

    def get_class(self) -> Type[ActionChunkBroker]:
        return _CLASS_MAPPING[self.value]

    def create(self, config) -> ActionChunkBroker:
        """Create broker from a config dataclass.

        Filters config parameters to only pass those accepted by the broker's __init__.
        """
        broker_class = self.get_class()
        config_dict = vars(config)

        # Get the broker's __init__ parameters
        import inspect

        sig = inspect.signature(broker_class.__init__)
        valid_params = set(sig.parameters.keys()) - {"self"}

        # Filter config to only include valid parameters
        filtered_config = {k: v for k, v in config_dict.items() if k in valid_params}

        return broker_class(**filtered_config)

    @classmethod
    def from_string(cls, value: str) -> "ActionChunkBrokerType":
        """Get enum member by value."""
        return cls(value.lower())
