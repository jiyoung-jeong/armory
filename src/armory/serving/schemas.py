from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, NamedTuple, TypeAlias

import numpy as np
from jaxtyping import Float

from armory_client.messages import (
    InferResponse,
    InferType,
    RTCParams,
    TrainTimeRTCParams,
    VlashParams,
)

if TYPE_CHECKING:
    from armory.serving.slots import SlotData

RobotID: TypeAlias = str


@dataclass(frozen=True)
class SlotRequest:
    """Flows end-to-end: built by WS → sent to Scheduler → put in batch_queue → received by GPU."""

    slot_index: int
    robot_id: RobotID
    request_id: int
    arrival_timestamp: float  # when WS received the request (server-side)
    observation_step: int
    action_index_start: int
    request_timestamp: float
    deadline: float
    execution_horizon: int
    infer_type: InferType
    params: RTCParams | VlashParams | TrainTimeRTCParams | None
    noise: np.ndarray | None
    control_hz: float
    estimated_d_param: int = 0  # filled by scheduler before batching
    is_padding: bool = False  # true for artificial slots used only to pad GPU batch size


@dataclass(frozen=True)
class AckNotification:
    """Sent from WS to scheduler when a client acks receipt of an InferResponse."""

    robot_id: RobotID
    request_id: int
    chunk_id: int
    observation_step: int
    receive_time: float
    server_send_time: float


@dataclass(frozen=True)
class BatchProfile:
    """Latency profile per batch size (seconds). Sent once from GPU to scheduler after warmup."""

    latencies: dict[int, float]


@dataclass
class WarmupSeed:
    robot_id: RobotID
    obs_samples: list[tuple[float, float]]  # (arrival_ts, request_ts) per ping
    delivery_samples: list[tuple[float, float]]  # (client_receive_time, server_send_time) per ack


# TODO: rename as ActionChunkMetadata
@dataclass(frozen=True)
class ActionChunk:
    chunk_id: int
    observation_step: int  # step when observation was captured
    arrival_time: float  # estimated/actual time the chunk lands on the robot
    action_index_start: int  # action index of the first action in the chunk
    execution_horizon: int
    arrived: bool = False


class RequestBatch(NamedTuple):
    requests: list[SlotRequest]
    chunks: list[ActionChunk]
    batch_id: int


class ResponseBatch(NamedTuple):
    responses: list[InferResponse]
    batch_id: int
    batch_size: int
    inference_start_time: float
    inference_duration: float


@dataclass
class SchedulerDecision:
    """A scheduler decision: a batch scheduling event."""

    scheduler_name: str
    metric_name: str
    duration: float
    recorded_at: float
    batch_id: int
    requests: list[dict] = field(default_factory=list)
    candidates: list[dict] = field(default_factory=list)
    scheduled: list[dict] = field(default_factory=list)

    @classmethod
    def from_json(cls, data: SchedulerDecision | dict) -> SchedulerDecision:
        if isinstance(data, cls):
            return data
        return cls(**data)


# TODO: copied over InferRequest, fix later
@dataclass(frozen=True)
class InternalRequest:
    robot_id: str
    observation: dict
    observation_step: int
    action_index_start: int
    request_timestamp: float
    deadline: float
    execution_horizon: int
    infer_type: InferType
    params: RTCParams | VlashParams | TrainTimeRTCParams | None = None
    noise: Float[np.ndarray, "action_horizon noise_dim"] | None = None
    type: str = "infer"  # FIXME: should be literal

    def __post_init__(self) -> None:
        if isinstance(self.infer_type, str):
            object.__setattr__(self, "infer_type", InferType(self.infer_type))

        if isinstance(self.params, dict):
            if self.infer_type == InferType.INFERENCE_TIME_RTC:
                object.__setattr__(self, "params", RTCParams(**self.params))

    @classmethod
    def from_slot_data(
        cls, slot_data: SlotData, params: RTCParams | VlashParams | TrainTimeRTCParams | None
    ) -> InternalRequest:
        return cls(
            robot_id=slot_data.robot_id,
            observation=slot_data.obs,
            observation_step=slot_data.observation_step,
            action_index_start=slot_data.action_index_start,
            request_timestamp=slot_data.request_timestamp,
            deadline=slot_data.deadline,
            execution_horizon=slot_data.execution_horizon,
            infer_type=slot_data.infer_type,
            params=params,
        )
