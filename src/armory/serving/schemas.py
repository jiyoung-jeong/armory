from __future__ import annotations

from dataclasses import dataclass, field
from typing import NamedTuple, TypeAlias

import numpy as np

from armory_client.messages import (
    InferResponse,
    InferType,
    RTCParams,
    TrainTimeRTCParams,
    VlashParams,
)

RobotID: TypeAlias = str


@dataclass(frozen=True)
class SlotRequest:
    """Flows end-to-end: built by WS → sent to Scheduler → put in batch_queue → received by GPU."""

    slot_index: int
    robot_id: str
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
class CompletionNotification:
    """Sent from GPU to scheduler after inference so the scheduler can update its state."""

    robot_id: str
    action_index_start: int
    request_id: int
    batch_size: int
    inference_duration: float
    observation_step: int
    execution_horizon: int
    server_arrival_time: float


@dataclass(frozen=True)
class AckNotification:
    """Sent from WS to scheduler when a client acks receipt of an InferResponse."""

    robot_id: str
    request_id: int
    observation_step: int
    receive_time: float
    server_send_time: float


@dataclass(frozen=True)
class BatchProfile:
    """Latency profile per batch size (seconds). Sent once from GPU to scheduler after warmup."""

    latencies: dict[int, float]


@dataclass
class WarmupSeed:
    robot_id: str
    obs_samples: list[tuple[float, float]]  # (arrival_ts, request_ts) per ping
    delivery_samples: list[tuple[float, float]]  # (client_receive_time, server_send_time) per ack


class RequestBatch(NamedTuple):
    requests: list[SlotRequest]
    batch_id: int


class ResponseBatch(NamedTuple):
    responses: list[InferResponse]
    batch_id: int
    batch_size: int | None = None


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
