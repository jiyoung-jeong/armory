from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, NamedTuple, TypeAlias

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
    action_index_start: int
    execution_horizon: int
    execution_start_step: int
    first_executed_index: int

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


@dataclass(frozen=True)
class ResetAll:
    """Server-internal: drop all per-robot AND mirror-wide state.

    Sent on /reset so next run doesn't inherit
    in-flight batches, last_batch_completed_time, or stale per-robot mirror
    entries from the previous one. Per-robot ResetRequests sent on websocket
    close already cover the per-robot half; this covers the mirror-wide half.
    """


# TODO: rename as ActionChunkMetadata
@dataclass(frozen=True)
class ActionChunk:
    chunk_id: int
    observation_step: int  # step when observation was captured
    action_index_start: int  # action index of the first action in the chunk
    execution_horizon: int
    arrival_time: float  # estimated/actual time the chunk lands on the robot
    execution_start_step: int = 0  # client step when new chunk became available
    first_executed_index: int = 0  # index within chunk where actual execution started
    # Provenance/lifecycle tag. Transitions:
    #   "queued"    -> queued by production scheduler (arrival_time predicted)
    #   "searched"  -> queued inside a lookahead search snapshot (never confirmed)
    #   "completed" -> GPU returned the batch (arrival_time refined from real completion)
    #   "confirmed" -> robot acked receipt (arrival_time = actual receive_time)
    origin: str = "queued"


class RequestBatch(NamedTuple):
    requests: list[SlotRequest]
    chunk_ids: list[int]
    batch_id: int


class ResponseBatch(NamedTuple):
    responses: list[InferResponse]
    batch_id: int
    batch_size: int
    inference_start_time: float
    inference_duration: float


@dataclass
class SchedulerDecision:
    """One pass of the scheduler's decision loop, recorded for debugging.

    Fields fall into three groups:
    - timing: when the decision started (`started_at`) and how long it took (`duration`),
    - state observed at decision time (mirror snapshot, `candidates`, `deadlines`,
      `next_server_available`, `in_flight_batches`),
    - outcome (`batch_id`, `scheduled`).

    `notes` is a free-form per-scheduler dict for algorithm-specific debug info
    (e.g. search nodes visited, slack budget, score components). The legacy
    `metric_name` defaults to "batch_scheduled" and is retained so older
    dashboard code keeps working.
    """

    scheduler_name: str
    started_at: float = 0.0
    duration: float = 0.0
    next_server_available: float = 0.0
    in_flight_batches: int = 0
    candidates: list[RobotID] = field(default_factory=list)
    deadlines: dict[RobotID, float] = field(default_factory=dict)
    batch_id: int | None = None
    scheduled: list[RobotID] = field(default_factory=list)
    notes: dict[str, Any] = field(default_factory=dict)
    metric_name: str = "batch_scheduled"

    @property
    def recorded_at(self) -> float:
        """Backwards-compat alias used by older snapshot code."""
        return self.started_at

    @classmethod
    def from_json(cls, data: SchedulerDecision | dict) -> SchedulerDecision:
        if isinstance(data, cls):
            return data
        # Tolerate older payloads where `recorded_at` was the canonical name.
        payload = dict(data)
        if "recorded_at" in payload and "started_at" not in payload:
            payload["started_at"] = payload.pop("recorded_at")
        else:
            payload.pop("recorded_at", None)
        # Drop fields the new schema no longer carries.
        payload.pop("requests", None)
        return cls(**payload)


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
