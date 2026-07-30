from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, NamedTuple, TypeAlias

import numpy as np

from armory.serving.protocol import SchedulerConfig
from armory.serving.rtc import InferType, RTCParams
from armory_client.messages import InferResponse

if TYPE_CHECKING:
    from armory.serving.slots import SlotData

RobotID: TypeAlias = str


@dataclass(frozen=True, slots=True)
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
    min_execution_horizon: int
    max_execution_horizon: int
    infer_type: InferType
    params: RTCParams | None
    noise: np.ndarray | None
    control_hz: float
    estimated_d_param: int = 0  # filled by scheduler before batching

    def can_serve(self, last_action_index_start: int, anticipated_action_index_start: int) -> bool:
        return (
            anticipated_action_index_start >= last_action_index_start + self.min_execution_horizon
        )


@dataclass(frozen=True, slots=True)
class AckNotification:
    """Sent from WS to scheduler when a client acks receipt of an InferResponse."""

    robot_id: RobotID
    request_id: int
    chunk_id: int
    observation_step: int
    action_index_start: int
    min_execution_horizon: int
    max_execution_horizon: int
    execution_start_step: int
    first_executed_index: int

    receive_time: float
    server_send_time: float


@dataclass(frozen=True, slots=True)
class BatchProfile:
    """Latency profile per batch size (seconds). Sent once from GPU to scheduler after warmup."""

    latencies: dict[int, float]


@dataclass(slots=True)
class WarmupSeed:
    robot_id: RobotID
    obs_samples: list[tuple[float, float]]  # (arrival_ts, request_ts) per ping
    delivery_samples: list[tuple[float, float]]  # (client_receive_time, server_send_time) per ack


@dataclass(frozen=True, slots=True)
class ResetAll:
    """Server-internal: drop all per-robot AND mirror-wide state.

    Sent on /reset so next run doesn't inherit
    in-flight batches, last_batch_completed_time, or stale per-robot mirror
    entries from the previous one. Per-robot ResetRequests sent on websocket
    close already cover the per-robot half; this covers the mirror-wide half.
    """


@dataclass(frozen=True, slots=True)
class Reconfigure:
    """Server-internal: rebuild the scheduler in-place from a new SchedulerConfig.

    Published from the WS main process on POST /reconfigure. The scheduler
    subprocess constructs a fresh ``RequestScheduler`` from
    ``SCHEDULER_REGISTRY[config.scheduling_algorithm]`` and swaps it in. The
    previously-seeded batch latency profile is re-applied to the new instance;
    per-robot latency state is left to be re-seeded by the next warmup phase.
    """

    config: SchedulerConfig


# TODO: this is the server-side version of ActionChunk that I think is used
# for the mirror. Remind me to discuss and also mention where it is used.
@dataclass(frozen=True, slots=True)
class ActionChunk:
    chunk_id: int
    observation_step: int  # step when observation was captured
    action_index_start: int  # action index of the first action in the chunk
    min_execution_horizon: int
    max_execution_horizon: int
    arrival_time: float  # estimated/actual time the chunk lands on the robot
    execution_start_step: int = 0  # client step when new chunk became available
    first_executed_index: int = 0  # index within chunk where actual execution started
    # Provenance/lifecycle tag. Transitions:
    #   "queued"    -> queued by production scheduler (arrival_time predicted)
    #   "searched"  -> queued inside a lookahead search snapshot (never confirmed)
    #   "completed" -> GPU returned the batch (arrival_time refined from real completion)
    #   "confirmed" -> robot acked receipt (arrival_time = actual receive_time)
    origin: str = "queued"
    debug_info: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_ack(cls, ack: AckNotification) -> ActionChunk:
        return cls(
            chunk_id=ack.chunk_id,
            observation_step=ack.observation_step,
            action_index_start=ack.action_index_start,
            min_execution_horizon=ack.min_execution_horizon,
            max_execution_horizon=ack.max_execution_horizon,
            execution_start_step=ack.execution_start_step,
            first_executed_index=ack.first_executed_index,
            origin="confirmed",
        )

    @property
    def last_action_index(self) -> int:
        """inclusive"""
        return self.action_index_start + self.max_execution_horizon - 1


@dataclass(frozen=True, slots=True)
class Idle:
    """Synthetic scheduler action: leave the GPU idle for ``duration`` seconds.

    Flows through the same dispatch path as a real batch — the scheduler queues
    it on the mirror (occupying server time without producing chunks) and the
    GPU worker sleeps for ``duration`` before returning an empty ResponseBatch.
    """

    duration: float


class RequestBatch(NamedTuple):
    requests: list[SlotRequest]
    chunk_ids: list[int]
    batch_id: int
    # > 0 marks a synthetic idle batch (empty ``requests``): the GPU sleeps this
    # long instead of inferring. See ``Idle``.
    idle_duration: float = 0.0


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
    (e.g. search nodes visited, slack budget, score components).
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


# TODO manual: think about how to name/define server-side dataclasses
# without being overly repetitive.
@dataclass(frozen=True, slots=True)
class InternalRequest:
    robot_id: str
    observation: dict
    observation_step: int
    action_index_start: int
    request_timestamp: float
    deadline: float
    min_execution_horizon: int
    max_execution_horizon: int
    infer_type: InferType
    params: RTCParams | None = None
    noise: np.ndarray | None = None  # action_horizon noise_dim
    type: Literal["infer"] = "infer"

    @classmethod
    def from_slot_data(cls, slot_data: SlotData, params: RTCParams | None) -> InternalRequest:
        return cls(
            robot_id=slot_data.robot_id,
            observation=slot_data.obs,
            observation_step=slot_data.observation_step,
            action_index_start=slot_data.action_index_start,
            request_timestamp=slot_data.request_timestamp,
            deadline=slot_data.deadline,
            min_execution_horizon=slot_data.min_execution_horizon,
            max_execution_horizon=slot_data.max_execution_horizon,
            infer_type=slot_data.infer_type,
            params=params,
        )
