from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any, Literal, NamedTuple, TypeAlias

import numpy as np

from armory.serving.config import ServerConfig
from armory.serving.rtc import InferType, RTCParams
from armory_client.messages import InferResponse, ResponseAck

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
    weight: float = 1.0

    def can_serve(self, last_action_index_start: int, anticipated_action_index_start: int) -> bool:
        return (
            anticipated_action_index_start >= last_action_index_start + self.min_execution_horizon
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class SlotData(SlotRequest):
    """Observation and request metadata written together atomically into a slot.

    This ensures that when the GPU worker reads a slot, the metadata (timestamps,
    step, etc.) always corresponds to the observation being inferred, even if the
    slot was overwritten by a newer request after the SlotRequest was enqueued.
    """

    observation: dict

    @property
    def request(self) -> SlotRequest:
        return SlotRequest(**{f.name: getattr(self, f.name) for f in fields(SlotRequest)})


@dataclass(frozen=True, slots=True)
class AckNotification:
    """Sent from WS to scheduler when a client acks receipt of an InferResponse."""

    ack: ResponseAck
    robot_id: RobotID
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
    """Server-internal: adopt a new ServerConfig in both subprocesses.

    Published from the WS main process on POST /reconfigure; the scheduler and
    the GPU worker both subscribe, and each applies the part it owns. The
    scheduler constructs a fresh ``RequestScheduler`` from
    ``SCHEDULER_REGISTRY[config.scheduler.scheduling_algorithm]`` and swaps it
    in. The previously-seeded batch latency profile is re-applied to the new
    instance; per-robot latency state is left to be re-seeded by the next
    warmup phase. The GPU worker records the config; its own knobs
    (``max_batch_size``, ``engine.num_steps``) are consumed at warmup and
    policy-construction time, so a change to them needs a restart.
    """

    config: ServerConfig


@dataclass(frozen=True, slots=True)
class PrepareScheduler:
    """Start an acknowledged, between-run scheduler transition."""

    operation_id: str
    config: ServerConfig


@dataclass(frozen=True, slots=True)
class PrepareGpu:
    """GPU queue barrier inserted after all work from the previous run."""

    operation_id: str


@dataclass(frozen=True, slots=True)
class GpuPrepared:
    """GPU→scheduler barrier acknowledgment, ordered after old completions."""

    operation_id: str


@dataclass(frozen=True, slots=True)
class PrepareAck:
    """Worker acknowledgment returned to the HTTP process."""

    operation_id: str
    worker: Literal["scheduler", "gpu", "router"]
    error: str | None = None


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
    def from_ack(cls, ack: ResponseAck) -> ActionChunk:
        return cls(
            chunk_id=ack.chunk_id,
            observation_step=ack.observation_step,
            action_index_start=ack.action_index_start,
            min_execution_horizon=ack.min_execution_horizon,
            max_execution_horizon=ack.max_execution_horizon,
            arrival_time=ack.receive_time,
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
