"""Deterministic driver that runs ``ActionChunkBrokerCore`` against a scripted
timeline and records the events the broker would have emitted.

The output (``TraceRecord``) is what feeds both:

  * Unit tests that check broker logic in isolation.
  * Comparison tests that replay the same events into ``Mirror`` and assert
    the mirror's predicted state matches the broker's recorded state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from armory.scheduling import mirror
from armory.serving.schemas import AckNotification, SlotRequest
from armory_client.action_chunkers.action_chunk_broker_core import ActionChunkBrokerCore
from armory_client.messages import InferResponse, InferType
from armory_client.schemas import Observation

ACTION_DIM = 7


@dataclass(frozen=True)
class ServerResponseSpec:
    """One server response. ``triggered_by_obs_index`` indexes into ``ScenarioSpec.obs_times``."""

    triggered_by_obs_index: int
    arrival_time: float  # when the chunk arrives at the client (= mirror.ActionChunk.arrival_time)
    action_start_step: int
    execution_horizon: int
    actions: np.ndarray | None = None  # default zeros((horizon, ACTION_DIM))


@dataclass(frozen=True)
class ScenarioSpec:
    control_hz: float
    execution_horizon: int  # broker's declared execution_horizon (independent of chunk horizon)
    obs_times: list[float]
    responses: list[ServerResponseSpec]
    robot_id: str = "r0"


@dataclass(frozen=True)
class StepRecord:
    """One row per broker.infer() call."""

    time: float
    observation_step: int  # the obs.step the broker saw
    action_step: int | None  # action index emitted; None for null/pause
    next_action_step: int  # core.next_action_step *after* this step
    actions_left: int  # queue length recorded in core.actions_left_history


@dataclass(frozen=True)
class TraceEvent:
    """An entry in the merged event timeline. Lower ``order`` fires earlier at the same time."""

    time: float
    order: int
    kind: Literal["obs", "schedule", "ack"]
    payload: object  # SlotRequest | tuple[mirror.ActionChunk, str] | AckNotification


@dataclass
class TraceRecord:
    spec: ScenarioSpec
    slot_requests: list[SlotRequest] = field(default_factory=list)
    acks: list[AckNotification] = field(default_factory=list)
    server_chunks: list[mirror.ActionChunk] = field(default_factory=list)
    per_step_actions: list[StepRecord] = field(default_factory=list)
    events: list[TraceEvent] = field(default_factory=list)


def run_scenario(spec: ScenarioSpec) -> TraceRecord:
    """Run the broker core against ``spec`` and return everything observable.

    Event ordering rule when timestamps tie: a chunk that arrives at the same
    instant as an observation is delivered *before* the observation. This matches
    the real broker's behavior when the daemon thread happens to win the lock
    race; the alternative ordering is exercised by spacing arrivals slightly
    earlier in your scenario.
    """
    _validate(spec)

    core = ActionChunkBrokerCore(
        control_hz=spec.control_hz,
        execution_horizon=spec.execution_horizon,
    )
    trace = TraceRecord(spec=spec)

    pending_responses = sorted(spec.responses, key=lambda r: r.arrival_time)
    pending_idx = 0
    request_id_counter = 0

    for obs_idx, obs_time in enumerate(spec.obs_times):
        # Deliver any chunks that have arrived since the last observation.
        while (
            pending_idx < len(pending_responses)
            and pending_responses[pending_idx].arrival_time <= obs_time
        ):
            response_spec = pending_responses[pending_idx]
            _deliver_response(
                core,
                trace,
                spec,
                response_spec,
                request_id=response_spec.triggered_by_obs_index + 1,
            )
            pending_idx += 1

        # Issue the observation.
        obs = Observation(
            state=np.zeros(1),
            step=obs_idx,
            image=np.zeros((1, 1, 3)),
            wrist_image=np.zeros((1, 1, 3)),
        )
        action, event = core.on_observation(obs, now=obs_time)

        request_id_counter += 1
        slot_request = SlotRequest(
            slot_index=0,
            robot_id=spec.robot_id,
            request_id=request_id_counter,
            arrival_timestamp=event.request_timestamp,
            observation_step=event.observation_step,
            action_start_step=event.action_start_step,
            request_timestamp=event.request_timestamp,
            deadline=event.deadline,
            execution_horizon=event.execution_horizon,
            infer_type=InferType.SYNC,
            params=None,
            noise=None,
            control_hz=spec.control_hz,
        )
        trace.slot_requests.append(slot_request)
        trace.events.append(TraceEvent(time=obs_time, order=1, kind="obs", payload=slot_request))

        action_step = action.step if action.action_chunk_index is not None else None
        trace.per_step_actions.append(
            StepRecord(
                time=obs_time,
                observation_step=obs_idx,
                action_step=action_step,
                next_action_step=core.next_action_step,
                actions_left=core.actions_left_history[-1],
            )
        )

    # Deliver any responses that arrive after the last observation. The broker's
    # daemon thread would still process these; they show up in ``acks`` /
    # ``server_chunks`` but produce no per_step_actions entry.
    while pending_idx < len(pending_responses):
        response_spec = pending_responses[pending_idx]
        _deliver_response(
            core,
            trace,
            spec,
            response_spec,
            request_id=response_spec.triggered_by_obs_index + 1,
        )
        pending_idx += 1

    trace.events.sort(key=lambda e: (e.time, e.order))
    return trace


def _deliver_response(
    core: ActionChunkBrokerCore,
    trace: TraceRecord,
    spec: ScenarioSpec,
    response_spec: ServerResponseSpec,
    request_id: int,
) -> None:
    triggering_obs_idx = response_spec.triggered_by_obs_index
    request_timestamp = spec.obs_times[triggering_obs_idx]
    actions = (
        response_spec.actions
        if response_spec.actions is not None
        else np.zeros((response_spec.execution_horizon, ACTION_DIM))
    )
    infer_response = InferResponse(
        robot_id=spec.robot_id,
        request_id=request_id,
        observation_step=triggering_obs_idx,
        action_start_step=response_spec.action_start_step,
        request_timestamp=request_timestamp,
        actions=actions,
        execution_horizon=response_spec.execution_horizon,
        noise=None,
        server_arrival_time=request_timestamp,
        inference_start_time=request_timestamp,
        inference_end_time=response_spec.arrival_time,
        server_send_time=response_spec.arrival_time,
    )

    ack_event = core.on_infer_response(infer_response, now=response_spec.arrival_time)

    server_chunk_pending = mirror.ActionChunk(
        observation_step=triggering_obs_idx,
        arrival_time=response_spec.arrival_time,
        action_start_step=response_spec.action_start_step,
        execution_horizon=response_spec.execution_horizon,
        arrived=False,
    )
    trace.server_chunks.append(server_chunk_pending)

    ack = AckNotification(
        robot_id=spec.robot_id,
        request_id=ack_event.request_id,
        observation_step=triggering_obs_idx,
        receive_time=response_spec.arrival_time,
        server_send_time=response_spec.arrival_time,
    )
    trace.acks.append(ack)

    # Schedule fires at request time (server "sent" the pending chunk at
    # inference end); ack fires at arrival_time. In our deterministic model the
    # server has zero compute latency, so schedule_time == request_timestamp.
    trace.events.append(
        TraceEvent(
            time=request_timestamp,
            order=2,
            kind="schedule",
            payload=(server_chunk_pending, spec.robot_id),
        )
    )
    trace.events.append(
        TraceEvent(time=response_spec.arrival_time, order=0, kind="ack", payload=ack)
    )


def _validate(spec: ScenarioSpec) -> None:
    assert spec.obs_times == sorted(spec.obs_times), "obs_times must be monotonic"
    assert all(0 <= r.triggered_by_obs_index < len(spec.obs_times) for r in spec.responses), (
        "triggered_by_obs_index out of range"
    )
    # Mirror keys ack→chunk by observation_step, so each response must come from a unique obs.
    triggering = [r.triggered_by_obs_index for r in spec.responses]
    assert len(triggering) == len(set(triggering)), (
        "each obs may trigger at most one response (mirror maps acks→chunks via observation_step)"
    )
    for r in spec.responses:
        assert r.arrival_time >= spec.obs_times[r.triggered_by_obs_index], (
            "response arrival_time must be >= triggering obs time"
        )
