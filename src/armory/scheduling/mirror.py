"""
Mirror of robot state on the scheduler side.

Two parallel sequences track a robot's progress:

- Control steps are the robot's discrete clock ticks at ``control_hz``. Each
  ``ControlStep`` records the wall time of the tick, the observation captured
  at it (``observation_step``, monotonically increasing), and which action
  index — if any — was executed at that tick (``action_step``).
- Action indexes are positions in the global, monotonically increasing
  sequence of actions produced by inference. Each ``ActionChunk`` covers
  ``[action_index_start, action_index_start + execution_horizon)``.

The two sequences are decoupled: a control step may execute no action (when
the next action index is not yet available on the robot), and a single chunk
spans many control steps. ``next_action_step`` on a control step is the
action index the robot will try to execute on its next tick.

Search and production both mutate Mirror through the same primitives:

- ``get_chunks`` builds (without queueing) anticipated chunks for a batch
  dispatched at ``dispatch_time``.
- ``fast_forward(time, robot_ids, chunks)`` queues those chunks and advances
  every robot's clock to ``time``.

Because both ``Robot.steps`` and ``Robot.chunks`` are append-only during
simulation, search uses ``checkpoint`` / ``restore`` to revert mutations
instead of deep-copying the mirror per DFS frame.
"""

from __future__ import annotations

import itertools
import logging
import time
from collections import deque
from dataclasses import dataclass

from armory.scheduling.latency import LatencyTracker
from armory.serving.schemas import (
    AckNotification,
    ActionChunk,
    ResponseBatch,
    RobotID,
    SlotRequest,
)

logger = logging.getLogger(__name__)


@dataclass
class ControlStep:
    time: float
    observation_step: int
    action_step: int | None  # which action index was executed at this step
    next_action_step: int


# TODO: another source of inconsistency is that we do not directly verify chunks against control steps on acks, we may need to just track which observation step it arrived before
class Robot:
    """Mirror of a single robot's control steps and action chunks.

    Invariant: once constructed, callers must seed the robot with an initial
    control step (``observation_step=0``, ``action_index_start=0``) via
    ``step()`` before invoking any other method. ``Mirror.receive_request``
    enforces this by calling ``step()`` immediately after construction.
    """

    def __init__(self, control_hz: float, execution_horizon: int):
        self.control_hz = control_hz
        self.execution_horizon = execution_horizon

        # Both lists are sorted increasing by time by assertion.
        self.steps: list[ControlStep] = []
        # Includes chunks that are in-transit.
        self.chunks: list[ActionChunk] = []

    def step(self, request: SlotRequest) -> bool:
        """Apply ``request`` to the control-step history. Returns False if the
        request is stale and was dropped.

        ``request.action_index_start`` is the action the *robot* wants to execute
        next — it does not advance until the robot actually consumes a chunk and
        moves on. Across multiple observations while the robot waits, the same
        ``action_index_start`` repeats. So this method just records the robot's
        reported state; it does not infer execution from a "match" between the
        request and the mirror's prediction. Simulated execution happens in
        ``advance_step`` via ``fast_forward``, not here.

        Drops fire when the request would either:
        - revisit an already-seen ``observation_step`` (out-of-order delivery), or
        - move ``next_action_step`` backward (a real regression; the robot
          shouldn't ever ask for a lower action index than it last reported).
        """
        if not self.steps:
            assert request.action_index_start == 0
            control_step = ControlStep(
                time=request.request_timestamp,
                observation_step=request.observation_step,
                action_step=None,
                next_action_step=0,
            )
        else:
            latest = self.steps[-1]
            if request.observation_step <= latest.observation_step:
                logger.warning(
                    "Robot.step dropping out-of-order request: "
                    "obs_step=%d <= latest=%d (action_index_start=%d, next_action_step=%d)",
                    request.observation_step,
                    latest.observation_step,
                    request.action_index_start,
                    latest.next_action_step,
                )
                return False
            if request.action_index_start < latest.next_action_step:
                logger.warning(
                    "Robot.step dropping backward request: "
                    "action_index_start=%d < next_action_step=%d (obs_step=%d)",
                    request.action_index_start,
                    latest.next_action_step,
                    request.observation_step,
                )
                return False
            control_step = ControlStep(
                time=request.request_timestamp,
                observation_step=request.observation_step,
                action_step=None,
                next_action_step=request.action_index_start,
            )

        self.steps.append(control_step)
        return True

    def queue_chunk(self, chunk: ActionChunk) -> None:
        self.chunks.append(chunk)

        # NOTE: chunk check
        for prev, curr in zip(self.chunks[:-1], self.chunks[1:]):
            if prev.action_index_start + prev.execution_horizon < curr.action_index_start:
                raise ValueError(
                    f"Gap in chunks between {prev.chunk_id} and {curr.chunk_id}: {self.chunks}"
                )

    def update_chunk(
        self,
        new_chunk: ActionChunk,
    ) -> None:
        for i, chunk in enumerate(self.chunks):
            if chunk.chunk_id == new_chunk.chunk_id:
                self.chunks[i] = new_chunk
                return

        # NOTE: chunk check
        for prev, curr in zip(self.chunks[:-1], self.chunks[1:]):
            if prev.action_index_start + prev.execution_horizon < curr.action_index_start:
                raise ValueError(
                    f"Gap in chunks between {prev.chunk_id} and {curr.chunk_id}: {self.chunks}"
                )

        raise ValueError(f"chunk {new_chunk.chunk_id} not found")

    @property
    def max_overall_action_step(self) -> int:
        return self.chunks[-1].action_index_start + self.chunks[-1].execution_horizon - 1

    def get_latest_control_step_before(self, time: float) -> ControlStep | None:
        for step in reversed(self.steps):
            if step.time < time:
                return step
        return None

    def actions_executed(self) -> int:
        steps = [s for s in self.steps if s.action_step is not None]
        if not steps:
            return 0
        return steps[-1].action_step - steps[0].action_step + 1

    def action_is_available(self, action_step: int, time: float) -> bool:
        for chunk in self.chunks:
            if (
                chunk.action_index_start
                <= action_step
                <= chunk.action_index_start + chunk.execution_horizon - 1
            ) and chunk.arrival_time <= time:
                return True
        return False

    def advance_step(self, prev_step: ControlStep) -> ControlStep:
        """Assumes there are no gaps in action steps of chunks."""
        next_time = prev_step.time + 1 / self.control_hz
        action_step = (
            prev_step.next_action_step
            if self.action_is_available(prev_step.next_action_step, next_time)
            else None
        )
        return ControlStep(
            time=next_time,
            observation_step=prev_step.observation_step + 1,
            action_step=action_step,
            next_action_step=action_step + 1
            if action_step is not None
            else prev_step.next_action_step,
        )

    def deadline(self) -> float:
        if not self.chunks:
            # TODO: should be first step when robot ran out of actions
            return self.steps[-1].time

        step = self.steps[-1]
        while step.next_action_step <= self.max_overall_action_step:
            # # If no chunk covers next_action_step, advance_step can never
            # # increment it (action_is_available stays False forever) and the
            # # loop spins indefinitely. Treat the gap as the stall point and
            # # return the current step time.
            # # NOTE Rohan: hack from Claude. fix properly
            if not any(
                chunk.action_index_start
                <= step.next_action_step
                <= chunk.action_index_start + chunk.execution_horizon - 1
                for chunk in self.chunks
            ):
                return step.time
            step = self.advance_step(step)

        return step.time

    def step_forward(self, time: float) -> None:
        while self.steps[-1].time < time:
            self.steps.append(self.advance_step(self.steps[-1]))

    def next_chunk_start(self, cs: ControlStep) -> int:
        """Action index the next queued chunk should start at, given control step ``cs``."""
        return cs.action_step if cs.action_step is not None else cs.next_action_step

    def remove_chunk(self, chunk_id: int) -> None:
        for i, chunk in enumerate(self.chunks):
            if chunk.chunk_id == chunk_id:
                del self.chunks[i]
                return
        raise KeyError(f"chunk_id {chunk_id} not found")

    def to_dict(self, now: float) -> dict:
        step = self.steps[-1] if self.steps else None
        return {
            "control_hz": self.control_hz,
            "execution_horizon": self.execution_horizon,
            "n_steps": len(self.steps),
            "n_chunks": len(self.chunks),
            "action_index_range": (
                [self.chunks[0].action_index_start, self.max_overall_action_step]
                if self.chunks
                else None
            ),
            "last_step": (
                {
                    "time_rel": step.time - now,
                    "observation_step": step.observation_step,
                    "next_action_step": step.next_action_step,
                }
                if step
                else None
            ),
            "steps": [str(s) for s in self.steps],
            "chunks": [str(c) for c in self.chunks],
        }


@dataclass
class Batch:
    batch_id: int
    robot_ids: list[RobotID]
    chunk_ids: list[int]
    completion_time: float = 0.0  # estimated wall time when inference finishes

    @property
    def size(self) -> int:
        return len(self.robot_ids)


@dataclass(frozen=True)
class Checkpoint:
    """Snapshot of a Mirror's append-only state. Restore resets robot and batch state.

    Checkpoints must preserve the actual list contents, not only list lengths:
    search restores divergent branches whose chunk/control-step prefixes may
    have the same length but different values.
    """

    robot_states: dict[RobotID, tuple[tuple[ControlStep, ...], tuple[ActionChunk, ...]]]
    in_flight_batches: tuple[Batch, ...]
    last_batch_completed_time: float


class Mirror:
    def __init__(self, latency_tracker: LatencyTracker | None = None):
        self.robots: dict[RobotID, Robot] = {}
        # Optional so existing tests can construct a Mirror without a tracker;
        # methods that need it (get_chunks, update_completion) assert non-None.
        self.latency_tracker = latency_tracker
        self.in_flight_batches: deque[Batch] = deque()
        self.last_batch_completed_time: float = 0.0
        self.chunk_id_counter = itertools.count(1)

    @property
    def in_flight_batches_count(self) -> int:
        return len(self.in_flight_batches)

    def reset_robot(self, robot_id: RobotID) -> None:
        self.robots.pop(robot_id, None)

    def receive_request(self, request: SlotRequest, control_hz: float) -> bool:
        """Returns False if the request was dropped as stale by ``Robot.step``."""
        if request.robot_id not in self.robots:
            # NOTE: for now, assume control_hz and execution_horizon are fixed for a robot's lifetime
            self.robots[request.robot_id] = Robot(control_hz, request.execution_horizon)
        return self.robots[request.robot_id].step(request)

    def _next_chunk_context(self, rid: RobotID, dispatch_time: float) -> tuple[ControlStep, int]:
        # Caller must fast-forward the mirror to at least ``dispatch_time`` first.
        robot = self.robots[rid]
        obs_cutoff = dispatch_time - self.latency_tracker.observation_latency(rid)
        cs = robot.get_latest_control_step_before(obs_cutoff)
        assert cs is not None, (
            f"robot {rid} has no control step before {obs_cutoff}, first control step: {robot.steps[0].time}"
        )
        return cs, robot.next_chunk_start(cs)

    def queue_batch(
        self, batch: list[RobotID], batch_id: int, *, origin: str = "queued"
    ) -> list[ActionChunk]:
        assert self.latency_tracker is not None
        dispatch_time = self.next_time_server_available()
        infer_lat = self.latency_tracker.infer_latency(len(batch))

        ckpt = self.checkpoint()
        self.fast_forward(dispatch_time)
        contexts = [self._next_chunk_context(rid, dispatch_time) for rid in batch]
        self.restore(ckpt)

        chunks: list[ActionChunk] = []
        for rid, (cs, action_index_start) in zip(batch, contexts):
            robot = self.robots[rid]
            chunk = ActionChunk(
                chunk_id=next(self.chunk_id_counter),
                observation_step=cs.observation_step,
                arrival_time=dispatch_time + infer_lat + self.latency_tracker.action_latency(rid),
                action_index_start=action_index_start,
                execution_horizon=robot.execution_horizon,
                origin=origin,
            )
            chunks.append(chunk)
            robot.queue_chunk(chunk)
        self.in_flight_batches.append(
            Batch(
                batch_id=batch_id,
                robot_ids=batch,
                chunk_ids=[c.chunk_id for c in chunks],
                completion_time=dispatch_time + infer_lat,
            )
        )
        return chunks

    def update_batch_completion(self, batch: ResponseBatch) -> None:
        """Refine each chunk's arrival_time once GPU inference has completed."""
        assert self.latency_tracker is not None
        # Stale ResponseBatches can arrive from before a ResetAll: the GPU was
        # already mid-flight when /reset cleared in_flight_batches. Ignore them
        # — batch_ids are monotonic, so a mismatch means we're seeing the past.
        if not self.in_flight_batches or self.in_flight_batches[0].batch_id != batch.batch_id:
            logger.debug(
                "Ignoring stale ResponseBatch %s (head=%s)",
                batch.batch_id,
                self.in_flight_batches[0].batch_id if self.in_flight_batches else None,
            )
            return
        in_flight = self.in_flight_batches.popleft()

        actual_completion = batch.inference_start_time + batch.inference_duration
        self.last_batch_completed_time = actual_completion

        served_chunk_ids = set([response.chunk_id for response in batch.responses])
        for (
            robot_id,
            chunk_id,
        ) in zip(in_flight.robot_ids, in_flight.chunk_ids):
            robot = self.robots.get(robot_id)
            if robot is None:
                logger.debug("Ignoring completion for unknown robot: %s", robot_id)
                continue
            if chunk_id not in served_chunk_ids:
                robot.remove_chunk(chunk_id)
            else:
                # kind ugly
                infer_response = next(
                    response for response in batch.responses if response.chunk_id == chunk_id
                )
                chunk = ActionChunk(
                    chunk_id=chunk_id,
                    observation_step=infer_response.observation_step,
                    arrival_time=actual_completion + self.latency_tracker.action_latency(robot_id),
                    action_index_start=infer_response.action_index_start,
                    execution_horizon=infer_response.execution_horizon,
                    execution_start_step=0,
                    origin="completed",
                )
                robot.update_chunk(chunk)

    def confirm_chunk(self, ack: AckNotification) -> None:
        robot = self.robots.get(ack.robot_id)
        if robot is None:
            logger.debug("Ignoring ack for unknown robot: %s", ack.robot_id)
            return
        chunk = ActionChunk(
            chunk_id=ack.chunk_id,
            observation_step=ack.observation_step,
            arrival_time=ack.receive_time,
            action_index_start=ack.action_index_start,
            execution_horizon=ack.execution_horizon,
            execution_start_step=ack.execution_start_step,
            first_executed_index=ack.first_executed_index,
            origin="confirmed",
        )
        robot.update_chunk(chunk)

    def next_time_server_available(self) -> float:
        if len(self.in_flight_batches) == 0:
            # During simulation last_batch_completed_time is a future simulated time;
            # during production it's in the past so time.time() dominates.
            return max(time.time(), self.last_batch_completed_time)
        # Each batch's completion_time is already chained: queue_batch sets it to
        # next_time_server_available() + infer_lat at enqueue time, so the tail
        # of the queue is exactly when the server next becomes free.
        return self.in_flight_batches[-1].completion_time

    def schedulable_requests(
        self,
        requests: dict[RobotID, SlotRequest],
        min_execution_horizon: int = 0,
    ) -> list[SlotRequest]:
        """Filter requests whose next-chunk start is at least ``min_execution_horizon`` past
        the last queued chunk. Mirrors the engine's _should_serve gate so the
        scheduler doesn't emit batches the engine will drop.
        """
        schedulable_requests: list[SlotRequest] = []

        dispatch_time = self.next_time_server_available()
        # logger.debug(
        #     "Dispatch time: %f, last batch completed time: %f, current time: %f",
        #     dispatch_time,
        #     self.last_batch_completed_time,
        #     time.time(),
        # )
        ckpt = self.checkpoint()
        self.fast_forward(dispatch_time)
        for robot_id, request in requests.items():
            robot = self.robots[robot_id]

            # added by Rohan. sometimes client clock is slightly off on reset, so new robot's first step lands after obs_cutoff,
            # then get_latest_control_step_before returns None and it blows up. I added the below to skip this robot initially,
            # it will get re-scheduled in the future.
            obs_cutoff = dispatch_time - self.latency_tracker.observation_latency(robot_id)
            if robot.get_latest_control_step_before(obs_cutoff) is None:
                continue
            #

            _, action_index_start = self._next_chunk_context(robot_id, dispatch_time)
            if (
                len(robot.chunks) == 0
                or action_index_start > robot.chunks[-1].action_index_start + min_execution_horizon
            ):
                schedulable_requests.append(request)
            # else:
            #     logger.debug("Request %s is not schedulable", request.robot_id)
            #     logger.debug("Action index start: %d", action_index_start)
            #     logger.debug("Request action index start: %d", request.action_index_start)
            #     logger.debug("Dispatch time: %f", dispatch_time)
            #     logger.debug("Request: %s", request)
            #     logger.debug("Robot: %s", robot_id)
            #     logger.debug("Robot steps: %s", self.robots[robot_id].steps)
            #     logger.debug("Robot chunks: %s", self.robots[robot_id].chunks)
        self.restore(ckpt)
        return schedulable_requests

    # below are methods only used by search
    def fast_forward(
        self,
        time: float,
    ) -> None:
        """Advance every robot's clock to ``time`` and mark completed batches."""
        for robot in self.robots.values():
            robot.step_forward(time)

        while self.in_flight_batches and self.in_flight_batches[0].completion_time <= time:
            completed = self.in_flight_batches.popleft()
            self.last_batch_completed_time = completed.completion_time

    def checkpoint(self) -> Checkpoint:
        return Checkpoint(
            robot_states={
                rid: (tuple(robot.steps), tuple(robot.chunks)) for rid, robot in self.robots.items()
            },
            in_flight_batches=tuple(self.in_flight_batches),
            last_batch_completed_time=self.last_batch_completed_time,
        )

    def restore(self, ckpt: Checkpoint) -> None:
        for rid in list(self.robots.keys()):
            if rid not in ckpt.robot_states:
                del self.robots[rid]
                continue
            steps, chunks = ckpt.robot_states[rid]
            robot = self.robots[rid]
            robot.steps = list(steps)
            robot.chunks = list(chunks)
        self.in_flight_batches = deque(ckpt.in_flight_batches)
        self.last_batch_completed_time = ckpt.last_batch_completed_time

    def deadlines(self) -> dict[RobotID, float]:
        deadlines = {rid: robot.deadline() for rid, robot in self.robots.items()}
        return deadlines

    def to_dict(self) -> dict:
        now = time.time()
        return {
            "robots": {rid: robot.to_dict(now) for rid, robot in sorted(self.robots.items())},
            "in_flight_batches": [
                {
                    "batch_id": b.batch_id,
                    "robot_ids": b.robot_ids,
                    "chunk_ids": b.chunk_ids,
                    "completion_time_rel": b.completion_time - now,
                }
                for b in self.in_flight_batches
            ],
            "last_batch_completed_time_rel": (
                self.last_batch_completed_time - now if self.last_batch_completed_time > 0 else None
            ),
        }
