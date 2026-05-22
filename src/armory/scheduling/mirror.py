"""
Mirror of robot state on the scheduler side.

Two parallel sequences track a robot's progress:

- Control steps are the robot's discrete clock ticks at ``control_hz``. Each
  ``ControlStep`` records the wall time of the tick, the observation captured
  at it (``observation_step``, monotonically increasing), and which action
  index — if any — was executed at that tick (``action_step``).
- Action indexes are positions in the global, monotonically increasing
  sequence of actions produced by inference. Each ``ActionChunk`` covers
  ``[action_index_start, action_index_start + max_execution_horizon)``.

The two sequences are decoupled: a control step may execute no action (when
the next action index is not yet available on the robot), and a single chunk
spans many control steps. ``next_action_step`` on a control step is the
action index the robot will try to execute on its next tick.

Search and production both mutate Mirror through the same primitives:

- ``get_chunks`` builds (without queueing) anticipated chunks for a batch
  dispatched at ``dispatch_time``.
- ``fast_forward(time, robot_ids, chunks)`` queues those chunks and advances
  every robot's clock to ``time``.
"""

from __future__ import annotations

import itertools
import logging
from itertools import pairwise
import time
from collections import deque
from dataclasses import dataclass, field, replace

from armory.scheduling.latency import LatencyTracker
from armory.serving.schemas import (
    AckNotification,
    ActionChunk,
    ResponseBatch,
    RobotID,
    SlotRequest,
)
from armory_client.messages import InferResponse

logger = logging.getLogger(__name__)


@dataclass
class ControlStep:
    time: float
    observation_step: int
    action_step: int | None  # which action index was executed at this step
    next_action_step: int


@dataclass
class ChunkContext:
    observation_step: int  # step when observation was captured
    action_index_start: int  # action index of the first action in the chunk
    min_execution_horizon: int
    max_execution_horizon: int
    arrival_time: float  # estimated/actual time the chunk lands on the robot
    execution_start_step: int = 0  # client step when new chunk became available
    first_executed_index: int = 0  # index within chunk where actual execution started
    debug_info: dict[str, Any] = field(default_factory=dict)

class Robot:
    """Mirror of a single robot's control steps and action chunks.

    The first ``step()`` after construction seeds the robot's clock from the
    request's ``observation_step`` and ``action_index_start`` — these need
    not be zero. A per-robot reset on the scheduler side wipes the mirror's
    Robot, but the real client keeps incrementing its own counters; when it
    re-registers we just pick up from wherever it is.
    """

    def __init__(
        self,
        robot_id: RobotID,
        control_hz: float,
        min_execution_horizon: int,
        max_execution_horizon: int,
        latency_tracker: LatencyTracker,
    ):
        # NOTE: saving some things here for convenience, pattern is quite bad
        self.robot_id = robot_id
        self.control_hz = control_hz
        self.min_execution_horizon = min_execution_horizon
        self.max_execution_horizon = max_execution_horizon
        self.latency_tracker = latency_tracker

        # Both lists are sorted increasing by time by assertion.
        self.steps: deque[ControlStep] = deque(maxlen=30)
        # Includes chunks that are in-transit.
        self.chunks: deque[ActionChunk] = deque(maxlen=5)
        self.last_request: SlotRequest | None = None

    def step(self, request: SlotRequest) -> bool:
        if not self.steps:
            self.min_execution_horizon = request.min_execution_horizon
            self.max_execution_horizon = request.max_execution_horizon
            control_step = ControlStep(
                time=request.request_timestamp,
                observation_step=request.observation_step,
                action_step=None,
                next_action_step=request.action_index_start,
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
                logger.warning(f"self.steps: {self.steps}")
                logger.warning(f"request: {request}")
                return False
            stepped_forward = request.action_index_start > self.last_request.action_index_start
            action_step = self.last_request.action_index_start if stepped_forward else None
            next_action_step = (
                action_step + 1 if action_step is not None else latest.next_action_step
            )
            control_step = ControlStep(
                time=request.request_timestamp,
                observation_step=request.observation_step,
                action_step=action_step,
                next_action_step=next_action_step,
            )

        self.steps.append(control_step)
        self.assert_step_consistency()
        self.last_request = request
        return True

    def queue_chunk(self, chunk: ActionChunk) -> None:
        self.chunks.append(chunk)
        self.assert_consistency()

    def apply_response(self, chunk_id: int, response: InferResponse, arrival_time: float) -> None:
        """A queued chunk has come back from inference. Refresh the chunk and
        re-derive downstream chunks since this chunk's arrival time shifted."""
        i = self._find_index(chunk_id)
        self.chunks[i] = ActionChunk(
            chunk_id=chunk_id,
            observation_step=response.observation_step,
            action_index_start=response.action_index_start,
            min_execution_horizon=response.min_execution_horizon,
            max_execution_horizon=response.max_execution_horizon,
            arrival_time=arrival_time,
            origin="completed",
        )
        self._recompute_from(i)
        self.assert_consistency()

    def bump_arrival(self, chunk_id: int, arrival_time: float) -> None:
        """An in-flight chunk's projected arrival shifted (e.g. an upstream
        batch finished earlier/later than expected). Re-derive this chunk's
        execution fields from the new arrival, then cascade downstream."""
        i = self._find_index(chunk_id)
        self.chunks[i] = replace(self.chunks[i], arrival_time=arrival_time)
        self._recompute_from(i)
        self.assert_consistency()

    def apply_ack(self, ack: AckNotification) -> None:
        """Client confirmed receipt: trust the ack's fields exactly, re-derive
        downstream chunks from the actual receive time."""
        i = self._find_index(ack.chunk_id)
        self.chunks[i] = ActionChunk(
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
        self._recompute_from(i + 1)
        self.assert_consistency()

    def drop_chunk(self, chunk_id: int) -> None:
        i = self._find_index(chunk_id)
        del self.chunks[i]
        self._recompute_from(i)
        self.assert_consistency()

    def get_chunk(self, chunk_id: int) -> ActionChunk:
        return self.chunks[self._find_index(chunk_id)]

    def _find_index(self, chunk_id: int) -> int:
        for i, chunk in enumerate(self.chunks):
            if chunk.chunk_id == chunk_id:
                return i
        raise KeyError(f"chunk_id {chunk_id} not found")

    def _recompute_from(self, start: int) -> None:
        """Re-derive observation_step, action_index_start, execution_start_step,
        and first_executed_index for self.chunks[start:] using each chunk's
        already-set arrival_time. Preserves chunk_id, origin, arrival_time,
        and min/max_execution_horizon."""
        for i in range(start, len(self.chunks)):
            chunk = self.chunks[i]
            ctx = self._context_at_arrival(chunk.arrival_time)
            self.chunks[i] = replace(
                chunk,
                observation_step=ctx.observation_step,
                action_index_start=ctx.action_index_start,
                execution_start_step=ctx.execution_start_step,
                first_executed_index=ctx.first_executed_index,
            )

    def calculate_chunk_context(
        self, dispatch_time: float, arrival_time: float | None = None
    ) -> ChunkContext:
        # NOTE: assumes time has been simulated up until dispatch_time
        assert self.steps[-1].time + (1 / self.control_hz) > dispatch_time, f"time has not been simulated up until dispatch_time {dispatch_time}, steps: {self.steps}, next step time would be {self.steps[-1].time + (1 / self.control_hz)}"
        if arrival_time is None:
            arrival_time = dispatch_time + self.latency_tracker.action_latency(self.robot_id)

        obs_cutoff = dispatch_time - self.latency_tracker.observation_latency(self.robot_id)
        control_step = self.get_latest_control_step_before(obs_cutoff)

        observation_step = control_step.observation_step
        action_start_index = control_step.action_step if control_step.action_step is not None else control_step.next_action_step

        step = self.steps[-1]
        while step.time < arrival_time:
            step = self.advance_step(step)

        execution_start_step = step.observation_step
        first_executed_index = max(
            0,
            step.action_step - action_start_index
            if step.action_step is not None
            else step.next_action_step - action_start_index,
        )

        return ChunkContext(
            observation_step=observation_step,
            action_index_start=action_start_index,
            min_execution_horizon=self.min_execution_horizon,
            max_execution_horizon=self.max_execution_horizon,
            arrival_time=arrival_time,
            execution_start_step=execution_start_step,
            first_executed_index=first_executed_index,
            debug_info={
                "steps": [str(s) for s in self.steps],
                "dispatch_time": dispatch_time,
                "arrival_time": arrival_time,
                "control_step": control_step,
                "step": step,
                "execution_start_step": execution_start_step,
                "first_executed_index": first_executed_index,
            },
        )

    def _context_at_arrival(self, arrival_time: float) -> ChunkContext:
        """Build a chunk context for a chunk whose arrival_time is fixed.

        obs_cutoff is rolled back from arrival_time through both action and
        observation latency, mirroring how the GPU saw the world when it
        produced this chunk. When obs_cutoff falls past the latest real step
        (typical for in-flight chunks recomputed by ``bump_arrival``), simulate
        forward past it so ``action_index_start`` reflects what the robot's
        ``next_action_step`` will be at obs time — not whatever it is right
        now, which doesn't yet account for upstream queued chunks."""
        action_latency = self.latency_tracker.action_latency(self.robot_id)
        obs_cutoff = (
            arrival_time - action_latency - self.latency_tracker.observation_latency(self.robot_id)
        )

        step = self.steps[-1]
        if step.time >= obs_cutoff:
            control_step = self.get_latest_control_step_before(obs_cutoff)
        else:
            while True:
                next_step = self.advance_step(step)
                if next_step.time >= obs_cutoff:
                    break
                step = next_step
            control_step = step

        observation_step = control_step.observation_step
        action_start_index = control_step.action_step if control_step.action_step is not None else control_step.next_action_step

        while step.time < arrival_time:
            step = self.advance_step(step)

        execution_start_step = step.observation_step
        first_executed_index = max(
            0,
            step.action_step - action_start_index
            if step.action_step is not None
            else step.next_action_step - action_start_index,
        )

        return ChunkContext(
            observation_step=observation_step,
            action_index_start=action_start_index,
            min_execution_horizon=self.min_execution_horizon,
            max_execution_horizon=self.max_execution_horizon,
            arrival_time=arrival_time,
            execution_start_step=execution_start_step,
            first_executed_index=first_executed_index,
        )

    def assert_consistency(self) -> None:
        """Debug-time invariant checks. Remove the call sites once we're
        confident the producers can't violate them."""
        for prev, curr in pairwise(self.chunks):
            assert prev.action_index_start + prev.max_execution_horizon >= curr.action_index_start, f"Gap in chunks between {prev.chunk_id} and {curr.chunk_id}: {self.chunks}"
            # NOTE: I wanted to add this assert, but it fails when a completion/ack causes a queued chunk to become redundant
            # assert prev.action_index_start < curr.action_index_start, f"Backward chunks {prev.chunk_id} and {curr.chunk_id}: {self.chunks}"

    def assert_step_consistency(self) -> None:
        for prev, curr in pairwise(self.steps):
            if curr.action_step is not None and curr.action_step != prev.next_action_step:
                logger.warning(f"self.steps: {self.steps}")
                logger.warning(f"prev: {prev}")
                logger.warning(f"curr: {curr}")
                raise ValueError(
                    f"Gap in steps between {prev.observation_step} and {curr.observation_step}: {prev.next_action_step} != {curr.action_step}"
                )

    @property
    def max_overall_action_step(self) -> int:
        if not self.chunks:
            return -1
        return self.chunks[-1].action_index_start + self.chunks[-1].max_execution_horizon - 1

    def get_latest_control_step_before(self, time: float) -> ControlStep | None:
        for step in reversed(self.steps):
            if step.time < time:
                return step
        # NOTE: might be very wrong, but just returning first step as a hack
        return step

    def action_is_available(self, action_step: int, time: float) -> bool:
        # Chunks are sorted by action_index_start (assert_consistency forbids
        # gaps), so we can break once action_index_start exceeds action_step.
        for chunk in self.chunks:
            if chunk.action_index_start > action_step:
                break
            if (
                chunk.action_index_start + chunk.first_executed_index
                <= action_step
                <= chunk.action_index_start + chunk.max_execution_horizon - 1
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
        step = self.steps[-1]
        if step.next_action_step < self.max_overall_action_step:
            while step.next_action_step <= self.max_overall_action_step:
                step = self.advance_step(step)
            return step.time
        elif step.next_action_step == self.max_overall_action_step:
            for prev_step in reversed(self.steps):
                if prev_step.action_step is not None:
                    return step.time
                step = prev_step

            assert False, "should not happen"
        else:
            # next_action_step == max_overall + 1 is the legitimate "just executed
            # the final action of the last chunk, now idle" state; anything beyond
            # that means we skipped indices.
            if (
                step.next_action_step > self.max_overall_action_step + 1
                and self.max_overall_action_step != -1
            ):
                logger.warning(f"self.steps: {self.steps}")
                logger.warning(f"step: {step}")
                logger.warning(f"self.max_overall_action_step: {self.max_overall_action_step}")
                logger.warning(f"self.chunks: {self.chunks}")
                raise ValueError(
                    f"step.next_action_step {step.next_action_step} is greater than max_overall_action_step {self.max_overall_action_step}"
                )
            return step.time

    def starved_steps(self) -> int:
        if not self.steps:
            return 0
        total_steps = self.steps[-1].observation_step + 1
        executed_steps = self.steps[-1].next_action_step
        return total_steps - executed_steps

    def step_forward(self, time_end: float) -> None:
        # Hot path: inlines advance_step + action_is_available and exploits the
        # fact that both ``action_step`` (= prev next_action_step) and ``time``
        # are monotonic non-decreasing across iterations. A cursor advances
        # past chunks whose entire range is below the current action_step so
        # we don't rescan them every tick.
        steps = self.steps
        prev_step = steps[-1]
        if prev_step.time >= time_end:
            return

        dt = 1.0 / self.control_hz

        time = prev_step.time + dt
        observation_step = prev_step.observation_step + 1
        action_step = prev_step.next_action_step

        chunk_idx = 0
        while time < time_end:
            # Go to the latest chunk that has arrived by the current time
            while chunk_idx + 1 < len(self.chunks) and self.chunks[chunk_idx + 1].arrival_time <= time:
                chunk_idx += 1
            
            is_available = (chunk_idx < len(self.chunks) and self.chunks[chunk_idx].arrival_time <= time and action_step <= self.chunks[chunk_idx].last_action_index)

            current_action_step = action_step if is_available else None
            next_action_step = action_step + 1 if is_available else action_step

            steps.append(
                ControlStep(
                    time=time,
                    observation_step=observation_step,
                    action_step=current_action_step,
                    next_action_step=next_action_step,
                )
            )
            time += dt
            observation_step += 1
            action_step = next_action_step

    def next_chunk_start(self, cs: ControlStep) -> int:
        """Action index the next queued chunk should start at, given control step ``cs``."""
        return cs.action_step if cs.action_step is not None else cs.next_action_step

    def _clone_for_twin(self) -> Robot:
        """Cheap shallow clone for speculative twins.

        ``steps`` and ``chunks`` get fresh list objects so twin-side appends
        and replacements don't leak back, but the items themselves are shared:
        ``ActionChunk`` is frozen, and ``ControlStep`` is never mutated in
        place anywhere in this module."""
        twin = Robot.__new__(Robot)
        twin.robot_id = self.robot_id
        twin.control_hz = self.control_hz
        twin.min_execution_horizon = self.min_execution_horizon
        twin.max_execution_horizon = self.max_execution_horizon
        twin.latency_tracker = self.latency_tracker
        twin.steps = deque(self.steps)
        twin.chunks = deque(self.chunks)
        twin.last_request = self.last_request  # NOTE: bad hack
        return twin

    def to_dict(self, now: float) -> dict:
        step = self.steps[-1] if self.steps else None
        return {
            "control_hz": self.control_hz,
            "min_execution_horizon": self.min_execution_horizon,
            "max_execution_horizon": self.max_execution_horizon,
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


class Mirror:
    """Tracks GPU timing + wraps Robots"""

    def __init__(self, latency_tracker: LatencyTracker):
        self.robots: dict[RobotID, Robot] = {}
        self.latency_tracker = latency_tracker
        self.in_flight_batches: deque[Batch] = deque()
        self.last_batch_completed_time: float = 0.0
        self.chunk_id_counter = itertools.count(1)
        # Persists past fast_forward popping the batch out of in_flight_batches;
        # used by schedulers that want a "no back-to-back" view of the most
        # recent dispatch.
        self.last_queued_batch_robot_ids: tuple[RobotID, ...] = ()
        self.latest_fast_forward_time: float = 0.0

    @property
    def in_flight_batches_count(self) -> int:
        return len(self.in_flight_batches)

    def next_time_server_available(self) -> float:
        if len(self.in_flight_batches) == 0:
            # During simulation last_batch_completed_time is a future simulated time;
            # during production it's in the past so time.time() dominates.
            return max(time.time(), self.last_batch_completed_time)
        # Each batch's completion_time is already chained: queue_batch sets it to
        # next_time_server_available() + infer_lat at enqueue time, so the tail
        # of the queue is exactly when the server next becomes free.
        return self.in_flight_batches[-1].completion_time

    def reset_robot(self, robot_id: RobotID) -> None:
        if self.robots.pop(robot_id, None) is None:
            return
        # Strip the robot's chunks from any in-flight batches so that a late
        # ResponseBatch (or downstream bump_arrival) doesn't look up a chunk_id
        # that no longer exists on the robot. The batch entries themselves are
        # kept (even if they end up empty) so GPU timing accounting —
        # next_time_server_available, batch_id matching in
        # update_batch_completion — stays consistent with the work the GPU is
        # actually still doing.
        for in_flight in self.in_flight_batches:
            kept_robots: list[RobotID] = []
            kept_chunks: list[int] = []
            for rid, cid in zip(in_flight.robot_ids, in_flight.chunk_ids):
                if rid == robot_id:
                    continue
                kept_robots.append(rid)
                kept_chunks.append(cid)
            in_flight.robot_ids = kept_robots
            in_flight.chunk_ids = kept_chunks

    def clear_all(self) -> None:
        self.robots.clear()
        self.in_flight_batches.clear()
        self.last_batch_completed_time = 0.0
        self.chunk_id_counter = itertools.count(1)
        self.last_queued_batch_robot_ids = ()

    def receive_request(self, request: SlotRequest) -> bool:
        """Returns False if the request was dropped as stale by ``Robot.step``."""
        if request.robot_id not in self.robots:
            # NOTE: for now, assume control_hz and max_execution_horizon are fixed for a robot's lifetime
            self.robots[request.robot_id] = Robot(
                request.robot_id,
                request.control_hz,
                request.min_execution_horizon,
                request.max_execution_horizon,
                self.latency_tracker,
            )
        return self.robots[request.robot_id].step(request)

    def queue_batch(
        self,
        batch: list[RobotID],
        batch_id: int,
        *,
        origin: str = "queued",
    ) -> list[ActionChunk]:
        dispatch_time = self.next_time_server_available()
        twin = self.get_twin()
        twin.fast_forward(dispatch_time)

        infer_lat = self.latency_tracker.infer_latency(len(batch))

        chunks: list[ActionChunk] = []
        for robot_id in batch:
            # The twin holds the simulated state at dispatch_time; the real
            # robot's clock may still be behind, so calculate against the twin.
            arrival_time = dispatch_time + infer_lat + self.latency_tracker.action_latency(robot_id)
            chunk_context = twin.robots[robot_id].calculate_chunk_context(
                dispatch_time,
                arrival_time=arrival_time,
            )
            chunk = ActionChunk(
                chunk_id=next(self.chunk_id_counter),
                observation_step=chunk_context.observation_step,
                action_index_start=chunk_context.action_index_start,
                min_execution_horizon=chunk_context.min_execution_horizon,
                max_execution_horizon=chunk_context.max_execution_horizon,
                arrival_time=chunk_context.arrival_time,
                execution_start_step=chunk_context.execution_start_step,
                first_executed_index=chunk_context.first_executed_index,
                origin=origin,
                debug_info={
                    "dispatch_time": dispatch_time,
                    "infer_lat": infer_lat,
                    "action_latency": self.latency_tracker.action_latency(robot_id),
                    "arrival_time": arrival_time,
                    "chunk_context": chunk_context,
                },
            )
            self.robots[robot_id].queue_chunk(chunk)
            chunks.append(chunk)
        self.in_flight_batches.append(
            Batch(
                batch_id=batch_id,
                robot_ids=batch,
                chunk_ids=[c.chunk_id for c in chunks],
                completion_time=dispatch_time + infer_lat,
            )
        )
        self.last_queued_batch_robot_ids = tuple(batch)
        return chunks

    def queue_idle(self, duration: float, batch_id: int) -> None:
        """Register a synthetic batch that occupies the server for ``duration``
        without producing any chunks.

        Mirrors ``queue_batch``'s GPU-timing bookkeeping (an in-flight Batch with
        a chained ``completion_time``) so ``next_time_server_available`` reflects
        the idle window — both in production and inside speculative search twins,
        where it keeps downstream ``schedulable_robot_ids`` / ``deadlines`` from
        being computed at a too-early dispatch time."""
        dispatch_time = self.next_time_server_available()
        self.in_flight_batches.append(
            Batch(
                batch_id=batch_id,
                robot_ids=[],
                chunk_ids=[],
                completion_time=dispatch_time + duration,
            )
        )
        self.last_queued_batch_robot_ids = ()

    def update_batch_completion(self, batch: ResponseBatch) -> None:
        """Refine each chunk's arrival_time once GPU inference has completed."""
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

        responses_by_chunk = {response.chunk_id: response for response in batch.responses}
        for robot_id, chunk_id in zip(in_flight.robot_ids, in_flight.chunk_ids):
            robot = self.robots.get(robot_id)
            if robot is None:
                logger.debug("Ignoring completion for unknown robot: %s", robot_id)
                continue
            response = responses_by_chunk.get(chunk_id)
            if response is None:
                robot.drop_chunk(chunk_id)
            else:
                arrival_time = actual_completion + self.latency_tracker.action_latency(robot_id)
                robot.apply_response(chunk_id, response, arrival_time)

        # Re-chain downstream batches off the actual completion of the head batch.
        prev_completion = actual_completion
        for queued_batch in self.in_flight_batches:
            queued_batch.completion_time = prev_completion + self.latency_tracker.infer_latency(
                len(queued_batch.robot_ids)
            )
            for robot_id, chunk_id in zip(queued_batch.robot_ids, queued_batch.chunk_ids):
                robot = self.robots.get(robot_id)
                if robot is None:
                    logger.debug("Ignoring completion for unknown robot: %s", robot_id)
                    continue
                arrival_time = queued_batch.completion_time + self.latency_tracker.action_latency(
                    robot_id
                )
                robot.bump_arrival(chunk_id, arrival_time)
            prev_completion = queued_batch.completion_time

    def confirm_chunk(self, ack: AckNotification) -> None:
        robot = self.robots.get(ack.robot_id)
        if robot is None:
            logger.debug("Ignoring ack for unknown robot: %s", ack.robot_id)
            return
        # The chunk may have been wiped by an intervening reset_robot (and the
        # robot since re-registered with a fresh empty chunks list). Acks are
        # advisory arrival-time refinements, so dropping a stale one is safe.
        if not any(c.chunk_id == ack.chunk_id for c in robot.chunks):
            logger.debug(
                "Ignoring ack for unknown chunk: robot=%s chunk_id=%s",
                ack.robot_id,
                ack.chunk_id,
            )
            return
        robot.apply_ack(ack)

    def schedulable_robot_ids(
        self,
    ) -> list[RobotID]:
        schedulable_robot_ids: list[RobotID] = []

        twin = self.get_twin()
        dispatch_time = twin.next_time_server_available()
        twin.fast_forward(dispatch_time)
        for robot_id in self.robots.keys():
            robot = twin.robots[robot_id]
            anticipated_chunk = robot.calculate_chunk_context(dispatch_time)
            if len(robot.chunks) == 0 or robot.last_request.can_serve(
                robot.chunks[-1].action_index_start, anticipated_chunk.action_index_start
            ):
                schedulable_robot_ids.append(robot_id)

        return schedulable_robot_ids

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

        self.latest_fast_forward_time = max(self.latest_fast_forward_time, time)

    def get_twin(self) -> Mirror:
        """Shallow twin for speculative planning.

        ``in_flight_batches`` and each robot's ``steps`` / ``chunks`` lists
        are copied so twin-side mutations don't leak back, but the items
        inside are shared. ``Batch.completion_time`` is mutated only by
        ``update_batch_completion`` on the real Mirror, never via a twin,
        so sharing ``Batch`` references is safe. ``latency_tracker`` and
        ``chunk_id_counter`` are read/never-touched on the twin path."""
        twin = Mirror.__new__(Mirror)
        twin.latency_tracker = self.latency_tracker
        twin.in_flight_batches = deque(self.in_flight_batches)
        twin.last_batch_completed_time = self.last_batch_completed_time
        twin.chunk_id_counter = self.chunk_id_counter
        twin.robots = {rid: robot._clone_for_twin() for rid, robot in self.robots.items()}
        twin.last_queued_batch_robot_ids = self.last_queued_batch_robot_ids
        twin.latest_fast_forward_time = self.latest_fast_forward_time
        return twin

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
            "latest_fast_forward_time": self.latest_fast_forward_time,
        }
