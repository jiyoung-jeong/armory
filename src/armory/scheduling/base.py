import itertools
import gc
import logging
import multiprocessing as mp
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

from armory.scheduling.latency import EMALatencyTracker

# Swap between Mirror and SimpleMirror here while mirror.py is being fixed.
from armory.scheduling.mirror import Mirror

# from armory.scheduling.simple_mirror import SimpleMirror as Mirror
from armory.serving.schemas import (
    AckNotification,
    RequestBatch,
    ResponseBatch,
    RobotID,
    SchedulerDecision,
    SlotRequest,
)

logger = logging.getLogger(__name__)


class RequestScheduler(ABC):
    def __init__(self, batch_queue: mp.Queue, max_batch_size: int = 1):
        self._batch_queue = batch_queue
        self._max_batch_size = max_batch_size

        self.latency_tracker = EMALatencyTracker()
        self.mirror = Mirror(self.latency_tracker)
        self._latest_requests: dict[RobotID, SlotRequest] = {}  # TODO: clean up later

        self.next_batch_id = itertools.count(1)
        self._in_flight = 0
        self._drain_fn: Callable[[], None] | None = None
        # gc.disable()

    def update(self, request: SlotRequest) -> None:
        self.latency_tracker.update_obs(
            request.robot_id, request.arrival_timestamp, request.request_timestamp
        )
        accepted = self.mirror.receive_request(request)
        if accepted:
            self._latest_requests[request.robot_id] = request

    def on_batch_completed(self, batch: ResponseBatch) -> None:
        self.latency_tracker.update_infer(batch.batch_size, batch.inference_duration)
        self.mirror.update_batch_completion(batch)

    def update_ack(self, notification: AckNotification) -> None:
        self.latency_tracker.update_action_delivery(
            notification.robot_id,
            notification.receive_time,
            notification.server_send_time,
        )
        self.mirror.confirm_chunk(notification)

    def schedule(self) -> list[SchedulerDecision]:
        """Run one decision pass: dispatch batches and emit a SchedulerDecision per call.

        Subclasses provide ``get_next_batches`` which returns the chosen batches
        plus a ``notes`` dict of algorithm-specific debug info; the base class
        snapshots common state (candidates, deadlines, mirror clock, slack) and
        wraps everything into ``SchedulerDecision`` records that flow into the
        metrics store.
        """
        started_at = time.time()
        logger.debug("schedule stage=mirror_next_avail")
        next_avail = self.mirror.next_time_server_available()
        logger.debug("schedule stage=mirror_in_flight_count")
        in_flight = self.mirror.in_flight_batches_count
        logger.debug(
            "schedule stage=mirror_schedulable latest_requests=%d", len(self._latest_requests)
        )
        candidate_ids = self.mirror.schedulable_robot_ids()
        candidates = [self._latest_requests[robot_id] for robot_id in candidate_ids]
        logger.debug("schedule stage=mirror_deadlines robots=%d", len(self.mirror.robots))
        deadlines = self.mirror.deadlines() if self.mirror.robots else {}

        logger.debug(
            "schedule stage=enter candidates=%d in_flight=%d slack=%+.3fs latest_requests=%d",
            len(candidates),
            in_flight,
            next_avail - started_at,
            len(self._latest_requests),
        )

        batches, notes = self.get_next_batches(candidates)
        post_return = time.time()
        logger.debug(
            "schedule stage=get_next_batches_done batches=%d mode=%s",
            len(batches),
            notes.get("mode") if isinstance(notes, dict) else None,
        )

        dispatch_start = time.time()
        # Phases recorded by the inner scheduler use timestamps captured before
        # `return`. The window between the last in-function phase and now covers
        # function return + local-variable dealloc (which can be slow when the
        # search held large frontiers) + this logger.debug. Surface it.
        if isinstance(notes, dict):
            phases = notes.setdefault("phases", [])
            last_end = max((float(p.get("end", 0.0)) for p in phases), default=0.0)
            if 0.0 < last_end < post_return:
                phases.append({"name": "return_overhead", "start": last_end, "end": post_return})
        decisions: list[SchedulerDecision] = []
        for batch in batches:
            batch_id = next(self.next_batch_id)
            chunks = self.mirror.queue_batch([slot.robot_id for slot in batch], batch_id)
            self._batch_queue.put_nowait(
                RequestBatch(
                    requests=batch,
                    chunk_ids=[chunk.chunk_id for chunk in chunks],
                    batch_id=batch_id,
                )
            )
            logger.debug(
                "schedule stage=dispatched batch_id=%d size=%d robots=%s",
                batch_id,
                len(batch),
                [slot.robot_id for slot in batch],
            )
            decisions.append(
                SchedulerDecision(
                    scheduler_name=type(self).__name__,
                    started_at=started_at,
                    duration=time.time() - started_at,
                    next_server_available=next_avail,
                    in_flight_batches=in_flight,
                    candidates=candidate_ids,
                    deadlines=dict(deadlines),
                    batch_id=batch_id,
                    scheduled=[slot.robot_id for slot in batch],
                    notes=dict(notes),
                )
            )

        if batches and isinstance(notes, dict):
            # gc_start = time.time()
            # gc.collect()
            # gc_end = time.time()

            phases = notes.setdefault("phases", [])
            # phases.append({"name": "gc", "start": gc_start, "end": gc_end})
            phases.append({"name": "dispatch", "start": dispatch_start, "end": time.time()})

        if not decisions:
            # Always emit at least one record per call so empty-batch ticks
            # (no_requests, dispatch_budget==0, search-with-no-commit, etc.)
            # show up in metrics instead of silently vanishing.
            decisions.append(
                SchedulerDecision(
                    scheduler_name=type(self).__name__,
                    started_at=started_at,
                    duration=time.time() - started_at,
                    next_server_available=next_avail,
                    in_flight_batches=in_flight,
                    candidates=candidate_ids,
                    deadlines=dict(deadlines),
                    batch_id=None,
                    scheduled=[],
                    notes=dict(notes) if isinstance(notes, dict) else {},
                )
            )

        return decisions

    @abstractmethod
    def get_next_batches(
        self, candidates: list[SlotRequest]
    ) -> tuple[list[list[SlotRequest]], dict[str, Any]]:
        """Return (batches_to_dispatch, debug_notes) for the current tick.

        ``candidates`` is the list of robots the mirror considers schedulable
        right now. Subclasses may consult additional state (latency tracker,
        in-flight batches, etc.) but should treat ``candidates`` as the
        authoritative pool to draw from.
        """
        ...

    def reset_robot(self, robot_id: str) -> None:
        self._latest_requests.pop(robot_id, None)
        self.mirror.reset_robot(robot_id)
        # self.latency_tracker.clear(robot_id)

    def reset_all(self) -> None:
        """Drop all scheduler + mirror state. For use on /reset between trials.

        Per-robot ResetRequests (sent on websocket close) only clear per-robot
        mirror state. They leave ``in_flight_batches`` and
        ``last_batch_completed_time`` intact — which means the GreedyDeadline
        gate ``in_flight_batches_count > 0`` stays tripped after a trial ends,
        and the next trial sees zero scheduling decisions.
        """
        self._latest_requests.clear()
        self.mirror.clear_all()
        self.latency_tracker.clear_all()
