import itertools
import logging
import multiprocessing as mp
import time
from abc import ABC, abstractmethod
from typing import Any

from armory.scheduling.latency import EMALatencyTracker
from armory.scheduling.mirror import Mirror
from armory.serving.schemas import (
    AckNotification,
    RequestBatch,
    ResponseBatch,
    RobotID,
    SchedulerDecision,
    SlotRequest,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class RequestScheduler(ABC):
    def __init__(
        self,
        batch_queue: mp.Queue,
        max_batch_size: int = 1,
        min_execution_horizon: int = 0,
    ):
        self._batch_queue = batch_queue
        self._max_batch_size = max_batch_size
        # Mirror the engine's _should_serve gate. The engine drops a request
        # whose action_index_start is not at least min_execution_horizon past what was last
        # served; if the scheduler doesn't apply the same gate, it keeps
        # emitting batches the engine will reject. Each rejected batch returns
        # an empty ResponseBatch which still pops in_flight, freeing the
        # GreedyDeadline gate to emit again — a tight loop that buries the
        # GPU's batch_queue.
        self._min_ex = min_execution_horizon

        self.latency_tracker = EMALatencyTracker()
        self.mirror = Mirror(self.latency_tracker)
        self._latest_requests: dict[RobotID, SlotRequest] = {}  # TODO: clean up later

        self.next_batch_id = itertools.count(1)
        self._in_flight = 0

    def update(self, request: SlotRequest) -> None:
        self.latency_tracker.update_obs(
            request.robot_id, request.arrival_timestamp, request.request_timestamp
        )
        accepted = self.mirror.receive_request(request, request.control_hz)
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
        candidates = self.mirror.schedulable_requests(self._latest_requests, min_ex=self._min_ex)
        candidate_ids = [r.robot_id for r in candidates]
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
        logger.debug(
            "schedule stage=get_next_batches_done batches=%d mode=%s",
            len(batches),
            notes.get("mode") if isinstance(notes, dict) else None,
        )

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
        self.mirror.robots.clear()
        self.mirror.in_flight_batches.clear()
        self.mirror.last_batch_completed_time = 0.0
        # Latency tracker keeps its profile — that's still valid across trials.
