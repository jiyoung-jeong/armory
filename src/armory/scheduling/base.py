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


class RequestScheduler(ABC):
    def __init__(
        self,
        batch_queue: mp.Queue,
        max_batch_size: int = 1,
    ):
        self._batch_queue = batch_queue
        self._max_batch_size = max_batch_size

        self.latency_tracker = EMALatencyTracker()
        self.mirror = Mirror(self.latency_tracker)
        self._latest_requests: dict[RobotID, SlotRequest] = {}  # TODO: clean up later

        self.next_batch_id = itertools.count(1)
        self._in_flight = 0

    def update(self, request: SlotRequest) -> None:
        self.latency_tracker.update_obs(
            request.robot_id, request.arrival_timestamp, request.request_timestamp
        )
        self.mirror.receive_request(request, request.control_hz)
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
        next_avail = self.mirror.next_time_server_available()
        in_flight = self.mirror.in_flight_batches_count
        candidates = self.mirror.schedulable_requests(self._latest_requests)
        candidate_ids = [r.robot_id for r in candidates]
        deadlines = self.mirror.deadlines() if self.mirror.robots else {}

        batches, notes = self.get_next_batches(candidates)

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
