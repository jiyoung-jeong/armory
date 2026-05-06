import itertools
import logging
import multiprocessing as mp
import time
from abc import ABC, abstractmethod

from armory.scheduling.latency import EMALatencyTracker
from armory.scheduling.mirror import Mirror
from armory.serving.schemas import (
    AckNotification,
    CompletionNotification,
    RequestBatch,
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

        self._latest_requests: dict[str, SlotRequest] = {}
        self.next_batch_id = itertools.count(1)
        self._in_flight = 0

    def update(self, request: SlotRequest) -> None:
        self._latest_requests[request.robot_id] = request
        self.latency_tracker.update_obs(
            request.robot_id, request.arrival_timestamp, request.request_timestamp
        )
        self.mirror.receive_request(request, request.control_hz)

    def update_completion(self, notification: CompletionNotification) -> None:
        self.latency_tracker.update_infer(notification.batch_size, notification.inference_duration)
        self.mirror.update_completion(notification, time.time())

    def update_ack(self, notification: AckNotification) -> None:
        self.latency_tracker.update_action_delivery(
            notification.robot_id,
            notification.receive_time,
            notification.server_send_time,
        )
        self.mirror.confirm_chunk(notification)

    def schedule(self) -> list[SchedulerDecision]:
        """Return a list of batches of requests to be sent to the GPU."""
        # TODO: better traces
        batches, decisions = self.get_next_batches()

        for batch in batches:
            now = time.time()
            robot_ids = [r.robot_id for r in batch]
            request_ids = [r.request_id for r in batch]
            chunks = self.mirror.get_chunks(robot_ids, request_ids, now)
            self.mirror.fast_forward(
                now + self.latency_tracker.infer_latency(len(batch)),
                robot_ids,
                chunks,
            )
            self._batch_queue.put_nowait(
                RequestBatch(requests=batch, batch_id=next(self.next_batch_id))
            )
            self._in_flight += 1

        return decisions

    def notify_batch_complete(self) -> None:
        self._in_flight = max(0, self._in_flight - 1)

    @property
    def in_flight(self) -> int:
        return self._in_flight

    @property
    def schedulable_requests(self) -> list[SlotRequest]:
        """Requests eligible to be batched on the next dispatch."""
        return list(self._latest_requests.values())

    @abstractmethod
    def get_next_batches(self) -> list[list[SlotRequest]]:
        pass

    def reset_robot(self, robot_id: str) -> None:
        self._latest_requests.pop(robot_id, None)
        self.mirror.reset_robot(robot_id)
        self.latency_tracker.clear(robot_id)
