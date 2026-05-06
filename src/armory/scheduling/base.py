import itertools
import logging
import multiprocessing as mp
from abc import ABC, abstractmethod

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
        """Return a list of batches of requests to be sent to the GPU."""
        # TODO: better traces
        batches = self.get_next_batches()
        decisions: list[SchedulerDecision] = []

        for batch in batches:
            chunks = self.mirror.queue_batch(batch)
            self._batch_queue.put_nowait(
                RequestBatch(
                    requests=batch,
                    chunk_ids=[
                        chunk.chunk_id for chunk in chunks
                    ],  # FIXME: can make clean up dataclasses later
                    batch_id=next(self.next_batch_id),
                )
            )

        return decisions

    @abstractmethod
    def get_next_batches(self) -> list[list[SlotRequest]]:
        pass

    def reset_robot(self, robot_id: str) -> None:
        self._latest_requests.pop(robot_id, None)
        self.mirror.reset_robot(robot_id)
        self.latency_tracker.clear(robot_id)
