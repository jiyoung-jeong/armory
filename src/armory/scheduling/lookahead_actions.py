from armory.scheduling import RequestScheduler
from armory.serving.schemas import SlotRequest
from armory.scheduling.mirror import search
import multiprocessing as mp


class LookaheadActionsScheduler(RequestScheduler):
    def __init__(
        self, batch_queue: mp.Queue, max_batch_size: int = 1, *, horizon: float = 0.5
    ) -> None:
        super().__init__(batch_queue, max_batch_size)
        self.horizon = horizon

    def get_next_batches(self) -> list[list[SlotRequest]]:
        if not self._batch_queue.empty() or self.schedulable_requests == []:
            return []

        return search(self.mirror, self.latency_tracker, self.horizon)
