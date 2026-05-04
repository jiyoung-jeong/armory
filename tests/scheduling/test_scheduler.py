from __future__ import annotations

from queue import Queue

import armory.scheduling.lookahead_actions as lookahead_actions
from armory.scheduling import RequestScheduler
from armory.scheduling.baselines import MaxBatchScheduler
from armory.scheduling.lookahead_actions import LookaheadActionsScheduler, Search
from armory.scheduling.mirror import ActionChunk, Mirror
from armory.serving.schemas import AckNotification, SlotRequest
from armory_client.messages import InferType


def _slot_request(
    *,
    robot_id: str = "r0",
    request_id: int = 1,
    timestamp: float = 0.0,
    observation_step: int = 0,
    action_start_step: int = 0,
) -> SlotRequest:
    return SlotRequest(
        slot_index=0,
        robot_id=robot_id,
        request_id=request_id,
        arrival_timestamp=timestamp,
        observation_step=observation_step,
        action_start_step=action_start_step,
        request_timestamp=timestamp,
        deadline=timestamp + 1.0,
        execution_horizon=4,
        infer_type=InferType.SYNC,
        params=None,
        noise=None,
        control_hz=10.0,
    )


def _seed_latencies(scheduler: RequestScheduler, robot_id: str = "r0") -> None:
    scheduler.latency_tracker.update_infer(1, 0.0)
    scheduler.latency_tracker.update_obs(robot_id, 0.0, 0.0)
    scheduler.latency_tracker.update_action_delivery(robot_id, 0.0, 0.0)


def test_reset_robot_clears_mirror_state_before_rescheduling_from_zero():
    scheduler = MaxBatchScheduler(Queue(), max_batch_size=1)
    _seed_latencies(scheduler)

    scheduler.update(_slot_request(timestamp=1.0, observation_step=10, action_start_step=10))
    scheduler.schedule()
    assert scheduler.mirror.robots["r0"].chunks[-1].observation_step == 10

    scheduler._batch_queue.get_nowait()
    scheduler.reset_robot("r0")
    scheduler.update(
        _slot_request(request_id=2, timestamp=2.0, observation_step=0, action_start_step=0)
    )
    scheduler.schedule()

    assert scheduler.mirror.robots["r0"].chunks[-1].observation_step == 0


def test_ack_after_reset_is_ignored_as_stale():
    scheduler = MaxBatchScheduler(Queue(), max_batch_size=1)
    _seed_latencies(scheduler)
    scheduler.update(_slot_request(timestamp=1.0, observation_step=10, action_start_step=10))
    scheduler.schedule()

    scheduler.reset_robot("r0")
    scheduler.update_ack(
        AckNotification(
            robot_id="r0",
            request_id=1,
            observation_step=10,
            receive_time=1.1,
            server_send_time=1.0,
        )
    )

    assert "r0" not in scheduler.mirror.robots


class _DuplicateBatchScheduler(RequestScheduler):
    def get_next_batches(self) -> list[list[SlotRequest]]:
        request = self.schedulable_requests[0]
        return [[request], [request]]


def test_schedule_rechecks_stale_requests_between_returned_batches():
    scheduler = _DuplicateBatchScheduler(Queue(), max_batch_size=1)
    _seed_latencies(scheduler)
    scheduler.update(_slot_request())

    scheduler.schedule()

    assert scheduler._batch_queue.qsize() == 1
    assert len(scheduler.mirror.robots["r0"].chunks) == 1


def test_lookahead_actions_uses_depth_one_for_initial_batch(monkeypatch):
    class FakeSearch:
        depths = []

        def __init__(self, mirror, latency_tracker, start_time, horizon, max_depth=3):
            self.max_depth = max_depth
            FakeSearch.depths.append(max_depth)

        def run(self):
            return [("r0",)]

    monkeypatch.setattr(lookahead_actions, "Search", FakeSearch)
    scheduler = LookaheadActionsScheduler(Queue(), max_batch_size=1)
    _seed_latencies(scheduler)
    try:
        scheduler.update(_slot_request())
        scheduler.schedule()

        assert scheduler._batch_queue.qsize() == 1
        assert FakeSearch.depths[0] == 1
    finally:
        scheduler._executor.shutdown(wait=True)


def test_lookahead_actions_uses_ready_background_schedule(monkeypatch):
    class FakeSearch:
        depths = []

        def __init__(self, mirror, latency_tracker, start_time, horizon, max_depth=3):
            self.max_depth = max_depth
            FakeSearch.depths.append(max_depth)

        def run(self):
            return [("r0",)]

    monkeypatch.setattr(lookahead_actions, "Search", FakeSearch)
    scheduler = LookaheadActionsScheduler(Queue(), max_batch_size=1)
    _seed_latencies(scheduler)
    try:
        scheduler.update(_slot_request(request_id=1, observation_step=0, action_start_step=0))
        scheduler.schedule()
        scheduler._batch_queue.get_nowait()

        scheduler.update(
            _slot_request(
                request_id=2,
                timestamp=0.1,
                observation_step=1,
                action_start_step=1,
            )
        )
        scheduler._search_future.result(timeout=1)
        scheduler.notify_batch_complete()
        scheduler.schedule()

        assert scheduler._batch_queue.qsize() == 1
        assert FakeSearch.depths.count(1) == 1
    finally:
        scheduler._executor.shutdown(wait=True)


def test_lookahead_action_search_skips_robot_without_newer_observation():
    request = _slot_request(timestamp=0.0, observation_step=0, action_start_step=0)
    mirror = Mirror()
    mirror.receive_request(request, request.control_hz)
    mirror.schedule_pending_chunk(
        request.robot_id,
        ActionChunk(
            observation_step=0,
            arrival_time=0.1,
            action_start_step=1,
            execution_horizon=4,
            arrived=False,
        ),
    )
    scheduler = MaxBatchScheduler(Queue(), max_batch_size=1)
    _seed_latencies(scheduler)

    assert Search(mirror, scheduler.latency_tracker, start_time=0.1, horizon=0.5).run() == []
