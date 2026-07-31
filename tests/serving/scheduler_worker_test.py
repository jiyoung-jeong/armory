from __future__ import annotations

from collections import deque
from pathlib import Path

import pytest

from armory.serving.config import ServerConfig
from armory.serving.protocol import SchedulerConfig
from armory.serving.rtc import InferType
from armory.serving.scheduler import SCHEDULER_REGISTRY, SchedulerWorker
from armory.serving.schemas import (
    AckNotification,
    Reconfigure,
    ResetAll,
    ResponseBatch,
    SlotRequest,
    WarmupSeed,
)
from armory_client.messages import ResetRequest, ResponseAck


def _slot_request(robot_id: str = "robot-new", request_id: int = 11) -> SlotRequest:
    return SlotRequest(
        slot_index=2,
        robot_id=robot_id,
        request_id=request_id,
        arrival_timestamp=20.0,
        observation_step=3,
        action_index_start=4,
        request_timestamp=19.5,
        deadline=30.0,
        min_execution_horizon=1,
        max_execution_horizon=8,
        infer_type=InferType.SYNC,
        params=None,
        noise=None,
        control_hz=10.0,
    )


def _ack(robot_id: str = "robot-new", request_id: int = 11) -> AckNotification:
    return AckNotification(
        ack=ResponseAck(
            request_id=request_id,
            chunk_id=5,
            observation_step=3,
            receive_time=22.0,
            action_index_start=4,
            min_execution_horizon=1,
            max_execution_horizon=8,
            execution_start_step=5,
            first_executed_index=1,
        ),
        robot_id=robot_id,
        server_send_time=21.0,
    )


class _MessageSocket:
    def __init__(self, messages: list[object] | None = None) -> None:
        self.messages = deque(messages or [])

    def poll(self, timeout: int = 0) -> bool:
        del timeout
        return bool(self.messages)

    def recv_pyobj(self, flags: int = 0) -> object:
        del flags
        return self.messages.popleft()

    def setsockopt(self, option: int, value: bytes) -> None:
        del option, value

    def connect(self, endpoint: str) -> None:
        del endpoint


class _SpyLatencyTracker:
    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def update_infer(self, batch_size: int, latency: float) -> None:
        self.calls.append(("infer", batch_size, latency))

    def update_obs(self, robot_id: str, arrival: float, requested: float) -> None:
        self.calls.append(("observation", robot_id, arrival, requested))

    def update_action_delivery(self, robot_id: str, received: float, sent: float) -> None:
        self.calls.append(("delivery", robot_id, received, sent))

    def observation_latency(self, robot_id: str) -> float:
        del robot_id
        return 0.1

    def action_latency(self, robot_id: str) -> float:
        del robot_id
        return 0.2


class _SpyScheduler:
    def __init__(self) -> None:
        self.calls: list[tuple] = []
        self.latency_tracker = _SpyLatencyTracker()

    def reset_robot(self, robot_id: str) -> None:
        self.calls.append(("reset_robot", robot_id))

    def reset_all(self) -> None:
        self.calls.append(("reset_all",))

    def update(self, request: SlotRequest) -> None:
        self.calls.append(("update", request.robot_id, request.request_id))

    def update_ack(self, notification: AckNotification) -> None:
        self.calls.append(("update_ack", notification.robot_id, notification.ack.request_id))


class _ReplacementScheduler(_SpyScheduler):
    instances: list[_ReplacementScheduler] = []

    def __init__(self, config: SchedulerConfig, batch_queue: object, max_batch_size: int) -> None:
        super().__init__()
        self.batch_queue = batch_queue
        self.max_batch_size = max_batch_size
        self.config = config
        self._drain_fn = None
        type(self).instances.append(self)


def test_schedulers_read_their_own_knobs_from_the_config() -> None:
    config = SchedulerConfig(alpha=0.25)
    batch_queue = object()

    dynamic = SCHEDULER_REGISTRY["dynamic-action"](config, batch_queue, max_batch_size=2)
    greedy = SCHEDULER_REGISTRY["greedy-deadline"](config, batch_queue, max_batch_size=2)

    assert dynamic._alpha == 0.25
    assert greedy._config is config


def test_server_messages_are_applied_in_fifo_order_across_reconfigure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    replacement_name = "_characterization-replacement"
    monkeypatch.setitem(SCHEDULER_REGISTRY, replacement_name, _ReplacementScheduler)
    _ReplacementScheduler.instances.clear()

    worker = SchedulerWorker(
        sched_in_ep="control",
        result_ep="results",
        batch_queue=object(),  # type: ignore[arg-type]
        metrics_dir=tmp_path,
        config=ServerConfig(
            max_batch_size=4,
            scheduler=SchedulerConfig(scheduling_algorithm="max-batch"),
        ),
        ready_event=object(),  # type: ignore[arg-type]
    )
    original = _SpyScheduler()
    result_socket = _MessageSocket()
    worker._current_scheduler = original
    worker._result_sock = result_socket
    worker._batch_profile = {1: 0.01, 4: 0.04}

    drained: list[tuple[object, object]] = []
    monkeypatch.setattr(
        worker,
        "_process_engine_messages",
        lambda active, socket: drained.append((active, socket)),
    )

    request = _slot_request()
    ack = _ack()
    control_socket = _MessageSocket(
        [
            ResetRequest(robot_id="robot-old"),
            ResetAll(),
            Reconfigure(
                config=ServerConfig(
                    max_batch_size=4,
                    scheduler=SchedulerConfig(scheduling_algorithm=replacement_name, alpha=0.25),
                )
            ),
            request,
            ack,
            WarmupSeed(
                robot_id="robot-new",
                obs_samples=[(12.0, 10.0)],
                delivery_samples=[(16.0, 13.0)],
            ),
        ]
    )

    worker._process_server_messages(control_socket)  # type: ignore[arg-type]

    assert original.calls == [("reset_robot", "robot-old"), ("reset_all",)]
    assert len(_ReplacementScheduler.instances) == 1
    replacement = _ReplacementScheduler.instances[0]
    assert worker._current_scheduler is replacement
    assert worker.config.scheduler.scheduling_algorithm == replacement_name
    assert worker.config.scheduler.alpha == 0.25
    assert replacement.max_batch_size == 4
    assert replacement.config.alpha == 0.25
    assert replacement.calls == [
        ("update", "robot-new", 11),
        ("update_ack", "robot-new", 11),
    ]
    assert replacement.latency_tracker.calls == [
        ("infer", 1, 0.01),
        ("infer", 4, 0.04),
        ("observation", "robot-new", 12.0, 10.0),
        ("delivery", "robot-new", 16.0, 13.0),
    ]
    assert control_socket.messages == deque()

    assert replacement._drain_fn is not None
    replacement._drain_fn()
    assert drained == [(replacement, result_socket)]


def test_failed_reconfigure_keeps_the_current_scheduler(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class _BrokenScheduler:
        def __init__(self, *args: object, **kwargs: object) -> None:
            del args, kwargs
            raise RuntimeError("construction failed")

    replacement_name = "_characterization-broken"
    monkeypatch.setitem(SCHEDULER_REGISTRY, replacement_name, _BrokenScheduler)  # type: ignore[arg-type]

    worker = SchedulerWorker(
        sched_in_ep="control",
        result_ep="results",
        batch_queue=object(),  # type: ignore[arg-type]
        metrics_dir=tmp_path,
        config=ServerConfig(
            max_batch_size=2,
            scheduler=SchedulerConfig(scheduling_algorithm="max-batch"),
        ),
        ready_event=object(),  # type: ignore[arg-type]
    )
    original = _SpyScheduler()
    worker._current_scheduler = original
    worker._batch_profile = {1: 0.1}
    worker._result_sock = _MessageSocket()

    worker._handle_reconfigure(
        Reconfigure(
            config=ServerConfig(
                max_batch_size=2,
                scheduler=SchedulerConfig(scheduling_algorithm=replacement_name),
            )
        )
    )

    assert worker._current_scheduler is original
    assert worker.config.scheduler.scheduling_algorithm == "max-batch"


def test_engine_completion_messages_are_fully_drained_in_fifo_order(tmp_path: Path) -> None:
    first = ResponseBatch([], 10, 0, 100.0, 0.0)
    second = ResponseBatch([], 11, 0, 101.0, 0.0)
    socket = _MessageSocket([first, second])

    class _CompletionSink:
        def __init__(self) -> None:
            self.completed: list[ResponseBatch] = []

        def on_batch_completed(self, batch: ResponseBatch) -> None:
            self.completed.append(batch)

    sink = _CompletionSink()
    worker = SchedulerWorker(
        sched_in_ep="control",
        result_ep="results",
        batch_queue=object(),  # type: ignore[arg-type]
        metrics_dir=tmp_path,
        config=ServerConfig(
            max_batch_size=2,
            scheduler=SchedulerConfig(scheduling_algorithm="max-batch"),
        ),
        ready_event=object(),  # type: ignore[arg-type]
    )

    worker._process_engine_messages(sink, socket)  # type: ignore[arg-type]

    assert sink.completed == [first, second]
    assert socket.messages == deque()
