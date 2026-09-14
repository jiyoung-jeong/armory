"""Decision logs must preserve the forecast used at dispatch time."""

import time
from dataclasses import replace
from queue import Queue

import pytest

from armory.backends.types import warmup_request
from armory.scheduling.baselines import RoundRobinScheduler
from armory.serving.protocol import SchedulerConfig


def test_prediction_is_a_snapshot_not_a_later_latency_estimate(monkeypatch):
    monkeypatch.setenv("ARMORY_RECORD_PREDICTIONS", "1")
    scheduler = RoundRobinScheduler(SchedulerConfig(scheduling_algorithm="round-robin"), Queue())
    now = time.time()
    request = replace(
        warmup_request({}),
        robot_id="robot",
        min_execution_horizon=1,
        max_execution_horizon=10,
        control_hz=20,
        request_timestamp=now - 0.002,
        arrival_timestamp=now,
        deadline=now,
    ).request
    scheduler.latency_tracker.update_infer(1, 0.2)
    scheduler.latency_tracker.update_action_delivery("robot", now, now - 0.001)
    scheduler.update(request)
    decision = scheduler.schedule()[0]
    prediction = decision.notes["prediction"]
    assert prediction["inference_duration"] == 0.2
    assert prediction["chunks"][0]["arrival_time"] == pytest.approx(
        prediction["completion_time"] + 0.001, abs=1e-5, rel=0
    )
    assert prediction["chunks"][0]["robot_id"] == "robot"
    scheduler.latency_tracker.update_infer(1, 0.8)
    assert scheduler.latency_tracker.infer_latency(1) != 0.2
    assert prediction["inference_duration"] == 0.2
