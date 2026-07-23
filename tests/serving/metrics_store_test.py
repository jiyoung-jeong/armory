import json
from dataclasses import asdict

from armory.serving.metrics.store import MetricsStore
from armory.serving.schemas import SchedulerDecision


def test_current_scheduler_decisions_remain_snapshot_and_json_serializable() -> None:
    decision = SchedulerDecision(
        scheduler_name="TestScheduler",
        started_at=2.0,
        duration=0.002,
        next_server_available=2.1,
        candidates=["robot-0", "robot-1"],
        scheduled=["robot-0"],
        batch_id=7,
    )
    store = MetricsStore(
        start_time=1.0,
        end_time=3.0,
        scheduler_decisions=[decision],
    )

    snapshot = store.snapshot()

    assert snapshot.scheduler_timing_ms == {"TestScheduler.batch_scheduled": [2.0]}
    assert snapshot.scheduling_decisions == [
        {
            "t": 1.0,
            "duration_ms": 2.0,
            "scheduler": "TestScheduler",
            "candidates": ["robot-0", "robot-1"],
            "scheduled": ["robot-0"],
            "batch_id": 7,
            "in_flight_batches": 0,
            "next_server_available_t": 1.1,
            "deadlines": {},
            "notes": {},
        }
    ]

    payload = asdict(store)
    json.dumps(payload)
    assert payload["scheduler_decisions"][0]["started_at"] == 2.0
    assert "metric_name" not in payload["scheduler_decisions"][0]
