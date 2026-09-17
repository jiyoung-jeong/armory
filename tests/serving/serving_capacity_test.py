import pytest
from scripts.analyze_serving_capacity import account
from scripts.benchmark_serving_capacity import request_tick


def test_slo_denominator_includes_unanswered_and_late_responses():
    sent = [dict(request_timestamp=t, observation_step=i) for i, t in enumerate([0, 0.5, 1, 1.5])]
    chunks = [
        dict(request_timestamp=0, response_timestamp=0.1, request_id=1, chunk_id=1),
        dict(request_timestamp=0.5, response_timestamp=0.7, request_id=2, chunk_id=2),
        dict(request_timestamp=0.5, response_timestamp=0.8, request_id=2, chunk_id=3),
    ]
    steps = [
        dict(time=t, local_chunk_index=v) for t, v in [(0, 0), (0.3, None), (0.6, 0), (1.2, None)]
    ]
    requests, measured, duplicates = account(sent, chunks, steps, 0, 1.5, 156)
    assert len(requests) == 3
    assert [r["slo_pass"] for r in requests] == [True, False, False]
    assert [r["responded"] for r in requests] == [True, True, False]
    assert [r["cycle_starved"] for r in requests] == [True, False, True]
    assert requests[1]["latency_ms"] == pytest.approx(200)
    assert duplicates == 1
    assert len(measured) == 4


def test_measurement_boundary_keeps_response_after_window():
    sent = [
        dict(request_timestamp=0.9, observation_step=0),
        dict(request_timestamp=1.4, observation_step=1),
    ]
    chunks = [dict(request_timestamp=0.9, response_timestamp=1.01, request_id=1, chunk_id=1)]
    requests, _, _ = account(sent, chunks, [dict(time=0.95, local_chunk_index=0)], 0.5, 1, 156)
    assert requests[0]["slo_pass"]
    assert requests[0]["cycle_starved"] is None


@pytest.mark.parametrize("hz", [10.0, 20.0, 40.0])
def test_request_count_independent_of_action_consumption_rate(hz):
    # Exactly two observations per second even while the control loop changes.
    assert sum(request_tick(i, hz, 2.0) for i in range(int(hz * 10))) == 20


def test_non_integral_request_period_rejected():
    with pytest.raises(ValueError):
        request_tick(0, 15.0, 2.0)
