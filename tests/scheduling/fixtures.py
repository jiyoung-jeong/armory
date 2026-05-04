"""Canonical scenarios for mirror unit and comparison tests.

Each fixture returns a ``ScenarioSpec`` exercising a specific shape of broker
behavior. ``no_overlap`` is the sanity baseline; the others target known edge
cases in queue/chunk management.

All scenarios assume control_hz=10 (dt=0.1) so timestamps are easy to read.
"""

from __future__ import annotations

import pytest

from tests.scheduling.driver import ScenarioSpec, ServerResponseSpec

CONTROL_HZ = 10.0
DT = 1 / CONTROL_HZ


def _obs_grid(n: int) -> list[float]:
    return [round(i * DT, 6) for i in range(n)]


@pytest.fixture
def no_overlap() -> ScenarioSpec:
    """Each chunk arrives well before its actions are needed; queue empties cleanly between chunks.

    obs 0 → null (no chunk yet)
    chunk arrives @ t=0.05 covering steps [1..4]
    obs 1..4 → execute steps 1..4
    chunk arrives @ t=0.45 covering steps [5..8]
    obs 5..8 → execute steps 5..8
    """
    return ScenarioSpec(
        control_hz=CONTROL_HZ,
        execution_horizon=4,
        obs_times=_obs_grid(9),
        responses=[
            ServerResponseSpec(0, arrival_time=0.05, action_start_step=1, execution_horizon=4),
            ServerResponseSpec(4, arrival_time=0.45, action_start_step=5, execution_horizon=4),
        ],
    )


@pytest.fixture
def late_chunk_with_pauses() -> ScenarioSpec:
    """Second chunk arrives late: broker emits null actions until it lands.

    obs 0 → null
    chunk1 arrives @ 0.05 → steps [1..3]
    obs 1..3 → exec steps 1..3
    obs 4 → null (chunk2 not arrived yet)
    obs 5 → null
    chunk2 arrives @ 0.55 → steps [4..7]
    obs 6..9 → exec steps 4..7
    """
    return ScenarioSpec(
        control_hz=CONTROL_HZ,
        execution_horizon=4,
        obs_times=_obs_grid(10),
        responses=[
            ServerResponseSpec(0, arrival_time=0.05, action_start_step=1, execution_horizon=3),
            ServerResponseSpec(3, arrival_time=0.55, action_start_step=4, execution_horizon=4),
        ],
    )


@pytest.fixture
def overriding_chunk() -> ScenarioSpec:
    """A new chunk arrives whose action_start_step precedes the queue's tail; the
    broker truncates and replaces.

    chunk1 arrives @ 0.05, steps [1..6] (horizon 6, longer than typical)
    obs 1..2 execute steps 1..2; queue still has [3..6]
    chunk2 arrives @ 0.25 with action_start_step=3 horizon=4 → steps [3..6]
        broker pops queue tail [3..6] and rebuilds
    obs 3..6 execute steps 3..6
    """
    return ScenarioSpec(
        control_hz=CONTROL_HZ,
        execution_horizon=4,
        obs_times=_obs_grid(7),
        responses=[
            ServerResponseSpec(0, arrival_time=0.05, action_start_step=1, execution_horizon=6),
            ServerResponseSpec(2, arrival_time=0.25, action_start_step=3, execution_horizon=4),
        ],
    )


@pytest.fixture
def back_to_back() -> ScenarioSpec:
    """chunk2 arrives the same step chunk1's last action runs.

    chunk1 @ 0.05, steps [1..3]
    chunk2 @ 0.35, steps [4..6]  (arrives just before obs 4)
    """
    return ScenarioSpec(
        control_hz=CONTROL_HZ,
        execution_horizon=3,
        obs_times=_obs_grid(7),
        responses=[
            ServerResponseSpec(0, arrival_time=0.05, action_start_step=1, execution_horizon=3),
            ServerResponseSpec(3, arrival_time=0.35, action_start_step=4, execution_horizon=3),
        ],
    )


@pytest.fixture
def queue_exhaustion() -> ScenarioSpec:
    """Only one chunk; queue drains and broker emits nulls afterwards."""
    return ScenarioSpec(
        control_hz=CONTROL_HZ,
        execution_horizon=3,
        obs_times=_obs_grid(7),
        responses=[
            ServerResponseSpec(0, arrival_time=0.05, action_start_step=1, execution_horizon=3),
        ],
    )


ALL_SCENARIOS = [
    "no_overlap",
    "late_chunk_with_pauses",
    "overriding_chunk",
    "back_to_back",
    "queue_exhaustion",
]
