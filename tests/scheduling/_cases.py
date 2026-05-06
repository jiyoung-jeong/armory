from __future__ import annotations

from dataclasses import dataclass

from armory.scheduling.mirror import ActionChunk, ControlStep

CONTROL_HZ = 1.0
CONTROL_PERIOD = 1.0 / CONTROL_HZ
EPS = 0.001


def arrives_before(observation_step: int) -> float:
    return observation_step * CONTROL_PERIOD - EPS


@dataclass(frozen=True)
class Scenario:
    name: str
    chunks: list[ActionChunk]
    obs_action_next: list[tuple[int, int | None, int]]

    def control_steps(self) -> list[ControlStep]:
        assert [x[0] for x in self.obs_action_next] == list(range(len(self.obs_action_next)))
        return [
            ControlStep(
                time=obs_step * CONTROL_PERIOD,
                observation_step=obs_step,
                action_step=action_step,
                next_action_step=next_action_step,
            )
            for obs_step, action_step, next_action_step in self.obs_action_next
        ]


CHUNK_OVERLAP = Scenario(
    name="chunk_overlap",
    chunks=[
        ActionChunk(
            request_id=0,
            observation_step=0,
            arrival_time=arrives_before(2),
            action_index_start=0,
            execution_horizon=5,
            arrived=False,
        ),
        ActionChunk(
            request_id=1,
            observation_step=4,
            arrival_time=arrives_before(4),
            action_index_start=2,
            execution_horizon=5,
            arrived=False,
        ),
    ],
    obs_action_next=[
        (0, None, 0),
        (1, None, 0),
        (2, 0, 1),
        (3, 1, 2),
        (4, 2, 3),
        (5, 3, 4),
        (6, 4, 5),
        (7, 5, 6),
        (8, 6, 7),
    ],
)

PAUSE_BEFORE_INFERENCE = Scenario(
    name="pause_before_inference",
    chunks=[
        ActionChunk(
            request_id=0,
            observation_step=0,
            arrival_time=arrives_before(2),
            action_index_start=0,
            execution_horizon=5,
            arrived=False,
        ),
        ActionChunk(
            request_id=1,
            observation_step=7,
            arrival_time=arrives_before(9),
            action_index_start=5,
            execution_horizon=5,
            arrived=False,
        ),
    ],
    obs_action_next=[
        (0, None, 0),
        (1, None, 0),
        (2, 0, 1),
        (3, 1, 2),
        (4, 2, 3),
        (5, 3, 4),
        (6, 4, 5),
        (7, None, 5),
        (8, None, 5),
        (9, 5, 6),
        (10, 6, 7),
        (11, 7, 8),
        (12, 8, 9),
        (13, 9, 10),
    ],
)

PAUSE_DURING_INFERENCE = Scenario(
    name="pause_during_inference",
    chunks=[
        ActionChunk(
            request_id=0,
            observation_step=0,
            arrival_time=arrives_before(2),
            action_index_start=0,
            execution_horizon=5,
            arrived=False,
        ),
        ActionChunk(
            request_id=1,
            observation_step=5,
            arrival_time=arrives_before(9),
            action_index_start=3,
            execution_horizon=5,
            arrived=False,
        ),
    ],
    obs_action_next=[
        (0, None, 0),
        (1, None, 0),
        (2, 0, 1),
        (3, 1, 2),
        (4, 2, 3),
        (5, 3, 4),
        (6, 4, 5),
        (7, None, 5),
        (8, None, 5),
        (9, 5, 6),
        (10, 6, 7),
        (11, 7, 8),
    ],
)

LONG_RUN = Scenario(
    name="long_run",
    chunks=[
        ActionChunk(
            request_id=0,
            observation_step=0,
            arrival_time=arrives_before(2),
            action_index_start=0,
            execution_horizon=5,
            arrived=False,
        ),
        ActionChunk(
            request_id=1,
            observation_step=5,
            arrival_time=arrives_before(7),
            action_index_start=5,
            execution_horizon=5,
            arrived=False,
        ),
        ActionChunk(
            request_id=2,
            observation_step=10,
            arrival_time=arrives_before(12),
            action_index_start=10,
            execution_horizon=5,
            arrived=False,
        ),
        ActionChunk(
            request_id=3,
            observation_step=15,
            arrival_time=arrives_before(17),
            action_index_start=15,
            execution_horizon=5,
            arrived=False,
        ),
    ],
    obs_action_next=[
        (0, None, 0),
        (1, None, 0),
        *[(n, n - 2, n - 1) for n in range(2, 22)],
    ],
)

ALL_SCENARIOS = [CHUNK_OVERLAP, PAUSE_BEFORE_INFERENCE, PAUSE_DURING_INFERENCE, LONG_RUN]
