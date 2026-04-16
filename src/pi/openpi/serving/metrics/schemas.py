from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from dataclasses import field
import itertools
from typing import NamedTuple, TypeAlias, TypeVar

import numpy as np
from openpi_client.messages import EpisodeEnd
from openpi_client.messages import EpisodeStart

RobotID: TypeAlias = str
T = TypeVar("T")


@dataclass
class RequestRecord:
    """Full lifecycle record for one inference request."""

    robot_id: RobotID
    request_id: int
    observation_step: int
    action_start_step: int
    execution_horizon: int
    request_timestamp: float  # client: when request was created
    server_arrival_time: float  # server: when observation arrived


@dataclass
class ResponseRecord:
    request: RequestRecord
    batch_id: int
    inference_start_time: float  # gpu: before infer_batch
    inference_end_time: float  # gpu: after infer_batch
    server_send_time: float = 0.0  # server: before websocket.send_bytes()
    receive_time: float = 0.0  # client: ResponseAck.receive_time
    execution_start_step: int = 0  # client: ResponseAck.execution_start_step
    first_executed_index: int = 0  # client: index within chunk where execution started
    execution_horizon: int = 0  # client: how many actions were in the response chunk

    def __post_init__(self) -> None:
        if isinstance(self.request, dict):
            self.request = RequestRecord(**self.request)

    @property
    def queue_delay_ms(self) -> float:
        return (self.inference_start_time - self.request.server_arrival_time) * 1000

    @property
    def gpu_time_ms(self) -> float:
        return (self.inference_end_time - self.inference_start_time) * 1000

    @property
    def total_latency_ms(self) -> float:
        return (self.inference_end_time - self.request.request_timestamp) * 1000

    @property
    def outbound_ms(self) -> float:
        """Only valid when receive_time > 0."""
        return (self.receive_time - self.server_send_time) * 1000


def window_filter(
    items: list[T],
    event_time_getter: Callable[[T], float],
    window_s: tuple[float, float],
) -> list[T]:
    start_timestamp, end_timestamp = window_s
    return [item for item in items if start_timestamp <= event_time_getter(item) < end_timestamp]


@dataclass
class Episode:
    task_suite_name: str
    task_id: int
    max_episode_steps: int
    task_language: str

    requests: list[RequestRecord]
    responses: list[ResponseRecord]
    success: bool | None = None
    num_observation_steps: int = 0
    step_timestamps: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.requests = [RequestRecord(**r) if isinstance(r, dict) else r for r in self.requests]
        self.responses = [ResponseRecord(**r) if isinstance(r, dict) else r for r in self.responses]
        assert all(
            next_request.action_start_step >= prev_request.action_start_step
            for prev_request, next_request in zip(self.requests[:-1], self.requests[1:], strict=True)
        )

    @property
    def start_time(self) -> float:
        return self.requests[0].request_timestamp

    @property
    def end_time(self) -> float:
        # TODO: approximately correct
        return self.requests[-1].request_timestamp

    @property
    def num_steps(self) -> int:
        return self.num_observation_steps

    @property
    def actions_left_history(self) -> np.ndarray[int, " num_steps"]:
        actions_left_history = np.zeros(self.num_steps, dtype=np.int32)
        for response in self.responses:
            # At execution_start_step the robot is on action first_executed_index of the chunk,
            # so it has (execution_horizon - first_executed_index) actions remaining, counting
            # down by 1 each step until the chunk is exhausted or the episode ends.
            remaining = response.execution_horizon - response.first_executed_index
            execution_end_step = min(response.execution_start_step + remaining, self.num_steps)
            n = execution_end_step - response.execution_start_step
            actions_left = np.arange(remaining, remaining - n, -1)
            actions_left_history[response.execution_start_step : execution_end_step] = np.maximum(
                actions_left_history[response.execution_start_step : execution_end_step],
                actions_left,
            )

        return actions_left_history

    def add_request(self, request: RequestRecord) -> None:
        assert request.observation_step == len(self.requests)
        self.requests.append(request)

    def add_response(self, response: ResponseRecord) -> None:
        self.responses.append(response)

    def get_requests(self, start_timestamp: float, end_timestamp: float) -> list[RequestRecord]:
        return window_filter(
            self.requests,
            lambda r: r.request_timestamp,
            (start_timestamp, end_timestamp),
        )

    def get_responses(self, start_timestamp: float, end_timestamp: float) -> list[ResponseRecord]:
        return window_filter(self.responses, lambda r: r.receive_time, (start_timestamp, end_timestamp))

    def get_windowed_actions_left(self, start_ts: float, end_ts: float) -> np.ndarray:
        """Slice of actions_left_history for control steps with timestamp in [start_ts, end_ts)."""
        history = self.actions_left_history
        return np.array(
            [history[i] for i, t in enumerate(self.step_timestamps) if start_ts <= t < end_ts],
            dtype=float,
        )

    def get_windowed_steps(self, start_ts: float, end_ts: float) -> list[tuple[float, float]]:
        """Return [(timestamp, actions_left), ...] for control steps in [start_ts, end_ts)."""
        history = self.actions_left_history
        return [(t, float(history[i])) for i, t in enumerate(self.step_timestamps) if start_ts <= t < end_ts]


@dataclass
class Robot:
    """Per-robot mutable state tracked during inference."""

    robot_id: str
    episodes: list[Episode]

    def __post_init__(self) -> None:
        self.episodes = [Episode(**e) if isinstance(e, dict) else e for e in self.episodes]

    @property
    def current_episode(self) -> Episode:
        assert len(self.episodes) > 0
        return self.episodes[-1]

    def start_episode(self, episode_start: EpisodeStart) -> None:
        self.episodes.append(
            Episode(
                task_suite_name=episode_start.task_suite_name,
                task_id=episode_start.task_id,
                max_episode_steps=episode_start.max_episode_steps,
                task_language=episode_start.task_language,
                requests=[],
                responses=[],
            )
        )

    def end_episode(self, episode_end: EpisodeEnd) -> None:
        episode = self.current_episode
        assert episode.task_suite_name == episode_end.task_suite_name
        assert episode.task_id == episode_end.task_id
        assert episode.num_steps == episode_end.steps_taken
        episode.success = episode_end.success

    def add_step(self, timestamp: float) -> None:
        self.current_episode.num_observation_steps += 1
        self.current_episode.step_timestamps.append(timestamp)

    def add_request(self, request: RequestRecord) -> None:
        self.current_episode.requests.append(request)

    def add_response(self, response: ResponseRecord) -> None:
        self.current_episode.responses.append(response)

    def get_request(self, request_id: int) -> RequestRecord:
        # NOTE: can only be called when store is live
        # search backward on current request
        return next(r for r in reversed(self.current_episode.requests) if r.request_id == request_id)

    @property
    def total_steps(self) -> int:
        return sum(e.num_steps for e in self.episodes)

    @property
    def total_starved_steps(self) -> int:
        return sum(np.sum(e.actions_left_history == 0) for e in self.episodes)

    def get_requests(self, start_timestamp: float, end_timestamp: float) -> list[RequestRecord]:
        return list(
            itertools.chain.from_iterable(e.get_requests(start_timestamp, end_timestamp) for e in self.episodes)
        )

    def get_responses(self, start_timestamp: float, end_timestamp: float) -> list[ResponseRecord]:
        return list(
            itertools.chain.from_iterable(e.get_responses(start_timestamp, end_timestamp) for e in self.episodes)
        )

    def get_actions_left_timed(self, start_ts: float, end_ts: float) -> tuple[np.ndarray, np.ndarray]:
        """Return (timestamps, actions_left_values) for steps in [start_ts, end_ts), with nan separators between episodes."""
        times_parts: list[np.ndarray] = []
        values_parts: list[np.ndarray] = []
        sep = np.array([np.nan])
        for episode in self.episodes:
            steps = episode.get_windowed_steps(start_ts, end_ts)
            if not steps:
                continue
            if times_parts:
                times_parts.append(sep)
                values_parts.append(sep)
            t_arr, v_arr = zip(*steps, strict=True)
            times_parts.append(np.array(t_arr, dtype=float))
            values_parts.append(np.array(v_arr, dtype=float))
        if not times_parts:
            return np.array([], dtype=float), np.array([], dtype=float)
        return np.concatenate(times_parts), np.concatenate(values_parts)

    def get_actions_left_concatenated(self, start_ts: float, end_ts: float) -> np.ndarray:
        """Concatenate windowed episode actions_left slices with nan separators.

        Each episode contributes the steps whose request_timestamp falls in [start_ts, end_ts).
        Episodes with no steps in the window are skipped. Nans mark episode boundaries.
        """
        parts: list[np.ndarray] = []
        sep = np.array([np.nan])
        for episode in self.episodes:
            arr = episode.get_windowed_actions_left(start_ts, end_ts)
            if len(arr) == 0:
                continue
            if parts:
                parts.append(sep)
            parts.append(arr)
        return np.concatenate(parts) if parts else np.array([], dtype=float)


class BatchSummary(NamedTuple):
    batch_id: int
    robot_ids: list[RobotID]
    request_ids: list[int]
    inference_start_time: float
    inference_end_time: float

    @classmethod
    def from_json(cls, data: BatchSummary | dict | list) -> BatchSummary:
        if isinstance(data, cls):
            return data
        if isinstance(data, dict):
            return cls(**data)
        return cls(*data)

    @property
    def gpu_time_ms(self) -> float:
        return (self.inference_end_time - self.inference_start_time) * 1000
