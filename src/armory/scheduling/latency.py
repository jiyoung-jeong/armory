from __future__ import annotations

from abc import ABC
from abc import abstractmethod


class LatencyTracker(ABC):
    """Per-robot and per-batch-size latency estimates (in seconds).

    Tracks three latencies:
    - observation_latency: time for observation to travel from robot to server
      (arrival_timestamp - request_timestamp)
    - infer_ms: GPU inference duration per batch size
    - action_delivery_ms: time for action to travel from server to robot
      (receive_time - server_send_time from ResponseAck)
    """

    def __init__(self) -> None:
        self._observation_latency: dict[str, float] = {}
        self._infer_latency: dict[int, float] = {}
        self._action_latency: dict[str, float] = {}

    @abstractmethod
    def _update_measurement(self, d: dict, key: object, value: float) -> None:
        pass

    # FIXME: rename time and ts
    def update_obs(self, robot_id: str, arrival_ts: float, request_ts: float) -> None:
        self._update_measurement(self._observation_latency, robot_id, arrival_ts - request_ts)

    def update_infer(self, batch_size: int, duration: float) -> None:
        self._update_measurement(self._infer_latency, batch_size, duration)

    def update_action_delivery(self, robot_id: str, receive_time: float, server_send_time: float) -> None:
        self._update_measurement(self._action_latency, robot_id, receive_time - server_send_time)

    # FIXME: how to make sure these are only called when these are available?
    def observation_latency(self, robot_id: str) -> float:
        return self._observation_latency[robot_id]

    def infer_latency(self, batch_size: int) -> float:
        return self._infer_latency[batch_size]

    def action_latency(self, robot_id: str) -> float:
        return self._action_latency[robot_id]

    def total_latency(self, robot_id: str, batch_size: int) -> float:
        """Total latency from observation timestep to robot receiving the action. Used as d param in RTC."""
        return self.observation_latency(robot_id) + self.infer_latency(batch_size) + self.action_latency(robot_id)

    def clear(self, robot_id: str) -> None:
        self._observation_latency.pop(robot_id, None)
        self._action_latency.pop(robot_id, None)


class EMALatencyTracker(LatencyTracker):
    def __init__(self, alpha: float = 0.1) -> None:
        super().__init__()
        self._alpha = alpha

    def _update_measurement(self, d: dict, key: object, value: float) -> None:
        if key not in d:
            d[key] = value
        else:
            d[key] = (1.0 - self._alpha) * d[key] + self._alpha * value


class JacobsonKarelsLatencyTracker(LatencyTracker):
    def update_measurement(self, d: dict, key: object, value: float) -> None:
        # FIXME: implement jacobson-karels tracker from https://github.com/jackvial/drtc/blob/0f6317703eae654d878956011e26fb50fa528162/src/lerobot/async_inference/utils/latency_estimation.py#L88 and compare accuracy
        raise NotImplementedError("JacobsonKarelsLatencyTracker is not implemented")
