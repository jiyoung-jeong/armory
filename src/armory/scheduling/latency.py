from __future__ import annotations


class LatencyTracker:
    """Per-robot and per-batch-size EMA latency estimates.

    Tracks three latencies:
    - obs_network_ms: time for observation to travel from robot to server
      (arrival_timestamp - request_timestamp; note: subject to clock skew between
      client and server — use for relative comparisons, not absolute values)
    - infer_ms: actual GPU inference duration per batch size
    - action_delivery_ms: time for action to travel from server to robot
      (receive_time - server_send_time from ResponseAck)
    """

    def __init__(self, alpha: float = 0.1) -> None:
        self._alpha = alpha
        self._obs_network_ms: dict[str, float] = {}
        self._infer_ms: dict[int, float] = {}
        self._action_delivery_ms: dict[str, float] = {}

    def _ema(self, d: dict, key: object, value: float) -> None:
        if key not in d:
            d[key] = value
        else:
            d[key] = (1.0 - self._alpha) * d[key] + self._alpha * value

    def update_obs(self, robot_id: str, arrival_ts: float, request_ts: float) -> None:
        """Update obs-to-server network latency estimate from timestamps."""
        self._ema(self._obs_network_ms, robot_id, (arrival_ts - request_ts) * 1e3)

    def update_infer(self, batch_size: int, duration_ms: float) -> None:
        """Update inference latency estimate for a given batch size."""
        self._ema(self._infer_ms, batch_size, duration_ms)

    def update_action_delivery(self, robot_id: str, receive_time: float, server_send_time: float) -> None:
        """Update action-delivery latency estimate from server-send and client-receive timestamps."""
        self._ema(self._action_delivery_ms, robot_id, (receive_time - server_send_time) * 1e3)

    def obs_network_ms(self, robot_id: str) -> float | None:
        return self._obs_network_ms.get(robot_id)

    def infer_ms(self, batch_size: int) -> float | None:
        return self._infer_ms.get(batch_size)

    def action_delivery_ms(self, robot_id: str) -> float | None:
        return self._action_delivery_ms.get(robot_id)

    def total_delivery_ms(self, robot_id: str, batch_size: int) -> float | None:
        """Total latency from inference-start to robot receiving the action."""
        infer = self._infer_ms.get(batch_size)
        delivery = self._action_delivery_ms.get(robot_id)
        if infer is None or delivery is None:
            return None
        return infer + delivery

    def reset_robot(self, robot_id: str) -> None:
        self._obs_network_ms.pop(robot_id, None)
        self._action_delivery_ms.pop(robot_id, None)
