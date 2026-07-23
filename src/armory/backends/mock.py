"""Lightweight serving-policy implementation used for CPU-only experiments."""

from __future__ import annotations

import json
import time
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from armory.backends.types import PolicyRequest, PolicyResult
from armory_client.messages import InferRequest


def _profile_path() -> Path:
    packaged = Path(__file__).with_name("inference_profiles.json")
    if packaged.exists():
        return packaged
    # Editable source tree: Hatch places the root config beside this module in
    # built wheels, while local development retains the existing configs path.
    return Path(__file__).parents[3] / "configs" / "inference_profiles.json"


with _profile_path().open() as f:
    INFERENCE_PROFILES = json.load(f)


class MockPolicy:
    """Stub policy implementing the engine interface without weights or a GPU."""

    def __init__(
        self,
        *,
        env: str,
        action_horizon: int,
        action_dim: int,
        inference_latency: dict[int, float],
    ) -> None:
        self._action_horizon = action_horizon
        self._action_dim = action_dim
        self._inference_latency = inference_latency
        self.metadata = {"env": env}

    def make_infer_request(self) -> InferRequest:
        # This is the legacy mock profiling request type. Live engine requests
        # are InternalRequest objects; infer_batch intentionally ignores fields.
        now = time.time()
        return InferRequest(
            robot_id="__warmup__",
            observation={},
            observation_step=0,
            action_index_start=0,
            request_timestamp=now,
            deadline=now + 60.0,
            min_execution_horizon=0,
            max_execution_horizon=0,
            params=None,
            noise=None,
        )

    def warmup(self, max_batch_size: int) -> None:
        del max_batch_size

    def infer_batch(self, requests: Sequence[PolicyRequest]) -> list[PolicyResult]:
        inference_latency = self._inference_latency[len(requests)]
        now = time.time()
        while time.time() - now < inference_latency:
            time.sleep(0.001)
        actions = np.zeros((self._action_horizon, self._action_dim), dtype=np.float32)
        return [{"actions": actions, "noise": None, "rtc_prev_actions": actions} for _ in requests]


class MockPolicyFactory:
    """Picklable factory for ``MockPolicy``."""

    def __init__(self, *, env: str, action_horizon: int, action_dim: int, model: str, gpu: str):
        self._env = env
        self._action_horizon = action_horizon
        self._action_dim = action_dim
        self._profile = f"{model}_{gpu}"
        self._inference_latency = {
            int(batch_size): latency
            for batch_size, latency in INFERENCE_PROFILES[model][gpu].items()
        }

    def __call__(self) -> MockPolicy:
        return MockPolicy(
            env=self._env,
            action_horizon=self._action_horizon,
            action_dim=self._action_dim,
            inference_latency=self._inference_latency,
        )
