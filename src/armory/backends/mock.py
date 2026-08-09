"""Lightweight serving-policy implementation used for CPU-only experiments."""

from __future__ import annotations

import json
import time
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from armory.backends.types import PolicyResult, warmup_request
from armory.serving.rtc import InferType
from armory.serving.schemas import SlotData

with Path(__file__).with_name("inference_profiles.json").open() as f:
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

    def make_infer_request(self) -> SlotData:
        # Only ever fed to infer_batch, which ignores every field but the count.
        return warmup_request({})

    def warmup(self, max_batch_size: int, infer_type: InferType) -> None:
        del max_batch_size, infer_type

    def infer_batch(self, requests: Sequence[SlotData]) -> list[PolicyResult]:
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
