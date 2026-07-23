"""Shared utilities for scripts."""

from __future__ import annotations

import json
import subprocess
import time
from collections.abc import Callable
from typing import Any, NamedTuple

import numpy as np

from armory.checkpoints import OPENPI_CHECKPOINT
from armory.serving.protocol import ServerMetadata
from armory_client.messages import InferRequest, InferType
from evaluation.types import JsonArgs  # noqa: F401  re-exported for scripts/serve.py
from gr00t_adapter.serve_factory import (  # noqa: E501
    create_gr00t_policy,
    get_gr00t_checkpoint_label,
    get_gr00t_model_dims,
    is_groot_model,
)
from openpi_adapter.serve_factory import EnvMode, create_policy, get_model_dims

with open("configs/inference_profiles.json") as f:
    INFERENCE_PROFILES = json.load(f)


def get_gpu_info() -> dict[str, Any]:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        gpu_info = result.stdout.strip().split(", ")
        return {
            "gpu_available": True,
            "gpu_name": gpu_info[0],
            "driver_version": gpu_info[1],
            "memory_total": gpu_info[2],
        }
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return {"gpu_available": False}


# ---------------------------------------------------------------------------
# Policy resolution – single entry point for all model backends
# ---------------------------------------------------------------------------


class ResolvedPolicy(NamedTuple):
    metadata: ServerMetadata
    factory: Callable


class _OpenPiFactory:
    """Picklable callable that constructs an OpenPI policy in the GPU subprocess."""

    def __init__(
        self,
        config_name: str,
        checkpoint_dir: str,
        num_steps: int,
        env_mode: EnvMode,
    ):
        self.config_name = config_name
        self.checkpoint_dir = checkpoint_dir
        self.num_steps = num_steps
        self.env_mode = env_mode

    def __call__(self):
        return create_policy(
            self.config_name,
            self.checkpoint_dir,
            sample_kwargs={"num_steps": self.num_steps},
            env_mode=self.env_mode,
        )


class _Gr00tFactory:
    """Picklable callable that constructs a GR00T policy in the GPU subprocess."""

    def __init__(self, model_family: str, env: EnvMode, checkpoint_dir: str | None):
        self.model_family = model_family
        self.env = env
        self.checkpoint_dir = checkpoint_dir

    def __call__(self):
        return create_gr00t_policy(self.model_family, self.env, self.checkpoint_dir)


def resolve_policy(
    *,
    model: str,
    env: EnvMode,
    policy_config: str | None,
    policy_dir: str | None,
    max_batch_size: int,
    num_steps: int,
    scheduling_algorithm: str,
    mock: Any = None,
) -> ResolvedPolicy:
    """Resolve model backend, checkpoint, and dims into a ResolvedPolicy.

    Dispatches to GR00T or OpenPI based on `model`. All routing lives here —
    callers (serve.py) stay model-agnostic.
    """
    if mock is not None:
        metadata = ServerMetadata(
            config_name="mock",
            checkpoint_dir="",
            action_horizon=mock.action_horizon,
            action_dim=mock.action_dim,
            num_steps=num_steps,
            max_batch_size=max_batch_size,
            env=env.value,
            scheduling_algorithm=scheduling_algorithm,
        )
        factory = _MockPolicyFactory(
            env=env.value,
            action_horizon=mock.action_horizon,
            action_dim=mock.action_dim,
            model=mock.model,
            gpu=mock.gpu,
        )
        return ResolvedPolicy(metadata=metadata, factory=factory)

    if is_groot_model(model):
        groot_ckpt_override = policy_dir
        checkpoint_label = groot_ckpt_override or get_gr00t_checkpoint_label(model, env)
        action_horizon, action_dim = get_gr00t_model_dims(model, env)
        metadata = ServerMetadata(
            config_name=f"{model}/{env.value}",
            checkpoint_dir=checkpoint_label,
            action_horizon=action_horizon,
            action_dim=action_dim,
            num_steps=num_steps,
            max_batch_size=max_batch_size,
            env=env.value,
            scheduling_algorithm=scheduling_algorithm,
        )
        factory = _Gr00tFactory(model, env, groot_ckpt_override)

    else:
        if policy_config is not None and policy_dir is not None:
            config_name, checkpoint_dir = policy_config, policy_dir
        elif ckpt := OPENPI_CHECKPOINT.get(env):
            config_name, checkpoint_dir = ckpt["config"], ckpt["dir"]
        else:
            raise ValueError(f"No default checkpoint for env={env}. Pass --policy explicitly.")
        action_horizon, action_dim = get_model_dims(config_name)
        metadata = ServerMetadata(
            config_name=config_name,
            checkpoint_dir=checkpoint_dir,
            action_horizon=action_horizon,
            action_dim=action_dim,
            num_steps=num_steps,
            max_batch_size=max_batch_size,
            env=env.value,
            scheduling_algorithm=scheduling_algorithm,
        )
        factory = _OpenPiFactory(config_name, checkpoint_dir, num_steps, env)

    return ResolvedPolicy(metadata=metadata, factory=factory)


def create_default_policy(
    env: EnvMode,
    *,
    batch_size: int = 1,
    sample_kwargs: dict | None = None,
):
    if checkpoint := OPENPI_CHECKPOINT.get(env):
        return create_policy(
            checkpoint["config"],
            checkpoint["dir"],
            sample_kwargs=sample_kwargs,
            use_triton_optimized=(env == EnvMode.LIBERO_REALTIME),
            batch_size=batch_size,
            env_mode=env,
        )
    raise ValueError(f"Unsupported environment mode: {env}")


class _MockPolicy:
    """Stub policy implementing the armory engine interface without weights/GPU."""

    def __init__(
        self, *, env: str, action_horizon: int, action_dim: int, inference_latency: dict[int, float]
    ):
        self._action_horizon = action_horizon
        self._action_dim = action_dim
        self._inference_latency = inference_latency
        self.metadata = {"env": env}

    def make_infer_request(self) -> InferRequest:
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
            infer_type=InferType.SYNC,
            params=None,
            noise=None,
        )

    def warmup(self, max_batch_size: int) -> None:
        del max_batch_size

    def infer_batch(self, requests: list[InferRequest]) -> list[dict[str, Any]]:
        inference_latency = self._inference_latency[len(requests)]
        now = time.time()
        while time.time() - now < inference_latency:
            time.sleep(0.001)
        actions = np.zeros((self._action_horizon, self._action_dim), dtype=np.float32)
        return [{"actions": actions, "noise": None, "rtc_prev_actions": actions} for _ in requests]


class _MockPolicyFactory:
    """Picklable factory for the mock policy."""

    def __init__(self, *, env: str, action_horizon: int, action_dim: int, model: str, gpu: str):
        self._env = env
        self._action_horizon = action_horizon
        self._action_dim = action_dim
        self._profile = f"{model}_{gpu}"
        self._inference_latency = {
            int(batch_size): latency
            for batch_size, latency in INFERENCE_PROFILES[model][gpu].items()
        }

    def __call__(self) -> _MockPolicy:
        return _MockPolicy(
            env=self._env,
            action_horizon=self._action_horizon,
            action_dim=self._action_dim,
            inference_latency=self._inference_latency,
        )
