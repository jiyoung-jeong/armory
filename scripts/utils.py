"""Shared utilities for scripts."""

import json
import time
import numpy as np
import subprocess
from typing import Any

from openpi_adapter.serve_factory import EnvMode
from openpi_adapter.serve_factory import create_policy

from armory_client.messages import InferRequest, InferType

with open("configs/inference_profiles.json", "r") as f:
    INFERENCE_PROFILES = {profile_name: {int(batch_size): latency for batch_size, latency in profile.items()} for profile_name, profile in json.load(f).items()}

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


DEFAULT_CHECKPOINT = {
    EnvMode.ALOHA: {
        "config": "pi05_aloha",
        "dir": "gs://openpi-assets/checkpoints/pi05_base",
    },
    EnvMode.ALOHA_SIM: {
        "config": "pi0_aloha_sim",
        "dir": "gs://openpi-assets/checkpoints/pi0_aloha_sim",
    },
    EnvMode.DROID: {
        "config": "pi05_droid",
        "dir": "gs://openpi-assets/checkpoints/pi05_droid",
    },
    EnvMode.LIBERO: {
        "config": "pi05_libero",
        "dir": "gs://openpi-assets/checkpoints/pi05_libero",
    },
    EnvMode.LIBERO_PI0: {
        "config": "pi0_libero",
        "dir": "gs://openpi-assets/checkpoints/pi0_libero",
    },
    EnvMode.LIBERO_PYTORCH: {
        "config": "pi0_libero",
        "dir": "/coc/flash8/rbansal66/openpi_rollout/openpi/.cache/openpi/openpi-assets/checkpoints/pi0_libero_pytorch_openpi",
    },
    EnvMode.LIBERO_REALTIME: {
        "config": "pi0_libero",
        "dir": "/coc/flash8/rbansal66/openpi_rollout/openpi/.cache/openpi/openpi-assets/checkpoints/pi0_libero_pytorch_dexmal_mokapots",
    },
}


def create_default_policy(env: EnvMode, *, batch_size: int = 1, default_prompt: str | None = None, sample_kwargs: dict | None = None):
    if checkpoint := DEFAULT_CHECKPOINT.get(env):
        return create_policy(
            checkpoint["config"],
            checkpoint["dir"],
            default_prompt=default_prompt,
            sample_kwargs=sample_kwargs,
            use_triton_optimized=(env == EnvMode.LIBERO_REALTIME),
            batch_size=batch_size,
            env_mode=env,
        )
    raise ValueError(f"Unsupported environment mode: {env}")



class _MockPolicy:
    """Stub policy implementing the armory engine interface without weights/GPU."""

    def __init__(self, *, env: str, action_horizon: int, action_dim: int, inference_latency: dict[int, float]):
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
            action_start_step=0,
            request_timestamp=now,
            deadline=now + 60.0,
            execution_horizon=0,
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
        return [
            {"actions": actions, "noise": None, "rtc_prev_actions": actions}
            for _ in requests
        ]


class _MockPolicyFactory:
    """Picklable factory for the mock policy."""

    def __init__(self, *, env: str, action_horizon: int, action_dim: int, profile: str):
        self._env = env
        self._action_horizon = action_horizon
        self._action_dim = action_dim
        self._profile = profile
        self._inference_latency = INFERENCE_PROFILES[profile]

    def __call__(self) -> _MockPolicy:
        return _MockPolicy(
            env=self._env,
            action_horizon=self._action_horizon,
            action_dim=self._action_dim,
            inference_latency=self._inference_latency,
        )
