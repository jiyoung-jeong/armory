"""Shared utilities for scripts."""

import subprocess
from typing import Any

from openpi_adapter.serve_factory import EnvMode
from openpi_adapter.serve_factory import create_policy


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


GROOT_CHECKPOINT: dict[str, dict] = {
    "gr00t-n1.7-libero": {
        "dir": "/coc/flash7/rbansal66/vvla/Isaac-GR00T/checkpoints/GR00T-N1.7-LIBERO/libero_10",
    },
}

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
