"""Shared utilities for benchmarking scripts.

Contains common functions used by benchmark.py, benchmark_offline.py, and serve_policy.py.
"""

import subprocess
from typing import Any

from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.policies.policy import EnvMode
from openpi.training import config as _config


def get_gpu_info() -> dict[str, Any]:
    """Get GPU information using nvidia-smi.

    Returns:
        Dictionary with GPU information:
        - gpu_available: bool
        - gpu_name: str (if available)
        - driver_version: str (if available)
        - memory_total: str (if available)
    """
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


# Default checkpoints for each environment
# These are referenced by create_default_policy()
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


def create_default_policy(
    env: EnvMode, *, batch_size: int = 1, default_prompt: str | None = None, sample_kwargs: dict[str, Any] | None = None
) -> _policy.Policy:
    """Create a default policy for the given environment.

    Args:
        env: Environment mode (e.g., LIBERO_PI0, LIBERO_PYTORCH, etc.)
        batch_size: Batch size for inference (default: 1)
        default_prompt: Default prompt if not provided in observation (default: None)
        sample_kwargs: Additional kwargs for sampling (e.g., {"num_steps": 10})

    Returns:
        Policy instance

    Raises:
        ValueError: If environment mode is not supported
    """
    if checkpoint := DEFAULT_CHECKPOINT.get(env):
        return _policy_config.create_trained_policy(
            _config.get_config(checkpoint["config"]),
            checkpoint["dir"],
            default_prompt=default_prompt,
            sample_kwargs=sample_kwargs,
            use_triton_optimized=(env == EnvMode.LIBERO_REALTIME),
            batch_size=batch_size,
        )
    raise ValueError(f"Unsupported environment mode: {env}")
