"""Shared utilities for scripts."""

import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from armory.checkpoints import OPENPI_CHECKPOINT
from armory_client.schemas import ServerMetadata
from openpi_adapter.serve_factory import EnvMode
from openpi_adapter.serve_factory import create_policy
from openpi_adapter.serve_factory import get_model_dims
from gr00t_adapter.serve_factory import (  # noqa: E501
    create_gr00t_policy,
    get_gr00t_checkpoint_label,
    get_gr00t_model_dims,
    is_groot_model,
)


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

@dataclass
class ResolvedPolicy:
    metadata: ServerMetadata
    factory: Callable


class _OpenPiFactory:
    """Picklable callable that constructs an OpenPI policy in the GPU subprocess."""

    def __init__(self, config_name: str, checkpoint_dir: str, default_prompt: str | None,
                 num_steps: int, env_mode: EnvMode):
        self.config_name = config_name
        self.checkpoint_dir = checkpoint_dir
        self.default_prompt = default_prompt
        self.num_steps = num_steps
        self.env_mode = env_mode

    def __call__(self):
        return create_policy(
            self.config_name,
            self.checkpoint_dir,
            default_prompt=self.default_prompt,
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
    default_prompt: str | None,
    scheduling_algorithm: str,
) -> ResolvedPolicy:
    """Resolve model backend, checkpoint, and dims into a ResolvedPolicy.

    Dispatches to GR00T or OpenPI based on `model`. All routing lives here —
    callers (serve.py) stay model-agnostic.
    """
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
        factory = _OpenPiFactory(config_name, checkpoint_dir, default_prompt, num_steps, env)

    return ResolvedPolicy(metadata=metadata, factory=factory)


def create_default_policy(env: EnvMode, *, batch_size: int = 1, default_prompt: str | None = None, sample_kwargs: dict | None = None):
    if checkpoint := OPENPI_CHECKPOINT.get(env):
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
