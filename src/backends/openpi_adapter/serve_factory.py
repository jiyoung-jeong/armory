"""Thin adapter: create an armory-compatible policy from an openpi checkpoint.

This is the only file in backends/openpi_adapter that scripts need to import.
Everything else (models, training configs) is an openpi internal.
"""

from __future__ import annotations

import enum
import pathlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from openpi_adapter.policy_adapter import OpenPiPolicyAdapter


class EnvMode(str, enum.Enum):
    ALOHA = "aloha"
    ALOHA_SIM = "aloha_sim"
    DROID = "droid"
    LIBERO = "libero"
    LIBERO_PI0 = "libero_pi0"
    LIBERO_PYTORCH = "libero_pytorch"
    LIBERO_REALTIME = "libero_realtime"


def _make_example_fn(env_mode: EnvMode | None):
    """Return the env-specific make_*_example() function for profiling/warmup."""
    if env_mode in (
        EnvMode.LIBERO,
        EnvMode.LIBERO_PI0,
        EnvMode.LIBERO_PYTORCH,
        EnvMode.LIBERO_REALTIME,
    ):
        from openpi.policies.libero_policy import make_libero_example

        return make_libero_example
    if env_mode == EnvMode.DROID:
        from openpi.policies.droid_policy import make_droid_example

        return make_droid_example
    from openpi.policies.aloha_policy import make_aloha_example

    return make_aloha_example


def create_policy(
    config_name: str,
    checkpoint_dir: str | pathlib.Path,
    *,
    default_prompt: str | None = None,
    sample_kwargs: dict[str, Any] | None = None,
    env_mode: EnvMode | None = None,
) -> OpenPiPolicyAdapter:
    """Create an armory-serving-compatible policy from a named openpi training config.

    Returns an OpenPiPolicyAdapter whose .infer_batch() / .warmup() / .make_infer_request()
    match the armory engine interface.
    """
    from openpi.policies import policy_config as _policy_config
    from openpi.training import config as _config

    from openpi_adapter.policy_adapter import OpenPiPolicyAdapter

    train_config = _config.get_config(config_name)
    policy = _policy_config.create_trained_policy(
        train_config,
        checkpoint_dir,
        default_prompt=default_prompt,
        sample_kwargs=sample_kwargs,
    )
    if env_mode is not None and "env" not in policy.metadata:
        policy._metadata["env"] = env_mode.value  # noqa: SLF001

    return OpenPiPolicyAdapter(policy, make_example_fn=_make_example_fn(env_mode))


def get_model_dims(config_name: str) -> tuple[int, int]:
    """Return (action_horizon, action_dim) for a training config without loading the model."""
    from openpi.training import config as _config

    train_config = _config.get_config(config_name)
    return train_config.model.action_horizon, train_config.model.action_dim


def get_config_name(config_name: str) -> str:
    """Return the canonical name for a training config (resolves aliases)."""
    from openpi.training import config as _config

    return _config.get_config(config_name).name
