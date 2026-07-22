"""Resolve CLI model selections into metadata and picklable policy factories."""

from __future__ import annotations

from collections.abc import Callable
from typing import NamedTuple

from armory.backends.mock import MockPolicyFactory
from armory.backends.types import EnvMode, MockPolicyConfig, PolicyFactory, ServingPolicy
from armory.checkpoints import OPENPI_CHECKPOINT
from armory.serving.protocol import ServerMetadata
from gr00t_adapter.serve_factory import (
    create_gr00t_policy,
    get_gr00t_checkpoint_label,
    get_gr00t_model_dims,
    is_groot_model,
)
from openpi_adapter.serve_factory import create_policy, get_model_dims


class ResolvedPolicy(NamedTuple):
    metadata: ServerMetadata
    factory: PolicyFactory


class OpenPiPolicyFactory:
    """Picklable callable that constructs an OpenPI policy in the GPU subprocess."""

    def __init__(
        self,
        config_name: str,
        checkpoint_dir: str,
        num_steps: int,
        env_mode: EnvMode,
    ) -> None:
        self.config_name = config_name
        self.checkpoint_dir = checkpoint_dir
        self.num_steps = num_steps
        self.env_mode = env_mode

    def __call__(self) -> ServingPolicy:
        return create_policy(
            self.config_name,
            self.checkpoint_dir,
            sample_kwargs={"num_steps": self.num_steps},
            env_mode=self.env_mode,
        )


class Gr00tPolicyFactory:
    """Picklable callable that constructs a GR00T policy in the GPU subprocess."""

    def __init__(self, model_family: str, env: EnvMode, checkpoint_dir: str | None) -> None:
        self.model_family = model_family
        self.env = env
        self.checkpoint_dir = checkpoint_dir

    def __call__(self) -> ServingPolicy:
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
    mock: MockPolicyConfig | None = None,
) -> ResolvedPolicy:
    """Resolve model backend, checkpoint, dimensions, and worker factory."""
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
        factory = MockPolicyFactory(
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
        factory: PolicyFactory = Gr00tPolicyFactory(model, env, groot_ckpt_override)
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
        factory = OpenPiPolicyFactory(config_name, checkpoint_dir, num_steps, env)

    return ResolvedPolicy(metadata=metadata, factory=factory)


def create_default_policy(
    env: EnvMode,
    *,
    batch_size: int = 1,
    sample_kwargs: dict | None = None,
):
    """Legacy convenience helper retained for compatibility."""
    if checkpoint := OPENPI_CHECKPOINT.get(env):
        # This call is intentionally preserved verbatim; the helper predates
        # the current create_policy signature and has no in-repo callers.
        creator: Callable = create_policy
        return creator(
            checkpoint["config"],
            checkpoint["dir"],
            sample_kwargs=sample_kwargs,
            use_triton_optimized=(env == EnvMode.LIBERO_REALTIME),
            batch_size=batch_size,
            env_mode=env,
        )
    raise ValueError(f"Unsupported environment mode: {env}")
