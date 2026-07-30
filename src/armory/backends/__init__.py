"""Shared backend contracts and model-selection types.

Backend implementations remain lazily imported by their factories. Keep this
module lightweight so importing CLI/config types never loads JAX, Torch, or
model weights.
"""

from armory.backends.types import (
    EnvMode,
    MockPolicyConfig,
    ModelFamily,
    PolicyFactory,
    PolicyResult,
    ServingPolicy,
    warmup_request,
)

__all__ = [
    "EnvMode",
    "MockPolicyConfig",
    "ModelFamily",
    "PolicyFactory",
    "PolicyResult",
    "ServingPolicy",
    "warmup_request",
]
