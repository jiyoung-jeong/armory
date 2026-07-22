"""Backend-neutral configuration and the policy interface used by the GPU worker."""

from __future__ import annotations

import enum
from collections.abc import Sequence
from typing import Protocol, TypeAlias, TypedDict

import numpy as np

from armory.serving.schemas import InternalRequest
from armory_client.messages import InferRequest

PolicyRequest: TypeAlias = InternalRequest | InferRequest


class EnvMode(str, enum.Enum):
    ALOHA = "aloha"
    ALOHA_SIM = "aloha_sim"
    DROID = "droid"
    LIBERO = "libero"
    LIBERO_20 = "libero_20"
    LIBERO_PI0 = "libero_pi0"
    LIBERO_PYTORCH = "libero_pytorch"
    LIBERO_REALTIME = "libero_realtime"
    REAL_SORT_LEGOS = "real_sort_legos"
    REAL_STACK_CUBES = "real_stack_cubes"
    REAL_MULTITASK = "real_multitask"
    REAL_ACT_20 = "real_act_20"
    REAL_ACT_40 = "real_act_40"
    REAL_ACT_60 = "real_act_60"
    REAL_ACT_80 = "real_act_80"
    REAL_ACT_100 = "real_act_100"


class ModelFamily(str, enum.Enum):
    PI05 = "pi05"
    GROOT_N17 = "gr00t-n1.7"


class MockPolicyConfig(Protocol):
    action_horizon: int
    action_dim: int
    model: str
    gpu: str


class PolicyResult(TypedDict):
    """One policy output row, in the same order as its input request.

    Backends may retain additional model-specific fields. The serving engine
    consumes these three keys directly and therefore requires all of them.
    """

    actions: np.ndarray
    noise: np.ndarray | None
    rtc_prev_actions: np.ndarray


class ServingPolicy(Protocol):
    """Synchronous batched-policy interface consumed by ``GpuWorker``."""

    def warmup(self, max_batch_size: int) -> None: ...

    def make_infer_request(self) -> PolicyRequest: ...

    def infer_batch(self, requests: Sequence[PolicyRequest]) -> list[PolicyResult]: ...


class PolicyFactory(Protocol):
    """Lightweight, picklable, zero-argument policy constructor."""

    def __call__(self) -> ServingPolicy: ...
