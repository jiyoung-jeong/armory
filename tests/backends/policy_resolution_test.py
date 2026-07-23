"""Characterization tests for policy selection at server startup.

These tests deliberately stop at the picklable policy factory boundary: loading a
factory would import model runtimes and weights, while resolving one should remain
CPU-only and deterministic.
"""

from __future__ import annotations

import pickle
from types import SimpleNamespace

import numpy as np
import pytest

from armory.backends import registry as backend_registry
from armory.backends.mock import MockPolicyFactory
from armory.backends.types import EnvMode
from armory_client.messages import InferRequest
from openpi_adapter.serve_factory import EnvMode as OpenPiEnvMode


@pytest.mark.parametrize(
    ("policy_config", "policy_dir", "expected_config", "expected_dir"),
    [
        (None, None, "pi05_libero", "gs://openpi-assets/checkpoints/pi05_libero"),
        ("custom_config", "/models/custom", "custom_config", "/models/custom"),
    ],
    ids=["default-checkpoint", "custom-checkpoint"],
)
def test_resolve_openpi_policy_metadata_and_factory(
    monkeypatch: pytest.MonkeyPatch,
    policy_config: str | None,
    policy_dir: str | None,
    expected_config: str,
    expected_dir: str,
) -> None:
    monkeypatch.setattr(backend_registry, "get_model_dims", lambda config_name: (17, 23))

    resolved = backend_registry.resolve_policy(
        model="pi05",
        env=EnvMode.LIBERO,
        policy_config=policy_config,
        policy_dir=policy_dir,
        max_batch_size=5,
        num_steps=9,
        scheduling_algorithm="round-robin",
    )

    assert resolved.metadata.config_name == expected_config
    assert resolved.metadata.checkpoint_dir == expected_dir
    assert resolved.metadata.action_horizon == 17
    assert resolved.metadata.action_dim == 23
    assert resolved.metadata.num_steps == 9
    assert resolved.metadata.max_batch_size == 5
    assert resolved.metadata.env == "libero"
    assert resolved.metadata.scheduling_algorithm == "round-robin"
    assert resolved.metadata.scheduler_kwargs is None

    assert isinstance(resolved.factory, backend_registry.OpenPiPolicyFactory)
    assert resolved.factory.config_name == expected_config
    assert resolved.factory.checkpoint_dir == expected_dir
    assert resolved.factory.num_steps == 9
    assert resolved.factory.env_mode is EnvMode.LIBERO


@pytest.mark.parametrize(
    ("policy_dir", "expected_checkpoint"),
    [
        (None, "nvidia/GR00T-N1.7-LIBERO/libero_10"),
        ("/models/gr00t", "/models/gr00t"),
    ],
    ids=["default-checkpoint", "custom-checkpoint"],
)
def test_resolve_gr00t_policy_metadata_and_factory(
    policy_dir: str | None,
    expected_checkpoint: str,
) -> None:
    resolved = backend_registry.resolve_policy(
        model="gr00t-n1.7",
        env=EnvMode.LIBERO,
        # Existing behavior: this OpenPI-oriented field is ignored by GR00T.
        policy_config="ignored-for-gr00t",
        policy_dir=policy_dir,
        max_batch_size=4,
        num_steps=11,
        scheduling_algorithm="lookahead-actions",
    )

    assert resolved.metadata.config_name == "gr00t-n1.7/libero"
    assert resolved.metadata.checkpoint_dir == expected_checkpoint
    assert resolved.metadata.action_horizon == 16
    assert resolved.metadata.action_dim == 7
    assert resolved.metadata.num_steps == 11
    assert resolved.metadata.max_batch_size == 4
    assert resolved.metadata.env == "libero"
    assert resolved.metadata.scheduling_algorithm == "lookahead-actions"
    assert resolved.metadata.scheduler_kwargs is None

    assert isinstance(resolved.factory, backend_registry.Gr00tPolicyFactory)
    assert resolved.factory.model_family == "gr00t-n1.7"
    assert resolved.factory.env is EnvMode.LIBERO
    assert resolved.factory.checkpoint_dir == policy_dir


def test_resolve_mock_policy_metadata_and_factory() -> None:
    mock = SimpleNamespace(action_horizon=20, action_dim=7, model="pi05", gpu="l40s")

    resolved = backend_registry.resolve_policy(
        model="pi05",
        env=EnvMode.LIBERO,
        policy_config=None,
        policy_dir=None,
        max_batch_size=5,
        num_steps=8,
        scheduling_algorithm="greedy-deadline",
        mock=mock,
    )

    assert resolved.metadata.config_name == "mock"
    assert resolved.metadata.checkpoint_dir == ""
    assert resolved.metadata.action_horizon == 20
    assert resolved.metadata.action_dim == 7
    assert resolved.metadata.num_steps == 8
    assert resolved.metadata.max_batch_size == 5
    assert resolved.metadata.env == "libero"
    assert resolved.metadata.scheduling_algorithm == "greedy-deadline"

    assert isinstance(resolved.factory, MockPolicyFactory)
    assert resolved.factory._env == "libero"
    assert resolved.factory._action_horizon == 20
    assert resolved.factory._action_dim == 7
    assert resolved.factory._profile == "pi05_l40s"
    assert resolved.factory._inference_latency[1] == pytest.approx(0.073)
    assert resolved.factory._inference_latency[5] == pytest.approx(0.2111)


def test_mock_factory_constructs_the_same_engine_policy_contract() -> None:
    factory = MockPolicyFactory(
        env="libero",
        action_horizon=20,
        action_dim=7,
        model="pi05",
        gpu="l40s",
    )
    policy = factory()
    request = policy.make_infer_request()
    assert isinstance(request, InferRequest)
    assert policy.metadata == {"env": "libero"}

    policy._inference_latency[1] = 0.0
    [result] = policy.infer_batch([request])

    assert set(result) == {"actions", "noise", "rtc_prev_actions"}
    assert result["actions"].shape == (20, 7)
    assert result["actions"].dtype == np.float32
    assert result["noise"] is None
    assert result["rtc_prev_actions"] is result["actions"]


@pytest.mark.parametrize("backend", ["openpi", "gr00t", "mock"])
def test_resolved_policy_factories_round_trip_through_pickle(
    monkeypatch: pytest.MonkeyPatch,
    backend: str,
) -> None:
    monkeypatch.setattr(backend_registry, "get_model_dims", lambda config_name: (17, 23))

    if backend == "openpi":
        resolved = backend_registry.resolve_policy(
            model="pi05",
            env=EnvMode.LIBERO,
            policy_config=None,
            policy_dir=None,
            max_batch_size=5,
            num_steps=9,
            scheduling_algorithm="round-robin",
        )
    elif backend == "gr00t":
        resolved = backend_registry.resolve_policy(
            model="gr00t-n1.7",
            env=EnvMode.LIBERO,
            policy_config=None,
            policy_dir=None,
            max_batch_size=5,
            num_steps=9,
            scheduling_algorithm="round-robin",
        )
    else:
        resolved = backend_registry.resolve_policy(
            model="pi05",
            env=EnvMode.LIBERO,
            policy_config=None,
            policy_dir=None,
            max_batch_size=5,
            num_steps=9,
            scheduling_algorithm="round-robin",
            mock=SimpleNamespace(action_horizon=20, action_dim=7, model="pi05", gpu="l40s"),
        )

    restored = pickle.loads(pickle.dumps(resolved.factory))

    assert type(restored) is type(resolved.factory)
    assert vars(restored) == vars(resolved.factory)
    assert type(restored).__module__.startswith("armory.backends.")


def test_backend_env_mode_is_canonical() -> None:
    assert OpenPiEnvMode is EnvMode
