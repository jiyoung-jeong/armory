"""Adapts openpi.Policy to the armory engine interface.

Ports the true GPU-parallel batch inference from the openpi fork into armory.
The armory engine expects infer_batch(), warmup(), and make_infer_request().
The official openpi Policy only exposes infer(). This class bridges the gap
while preserving RTC support by splitting RTC/non-RTC into sub-batches.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from armory.backends.types import PolicyResult, warmup_request
from armory.serving.rtc import InferType, RTCParams
from armory.serving.schemas import SlotData

logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG)


def _recursive_stack(list_of_dicts: list[dict]) -> dict:
    """Recursively stack a list of dicts-of-arrays into a single dict-of-batched-arrays."""
    result = {}
    for key in list_of_dicts[0]:
        values = [d[key] for d in list_of_dicts]
        if isinstance(values[0], dict):
            result[key] = _recursive_stack(values)
        elif isinstance(values[0], np.ndarray):
            result[key] = np.stack(values, axis=0)
        else:
            # scalars (np.bool_, np.float32, etc.) — convert to array
            result[key] = np.asarray(values)
    return result


def _rename_keys(obs: dict) -> dict:
    """Convert observation keys to the openpi policy input format.

    armory_client sends short keys (state/image/wrist_image/prompt).
    The official openpi make_*_example() returns long keys (observation/state etc.).
    Handle both so warmup and live inference use the same code path.
    """
    if "state" in obs:
        return {
            "observation/state": obs["state"],
            "observation/image": obs["image"],
            "observation/wrist_image": obs["wrist_image"],
            "prompt": obs["prompt"],
        }
    # Already long-key format (from official openpi make_*_example)
    return obs


class OpenPiPolicyAdapter:
    """Wraps openpi.policies.policy.Policy for the armory GPU worker.

    Adds GPU-parallel batch inference (infer_batch), warmup, and dummy request
    generation on top of the official openpi Policy which only exposes infer().
    """

    def __init__(self, policy, *, make_example_fn: Callable[[], dict]):
        self._policy = policy
        self._make_example_fn = make_example_fn

    # ------------------------------------------------------------------
    # Convenience accessors into the wrapped policy's internals
    # ------------------------------------------------------------------

    @property
    def _model(self):
        return self._policy._model

    @property
    def _input_transform(self):
        return self._policy._input_transform

    @property
    def _output_transform(self):
        return self._policy._output_transform

    @property
    def _sample_kwargs(self) -> dict:
        return self._policy._sample_kwargs

    @property
    def _sample_actions(self):
        return self._policy._sample_actions

    @property
    def _is_pytorch_model(self) -> bool:
        return self._policy._is_pytorch_model

    @property
    def _pytorch_device(self) -> str:
        return self._policy._pytorch_device

    @property
    def _is_triton_optimized(self) -> bool:
        return getattr(self._policy, "_is_triton_optimized", False)

    def _get_rng(self):
        return self._policy._rng

    def _split_rng(self):
        self._policy._rng, key = jax.random.split(self._policy._rng)
        return key

    # ------------------------------------------------------------------
    # Batch inference internals
    # ------------------------------------------------------------------

    def _sample_noise(self, rng_or_device, batch_size: int):
        """Sample noise for a batch. Falls back to None if model lacks sample_noise."""
        if hasattr(self._model, "sample_noise"):
            return self._model.sample_noise(rng_or_device, batch_size=batch_size)
        return None

    def create_batch_obs(self, observations: list[dict]):
        """Stack a list of observation dicts into a batched model Observation.

        Applies _input_transform to each observation independently (so per-observation
        ops like prompt tokenization work correctly) then stacks the numerical outputs.
        """
        from openpi.models import model as _model

        if self._is_triton_optimized:
            return {
                "state": np.stack([obs["observation/state"] for obs in observations]),
                "base_0_rgb": np.stack([obs["observation/image"] for obs in observations]),
                "left_wrist_0_rgb": np.stack(
                    [obs["observation/wrist_image"] for obs in observations]
                ),
                "right_wrist_0_rgb": np.stack(
                    [obs["observation/wrist_image"] for obs in observations]
                ),
                "prompt": np.stack([obs["prompt"] for obs in observations]),
            }, None

        # Apply transform to each observation independently (tokenization, image parsing etc.)
        transformed = [
            self._input_transform(jax.tree.map(lambda x: x, obs)) for obs in observations
        ]

        # Recursively stack nested dicts/arrays into a single batched dict
        batched = _recursive_stack(transformed)

        if not self._is_pytorch_model:
            batched = jax.tree.map(
                lambda x: jnp.asarray(x) if isinstance(x, np.ndarray) else x, batched
            )
        else:
            import torch

            batched = jax.tree.map(
                lambda x: (
                    torch.from_numpy(np.array(x)).to(self._pytorch_device)
                    if isinstance(x, np.ndarray)
                    else x
                ),
                batched,
            )

        prev_actions = batched.pop("actions", None)
        return _model.Observation.from_dict(batched), prev_actions

    def _infer_batch_group(
        self, requests: Sequence[SlotData], *, infer_type: InferType
    ) -> list[PolicyResult]:
        """Run a homogeneous sub-batch (all RTC or all non-RTC) in a single GPU call."""
        batch_size = len(requests)

        # Sample or collect noise for the batch
        if self._is_pytorch_model:
            rng_or_device = self._pytorch_device
            noise_to_use = self._sample_noise(rng_or_device, batch_size)
            if noise_to_use is not None:
                for i, req in enumerate(requests):
                    if req.noise is not None:
                        noise_to_use[i] = req.noise
        else:
            rng_or_device = self._split_rng()
            noise_to_use = self._sample_noise(rng_or_device, batch_size)
            if noise_to_use is not None:
                for i, req in enumerate(requests):
                    if req.noise is not None:
                        noise_to_use = noise_to_use.at[i].set(req.noise)

        obs_dicts = [_rename_keys(req.observation) for req in requests]
        if infer_type != InferType.SYNC:
            obs_dicts = [
                {**obs, "actions": np.array(req.params.prev_action)}
                for obs, req in zip(obs_dicts, requests, strict=True)
            ]
        observation, prev_actions = self.create_batch_obs(obs_dicts)

        if self._is_triton_optimized:
            sample_kwargs = dict(self._sample_kwargs)
            if noise_to_use is not None:
                sample_kwargs["noise"] = np.asarray(noise_to_use)
            actions, state_norm = self._sample_actions(rng_or_device, observation, **sample_kwargs)
            raw_actions = np.asarray(actions)
            results = []
            for i in range(batch_size):
                result: dict[str, Any] = {"state": state_norm[i], "actions": raw_actions[i]}
                result = self._output_transform(result)
                noise_np = np.asarray(noise_to_use) if noise_to_use is not None else None
                result["noise"] = (
                    noise_np[i] if (noise_np is not None and noise_np.ndim == 3) else noise_np
                )
                result["rtc_prev_actions"] = result["actions"]
                results.append(result)
            return results

        sample_kwargs = dict(self._sample_kwargs)
        if noise_to_use is not None:
            sample_kwargs["noise"] = noise_to_use

        if infer_type != InferType.SYNC:
            rtc_params = [req.params for req in requests]
            s_values = np.asarray([p.s_param for p in rtc_params], dtype=np.int32)
            d_values = np.asarray([p.d_param for p in rtc_params], dtype=np.int32)
            eh_values = np.asarray([req.max_execution_horizon for req in requests], dtype=np.int32)
            logger.debug(
                "RTC sub-batch: size=%d s=%s d=%s eh=%s",
                batch_size,
                s_values.tolist(),
                d_values.tolist(),
                eh_values.tolist(),
            )
            if infer_type == InferType.TRAIN_TIME_RTC:
                sample_kwargs["use_train_rtc"] = True
            else:
                sample_kwargs["use_rtc"] = True
            sample_kwargs["prev_action"] = prev_actions
            sample_kwargs["s"] = jnp.asarray(s_values)
            sample_kwargs["d"] = jnp.asarray(d_values)
            sample_kwargs["execution_horizon"] = jnp.asarray(eh_values)

        actions = self._sample_actions(rng_or_device, observation, **sample_kwargs)
        if self._is_pytorch_model:
            raw_actions = np.asarray(actions.detach().cpu())
            raw_state = np.asarray(observation.state.detach().cpu())
        else:
            raw_actions = np.asarray(actions)
            raw_state = np.asarray(observation.state)

        # Apply output_transform per-sample: transforms like LiberoOutputs use [:, :7] designed
        # for single-sample shapes (action_horizon, action_dim_padded), not batched shapes.
        noise_np = np.asarray(noise_to_use) if noise_to_use is not None else None
        results = []
        for i in range(batch_size):
            sample = {"actions": raw_actions[i], "state": raw_state[i]}
            result = self._output_transform(sample)
            result["noise"] = (
                noise_np[i] if (noise_np is not None and noise_np.ndim == 3) else noise_np
            )
            result["rtc_prev_actions"] = result["actions"]
            results.append(result)

        return results

    # ------------------------------------------------------------------
    # Armory engine interface
    # ------------------------------------------------------------------

    def infer_batch(self, requests: Sequence[SlotData]) -> list[PolicyResult]:
        """GPU-parallel batch inference, splitting RTC and non-RTC into sub-batches."""
        if not requests:
            return []

        results: list[PolicyResult | None] = [None] * len(requests)
        grouped: dict[InferType, list[int]] = {}

        for i, req in enumerate(requests):
            can_rtc = (
                not self._is_pytorch_model
                and not self._is_triton_optimized
                and req.infer_type in (InferType.INFERENCE_TIME_RTC, InferType.TRAIN_TIME_RTC)
                and isinstance(req.params, RTCParams)
            )
            logger.debug(
                f"can_rtc: {can_rtc}, pytorch_model: {self._is_pytorch_model}, triton_optimized: {self._is_triton_optimized}, infer_type: {req.infer_type}, params: {req.params}"
            )
            grouped.setdefault(req.infer_type if can_rtc else InferType.SYNC, []).append(i)

        for infer_type, indices in grouped.items():
            sub_requests = [requests[i] for i in indices]
            sub_results = self._infer_batch_group(sub_requests, infer_type=infer_type)
            for i, result in zip(indices, sub_results, strict=True):
                results[i] = result

        assert all(r is not None for r in results)
        return list(results)  # type: ignore[return-value]

    def make_infer_request(self) -> SlotData:
        return warmup_request(self._make_example_fn())

    def warmup(self, max_batch_size: int, infer_type: InferType) -> None:
        example_obs = self._make_example_fn()
        for batch_size in range(1, max_batch_size + 1):
            logger.info("Warming up %s batch_size=%d", InferType.SYNC, batch_size)
            result = self.infer_batch([warmup_request(example_obs)] * batch_size)

        if (
            infer_type != InferType.SYNC
            and not self._is_pytorch_model
            and not self._is_triton_optimized
        ):
            req = warmup_request(
                example_obs,
                infer_type=infer_type,
                params=RTCParams(
                    prev_action=np.zeros_like(result[0]["actions"]), s_param=5, d_param=3
                ),
            )
            for batch_size in range(1, max_batch_size + 1):
                logger.info("Warming up %s batch_size=%d", infer_type, batch_size)
                result = self.infer_batch([req] * batch_size)

        logger.info("Warmup complete; output shape: %s", result[0]["actions"].shape)

    @property
    def metadata(self) -> dict[str, Any]:
        return self._policy.metadata
