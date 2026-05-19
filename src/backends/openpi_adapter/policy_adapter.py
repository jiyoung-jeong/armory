"""Adapts openpi.Policy to the armory engine interface.

Ports the true GPU-parallel batch inference from the openpi fork into armory.
The armory engine expects infer_batch(), warmup(), and make_infer_request().
The official openpi Policy only exposes infer(). This class bridges the gap
while preserving RTC support by splitting RTC/non-RTC into sub-batches.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from armory.serving.schemas import InternalRequest
from armory_client.messages import InferType, RTCParams

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
            }

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

        return _model.Observation.from_dict(batched)

    def _infer_batch_group(
        self, requests: list[InternalRequest], *, use_rtc: bool
    ) -> list[dict[str, Any]]:
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

        observation = self.create_batch_obs([_rename_keys(req.observation) for req in requests])

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
                result["rtc_prev_actions"] = raw_actions[i]
                results.append(result)
            return results

        sample_kwargs = dict(self._sample_kwargs)
        if noise_to_use is not None:
            sample_kwargs["noise"] = noise_to_use

        if use_rtc:
            rtc_params = [req.params for req in requests]
            prev_actions = np.stack([np.asarray(p.prev_action) for p in rtc_params], axis=0)
            s_values = np.asarray([p.s_param for p in rtc_params], dtype=np.int32)
            d_values = np.asarray([p.d_param for p in rtc_params], dtype=np.int32)
            eh_values = np.asarray([req.max_execution_horizon for req in requests], dtype=np.int32)
            logger.debug(
                "RTC sub-batch: size=%d s=%s d=%s eh=%s",
                batch_size, s_values.tolist(), d_values.tolist(), eh_values.tolist(),
            )
            sample_kwargs["use_rtc"] = True
            sample_kwargs["prev_action"] = jnp.asarray(prev_actions)
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
            result["rtc_prev_actions"] = raw_actions[i]
            results.append(result)

        return results

    # ------------------------------------------------------------------
    # Armory engine interface
    # ------------------------------------------------------------------

    def infer_batch(self, requests: list[InternalRequest]) -> list[dict[str, Any]]:
        """GPU-parallel batch inference, splitting RTC and non-RTC into sub-batches."""
        if not requests:
            return []

        results: list[dict[str, Any] | None] = [None] * len(requests)
        grouped: dict[bool, list[int]] = {False: [], True: []}

        for i, req in enumerate(requests):
            can_rtc = (
                not self._is_pytorch_model
                and not self._is_triton_optimized
                and req.infer_type == InferType.INFERENCE_TIME_RTC
                and isinstance(req.params, RTCParams)
            )
            logger.debug(
                f"can_rtc: {can_rtc}, pytorch_model: {self._is_pytorch_model}, triton_optimized: {self._is_triton_optimized}, infer_type: {req.infer_type}, params: {req.params}"
            )
            grouped[can_rtc].append(i)

        for use_rtc, indices in grouped.items():
            if not indices:
                continue
            sub_requests = [requests[i] for i in indices]
            sub_results = self._infer_batch_group(sub_requests, use_rtc=use_rtc)
            for i, result in zip(indices, sub_results, strict=True):
                results[i] = result

        assert all(r is not None for r in results)
        return list(results)  # type: ignore[return-value]

    def make_infer_request(self) -> InternalRequest:
        return InternalRequest(
            robot_id="__warmup__",
            observation=self._make_example_fn(),
            observation_step=0,
            action_index_start=0,
            request_timestamp=time.time(),
            deadline=time.time() + 60.0,
            min_execution_horizon=0,
            max_execution_horizon=0,
            infer_type=InferType.SYNC,
            params=None,
            noise=None,
        )

    def warmup(self, max_batch_size: int) -> None:
        """Warm up both SYNC and RTC paths to trigger JAX JIT compilation."""
        example_obs = self._make_example_fn()
        warmup_requests = [
            InternalRequest(
                robot_id="__warmup__",
                observation=example_obs,
                observation_step=0,
                action_index_start=0,
                request_timestamp=0,
                deadline=0,
                min_execution_horizon=0,
                max_execution_horizon=0,
                infer_type=InferType.SYNC,
                params=None,
                noise=None,
            ),
        ]

        # Add RTC warmup if the model supports it
        if not self._is_pytorch_model and not self._is_triton_optimized:
            example_actions = (
                np.asarray(self._model.make_example_actions())
                if hasattr(self._model, "make_example_actions")
                else np.zeros((8, 7), dtype=np.float32)
            )
            warmup_requests.append(
                InternalRequest(
                    robot_id="__warmup__",
                    observation=example_obs,
                    observation_step=0,
                    action_index_start=0,
                    request_timestamp=0,
                    deadline=0,
                    min_execution_horizon=0,
                    max_execution_horizon=0,
                    infer_type=InferType.INFERENCE_TIME_RTC,
                    params=RTCParams(prev_action=example_actions, s_param=5, d_param=3),
                    noise=None,
                ),
            )

        for req in warmup_requests:
            for batch_size in range(1, max_batch_size + 1):
                logger.info("Warming up %s batch_size=%d", req.infer_type, batch_size)
                result = self.infer_batch([req] * batch_size)
        logger.info("Warmup complete; output shape: %s", result[0]["actions"].shape)

    @property
    def metadata(self) -> dict[str, Any]:
        return self._policy.metadata
