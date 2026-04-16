from collections.abc import Sequence
import enum
import logging
import pathlib
from typing import Any, TypeAlias

import flax
import flax.traverse_util
import jax
import jax.numpy as jnp
import numpy as np
from openpi_client import base_policy as _base_policy
from openpi_client.messages import InferRequest
from openpi_client.messages import InferResponse
from openpi_client.messages import InferType
from openpi_client.messages import RTCParams
import torch
from typing_extensions import override

from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.policies.aloha_policy import make_aloha_example
from openpi.policies.droid_policy import make_droid_example
from openpi.policies.libero_policy import make_libero_example
from openpi.shared import array_typing as at
from openpi.shared import nnx_utils

BasePolicy: TypeAlias = _base_policy.BasePolicy

logger = logging.getLogger(__name__)


class EnvMode(enum.Enum):
    """Supported environments."""

    ALOHA = "aloha"
    ALOHA_SIM = "aloha_sim"
    DROID = "droid"
    LIBERO = "libero"
    LIBERO_PI0 = "libero_pi0"
    LIBERO_PYTORCH = "libero_pytorch"
    LIBERO_REALTIME = "libero_realtime"


class Policy(BasePolicy):
    def __init__(
        self,
        model: _model.BaseModel,
        *,
        rng: at.KeyArrayLike | None = None,
        transforms: Sequence[_transforms.DataTransformFn] = (),
        output_transforms: Sequence[_transforms.DataTransformFn] = (),
        sample_kwargs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        pytorch_device: str = "cpu",
        is_pytorch: bool = False,
        is_triton_optimized: bool = False,
    ):
        """Initialize the Policy.

        Args:
            model: The model to use for action sampling.
            rng: Random number generator key for JAX models. Ignored for PyTorch models.
            transforms: Input data transformations to apply before inference.
            output_transforms: Output data transformations to apply after inference.
            sample_kwargs: Additional keyword arguments to pass to model.sample_actions.
            metadata: Additional metadata to store with the policy.
            pytorch_device: Device to use for PyTorch models (e.g., "cpu", "cuda:0").
                          Only relevant when is_pytorch=True.
            is_pytorch: Whether the model is a PyTorch model. If False, assumes JAX model.
        """
        self._model = model
        self._input_transform = _transforms.compose(transforms)
        self._output_transform = _transforms.compose(output_transforms)
        self._sample_kwargs = sample_kwargs or {}
        self._metadata = metadata or {}
        self._is_pytorch_model = is_pytorch
        self._pytorch_device = pytorch_device
        self._is_triton_optimized = is_triton_optimized

        if self._is_pytorch_model:
            # assert isinstance(self._model, torch.nn.Module), "Model must be a PyTorch model"
            self._model = self._model.to(pytorch_device)
            self._model.eval()
        else:
            self._rng = rng or jax.random.key(0)
            self._model.sample_actions = nnx_utils.module_jit(
                self._model.sample_actions,
                static_argnames=["use_rtc"],
            )
        self._sample_actions = model.sample_actions

    @override
    def infer(
        self,
        obs: dict,
        *,
        prev_action: np.ndarray | None = None,
        use_rtc: bool = False,
        noise: np.ndarray | None = None,
        s_param: int = 5,
        d_param: int = 4,
    ) -> dict:  # type: ignore[misc]
        # Use provided noise, or sample from model if not provided
        if noise is not None:
            noise_to_use = noise
        # Sample noise from model
        elif self._is_pytorch_model:
            # PyTorch models don't use RNG, they use torch.randn internally
            noise_to_use = self._model.sample_noise(self._pytorch_device, batch_size=1)
        else:
            self._rng, noise_rng = jax.random.split(self._rng)
            noise_to_use = self._model.sample_noise(noise_rng, batch_size=1)[0]  # Remove batch dim

        # Make a copy since transformations may modify the inputs in place.
        inputs = jax.tree.map(lambda x: x, obs)
        if self._is_triton_optimized:
            # Triton policy expects already-repacked LIBERO dict keys (base_0_rgb, etc) and
            # applies its own preprocessing/normalization internally.
            triton_obs: dict[str, Any] = {
                "state": np.asarray(inputs["observation/state"])[None, ...],
                "base_0_rgb": np.asarray(inputs["observation/image"])[None, ...],
                "left_wrist_0_rgb": np.asarray(inputs["observation/wrist_image"])[None, ...],
                "right_wrist_0_rgb": np.asarray(inputs["observation/wrist_image"])[None, ...],
                # Keep prompt as object array so downstream tokenization can `.item()` if needed.
                "prompt": np.asarray([inputs.get("prompt", "")], dtype=object),
            }

            sample_kwargs = dict(self._sample_kwargs)
            # Always pass noise to model (either provided or sampled)
            sample_kwargs["noise"] = np.asarray(noise_to_use)

            sample_rng_or_pytorch_device = self._pytorch_device
            actions = self._sample_actions(sample_rng_or_pytorch_device, triton_obs, **sample_kwargs)

            actions_np = np.asarray(actions)
            if actions_np.ndim == 3 and actions_np.shape[0] == 1:
                actions_np = actions_np[0]

            # Build outputs in the same *normalized* space as the JAX model path:
            # - `actions`: normalized (H, 32) from Triton model
            # - `state`: normalized (32,) computed from raw state and checkpoint norm stats
            # FIXME later: this should go inside the triton model
            ns = getattr(self._model, "norm_stats", None)
            raw_state = np.asarray(inputs["observation/state"], dtype=np.float32)
            state_norm = np.pad(raw_state, (0, max(0, 32 - raw_state.shape[-1])), constant_values=0.0)
            if ns is not None and "state" in ns:
                mean = np.asarray(ns["state"]["mean"], dtype=np.float32)
                std = np.asarray(ns["state"]["std"], dtype=np.float32)
                mean = np.pad(mean, (0, max(0, 32 - mean.shape[-1])), constant_values=0.0)
                std = np.pad(std, (0, max(0, 32 - std.shape[-1])), constant_values=1.0)
                state_norm = (state_norm - mean) / (std + 1e-6)

            outputs: dict[str, Any] = {"state": state_norm, "actions": actions_np}

            # Apply the full output transform (including Unnormalize) to match the JAX policy.
            outputs = self._output_transform(outputs)

            # Return noise as numpy array
            outputs["noise"] = np.asarray(noise_to_use)
        else:
            inputs = self._input_transform(inputs)
            if not self._is_pytorch_model:
                # Make a batch and convert to jax.Array.
                inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
                self._rng, sample_rng_or_pytorch_device = jax.random.split(self._rng)
            else:
                # Convert inputs to PyTorch tensors and move to correct device
                inputs = jax.tree.map(
                    lambda x: torch.from_numpy(np.array(x)).to(self._pytorch_device)[None, ...],
                    inputs,
                )
                sample_rng_or_pytorch_device = self._pytorch_device

            # Prepare kwargs for sample_actions
            sample_kwargs = dict(self._sample_kwargs)
            # Convert noise_to_use to appropriate format and add batch dimension
            if self._is_pytorch_model:
                if isinstance(noise_to_use, np.ndarray):
                    noise_batched = torch.from_numpy(noise_to_use).to(self._pytorch_device)[None, ...]
                else:
                    noise_batched = noise_to_use[None, ...] if noise_to_use.ndim == 2 else noise_to_use
            else:
                noise_batched = jnp.asarray(noise_to_use)
                if noise_batched.ndim == 2:
                    noise_batched = noise_batched[None, ...]
            sample_kwargs["noise"] = noise_batched

            observation = _model.Observation.from_dict(inputs)

            actions = self._sample_actions(
                sample_rng_or_pytorch_device,
                observation,
                prev_action=prev_action,
                use_rtc=use_rtc,
                s=s_param,
                d=d_param,
                **sample_kwargs,
            )
            outputs = {
                "state": observation.state,
                "actions": actions,
            }

            # Collect data for JAX models (after JIT execution)
            if not self._is_pytorch_model and hasattr(self._model, "output_actions_save"):
                self._model.output_actions_save.append(actions)

            if self._is_pytorch_model:
                outputs = jax.tree.map(lambda x: np.asarray(x[0, ...].detach().cpu()), outputs)
            else:
                outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)

            outputs = self._output_transform(outputs)

            # Return noise as numpy array (unbatched)
            if self._is_pytorch_model and hasattr(noise_to_use, "detach"):
                outputs["noise"] = np.asarray(noise_to_use.detach().cpu())
            else:
                outputs["noise"] = np.asarray(noise_to_use)

        return outputs

    def create_batch_obs(self, observations: list[dict]) -> _model.Observation:
        # Stack observations into batch format
        batched_obs = {}

        # FIXME: no idea how the typing here works
        if self._is_triton_optimized:
            return {
                "state": np.stack([obs["observation/state"] for obs in observations], axis=0),
                "base_0_rgb": np.stack([obs["observation/image"] for obs in observations], axis=0),
                "left_wrist_0_rgb": np.stack([obs["observation/wrist_image"] for obs in observations], axis=0),
                "right_wrist_0_rgb": np.stack([obs["observation/wrist_image"] for obs in observations], axis=0),
                "prompt": np.stack([obs["prompt"] for obs in observations], axis=0),
            }

        # FIXME: don't hardcode these values
        keys = (
            "observation/state",
            "observation/image",
            "observation/wrist_image",
            "prompt",
        )
        for key in keys:
            # Stack all values for this key
            values = [obs[key] for obs in observations]
            if isinstance(values[0], np.ndarray):
                batched_obs[key] = np.stack(values, axis=0)
            elif isinstance(values[0], dict):
                # Handle nested dictionaries (like images)
                batched_obs[key] = {}
                for subkey in values[0]:
                    subvalues = [obs[key][subkey] for obs in observations]
                    if isinstance(subvalues[0], np.ndarray):
                        batched_obs[key][subkey] = np.stack(subvalues, axis=0)
                    else:
                        batched_obs[key][subkey] = subvalues
            else:
                batched_obs[key] = values

        # Make a copy since transformations may modify the inputs in place.
        inputs = jax.tree.map(lambda x: x, batched_obs)
        # Apply transforms to batched observation
        inputs = self._input_transform(inputs)

        if not self._is_pytorch_model:
            # Convert to jax.Array (already batched)
            inputs = jax.tree.map(lambda x: jnp.asarray(x), inputs)
        else:
            # Convert inputs to PyTorch tensors and move to correct device
            inputs = jax.tree.map(lambda x: torch.from_numpy(np.array(x)).to(self._pytorch_device), inputs)

        return _model.Observation.from_dict(inputs)

    def _infer_batch_group(self, requests: list[InferRequest], *, use_rtc: bool) -> list[dict[str, Any]]:
        """Run a homogeneous sub-batch.

        RTC requests require JAX model support plus per-request RTCParams. We keep
        RTC and non-RTC requests in separate sub-batches so mixed batches still work.
        """
        batch_size = len(requests)

        # Sample batched noise from model.
        if self._is_pytorch_model:
            sample_rng_or_pytorch_device = self._pytorch_device
            noise_to_use = self._model.sample_noise(self._pytorch_device, batch_size=batch_size)
            for i, request in enumerate(requests):
                noise_to_use[i] = request.noise if request.noise is not None else noise_to_use[i]
        else:
            self._rng, sample_rng_or_pytorch_device = jax.random.split(self._rng)
            noise_to_use = self._model.sample_noise(sample_rng_or_pytorch_device, batch_size=batch_size)
            for i, request in enumerate(requests):
                noise_to_use = noise_to_use.at[i].set(request.noise if request.noise is not None else noise_to_use[i])

        # FIXME: temporary hack
        def rename_keys(obs: dict) -> dict:
            return {
                "observation/state": obs["state"],
                "observation/image": obs["image"],
                "observation/wrist_image": obs["wrist_image"],
                "prompt": obs["prompt"],
            }

        observation = self.create_batch_obs([rename_keys(req.observation) for req in requests])

        if self._is_triton_optimized:
            if use_rtc:
                logger.warning(
                    "RTC requested for a Triton-optimized policy batch, but Triton batching does not implement RTC; falling back to standard sampling."
                )

            sample_kwargs = dict(self._sample_kwargs)
            sample_kwargs["noise"] = np.asarray(noise_to_use)

            # TODO Rohan: return state_norm since Triton kernels bypass input_transform for internal method. Figure out why input_transform doesn't work
            actions, state_norm = self._sample_actions(sample_rng_or_pytorch_device, observation, **sample_kwargs)
            raw_actions = np.asarray(actions)

            results: list[dict[str, Any]] = []
            for i in range(len(requests)):
                result: dict[str, Any] = {"state": state_norm[i], "actions": raw_actions[i]}
                result = self._output_transform(result)

                noise_np = np.asarray(noise_to_use)
                noise_i = noise_np[i] if noise_np.ndim == 3 else noise_np
                result["noise"] = noise_i
                # RTC guidance expects actions in model space, before output unnormalization.
                result["rtc_prev_actions"] = raw_actions[i]
                results.append(result)

            return results

        sample_kwargs = dict(self._sample_kwargs)
        sample_kwargs["noise"] = noise_to_use

        if use_rtc:
            assert not self._is_pytorch_model, "RTC is not supported for PyTorch models"

            rtc_params = [request.params for request in requests]
            assert all(isinstance(params, RTCParams) for params in rtc_params), "RTC batch requires RTCParams"
            prev_actions = np.stack([np.asarray(params.prev_action) for params in rtc_params], axis=0)
            s_values = np.asarray([params.s_param for params in rtc_params], dtype=np.int32)
            d_values = np.asarray([params.d_param for params in rtc_params], dtype=np.int32)
            logger.debug(
                "Executing RTC sub-batch: batch_size=%d s=%s d=%s prev_action_shape=%s",
                len(requests),
                s_values.tolist(),
                d_values.tolist(),
                tuple(prev_actions.shape),
            )

            sample_kwargs["use_rtc"] = True
            sample_kwargs["prev_action"] = jnp.asarray(prev_actions)
            sample_kwargs["s"] = jnp.asarray(s_values)
            sample_kwargs["d"] = jnp.asarray(d_values)

        actions = self._sample_actions(sample_rng_or_pytorch_device, observation, **sample_kwargs)
        raw_actions = np.asarray(actions.detach().cpu()) if self._is_pytorch_model else np.asarray(actions)
        outputs = {
            "state": observation.state,
            "actions": actions,
        }

        if self._is_pytorch_model:
            outputs = jax.tree.map(lambda x: np.asarray(x.detach().cpu()), outputs)
        else:
            outputs = jax.tree.map(lambda x: np.asarray(x), outputs)

        outputs = self._output_transform(outputs)

        results = []
        for i in range(len(requests)):
            result = {}
            for key, value in outputs.items():
                if isinstance(value, np.ndarray) and len(value.shape) > 0:
                    result[key] = value[i]
                else:
                    result[key] = value

            if self._is_pytorch_model and hasattr(noise_to_use, "detach"):
                noise_batch = np.asarray(noise_to_use.detach().cpu())
            else:
                noise_batch = np.asarray(noise_to_use)
            noise_i = noise_batch[i] if noise_batch.ndim == 3 else noise_batch
            result["noise"] = noise_i
            # RTC guidance expects actions in model space, before output unnormalization.
            result["rtc_prev_actions"] = raw_actions[i]
            results.append(result)

        return results

    def infer_batch(self, requests: list[InferRequest]) -> list[InferResponse]:  # FIXME: return type is wrong
        """Run inference on a batch of request.

        Args:
            obs_batch: List of InferRequest objects of the same infer_type.
            noise: Optional noise tensor for batch (shape: batch_size, action_horizon, action_dim)

        Returns:
            List of InferResponse objects, one for each input request.
        """
        if not requests:
            return []

        results: list[dict[str, Any] | None] = [None] * len(requests)
        grouped_indices: dict[bool, list[int]] = {False: [], True: []}

        for index, request in enumerate(requests):
            can_use_rtc = (
                not self._is_pytorch_model
                and not self._is_triton_optimized
                and request.infer_type == InferType.INFERENCE_TIME_RTC
                and isinstance(request.params, RTCParams)
            )
            grouped_indices[can_use_rtc].append(index)

        for use_rtc, indices in grouped_indices.items():
            if not indices:
                continue
            grouped_requests = [requests[index] for index in indices]
            grouped_results = self._infer_batch_group(grouped_requests, use_rtc=use_rtc)
            for index, result in zip(indices, grouped_results, strict=True):
                results[index] = result

        assert all(result is not None for result in results)
        return list(results)

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata

    def make_example(self) -> dict:
        assert "env" in self._metadata, "Environment not set in metadata"
        env = EnvMode(self._metadata["env"])
        if env == EnvMode.ALOHA:
            return make_aloha_example()
        if env == EnvMode.DROID:
            return make_droid_example()
        if env in [
            EnvMode.LIBERO,
            EnvMode.LIBERO_REALTIME,
            EnvMode.LIBERO_PYTORCH,
            EnvMode.LIBERO_PI0,
        ]:
            return make_libero_example()

        raise ValueError(f"Unknown environment: {env}")

    def make_example_actions(self) -> np.ndarray:
        return self._model.make_example_actions()

    # FIXME: reorganize warmup code later
    def make_infer_request(self) -> InferRequest:
        observation = self.make_example()
        return InferRequest(
            robot_id="test_robot",
            observation_step=0,
            action_start_step=0,
            execution_horizon=0,
            request_timestamp=0,
            deadline=0,
            observation=observation,
            infer_type=InferType.SYNC,
            params=None,
            noise=None,
        )

    def warmup(self, max_batch_size: int) -> None:
        """Warm up policy by running inference to trigger JIT compilation."""
        observation = self.make_example()

        requests = [
            InferRequest(
                robot_id="test_robot",
                observation_step=0,
                action_start_step=0,
                execution_horizon=0,
                request_timestamp=0,
                deadline=0,
                observation=observation,
                infer_type=InferType.SYNC,
                params=None,
                noise=None,
            ),
            InferRequest(
                robot_id="test_robot",
                observation_step=0,
                action_start_step=0,
                execution_horizon=0,
                request_timestamp=0,
                deadline=0,
                observation=observation,
                infer_type=InferType.INFERENCE_TIME_RTC,
                params=RTCParams(prev_action=self.make_example_actions(), s_param=5, d_param=3),
                noise=None,
            ),
        ]

        for request in requests:
            for batch_size in range(1, max_batch_size + 1):
                logger.info(f"Warming up {request.infer_type} for batch_size={batch_size}")
                # Warm up with full batch_size (we always pad to this size)
                batch = [request] * batch_size
                actions = self.infer_batch(batch)

        # FIXME: fix after fixing typing on infer_batch
        logger.info(f"Output size: {actions[0]['actions'].shape}")


class PolicyRecorder(_base_policy.BasePolicy):
    """Records the policy's behavior to disk."""

    def __init__(self, policy: _base_policy.BasePolicy, record_dir: str):
        self._policy = policy

        logging.info(f"Dumping policy records to: {record_dir}")
        self._record_dir = pathlib.Path(record_dir)
        self._record_dir.mkdir(parents=True, exist_ok=True)
        self._record_step = 0

    @override
    def infer(self, obs: dict) -> dict:  # type: ignore[misc]
        results = self._policy.infer(obs)

        data = {"inputs": obs, "outputs": results}
        data = flax.traverse_util.flatten_dict(data, sep="/")

        output_path = self._record_dir / f"step_{self._record_step}"
        self._record_step += 1

        np.save(output_path, np.asarray(data))
        return results
