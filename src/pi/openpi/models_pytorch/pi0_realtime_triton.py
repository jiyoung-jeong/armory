import json
import os
import pickle

import einops
import numpy as np
from PIL import Image
import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F  # noqa: N812

from openpi.models import model as _model
from openpi.shared.pi0_infer_batched import Pi0InferenceBatched


class Pi0TritonPytorch(nn.Module):
    def __init__(self, config, converted_checkpoint_path: str, num_views: int = 2, batch_size: int = 1):
        super().__init__()
        self.config = config
        self.pi05 = False
        self.is_triton_optimized = True
        self.batch_size = batch_size
        self.num_views = num_views

        self.all_views = ["base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"]
        self.visible_views = self.all_views[: self.num_views]

        norm_stats_dir = os.path.join(os.path.dirname(converted_checkpoint_path), "assets/physical-intelligence/libero")
        self.norm_stats = self._load_norm_stats(norm_stats_dir)

        self.policy = self._load_model(converted_checkpoint_path)

        self.action_horizon = 50
        self.action_dim = 32

    def _load_norm_stats(self, norm_stats_dir: str) -> dict:
        norm_stats_path = os.path.join(norm_stats_dir, "norm_stats.json")
        if os.path.exists(norm_stats_path):
            with open(norm_stats_path) as f:
                return json.load(f)["norm_stats"]
        return None

    def _load_model(self, checkpoint_path: str):
        with open(checkpoint_path, "rb") as f:
            weights = pickle.load(f)
        return Pi0InferenceBatched(
            checkpoint=weights, num_views=self.num_views, chunk_size=50, batch_size=self.batch_size
        )

    def sample_noise(self, device: str, batch_size: int = 1) -> torch.Tensor:
        """Sample noise for diffusion process."""
        noise_shape = (batch_size, self.action_horizon, self.action_dim)
        return torch.randn(noise_shape, device=device)

    def make_example_actions(self) -> _model.Actions:
        return torch.zeros((self.action_horizon, self.action_dim))

    def _parse_image(self, image) -> np.ndarray:
        image = np.asarray(image)
        if np.issubdtype(image.dtype, np.floating):
            image = (255 * image).astype(np.uint8)
        if image.shape[0] == 3:
            image = einops.rearrange(image, "c h w -> h w c")
        return image

    def _pad_to_dim(self, x: np.ndarray, target_dim: int, axis: int = -1) -> np.ndarray:
        current_dim = x.shape[axis]
        if current_dim < target_dim:
            pad_width = [(0, 0)] * len(x.shape)
            pad_width[axis] = (0, target_dim - current_dim)
            return np.pad(x, pad_width)
        return x

    def _resize_with_pad(self, image: np.ndarray, height: int = 224, width: int = 224) -> np.ndarray:
        pil_image = Image.fromarray(image)
        cur_width, cur_height = pil_image.size
        if cur_width == width and cur_height == height:
            return image

        ratio = max(cur_width / width, cur_height / height)
        resized_height = int(cur_height / ratio)
        resized_width = int(cur_width / ratio)
        resized_image = pil_image.resize((resized_width, resized_height), resample=Image.BILINEAR)
        zero_image = Image.new(resized_image.mode, (width, height), 0)
        pad_height = max(0, int((height - resized_height) / 2))
        pad_width = max(0, int((width - resized_width) / 2))
        zero_image.paste(resized_image, (pad_width, pad_height))
        return np.array(zero_image)

    def _normalize_image(self, image: np.ndarray) -> np.ndarray:
        return image.astype(np.float32) / 255.0 * 2.0 - 1.0

    def _normalize_state(self, state: np.ndarray, norm_stats: dict) -> np.ndarray:
        if norm_stats and "state" in norm_stats:
            state_mean = np.array(norm_stats["state"]["mean"])
            state_mean = self._pad_to_dim(state_mean, 32)
            state_std = np.array(norm_stats["state"]["std"])
            state_std = self._pad_to_dim(state_std, 32)
            return (state - state_mean) / (state_std + 1e-6)
        return None

    def _unnormalize_state(self, actions: np.ndarray, norm_stats: dict) -> np.ndarray:
        if norm_stats and "actions" in norm_stats:
            actions_mean = np.array(norm_stats["actions"]["mean"])
            actions_mean = self._pad_to_dim(actions_mean, 32)
            actions_std = np.array(norm_stats["actions"]["std"])
            actions_std = self._pad_to_dim(actions_std, 32)
            return actions * (actions_std + 1e-6) + actions_mean
        return None

    def _resize_with_pad_batch(self, images: np.ndarray, height: int = 224, width: int = 224) -> np.ndarray:
        """Fully vectorized resize with padding for batch of images using PyTorch.

        Args:
            images: (batch, H, W, C) array
            height: Target height
            width: Target width

        Returns:
            Resized images (batch, height, width, C)
        """
        batch_size, cur_height, cur_width, channels = images.shape

        # Quick path: if already correct size
        if cur_height == height and cur_width == width:
            return images

        # Convert to torch: BHWC -> BCHW
        images_torch = torch.from_numpy(images).permute(0, 3, 1, 2).float()

        # Calculate resize dimensions (maintain aspect ratio)
        ratio = max(cur_width / width, cur_height / height)
        resized_height = int(cur_height / ratio)
        resized_width = int(cur_width / ratio)

        # Resize entire batch at once
        resized = F.interpolate(
            images_torch, size=(resized_height, resized_width), mode="bilinear", align_corners=False
        )

        # Create padded output (all zeros)
        output = torch.zeros((batch_size, channels, height, width), dtype=resized.dtype, device=resized.device)

        # Calculate padding offsets (vectorized)
        pad_height = max(0, (height - resized_height) // 2)
        pad_width = max(0, (width - resized_width) // 2)

        # Place resized images in output (all at once)
        output[:, :, pad_height : pad_height + resized_height, pad_width : pad_width + resized_width] = resized

        # Convert back: BCHW -> BHWC
        output = output.permute(0, 2, 3, 1).numpy()

        return output.astype(images.dtype)

    def _apply_input_transforms(self, data: dict, action_dim: int = 32, norm_stats: dict | None = None) -> dict:
        """Apply input transforms to a batch of observations (vectorized).

        Args:
            data: Dict with batched data (batch_size, ...)
            action_dim: Target action dimension
            norm_stats: Normalization statistics

        Returns:
            Dict with transformed batched data
        """

        # Vectorized state processing
        # Pad all states at once
        current_dim = data["state"].shape[-1]
        if current_dim < action_dim:
            pad_width = [(0, 0)] * (len(data["state"].shape) - 1) + [(0, action_dim - current_dim)]
            states = np.pad(data["state"], pad_width)
        else:
            states = data["state"]

        # Vectorized state normalization
        if norm_stats and "state" in norm_stats:
            state_mean = np.array(norm_stats["state"]["mean"])
            state_mean = self._pad_to_dim(state_mean, action_dim)
            state_std = np.array(norm_stats["state"]["std"])
            state_std = self._pad_to_dim(state_std, action_dim)
            states = (states - state_mean[None, :]) / (state_std[None, :] + 1e-6)

        # Vectorized image processing
        batch_images = {}
        for view in self.all_views:
            images = data[view]  # (batch, H, W, C) or (batch, C, H, W)

            # Parse images (handle BCHW -> BHWC if needed)
            if images.ndim != 4:
                raise ValueError(f"Expected 4D image tensor, got shape {images.shape}")

            # Convert to numpy if torch tensor
            if hasattr(images, "numpy"):
                images = images.cpu().numpy()

            # Handle different layouts and dtypes
            if np.issubdtype(images.dtype, np.floating):
                images = (255 * images).astype(np.uint8)

            # BCHW -> BHWC conversion if needed
            if images.shape[1] == 3:  # BCHW
                images = np.transpose(images, (0, 2, 3, 1))

            # Resize all images in batch
            images = self._resize_with_pad_batch(images, 224, 224)

            # Normalize: vectorized across batch
            images = images.astype(np.float32) / 255.0 * 2.0 - 1.0

            batch_images[view] = images

        image_mask_dict = {
            "base_0_rgb": np.True_,
            "left_wrist_0_rgb": np.True_,
            "right_wrist_0_rgb": np.False_,
        }

        inputs = {
            "state": states,
            "image": batch_images,
            "image_mask": image_mask_dict,
        }
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]
        return inputs

    # num_steps is unused for compatibility
    @torch.no_grad()
    def sample_actions(
        self,
        device,
        observation,
        noise=None,
        num_steps=10,
    ) -> tuple[Tensor, np.ndarray]:
        # Get actual batch size from observation
        actual_batch_size = observation["state"].shape[0]

        # Check if batch size matches what the model was initialized with
        if actual_batch_size != self.batch_size:
            raise ValueError(
                f"Batch size mismatch: model initialized with batch_size={self.batch_size}, "
                f"but received observation with batch_size={actual_batch_size}. "
                f"Please reinitialize the model with the correct batch_size."
            )

        if noise is None:
            noise = self.sample_noise(device, actual_batch_size).numpy()

        transformed_inputs = self._apply_input_transforms(observation, action_dim=32, norm_stats=self.norm_stats)

        # Stack images: (batch, num_views=2, H, W, C)
        images = []
        for view in self.visible_views:
            img = transformed_inputs["image"][view]  # (batch, H, W, C)
            images.append(torch.from_numpy(img))

        observation_images = torch.stack(images, dim=1).to(torch.float32)  # (batch, 2, H, W, C)
        observation_state = torch.from_numpy(transformed_inputs["state"].astype(np.float32)).to(
            torch.float32
        )  # (batch, 32)

        normalized_observation_images = observation_images.cuda()
        normalized_observation_state = observation_state.cuda()
        diffusion_noise = torch.from_numpy(noise).to(torch.float32).cuda()

        # Forward pass through batched policy
        # action shape (batch, action_horizon, action_dim)
        actions = self.policy.forward(normalized_observation_images, normalized_observation_state, diffusion_noise)

        # Convert to numpy
        actions = actions.cpu().float().numpy()

        # FIXME: overrides superclass return type
        return actions, transformed_inputs["state"]
