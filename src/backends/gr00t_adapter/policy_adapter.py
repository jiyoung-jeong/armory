"""Adapts gr00t.policy.Gr00tPolicy to the armory engine interface.

Observation layout from armory clients (flat dict):
  state       : np.float32 (8,)  [x, y, z, roll, pitch, yaw, grip0, grip1]
  image       : np.uint8   (H, W, 3)
  wrist_image : np.uint8   (H, W, 3)
  prompt      : str

GR00T LIBERO layout (nested, batched):
  video  / image       : (B, 1, H, W, 3) uint8
  video  / wrist_image : (B, 1, H, W, 3) uint8
  state  / x           : (B, 1, 1)  float32
  state  / y,z,roll,pitch,yaw : same
  state  / gripper     : (B, 1, 2)  float32
  language / annotation.human.action.task_description : [[str]] * B

Output from GR00T (dict of action arrays):
  {action_key: np.ndarray (B, horizon, dim), ...}
→ concatenated per-sample into actions (horizon, action_dim).
"""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence

import numpy as np

from armory.backends.types import PolicyRequest, PolicyResult
from armory.serving.rtc import InferType
from armory.serving.schemas import InternalRequest

logger = logging.getLogger(__name__)

# LIBERO state layout within the flat (8,) state vector
_STATE_SLICES: list[tuple[str, slice]] = [
    ("x", slice(0, 1)),
    ("y", slice(1, 2)),
    ("z", slice(2, 3)),
    ("roll", slice(3, 4)),
    ("pitch", slice(4, 5)),
    ("yaw", slice(5, 6)),
    ("gripper", slice(6, 8)),  # 2-element gripper joint positions
]

# Action keys in concatenation order (must match modality config)
_ACTION_KEYS = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]

LANGUAGE_KEY = "annotation.human.action.task_description"


def _obs_to_groot(requests: Sequence[PolicyRequest]) -> dict:
    """Convert a list of armory InternalRequests into a single batched GR00T observation."""
    B = len(requests)
    images = [req.observation["image"] for req in requests]
    wrist_images = [req.observation["wrist_image"] for req in requests]
    states = [req.observation["state"] for req in requests]
    prompts = [req.observation["prompt"] for req in requests]

    H, W, _ = images[0].shape

    video_image = np.stack(images).reshape(B, 1, H, W, 3)  # (B, 1, H, W, 3)
    video_wrist = np.stack(wrist_images).reshape(B, 1, H, W, 3)  # (B, 1, H, W, 3)

    state_dict = {}
    for key, sl in _STATE_SLICES:
        dim = sl.stop - sl.start
        state_dict[key] = np.stack([s[sl] for s in states]).reshape(B, 1, dim).astype(np.float32)

    return {
        "video": {
            "image": video_image,
            "wrist_image": video_wrist,
        },
        "state": state_dict,
        "language": {
            LANGUAGE_KEY: [[p] for p in prompts],
        },
    }


def _convert_gripper(actions: np.ndarray) -> np.ndarray:
    """Convert gripper from GR00T's training space to robosuite convention.

    GR00T outputs gripper in RLDS/LeRobot space: 0 = close, 1 = open.
    Robosuite (LIBERO) expects:                 +1 = close, -1 = open.

    Step 1 – normalize [0,1] → [-1,+1] with binarize:  sign(2x - 1)
    Step 2 – invert sign (robosuite polarity flip):     * -1
    Net:  0 → sign(-1)*-1 = +1 (close)
          1 → sign(+1)*-1 = -1 (open)
    """
    actions = actions.copy()
    actions[..., -1] = np.sign(2.0 * actions[..., -1] - 1.0) * -1.0
    return actions


def _groot_action_to_armory(action_dict: dict, batch_size: int) -> list[PolicyResult]:
    """Convert GR00T's batched action dict to a list of per-sample armory result dicts."""
    results: list[PolicyResult] = []
    for i in range(batch_size):
        parts = []
        for key in _ACTION_KEYS:
            arr = action_dict[key][i]  # (horizon, dim)
            parts.append(arr)
        actions = np.concatenate(parts, axis=-1)  # (horizon, total_action_dim)
        actions = _convert_gripper(actions)
        results.append(
            {
                "actions": actions,
                "noise": None,  # GR00T does not expose diffusion noise
                "rtc_prev_actions": actions,
            }
        )
    return results


def _make_example_obs() -> dict:
    """Return a single dummy observation with the correct armory flat layout."""
    return {
        "state": np.zeros(8, dtype=np.float32),
        "image": np.zeros((256, 256, 3), dtype=np.uint8),
        "wrist_image": np.zeros((256, 256, 3), dtype=np.uint8),
        "prompt": "pick up the cube",
    }


class Gr00tPolicyAdapter:
    """Wraps Gr00tPolicy for the armory GPU worker.

    Implements:
      warmup(max_batch_size)
      infer_batch(requests) -> list[dict]
      make_infer_request() -> InternalRequest
    """

    def __init__(self, policy):
        self._policy = policy

    def infer_batch(self, requests: Sequence[PolicyRequest]) -> list[PolicyResult]:
        if not requests:
            return []
        obs = _obs_to_groot(requests)
        action_dict, _ = self._policy.get_action(obs)
        return _groot_action_to_armory(action_dict, len(requests))

    def make_infer_request(self) -> InternalRequest:
        return InternalRequest(
            robot_id="__warmup__",
            observation=_make_example_obs(),
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
        request = self.make_infer_request()
        for batch_size in range(1, max_batch_size + 1):
            logger.info("Warming up GR00T batch_size=%d", batch_size)
            result = self.infer_batch([request] * batch_size)
        logger.info("Warmup complete; output shape: %s", result[0]["actions"].shape)
