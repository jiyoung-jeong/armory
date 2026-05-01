"""Factory for creating armory-compatible GR00T policies.

This is the only file in gr00t_adapter that scripts need to import.
"""

from __future__ import annotations

import pathlib

from gr00t_adapter.policy_adapter import Gr00tPolicyAdapter

# GR00T model registry: model_name -> (embodiment_tag, checkpoint_path, action_horizon, action_dim)
_GROOT_MODELS: dict[str, dict] = {
    "gr00t-n1.7-libero": {
        "embodiment_tag": "LIBERO_PANDA",
        "default_checkpoint": "/coc/flash7/rbansal66/vvla/Isaac-GR00T/checkpoints/GR00T-N1.7-LIBERO/libero_10",
        "action_horizon": 16,
        "action_dim": 7,   # x, y, z, roll, pitch, yaw, gripper (1+1+1+1+1+1+1)
    },
}


def get_gr00t_model_dims(model_name: str) -> tuple[int, int]:
    """Return (action_horizon, action_dim) for a GR00T model name."""
    cfg = _resolve_model(model_name)
    return cfg["action_horizon"], cfg["action_dim"]


def _resolve_model(model_name: str) -> dict:
    model_name = model_name.lower()
    if model_name not in _GROOT_MODELS:
        raise ValueError(
            f"Unknown GR00T model '{model_name}'. "
            f"Available: {list(_GROOT_MODELS.keys())}"
        )
    return _GROOT_MODELS[model_name]


def create_gr00t_policy(
    model_name: str,
    checkpoint_dir: str | pathlib.Path | None = None,
) -> Gr00tPolicyAdapter:
    """Create an armory-serving-compatible GR00T policy.

    Args:
        model_name: Model identifier, e.g. "gr00t-n1.7-libero".
        checkpoint_dir: Path to the checkpoint directory.
            If None, uses the default checkpoint for the model.

    Returns:
        A Gr00tPolicyAdapter with .warmup() / .infer_batch() / .make_infer_request().
    """
    import torch
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.policy.gr00t_policy import Gr00tPolicy

    cfg = _resolve_model(model_name)
    ckpt = str(checkpoint_dir) if checkpoint_dir is not None else cfg["default_checkpoint"]
    tag = EmbodimentTag[cfg["embodiment_tag"]]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    policy = Gr00tPolicy(
        embodiment_tag=tag,
        model_path=ckpt,
        device=device,
    )
    return Gr00tPolicyAdapter(policy)


def is_groot_model(model_name: str) -> bool:
    """Return True if model_name refers to a GR00T model."""
    return model_name.lower() in _GROOT_MODELS
