"""Factory for creating armory-compatible GR00T policies."""

from __future__ import annotations

import pathlib

from armory.checkpoints import GROOT_CHECKPOINT
from gr00t_adapter.policy_adapter import Gr00tPolicyAdapter
from openpi_adapter.serve_factory import EnvMode


def _resolve(model_family: str, env: EnvMode) -> dict:
    family = GROOT_CHECKPOINT.get(model_family.lower())
    if family is None:
        raise ValueError(
            f"Unknown GR00T model family '{model_family}'. "
            f"Available: {list(GROOT_CHECKPOINT)}"
        )
    cfg = family.get(env)
    if cfg is None:
        avail = sorted(e.value for e in family)
        raise ValueError(
            f"No GR00T checkpoint for model='{model_family}' env={env.value!r}. "
            f"Available envs for this family: {avail}"
        )
    return cfg


def is_groot_model(model_family: str) -> bool:
    return model_family.lower() in GROOT_CHECKPOINT


def get_gr00t_model_dims(model_family: str, env: EnvMode) -> tuple[int, int]:
    cfg = _resolve(model_family, env)
    return cfg["action_horizon"], cfg["action_dim"]


def get_gr00t_checkpoint_label(model_family: str, env: EnvMode) -> str:
    """Human-readable checkpoint id for server metadata (Hub id + subfolder)."""
    cfg = _resolve(model_family, env)
    return f'{cfg["hub_model_id"]}/{cfg["hub_subfolder"]}'


def create_gr00t_policy(
    model_family: str,
    env: EnvMode,
    checkpoint_dir: str | pathlib.Path | None = None,
) -> Gr00tPolicyAdapter:
    """Create an armory-serving-compatible GR00T policy."""
    import torch
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.policy.gr00t_policy import Gr00tPolicy

    cfg = _resolve(model_family, env)
    tag = EmbodimentTag[cfg["embodiment_tag"]]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if checkpoint_dir is not None:
        root = pathlib.Path(checkpoint_dir).expanduser()
        subfolder = cfg["hub_subfolder"]
        # ``hf download --local-dir checkpoints/GR00T-N1.7-LIBERO`` nests weights under
        # ``libero_10/``. If the user passes the parent dir, append the known subfolder.
        if root.is_dir():
            has_root_config = (root / "config.json").exists()
            nested = root / subfolder
            if not has_root_config and nested.is_dir() and (nested / "config.json").exists():
                policy = Gr00tPolicy(
                    embodiment_tag=tag,
                    model_path=str(root),
                    checkpoint_subfolder=subfolder,
                    device=device,
                )
            else:
                policy = Gr00tPolicy(
                    embodiment_tag=tag,
                    model_path=str(checkpoint_dir),
                    checkpoint_subfolder=None,
                    device=device,
                )
        else:
            policy = Gr00tPolicy(
                embodiment_tag=tag,
                model_path=str(checkpoint_dir),
                checkpoint_subfolder=None,
                device=device,
            )
    else:
        policy = Gr00tPolicy(
            embodiment_tag=tag,
            model_path=cfg["hub_model_id"],
            checkpoint_subfolder=cfg["hub_subfolder"],
            device=device,
        )
    return Gr00tPolicyAdapter(policy)
