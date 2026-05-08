"""Default checkpoint paths and per-env metadata for serving."""

from __future__ import annotations

from typing import TypedDict

from openpi_adapter.serve_factory import EnvMode


class OpenPiCheckpointEntry(TypedDict):
    config: str
    dir: str


OPENPI_CHECKPOINT: dict[EnvMode, OpenPiCheckpointEntry] = {
    EnvMode.ALOHA: {
        "config": "pi05_aloha",
        "dir": "gs://openpi-assets/checkpoints/pi05_base",
    },
    EnvMode.ALOHA_SIM: {
        "config": "pi0_aloha_sim",
        "dir": "gs://openpi-assets/checkpoints/pi0_aloha_sim",
    },
    EnvMode.DROID: {
        "config": "pi05_droid",
        "dir": "gs://openpi-assets/checkpoints/pi05_droid",
    },
    EnvMode.LIBERO: {
        "config": "pi05_libero",
        "dir": "gs://openpi-assets/checkpoints/pi05_libero",
    },
    EnvMode.LIBERO_PI0: {
        "config": "pi0_libero",
        "dir": "gs://openpi-assets/checkpoints/pi0_libero",
    },
    EnvMode.LIBERO_PYTORCH: {
        "config": "pi0_libero",
        "dir": "/coc/flash8/rbansal66/openpi_rollout/openpi/.cache/openpi/openpi-assets/checkpoints/pi0_libero_pytorch_openpi",
    },
    EnvMode.LIBERO_REALTIME: {
        "config": "pi0_libero",
        "dir": "/coc/flash8/rbansal66/openpi_rollout/openpi/.cache/openpi/openpi-assets/checkpoints/pi0_libero_pytorch_dexmal_mokapots",
    },
    EnvMode.REAL_SORT_LEGOS: {
        "config": "pi05_sort_legos_correct_bins_extra_data",
        "dir": "/coc/flash7/rbansal66/vvla/openpi-training/checkpoints/pi05_sort_legos_correct_bins_extra_data/sort_legos_extra_data_finetune/11000",
    },
    EnvMode.REAL_STACK_CUBES: {
        "config": "pi05_stack_cubes",
        "dir": "/coc/flash7/rbansal66/vvla/openpi-training/checkpoints/pi05_stack_cubes/stack_cubes_finetune/8000",
    },
}


class GrootCheckpointEntry(TypedDict):
    embodiment_tag: str
    hub_model_id: str
    hub_subfolder: str
    action_horizon: int
    action_dim: int


# note: Can either rely on the HF cache or point ``--policy.dir`` at ``checkpoints/GR00T-N1.7-LIBERO`` if downloaded locally
GROOT_CHECKPOINT: dict[str, dict[EnvMode, GrootCheckpointEntry]] = {
    "gr00t-n1.7": {
        EnvMode.LIBERO: {
            "embodiment_tag": "LIBERO_PANDA",
            "hub_model_id": "nvidia/GR00T-N1.7-LIBERO",
            "hub_subfolder": "libero_10",
            "action_horizon": 16,
            "action_dim": 7,
        },
    },
}
