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
}


class GrootCheckpointEntry(TypedDict):
    embodiment_tag: str
    dir: str
    action_horizon: int
    action_dim: int


# model_family -> EnvMode -> checkpoint row (parallel to OPENPI_CHECKPOINT)
GROOT_CHECKPOINT: dict[str, dict[EnvMode, GrootCheckpointEntry]] = {
    "gr00t-n1.7": {
        EnvMode.LIBERO: {
            "embodiment_tag": "LIBERO_PANDA",
            "dir": "/coc/flash7/rbansal66/vvla/Isaac-GR00T/checkpoints/GR00T-N1.7-LIBERO/libero_10",
            "action_horizon": 16,
            "action_dim": 7,
        },
    },
}
