"""Shared helpers for episode data persistence.

Used by both the sim ``Saver`` (sims.libero.subscribers.saver) and the
real-robot ``RealSaver`` (armory_client.runtime.real_saver) so they emit the
same on-disk layout for the offline metrics pipeline to consume.
"""

from __future__ import annotations

import pathlib
from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from armory_client.schemas import (
    ActionChunk,
    JSONDataclass,
    Observation,
    Timestamp,
)


@dataclass(frozen=True)
class Result(JSONDataclass):
    """Per-episode metadata persisted to ``metadata.json``."""

    robot_idx: int
    success: bool
    steps_taken: int
    task_suite_name: str
    task_id: int
    task_language: str
    episode_idx: int


@dataclass
class EpisodeSaveData:
    """Snapshot of one episode's data, safe to hand off to a background thread."""

    timestamps: list[Timestamp]
    observations_buffer: dict[int, Observation]
    action_chunks: list[ActionChunk]
    actions_left_snapshot: list[int]
    cost_history: list[float]
    success: bool
    episode_idx: int
    initial_state: np.ndarray | None


def save_timestamps(timestamps: list[Timestamp], out_folder: pathlib.Path) -> None:
    Timestamp.to_csv(timestamps, out_folder / "timestamps.csv")


def save_action_chunks(
    action_chunks: list[ActionChunk], out_folder: pathlib.Path
) -> None:
    ActionChunk.to_parquet(action_chunks, out_folder / "action_chunks.parquet")


def save_actions_left(
    actions_left_snapshot: list[int], out_folder: pathlib.Path
) -> None:
    np.save(
        out_folder / "actions_left.npy",
        np.array(actions_left_snapshot, dtype=np.int32),
    )


def save_cost_history_npy(
    cost_history: list[float], out_folder: pathlib.Path
) -> np.ndarray:
    costs = np.array(cost_history, dtype=np.float64)
    np.save(out_folder / "cost_history.npy", costs)
    return costs


def plot_cost_history(
    costs: np.ndarray,
    out_folder: pathlib.Path,
    robot_idx: int,
    task_suite_name: str,
    task_id: int,
) -> None:
    steps = np.arange(len(costs))
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(steps, costs, linewidth=0.8, color="steelblue")
    ax.set_xlabel("Environment step")
    ax.set_ylabel("Cost (s)")
    ax.set_title(
        f"Cost per step — robot {robot_idx} | "
        f"{task_suite_name} task {task_id}"
    )
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_folder / "cost_history.png", dpi=150)
    plt.close(fig)
