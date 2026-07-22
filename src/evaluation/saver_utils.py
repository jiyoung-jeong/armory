"""Shared helpers for episode data persistence.

Used by both the sim ``Saver`` (sims.libero.subscribers.saver) and the
real-robot ``RealSaver`` (runtime.real_saver) so they emit the same on-disk
layout for the offline metrics pipeline to consume.
"""

from __future__ import annotations

import pathlib
from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from armory_client.schemas import ActionChunk, Observation
from evaluation.recording import JSONDataclass, Timestamp


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


def save_action_chunks(action_chunks: list[ActionChunk], out_folder: pathlib.Path) -> None:
    if not action_chunks:
        return
    data: dict[str, list] = {
        "chunk_id": [],
        "observation_step": [],
        "action_index_start": [],
        "execution_start_step": [],
        "actions": [],
        "min_execution_horizon": [],
        "max_execution_horizon": [],
        "request_timestamp": [],
        "response_timestamp": [],
        "request_id": [],
        "noise": [],
    }
    for chunk in action_chunks:
        data["chunk_id"].append(chunk.chunk_id)
        data["observation_step"].append(chunk.observation_step)
        data["action_index_start"].append(chunk.action_index_start)
        data["execution_start_step"].append(chunk.execution_start_step)
        data["actions"].append(chunk.actions.tolist())
        data["min_execution_horizon"].append(chunk.min_execution_horizon)
        data["max_execution_horizon"].append(chunk.max_execution_horizon)
        data["request_timestamp"].append(chunk.request_timestamp)
        data["response_timestamp"].append(chunk.response_timestamp)
        data["request_id"].append(chunk.request_id)
        data["noise"].append(chunk.noise.tolist() if chunk.noise is not None else None)
    pd.DataFrame(data).to_parquet(
        out_folder / "action_chunks.parquet", engine="pyarrow", index=False
    )


def save_actions_left(actions_left_snapshot: list[int], out_folder: pathlib.Path) -> None:
    np.save(
        out_folder / "actions_left.npy",
        np.array(actions_left_snapshot, dtype=np.int32),
    )


def save_cost_history_npy(cost_history: list[float], out_folder: pathlib.Path) -> np.ndarray:
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
    ax.set_title(f"Cost per step — robot {robot_idx} | {task_suite_name} task {task_id}")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_folder / "cost_history.png", dpi=150)
    plt.close(fig)
