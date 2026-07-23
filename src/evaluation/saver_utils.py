from __future__ import annotations

import pathlib
from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd

from armory_client.schemas import ActionChunk
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


def save_timestamps(timestamps: tuple[Timestamp, ...], out_folder: pathlib.Path) -> None:
    Timestamp.to_csv(timestamps, out_folder / "timestamps.csv")


def save_action_chunks(action_chunks: tuple[ActionChunk, ...], out_folder: pathlib.Path) -> None:
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


def save_actions_left(actions_left_snapshot: tuple[int, ...], out_folder: pathlib.Path) -> None:
    np.save(
        out_folder / "actions_left.npy",
        np.array(actions_left_snapshot, dtype=np.int32),
    )


def save_cost_history_npy(cost_history: list[float], out_folder: pathlib.Path) -> np.ndarray:
    costs = np.array(cost_history, dtype=np.float64)
    np.save(out_folder / "cost_history.npy", costs)
    return costs
