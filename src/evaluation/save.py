"""Episode serialization helpers used by ``evaluation.runtime.Runtime`` sinks."""

from __future__ import annotations

import dataclasses
import logging
import pathlib
from dataclasses import dataclass

import imageio
import numpy as np
import pandas as pd

from armory_client.schemas import ActionChunk
from evaluation.recording import JSONDataclass, Timestamp
from evaluation.runtime import Rollout

logger = logging.getLogger(__name__)


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


@dataclasses.dataclass(frozen=True)
class SaveMeta:
    """Static, per-robot metadata needed to lay an episode out on disk."""

    out_dir: pathlib.Path
    robot_idx: int
    control_hz: float

    # TODO: per task metadata typing
    task_suite_name: str
    task_id: int
    task_language: str
    save_video: bool = True


def save_episode(rollout: Rollout, meta: SaveMeta) -> None:
    """Persist a completed rollout. The caller transfers ownership to this function."""
    out_folder, episode_idx = _next_out_folder(meta, success=rollout.success)

    _save_metadata(out_folder, rollout, meta, episode_idx)
    Timestamp.to_csv(rollout.timestamps, out_folder / "timestamps.csv")
    save_action_chunks(rollout.action_chunks, out_folder)
    if meta.save_video:
        _save_video(out_folder, rollout, meta.control_hz)
    # _save_debug_data(out_folder, rollout)
    np.save(
        out_folder / "actions_left.npy",
        np.array(rollout.actions_left, dtype=np.int32),
    )

    np.save(out_folder / "cost_history.npy", cost_history(rollout))
    logger.info("Saved episode %d to %s", episode_idx, out_folder)


def save_action_chunks(action_chunks: tuple[ActionChunk, ...], out_folder: pathlib.Path) -> None:
    if not action_chunks:
        return
    pd.DataFrame.from_records(_action_chunk_record(chunk) for chunk in action_chunks).to_parquet(
        out_folder / "action_chunks.parquet", engine="pyarrow", index=False
    )


def _action_chunk_record(chunk: ActionChunk) -> dict[str, object]:
    record = {field.name: getattr(chunk, field.name) for field in dataclasses.fields(chunk)}
    record["actions"] = chunk.actions.tolist()
    record["noise"] = chunk.noise.tolist() if chunk.noise is not None else None
    return record


def cost_history(rollout: Rollout) -> list[float]:
    """Elapsed time from a chunk's inference request to each step that executes it."""
    costs: list[float] = []
    for ts in rollout.timestamps:
        idx = ts.action_chunk_index
        if idx is not None and idx < len(rollout.action_chunks):
            costs.append(ts.timestamp - rollout.action_chunks[idx].request_timestamp)
        else:
            costs.append(float("nan"))
    return costs


def _next_out_folder(meta: SaveMeta, success: bool) -> tuple[pathlib.Path, int]:
    robot_folder = meta.out_dir / str(meta.robot_idx)
    robot_folder.mkdir(parents=True, exist_ok=True)

    existing = [p for p in robot_folder.iterdir() if p.is_dir()]
    next_idx = max((int(p.name.split("_")[0]) for p in existing), default=-1) + 1
    success_str = "success" if success else "failure"
    out_folder = robot_folder / f"{next_idx}_{meta.task_suite_name}_{meta.task_id}_{success_str}"
    out_folder.mkdir(parents=True, exist_ok=True)
    return out_folder, next_idx


def _save_metadata(
    out_folder: pathlib.Path, rollout: Rollout, meta: SaveMeta, episode_idx: int
) -> None:
    Result(
        success=rollout.success,
        robot_idx=meta.robot_idx,
        steps_taken=len(rollout.timestamps),
        task_suite_name=meta.task_suite_name,
        task_id=meta.task_id,
        task_language=meta.task_language,
        episode_idx=episode_idx,
    ).to_json(out_folder / "metadata.json")


def _save_video(out_folder: pathlib.Path, rollout: Rollout, control_hz: float) -> None:
    images = [np.asarray(obs.image) for obs in rollout.observations]
    if not images:
        return
    imageio.mimwrite(out_folder / "out.mp4", images, fps=control_hz)


def _save_debug_data(out_folder: pathlib.Path, rollout: Rollout) -> None:
    """Save observations, noise, and actions as a single .npz — only if noise is present."""
    if not any(chunk.noise is not None for chunk in rollout.action_chunks):
        logger.debug("No debug data to save (no noise present)")
        return

    to_save: dict[str, np.ndarray] = {}
    if rollout.initial_state is not None:
        to_save["initial_state"] = rollout.initial_state

    observations = {obs.step: obs for obs in rollout.observations}
    for i, chunk in enumerate(rollout.action_chunks):
        prefix = f"chunk_{i:04d}"
        obs = observations.get(chunk.observation_step)
        if obs is not None:
            to_save[f"{prefix}/observation/state"] = obs.state
            to_save[f"{prefix}/observation/image"] = obs.image
            to_save[f"{prefix}/observation/wrist_image"] = obs.wrist_image
            if hasattr(obs, "prompt"):
                to_save[f"{prefix}/observation/prompt"] = obs.prompt
        else:
            logger.warning("No observation for chunk %d at step %d", i, chunk.observation_step)
        if chunk.noise is not None:
            to_save[f"{prefix}/noise"] = chunk.noise
        to_save[f"{prefix}/actions"] = chunk.actions
        to_save[f"{prefix}/start_step"] = chunk.observation_step
        to_save[f"{prefix}/max_execution_horizon"] = chunk.max_execution_horizon

    debug_file = out_folder / "debug_data.npz"
    np.savez_compressed(debug_file, **to_save)
    logger.info("Saved %d chunks to %s", len(rollout.action_chunks), debug_file)
