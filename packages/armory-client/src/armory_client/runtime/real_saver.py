"""Per-robot data capture for real-robot client nodes.

Mirrors ``sims.libero.subscribers.saver.Saver`` exactly on disk so that the
existing offline metrics pipeline (``sims.libero.metrics.calculate_metrics``
and ``generate_all_plots``) operates on real-robot output without
modification.

Differences from the sim Saver:
- No environment object — accepts primitives (control_hz, prompt, success).
- Manages ``episode_idx`` internally (sim got it from the env).
- Captures ``initial_state`` from the first observation.
- Defaults ``save_video=False`` (real episodes are open-ended and video can
  be hundreds of MB).
"""

from __future__ import annotations

import logging
import pathlib
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from typing_extensions import override

from armory_client.action_chunkers.action_chunk_broker import ActionChunkBroker
from armory_client.runtime import subscriber as _subscriber
from armory_client.schemas import (
    Action,
    ActionChunk,
    JSONDataclass,
    Observation,
    Timestamp,
)

logger = logging.getLogger(__name__)


_ROBOT_IDX_RE = re.compile(r"(\d+)")


def _robot_idx_from_id(robot_id: str) -> int:
    """Extract a numeric robot_idx from a string id like 'robot_11'.

    Falls back to 0 if no digits are present (so plotting still works).
    """
    match = _ROBOT_IDX_RE.search(robot_id)
    return int(match.group(1)) if match else 0


@dataclass(frozen=True)
class Result(JSONDataclass):
    """Per-episode metadata. Schema matches sims.libero.subscribers.saver.Result."""

    robot_idx: int
    success: bool
    steps_taken: int
    task_suite_name: str
    task_id: int
    task_language: str
    episode_idx: int


@dataclass
class _EpisodeSaveData:
    """Snapshot of one episode's data, safe to hand off to a background thread."""

    timestamps: list[Timestamp]
    observations_buffer: dict[int, Observation]
    action_chunks: list[ActionChunk]
    actions_left_snapshot: list[int]
    cost_history: list[float]
    success: bool
    episode_idx: int
    initial_state: np.ndarray | None


class RealSaver(_subscriber.Subscriber):
    """Saves real-robot trajectory data; on-disk layout matches the sim Saver."""

    def __init__(
        self,
        out_dir: pathlib.Path,
        robot_id: str,
        prompt: str,
        control_hz: int,
        action_chunk_broker: ActionChunkBroker,
        save_video: bool = False,
        task_suite_name: str = "real",
        task_id: int = 0,
        success_default: bool = False,
    ) -> None:
        out_dir = pathlib.Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        self._out_dir = out_dir
        self._robot_id = robot_id
        self._robot_idx = _robot_idx_from_id(robot_id)
        self._prompt = prompt
        self._control_hz = int(control_hz)
        self._action_chunk_broker = action_chunk_broker
        self._save_video_enabled = bool(save_video)
        self._task_suite_name = task_suite_name
        self._task_id = task_id
        self._success_default = bool(success_default)
        self._episode_counter = 0
        self._executor = ThreadPoolExecutor(max_workers=5)

        # Per-episode buffers (initialized in on_episode_start).
        self._timestamps: list[Timestamp] = []
        self._observations_buffer: dict[int, Observation] = {}
        self._actions_left_snapshot: list[int] = []
        self._cost_history: list[float] = []
        self._initial_state: np.ndarray | None = None

    # ── lifecycle ────────────────────────────────────────────────

    @override
    def on_episode_start(self) -> None:
        self._timestamps = []
        self._observations_buffer = {}
        self._actions_left_snapshot = []
        self._cost_history = []
        self._initial_state = None

    @override
    def on_step(self, observation: Observation, action: Action) -> None:
        if self._initial_state is None and observation.state is not None:
            # Snapshot the very first state we see — the equivalent of sim's
            # current_initial_state.
            self._initial_state = np.asarray(observation.state).copy()

        self._observations_buffer[observation.step] = observation

        self._timestamps.append(
            Timestamp(
                timestamp=time.perf_counter(),
                action_chunk_index=action.action_chunk_index,
                action_index=action.index_in_chunk,
                env_step=observation.step,
            )
        )

        # Mirror sim Saver: snapshot the broker's queue depth after consuming
        # this step's action.
        history = self._action_chunk_broker.actions_left_history
        self._actions_left_snapshot.append(history[-1] if history else 0)

        # Cost = wall time from when the chunk's observation was sent for
        # inference to when this step actually executed an action from it.
        if action.action_chunk_index is not None:
            chunk = self._action_chunk_broker.action_chunks[action.action_chunk_index]
            cost = time.time() - chunk.request_timestamp
        else:
            cost = float("nan")
        self._cost_history.append(cost)

    @override
    def on_episode_end(self) -> None:
        data = _EpisodeSaveData(
            timestamps=self._timestamps,
            observations_buffer=self._observations_buffer,
            action_chunks=list(self._action_chunk_broker.action_chunks),
            actions_left_snapshot=self._actions_left_snapshot,
            cost_history=self._cost_history,
            success=self._success_default,
            episode_idx=self._episode_counter,
            initial_state=self._initial_state,
        )
        self._episode_counter += 1
        self._executor.submit(self._save_all, data)

    @override
    def close(self) -> None:
        self._executor.shutdown(wait=True)

    # ── disk writes (same names + formats as sim Saver) ─────────

    def _save_all(self, data: _EpisodeSaveData) -> None:
        out_folder = self._get_out_folder(data)
        try:
            self._save_metadata(out_folder, data)
            self._save_timestamps(out_folder, data)
            self._save_action_chunks(out_folder, data)
            if self._save_video_enabled:
                self._save_video(out_folder, data)
            self._save_debug_data(out_folder, data)
            self._save_actions_left(out_folder, data)
            self._save_cost_history(out_folder, data)
        except Exception:
            logger.exception("RealSaver: error writing episode %s", out_folder)

    def _get_out_folder(self, data: _EpisodeSaveData) -> pathlib.Path:
        # Use the snapshot's episode_idx (assigned at on_episode_end time) so
        # concurrent flushes don't collide on a disk-scan.
        robot_folder = self._out_dir / str(self._robot_idx)
        robot_folder.mkdir(parents=True, exist_ok=True)
        success_str = "success" if data.success else "failure"
        out_folder = (
            robot_folder
            / f"{data.episode_idx}_{self._task_suite_name}_{self._task_id}_{success_str}"
        )
        out_folder.mkdir(parents=True, exist_ok=True)
        return out_folder

    def _save_metadata(self, out_folder: pathlib.Path, data: _EpisodeSaveData) -> None:
        result = Result(
            success=data.success,
            robot_idx=self._robot_idx,
            steps_taken=len(data.timestamps),
            task_suite_name=self._task_suite_name,
            task_id=self._task_id,
            task_language=self._prompt,
            episode_idx=data.episode_idx,
        )
        result.to_json(out_folder / "metadata.json")

    def _save_timestamps(self, out_folder: pathlib.Path, data: _EpisodeSaveData) -> None:
        Timestamp.to_csv(data.timestamps, out_folder / "timestamps.csv")

    def _save_action_chunks(self, out_folder: pathlib.Path, data: _EpisodeSaveData) -> None:
        ActionChunk.to_parquet(data.action_chunks, out_folder / "action_chunks.parquet")

    def _save_video(self, out_folder: pathlib.Path, data: _EpisodeSaveData) -> None:
        images = [
            obs.image
            for obs in data.observations_buffer.values()
            if obs.image is not None
        ]
        if not images:
            return
        imageio.mimwrite(
            out_folder / "out.mp4",
            [np.asarray(x) for x in images],
            fps=self._control_hz,
        )

    def _save_debug_data(self, out_folder: pathlib.Path, data: _EpisodeSaveData) -> None:
        has_noise = any(
            getattr(chunk, "noise", None) is not None for chunk in data.action_chunks
        )
        if not has_noise:
            return

        debug_data_file = out_folder / "debug_data.npz"
        data_to_save: dict[str, np.ndarray | str] = {}

        if data.initial_state is not None:
            data_to_save["initial_state"] = data.initial_state

        for i, chunk in enumerate(data.action_chunks):
            prefix = f"chunk_{i:04d}"
            obs = data.observations_buffer.get(chunk.observation_step)
            if obs is not None:
                if obs.state is not None:
                    data_to_save[f"{prefix}/observation/state"] = obs.state
                if obs.image is not None:
                    data_to_save[f"{prefix}/observation/image"] = obs.image
                if obs.wrist_image is not None:
                    data_to_save[f"{prefix}/observation/wrist_image"] = obs.wrist_image
                if hasattr(obs, "prompt"):
                    data_to_save[f"{prefix}/observation/prompt"] = obs.prompt
            if chunk.noise is not None:
                data_to_save[f"{prefix}/noise"] = chunk.noise
            data_to_save[f"{prefix}/actions"] = chunk.actions
            data_to_save[f"{prefix}/start_step"] = chunk.observation_step
            data_to_save[f"{prefix}/execution_horizon"] = chunk.execution_horizon

        np.savez_compressed(debug_data_file, **data_to_save)

    def _save_actions_left(self, out_folder: pathlib.Path, data: _EpisodeSaveData) -> None:
        np.save(
            out_folder / "actions_left.npy",
            np.array(data.actions_left_snapshot, dtype=np.int32),
        )

    def _save_cost_history(self, out_folder: pathlib.Path, data: _EpisodeSaveData) -> None:
        costs = np.array(data.cost_history, dtype=np.float64)
        np.save(out_folder / "cost_history.npy", costs)

        if costs.size == 0:
            return

        steps = np.arange(len(costs))
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(steps, costs, linewidth=0.8, color="steelblue")
        ax.set_xlabel("Environment step")
        ax.set_ylabel("Cost (s)")
        ax.set_title(
            f"Cost per step — robot {self._robot_idx} | "
            f"{self._task_suite_name} task {self._task_id}"
        )
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_folder / "cost_history.png", dpi=150)
        plt.close(fig)
