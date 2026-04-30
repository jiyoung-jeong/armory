import logging
import pathlib
import time
import imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Optional, Tuple
import dataclasses
from dataclasses import dataclass
from armory_client.runtime import subscriber as _subscriber
from typing_extensions import override
from armory_client.action_chunkers.action_chunk_broker import ActionChunkBroker
from armory_client.schemas import (
    Timestamp,
    JSONDataclass,
    ActionChunk,
    Observation,
    Action,
)
from libero.libero import benchmark
from sims.libero.env import LiberoSimEnvironment

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Result(JSONDataclass):
    robot_idx: int
    success: bool
    steps_taken: int
    task_suite_name: str
    task_id: int
    task_language: str
    episode_idx: int


@dataclass
class _EpisodeSaveData:
    """Snapshot of all data needed to persist one episode, safe to hand off to a thread."""

    timestamps: List[Timestamp]
    observations_buffer: Dict[int, Observation]
    action_chunks: List[ActionChunk]
    actions_left_snapshot: List[int]
    cost_history: List[float]
    current_success: bool
    episode_idx: int
    initial_state: Optional[np.ndarray]


class Saver(_subscriber.Subscriber):
    """Saves episode data by offloading I/O to a background thread pool."""

    def __init__(
        self,
        out_dir: pathlib.Path,
        environment: LiberoSimEnvironment,
        action_chunk_broker: ActionChunkBroker,
        task_suite_name: str,
        task_id: int,
        task: benchmark.Task,
        robot_idx: int,
    ) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        self._out_dir = out_dir
        self._task_suite_name = task_suite_name
        self._task_id = task_id
        self._task = task
        self._robot_idx = robot_idx
        self._environment = environment
        self._action_chunk_broker = action_chunk_broker
        self._timestamps: List[Timestamp] = []
        self._control_hz = environment.control_hz
        self._observations_buffer: Dict[int, Observation] = {}
        self._executor = ThreadPoolExecutor(max_workers=5)

    @override
    def on_episode_start(self) -> None:
        self._timestamps = []
        self._action_chunk_indices = []
        self._observations_buffer = {}
        self._actions_left_snapshot: list[int] = []
        self._cost_history: list[float] = []

    @override
    def on_step(self, observation: Observation, action: Action) -> None:
        # Store observation for debug data and video reconstruction
        self._observations_buffer[observation.step] = observation

        self._timestamps.append(
            Timestamp(
                timestamp=time.perf_counter(),
                action_chunk_index=action.action_chunk_index,
                action_index=action.index_in_chunk,
                env_step=observation.step,
            )
        )

        # Snapshot broker queue length after this step's action was consumed.
        # broker.infer() already recorded into _actions_left_history; mirror it here
        # by reading the latest entry (avoids a second lock acquisition).
        history = self._action_chunk_broker.actions_left_history
        self._actions_left_snapshot.append(history[-1] if history else 0)

        # Cost: elapsed time from when the observation was sent for inference
        # until this step executes that chunk's action.
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
            # Shallow-copy the broker list in case it gets reset between episodes.
            action_chunks=list(self._action_chunk_broker.action_chunks),
            actions_left_snapshot=self._actions_left_snapshot,
            cost_history=self._cost_history,
            current_success=self._environment.current_success,
            episode_idx=self._environment.episode_idx,
            initial_state=self._environment.current_initial_state,
        )

        self._executor.submit(self._save_all, data)

    def close(self) -> None:
        self._executor.shutdown(wait=True)

    def _save_all(self, data: _EpisodeSaveData) -> None:
        out_folder, dir_episode_idx = self._get_out_folder(data)
        data = dataclasses.replace(data, episode_idx=dir_episode_idx)
        self._save_metadata(out_folder, data)
        self._save_timestamps(out_folder, data)
        self._save_action_chunks(out_folder, data)
        self._save_video(out_folder, data)
        self._save_debug_data(out_folder, data)
        self._save_actions_left(out_folder, data)
        self._save_cost_history(out_folder, data)

    def _get_out_folder(self, data: _EpisodeSaveData) -> Tuple[pathlib.Path, int]:
        robot_folder = self._out_dir / str(self._robot_idx)
        pathlib.Path(robot_folder).mkdir(parents=True, exist_ok=True)

        existing = list(robot_folder.iterdir())
        next_idx = (
            max([int(p.name.split("_")[0]) for p in existing if p.is_dir()], default=-1)
            + 1
        )
        success_str = "success" if data.current_success else "failure"
        out_folder = (
            robot_folder
            / f"{next_idx}_{self._task_suite_name}_{self._task_id}_{success_str}"
        )
        pathlib.Path(out_folder).mkdir(parents=True, exist_ok=True)
        return pathlib.Path(out_folder), next_idx

    def _save_metadata(self, out_folder: pathlib.Path, data: _EpisodeSaveData) -> None:
        logger.info(f"Saving metadata to {out_folder / 'metadata.json'}")
        result = Result(
            success=data.current_success,
            robot_idx=self._robot_idx,
            steps_taken=len(data.timestamps),
            task_suite_name=self._task_suite_name,
            task_id=self._task_id,
            task_language=self._task.language,
            episode_idx=data.episode_idx,
        )
        result.to_json(out_folder / "metadata.json")

    def _save_timestamps(
        self, out_folder: pathlib.Path, data: _EpisodeSaveData
    ) -> None:
        logger.info(f"Saving timestamps to {out_folder / 'timestamps.csv'}")
        Timestamp.to_csv(data.timestamps, out_folder / "timestamps.csv")

    def _save_action_chunks(
        self, out_folder: pathlib.Path, data: _EpisodeSaveData
    ) -> None:
        logger.info(f"Saving action chunks to {out_folder}")
        ActionChunk.to_parquet(data.action_chunks, out_folder / "action_chunks.parquet")

    def _save_video(self, out_folder: pathlib.Path, data: _EpisodeSaveData) -> None:
        logger.info(f"Saving video to {out_folder / 'out.mp4'}")
        images = [obs.image for obs in data.observations_buffer.values()]
        imageio.mimwrite(
            out_folder / "out.mp4",
            [np.asarray(x) for x in images],
            fps=self._control_hz,  # NOTE: saving in control hz fps for now
        )

    def _save_debug_data(
        self, out_folder: pathlib.Path, data: _EpisodeSaveData
    ) -> None:
        """Save debug data as a single .npz file with observations, noise, and actions."""
        # Check if we have noise data
        has_noise = any(chunk.noise is not None for chunk in data.action_chunks)
        if not has_noise:
            logger.debug("No debug data to save (no noise present)")
            return

        debug_data_file = out_folder / "debug_data.npz"
        logger.info(f"Saving debug data to {debug_data_file}")

        # Build data dict for .npz file
        data_to_save = {}

        # Save the initial state used for this episode
        if data.initial_state is not None:
            data_to_save["initial_state"] = data.initial_state

        for i, chunk in enumerate(data.action_chunks):
            prefix = f"chunk_{i:04d}"

            # Save observation that triggered this inference
            obs = data.observations_buffer.get(chunk.observation_step)
            if obs is not None:
                data_to_save[f"{prefix}/observation/state"] = obs.state
                data_to_save[f"{prefix}/observation/image"] = obs.image
                data_to_save[f"{prefix}/observation/wrist_image"] = obs.wrist_image
                if hasattr(obs, "prompt"):
                    data_to_save[f"{prefix}/observation/prompt"] = obs.prompt
            else:
                logger.warning(
                    f"No observation found for chunk {i} at step {chunk.observation_step}"
                )

            # Save noise (should always be present)
            if chunk.noise is not None:
                data_to_save[f"{prefix}/noise"] = chunk.noise

            # Save actions (final robot-ready actions)
            data_to_save[f"{prefix}/actions"] = chunk.actions

            # Save metadata
            # TODO: fix this
            data_to_save[f"{prefix}/start_step"] = chunk.observation_step
            data_to_save[f"{prefix}/execution_horizon"] = chunk.execution_horizon

        # Save as compressed npz
        np.savez_compressed(debug_data_file, **data_to_save)
        logger.info(f"Saved {len(data.action_chunks)} chunks to {debug_data_file}")

    def _save_actions_left(
        self, out_folder: pathlib.Path, data: _EpisodeSaveData
    ) -> None:
        path = out_folder / "actions_left.npy"
        np.save(path, np.array(data.actions_left_snapshot, dtype=np.int32))
        logger.info(f"Saved actions_left to {path}")

    def _save_cost_history(
        self, out_folder: pathlib.Path, data: _EpisodeSaveData
    ) -> None:
        costs = np.array(data.cost_history, dtype=np.float64)
        npy_path = out_folder / "cost_history.npy"
        np.save(npy_path, costs)
        logger.info(f"Saved cost_history to {npy_path}")

        plot_path = out_folder / "cost_history.png"
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
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)
        logger.info(f"Saved cost_history plot to {plot_path}")
