from __future__ import annotations

import dataclasses
import logging
import pathlib
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING

import imageio
import numpy as np
from typing_extensions import override

from armory_client.action_chunkers.action_chunk_broker import ActionChunkBroker
from armory_client.schemas import Action, Observation
from armory_evaluation.recording import Timestamp
from armory_evaluation.runtime import subscriber as _subscriber
from armory_evaluation.runtime.saver_utils import (
    EpisodeSaveData,
    Result,
    plot_cost_history,
    save_action_chunks,
    save_actions_left,
    save_cost_history_npy,
    save_timestamps,
)

if TYPE_CHECKING:
    from libero.libero import benchmark

    from armory_evaluation.sims.libero.env import LiberoSimEnvironment

logger = logging.getLogger(__name__)


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
        save_video: bool = True,
        executor: ThreadPoolExecutor | None = None,
        pending_slots: threading.BoundedSemaphore | None = None,
    ) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        self._out_dir = out_dir
        self._task_suite_name = task_suite_name
        self._task_id = task_id
        self._task = task
        self._robot_idx = robot_idx
        self._environment = environment
        self._action_chunk_broker = action_chunk_broker
        self._save_video_enabled = save_video
        self._timestamps: list[Timestamp] = []
        self._control_hz = environment.control_hz
        self._observations_buffer: dict[int, Observation] = {}
        # When ``executor`` is provided, the caller owns its lifetime and the
        # per-episode ``close()`` is a no-op — this lets repeated episodes
        # within one worker submit saves to a shared pool without blocking
        # the worker between episodes (mp4 encoding can take several seconds).
        if executor is not None:
            self._executor = executor
            self._owns_executor = False
        else:
            self._executor = ThreadPoolExecutor(max_workers=5)
            self._owns_executor = True
        # Bounds the number of in-flight save jobs. Each EpisodeSaveData pins
        # the full episode's observation images (~60MB at 200 steps), so an
        # unbounded backlog can OOM a long run if encoding falls behind.
        # When the limit is hit, on_episode_end blocks (backpressure on the
        # worker) instead of queueing another episode's buffers.
        self._pending_slots = pending_slots

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
        data = EpisodeSaveData(
            timestamps=self._timestamps,
            observations_buffer=self._observations_buffer,
            # Shallow-copy the broker list in case it gets reset between episodes.
            action_chunks=list(self._action_chunk_broker.action_chunks),
            actions_left_snapshot=self._actions_left_snapshot,
            cost_history=self._cost_history,
            success=self._environment.current_success,
            episode_idx=self._environment.episode_idx,
            initial_state=self._environment.current_initial_state,
        )

        if self._pending_slots is not None:
            self._pending_slots.acquire()
            try:
                self._executor.submit(self._save_all_bounded, data)
            except BaseException:
                self._pending_slots.release()
                raise
        else:
            self._executor.submit(self._save_all, data)

    def close(self) -> None:
        # Only drain the executor if this Saver owns it. When a shared
        # executor was injected, the worker process is responsible for
        # shutting it down once at the very end.
        if self._owns_executor:
            self._executor.shutdown(wait=True)

    def _save_all_bounded(self, data: EpisodeSaveData) -> None:
        try:
            self._save_all(data)
        finally:
            self._pending_slots.release()

    def _save_all(self, data: EpisodeSaveData) -> None:
        out_folder, dir_episode_idx = self._get_out_folder(data)
        data = dataclasses.replace(data, episode_idx=dir_episode_idx)
        self._save_metadata(out_folder, data)
        logger.info(f"Saving timestamps to {out_folder / 'timestamps.csv'}")
        save_timestamps(data.timestamps, out_folder)
        logger.info(f"Saving action chunks to {out_folder}")
        save_action_chunks(data.action_chunks, out_folder)
        if self._save_video_enabled:
            self._save_video(out_folder, data)
        self._save_debug_data(out_folder, data)
        path = out_folder / "actions_left.npy"
        save_actions_left(data.actions_left_snapshot, out_folder)
        logger.info(f"Saved actions_left to {path}")
        self._save_cost_history(out_folder, data)

    def _get_out_folder(self, data: EpisodeSaveData) -> tuple[pathlib.Path, int]:
        robot_folder = self._out_dir / str(self._robot_idx)
        pathlib.Path(robot_folder).mkdir(parents=True, exist_ok=True)

        existing = list(robot_folder.iterdir())
        next_idx = max([int(p.name.split("_")[0]) for p in existing if p.is_dir()], default=-1) + 1
        success_str = "success" if data.success else "failure"
        out_folder = (
            robot_folder / f"{next_idx}_{self._task_suite_name}_{self._task_id}_{success_str}"
        )
        pathlib.Path(out_folder).mkdir(parents=True, exist_ok=True)
        return pathlib.Path(out_folder), next_idx

    def _save_metadata(self, out_folder: pathlib.Path, data: EpisodeSaveData) -> None:
        logger.info(f"Saving metadata to {out_folder / 'metadata.json'}")
        result = Result(
            success=data.success,
            robot_idx=self._robot_idx,
            steps_taken=len(data.timestamps),
            task_suite_name=self._task_suite_name,
            task_id=self._task_id,
            task_language=self._task.language,
            episode_idx=data.episode_idx,
        )
        result.to_json(out_folder / "metadata.json")

    def _save_video(self, out_folder: pathlib.Path, data: EpisodeSaveData) -> None:
        logger.info(f"Saving video to {out_folder / 'out.mp4'}")
        images = [obs.image for obs in data.observations_buffer.values()]
        imageio.mimwrite(
            out_folder / "out.mp4",
            [np.asarray(x) for x in images],
            fps=self._control_hz,  # NOTE: saving in control hz fps for now
        )

    def _save_debug_data(self, out_folder: pathlib.Path, data: EpisodeSaveData) -> None:
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
            data_to_save[f"{prefix}/max_execution_horizon"] = chunk.max_execution_horizon

        # Save as compressed npz
        np.savez_compressed(debug_data_file, **data_to_save)
        logger.info(f"Saved {len(data.action_chunks)} chunks to {debug_data_file}")

    def _save_cost_history(self, out_folder: pathlib.Path, data: EpisodeSaveData) -> None:
        npy_path = out_folder / "cost_history.npy"
        costs = save_cost_history_npy(data.cost_history, out_folder)
        logger.info(f"Saved cost_history to {npy_path}")

        plot_path = out_folder / "cost_history.png"
        plot_cost_history(
            costs,
            out_folder,
            robot_idx=self._robot_idx,
            task_suite_name=self._task_suite_name,
            task_id=self._task_id,
        )
        logger.info(f"Saved cost_history plot to {plot_path}")
