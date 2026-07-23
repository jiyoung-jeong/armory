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

# TODO: leaving this here, this needs to be incorporated somehow
from __future__ import annotations

import abc
import logging
import pathlib
import re
import time
from concurrent.futures import ThreadPoolExecutor

import imageio
import numpy as np
from typing_extensions import override

from armory_client.action_chunkers.action_chunk_broker import ActionChunkBroker
from armory_client.schemas import Action, Observation
from evaluation.recording import Timestamp
from evaluation.runtime import Rollout
from evaluation.save import cost_history
from evaluation.saver_utils import (
    Result,
    plot_cost_history,
    save_action_chunks,
    save_actions_left,
    save_cost_history_npy,
    save_timestamps,
)

logger = logging.getLogger(__name__)


_ROBOT_IDX_RE = re.compile(r"(\d+)")


class Subscriber(abc.ABC):
    """Subscribes to events in the runtime.

    Subscribers can be used to save data, visualize, etc.
    """

    @abc.abstractmethod
    def on_episode_start(self) -> None:
        """Called when an episode starts."""

    @abc.abstractmethod
    def on_step(self, observation: Observation, action: Action) -> None:
        """Append a step to the episode."""

    @abc.abstractmethod
    def on_episode_end(self) -> None:
        """Called when an episode ends."""

    def close(self) -> None:
        """Called when the runtime is closing."""
        pass


def _robot_idx_from_id(robot_id: str) -> int:
    """Extract a numeric robot_idx from a string id like 'robot_11'.

    Falls back to 0 if no digits are present (so plotting still works).
    """
    match = _ROBOT_IDX_RE.search(robot_id)
    return int(match.group(1)) if match else 0


class RealSaver(Subscriber):
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
        self._observations: list[Observation] = []
        self._initial_state: np.ndarray | None = None

    # ── lifecycle ────────────────────────────────────────────────

    @override
    def on_episode_start(self) -> None:
        self._timestamps = []
        self._observations = []
        self._initial_state = None

    @override
    def on_step(self, observation: Observation, action: Action) -> None:
        if self._initial_state is None and observation.state is not None:
            # Snapshot the very first state we see — the equivalent of sim's
            # current_initial_state.
            self._initial_state = np.asarray(observation.state).copy()

        self._observations.append(observation)

        self._timestamps.append(
            Timestamp(
                # Wall-clock (seconds since epoch). Real robots run in separate
                # processes on separate machines, so time.perf_counter() origins
                # diverge — using it here would push cross-robot column offsets
                # in metrics._build_actions_left_matrix into the millions.
                timestamp=time.time(),
                action_chunk_index=action.action_chunk_index,
                action_index=action.index_in_chunk,
                env_step=observation.step,
            )
        )

    @override
    def on_episode_end(self) -> None:
        action_chunks, actions_left = self._action_chunk_broker.snapshot_episode_data()
        rollout = Rollout(
            observations=tuple(self._observations),
            timestamps=tuple(self._timestamps),
            success=self._success_default,
            initial_state=self._initial_state,
            action_chunks=tuple(action_chunks),
            actions_left=tuple(actions_left),
        )
        episode_idx = self._episode_counter
        self._episode_counter += 1
        self._executor.submit(self._save_all, rollout, episode_idx)

    @override
    def close(self) -> None:
        self._executor.shutdown(wait=True)

    # ── disk writes (same names + formats as sim Saver) ─────────

    def _save_all(self, rollout: Rollout, episode_idx: int) -> None:
        out_folder = self._get_out_folder(rollout, episode_idx)
        try:
            self._save_metadata(out_folder, rollout, episode_idx)
            save_timestamps(rollout.timestamps, out_folder)
            save_action_chunks(rollout.action_chunks, out_folder)
            if self._save_video_enabled:
                self._save_video(out_folder, rollout)
            self._save_debug_data(out_folder, rollout)
            save_actions_left(rollout.actions_left, out_folder)
            self._save_cost_history(out_folder, rollout)
        except Exception:
            logger.exception("RealSaver: error writing episode %s", out_folder)

    def _get_out_folder(self, rollout: Rollout, episode_idx: int) -> pathlib.Path:
        # Use the snapshot's episode_idx (assigned at on_episode_end time) so
        # concurrent flushes don't collide on a disk-scan.
        robot_folder = self._out_dir / str(self._robot_idx)
        robot_folder.mkdir(parents=True, exist_ok=True)
        success_str = "success" if rollout.success else "failure"
        out_folder = (
            robot_folder / f"{episode_idx}_{self._task_suite_name}_{self._task_id}_{success_str}"
        )
        out_folder.mkdir(parents=True, exist_ok=True)
        return out_folder

    def _save_metadata(self, out_folder: pathlib.Path, rollout: Rollout, episode_idx: int) -> None:
        result = Result(
            success=rollout.success,
            robot_idx=self._robot_idx,
            steps_taken=len(rollout.timestamps),
            task_suite_name=self._task_suite_name,
            task_id=self._task_id,
            task_language=self._prompt,
            episode_idx=episode_idx,
        )
        result.to_json(out_folder / "metadata.json")

    def _save_video(self, out_folder: pathlib.Path, rollout: Rollout) -> None:
        images = [obs.image for obs in rollout.observations if obs.image is not None]
        if not images:
            return
        imageio.mimwrite(
            out_folder / "out.mp4",
            [np.asarray(x) for x in images],
            fps=self._control_hz,
        )

    def _save_debug_data(self, out_folder: pathlib.Path, rollout: Rollout) -> None:
        has_noise = any(
            getattr(chunk, "noise", None) is not None for chunk in rollout.action_chunks
        )
        if not has_noise:
            return

        debug_data_file = out_folder / "debug_data.npz"
        data_to_save: dict[str, np.ndarray | str] = {}

        if rollout.initial_state is not None:
            data_to_save["initial_state"] = rollout.initial_state

        observations = {obs.step: obs for obs in rollout.observations}
        for i, chunk in enumerate(rollout.action_chunks):
            prefix = f"chunk_{i:04d}"
            obs = observations.get(chunk.observation_step)
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
            data_to_save[f"{prefix}/max_execution_horizon"] = chunk.max_execution_horizon

        np.savez_compressed(debug_data_file, **data_to_save)

    def _save_cost_history(self, out_folder: pathlib.Path, rollout: Rollout) -> None:
        costs = save_cost_history_npy(cost_history(rollout), out_folder)
        if costs.size == 0:
            return
        plot_cost_history(
            costs,
            out_folder,
            robot_idx=self._robot_idx,
            task_suite_name=self._task_suite_name,
            task_id=self._task_id,
        )
