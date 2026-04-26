from dataclasses import dataclass
from armory_client.runtime import environment as _environment
from armory_client.schemas import Action
from armory_client.schemas import Observation
import numpy as np
from armory_client import image_tools
from libero.libero.envs import OffScreenRenderEnv
from sims.libero import utils
from typing import List
from typing_extensions import override

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
NUM_STEPS_WAIT = 10


@dataclass
class LiberoObservation(Observation):
    image: np.ndarray
    wrist_image: np.ndarray
    prompt: str


class LiberoSimEnvironment(_environment.Environment):
    """Wraps OffScreenRenderEnv into the openpi_client.runtime.Environment interface.
    Runs episodes for different initial states for ONLY ONE task.

    This environment:
    - Uses pre-collected initial states from LIBERO.
    - On reset(), loads the next initial state and waits for objects to settle.
    - get_observation() returns the dict expected by the Libero VLA policy.
    - apply_action() steps the underlying simulator with the provided action.
    """

    def __init__(
        self,
        env: OffScreenRenderEnv,
        task_description: str,
        initial_states: np.ndarray,
        *,
        resize_size: int = 224,
        max_episode_steps: int = 300,
        control_hz: float = 100.0,
    ) -> None:
        self._env = env
        self._task_description = task_description
        self._initial_states = initial_states
        self._resize_size = resize_size
        self._max_episode_steps = max_episode_steps
        self._control_hz = control_hz

        self._episode_idx = 0
        self._done = True
        self._step_counter = 0
        self._last_obs = None
        self._episode_results: List[bool] = []
        self._current_frames: List[np.ndarray] = []
        self._current_success = False

    def reset(self) -> None:
        """Reset environment to next initial state and wait for object stabilization."""
        if self._episode_idx >= len(self._initial_states):
            # TODO: need to decide semantics here
            return

        self._env.reset()
        obs = self._env.set_init_state(self._initial_states[self._episode_idx])

        # Let objects fall / settle
        for _ in range(NUM_STEPS_WAIT):
            obs, _, _, _ = self._env.step(LIBERO_DUMMY_ACTION)

        self._last_obs = obs
        self._done = False
        self._step_counter = 0
        self._current_success = False
        self._episode_idx += 1

    def is_episode_complete(self) -> bool:
        return self._done

    def get_observation(self) -> LiberoObservation:
        if self._last_obs is None:
            raise RuntimeError("Observation is not set. Call reset() first.")

        obs = self._last_obs

        # IMPORTANT: rotate 180 degrees to match train preprocessing
        img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
        wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
        img = image_tools.convert_to_uint8(
            image_tools.resize_with_pad(img, self._resize_size, self._resize_size)
        )
        wrist_img = image_tools.convert_to_uint8(
            image_tools.resize_with_pad(wrist_img, self._resize_size, self._resize_size)
        )

        state = np.concatenate(
            (
                obs["robot0_eef_pos"],
                utils._quat2axisangle(obs["robot0_eef_quat"]),
                obs["robot0_gripper_qpos"],
            )
        )

        return LiberoObservation(
            step=self._step_counter,
            state=state,
            image=img,
            wrist_image=wrist_img,
            prompt=str(self._task_description),
        )

    def apply_action(self, action: Action) -> None:
        """Take one or more low-level action steps in the LIBERO simulator."""
        act = action.action

        # Always execute at least one step with the new action
        obs, _, libero_success, info = self._env.step(act.tolist())
        # LIBERO overrides `done` with _check_success(), losing horizon-based termination.
        # Also check the underlying robosuite env's done flag to catch that case.
        done = libero_success or self._env.env.done
        self._last_obs = obs
        self._step_counter += 1

        if libero_success:
            self._current_success = True

        if done or self._step_counter >= self._max_episode_steps:
            self._done = True
            self._episode_results.append(self._current_success)

    @property
    def episode_idx(self) -> int:
        return self._episode_idx

    @property
    def current_initial_state(self) -> np.ndarray:
        """Return the initial state used for the current episode.

        Note: episode_idx is incremented after reset, so we need to subtract 1
        to get the initial state that was actually used.
        """
        if self._episode_idx == 0:
            raise RuntimeError("No episode has been started yet. Call reset() first.")
        return self._initial_states[self._episode_idx - 1]

    @property
    def episode_results(self) -> List[bool]:
        """Per-episode success flags accumulated so far."""
        return self._episode_results

    @property
    def current_success(self) -> bool:
        return self._current_success

    @property
    def control_hz(self) -> float:
        return self._control_hz

    @property
    def max_episode_steps(self) -> int:
        return self._max_episode_steps

    @property
    def total_episodes(self) -> int:
        return len(self._initial_states)

    @override
    def close(self) -> None:
        self._env.close()
