import math
import pathlib
import random
from dataclasses import dataclass

import numpy as np
from typing_extensions import override

from armory_client.schemas import Action, ActionChunk, Observation
from evaluation import image_tools
from evaluation.envs.base import Environment
from evaluation.envs.config import LiberoConfig

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
NUM_STEPS_WAIT = 10
LIBERO_ENV_RESOLUTION = 256
RESIZE_SIZE = 224


@dataclass
class LiberoObservation(Observation):
    state: np.ndarray
    image: np.ndarray
    wrist_image: np.ndarray
    prompt: str


@dataclass(frozen=True)
class LiberoRobotSpec:
    """The fixed LIBERO task assigned to one robot for an entire run."""

    task_suite_name: str
    task_id: int


def plan_robot_specs(
    config: LiberoConfig, *, num_robots: int, experiment_seed: int
) -> list[LiberoRobotSpec]:
    """Assign distinct, reproducible tasks to every robot in a fleet.

    Planning happens once in the parent process.  Sampling within worker
    processes would make a without-replacement guarantee impossible.
    """
    from libero.libero.benchmark import get_benchmark_dict

    task_suite = get_benchmark_dict()[config.task_suite_name]()
    if num_robots > task_suite.n_tasks:
        raise ValueError(
            f"LIBERO suite {config.task_suite_name!r} has {task_suite.n_tasks} tasks, "
            f"but the experiment has {num_robots} robots; task assignment is without replacement."
        )

    seed = experiment_seed if config.task_seed is None else config.task_seed
    task_ids = sample_task_ids(num_tasks=task_suite.n_tasks, num_robots=num_robots, seed=seed)
    return [
        LiberoRobotSpec(task_suite_name=config.task_suite_name, task_id=task_id)
        for task_id in task_ids
    ]


def sample_task_ids(*, num_tasks: int, num_robots: int, seed: int) -> list[int]:
    """Sample one distinct task ID for every robot, reproducibly."""
    if num_robots > num_tasks:
        raise ValueError(
            f"Cannot assign {num_robots} robots to {num_tasks} tasks without replacement."
        )
    return random.Random(seed).sample(range(num_tasks), num_robots)


def get_libero_env(task, seed):
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    task_bddl_file = (
        pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    )
    # TODO: figure out why we don't pass RESIZE_SIZE directly here, and then comment
    env_args = {
        "bddl_file_name": task_bddl_file,
        "camera_heights": LIBERO_ENV_RESOLUTION,
        "camera_widths": LIBERO_ENV_RESOLUTION,
    }
    env = OffScreenRenderEnv(**env_args)
    env.seed(
        seed
    )  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env


def quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


class LiberoSimEnvironment(Environment):
    """Wraps a LIBERO ``OffScreenRenderEnv`` in the eval ``Environment`` interface.

    Bound to a single task. Each ``reset()`` loads the next pre-collected initial
    state (cycling once exhausted) and waits for objects to settle. The driver
    decides how many episodes to run and any wall-clock budget.
    """

    def __init__(
        self,
        spec: LiberoRobotSpec,
        *,
        max_episode_steps: int = 300,
        seed: int = 42,
    ) -> None:
        from libero.libero.benchmark import get_benchmark_dict

        benchmark_dict = get_benchmark_dict()
        task_suite = benchmark_dict[spec.task_suite_name]()

        task = task_suite.get_task(spec.task_id)
        self._env = get_libero_env(task, seed=seed)

        self._task_description = task.language
        self._initial_states = task_suite.get_task_init_states(spec.task_id)
        self._max_episode_steps = max_episode_steps

        self._episode_idx = 0
        self._done = True
        self._success = False
        self._step_counter = 0
        self._last_obs = None
        self._current_initial_state: np.ndarray | None = None

    @override
    def reset(self) -> None:
        """Reset to the next initial state (cycling) and wait for stabilization."""
        state = self._initial_states[self._episode_idx % len(self._initial_states)]
        self._current_initial_state = state

        self._env.reset()
        obs = self._env.set_init_state(state)

        # ``set_init_state`` renders the first offscreen camera frame, which
        # creates the EGL context. Checking earlier reports no renderer even
        # when the NVIDIA backend is correctly configured.
        if self._episode_idx == 0:
            from utils import assert_egl_rendering

            assert_egl_rendering()

        # Let objects fall / settle.
        for _ in range(NUM_STEPS_WAIT):
            obs, _, _, _ = self._env.step(LIBERO_DUMMY_ACTION)

        self._last_obs = obs
        self._done = False
        self._success = False
        self._step_counter = 0
        self._episode_idx += 1

    @override
    def is_episode_complete(self) -> bool:
        return self._done

    @override
    def get_observation(self) -> LiberoObservation:
        if self._last_obs is None:
            raise RuntimeError("Observation is not set. Call reset() first.")

        obs = self._last_obs

        # IMPORTANT: rotate 180 degrees to match train preprocessing.
        img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
        wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
        img = image_tools.convert_to_uint8(
            image_tools.resize_with_pad(img, RESIZE_SIZE, RESIZE_SIZE)
        )
        wrist_img = image_tools.convert_to_uint8(
            image_tools.resize_with_pad(wrist_img, RESIZE_SIZE, RESIZE_SIZE)
        )

        state = np.concatenate(
            (
                obs["robot0_eef_pos"],
                quat2axisangle(obs["robot0_eef_quat"]),
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

    @override
    def apply_action(self, action: Action) -> None:
        """Take one low-level action step in the LIBERO simulator."""
        obs, _, libero_success, _ = self._env.step(action.action.tolist())
        # LIBERO overrides `done` with _check_success(), losing horizon-based
        # termination, so also check the underlying robosuite env's done flag.
        self._last_obs = obs
        self._step_counter += 1

        if libero_success:
            self._success = True
            self._done = True
        elif self._env.env.done or self._step_counter >= self._max_episode_steps:
            self._done = True

    @override
    def create_null_action(
        self, observation: Observation, current_action_chunk: ActionChunk | None
    ) -> Action:
        action = np.zeros(7, dtype=np.float32)
        if current_action_chunk is not None:
            action[-1] = current_action_chunk.get_action(-1)[-1]
        return Action(
            step=None,
            action=action,
            action_chunk_index=None,
            index_in_chunk=None,
        )

    @override
    def close(self) -> None:
        self._env.close()

    @property
    @override
    def current_success(self) -> bool:
        return self._success

    @property
    @override
    def current_initial_state(self) -> np.ndarray | None:
        return self._current_initial_state

    @property
    @override
    def task_language(self) -> str:
        return str(self._task_description)
