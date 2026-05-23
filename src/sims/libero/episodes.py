from __future__ import annotations

import random
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from jaxtyping import Float

if TYPE_CHECKING:
    from libero.libero import benchmark


@dataclass
class Episode:
    """A single episode: one task, one initial state."""

    idx: int  # 1-indexed
    task_suite_name: str
    task_id: int
    task: Any  # benchmark.Task in libero mode; duck-typed (.language) elsewhere
    initial_state: np.ndarray

    def __str__(self) -> str:
        return f"Episode(task_suite_name={self.task_suite_name}, task_id={self.task_id}, task={self.task.language})"


@dataclass
class _MockTask:
    language: str


def create_mock_episodes(num_episodes: int) -> list[Episode]:
    """Synthetic episodes for the mock env — no libero dependency."""
    return [
        Episode(
            idx=i + 1,
            task_suite_name="mock",
            task_id=0,
            task=_MockTask(language="mock task"),
            initial_state=np.zeros(1, dtype=np.float32),
        )
        for i in range(num_episodes)
    ]


def pick_subset_task_ids(task_suite_name: str, subset_size: int, seed: int) -> list[int]:
    """Deterministically pick ``subset_size`` task ids from a task suite.

    The selection is seeded by ``seed`` (independent of the global RNG) so that
    each sweep seed → trial picks a different but reproducible subset.
    ``subset_size <= 0`` or ``>= n_tasks`` returns all task ids.
    """
    from libero.libero import benchmark

    task_suite: benchmark.Benchmark = benchmark.get_benchmark_dict()[task_suite_name]()
    n_tasks = task_suite.n_tasks
    if subset_size <= 0 or subset_size >= n_tasks:
        return list(range(n_tasks))
    return sorted(random.Random(seed).sample(range(n_tasks), subset_size))


def assign_robots_to_tasks(num_robots: int, task_ids: list[int]) -> list[int]:
    """Round-robin assign ``num_robots`` robots to ``task_ids`` (with wraparound)."""
    if not task_ids:
        raise ValueError("task_ids must be non-empty")
    return [task_ids[i % len(task_ids)] for i in range(num_robots)]


def create_episodes(task_suite_name: str, num_trials_per_task: int) -> list[Episode]:
    from libero.libero import benchmark

    benchmark_dict: dict[str, type[benchmark.Benchmark]] = benchmark.get_benchmark_dict()
    task_suite: benchmark.Benchmark = benchmark_dict[task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks

    episodes: list[Episode] = []
    for task_id in range(num_tasks_in_suite):
        task: benchmark.Task = task_suite.get_task(task_id)
        all_initial_states: Float[np.ndarray, "n_initial_states state_dim"] = (
            task_suite.get_task_init_states(task_id)
        )

        if len(all_initial_states) < num_trials_per_task:
            raise ValueError(f"Task {task_id} has less initial states than trials per task")

        initial_states = all_initial_states[:num_trials_per_task]
        for state in initial_states:
            episodes.append(
                Episode(
                    idx=len(episodes) + 1,
                    task_suite_name=task_suite_name,
                    task_id=task_id,
                    task=task,
                    initial_state=state,
                )
            )
    random.shuffle(episodes)
    for i, episode in enumerate(episodes):
        episode.idx = i + 1

    return episodes
