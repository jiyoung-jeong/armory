from __future__ import annotations

import random
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

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


# Fixed libero_10 trial layout: seed -> contiguous pair of task ids. Lets the
# 5-trial sweep (seeds 1..5) cover all 10 libero_10 tasks exactly once with a
# stable, paper-ready mapping. Falls back to seeded random sampling for any
# (task_suite_name, subset_size, seed) combination outside this table.
_LIBERO_10_SEED_PAIRS: dict[int, list[int]] = {
    1: [0, 1],
    2: [2, 3],
    3: [4, 5],
    4: [6, 7],
    5: [8, 9],
}


def pick_subset_task_ids(task_suite_name: str, subset_size: int, seed: int) -> list[int]:
    """Pick ``subset_size`` task ids from a task suite.

    For the canonical libero_10 5-trial layout (``task_suite_name="libero_10"``,
    ``subset_size=2``, ``seed in {1,2,3,4,5}``), returns the hardcoded pair from
    ``_LIBERO_10_SEED_PAIRS``. Otherwise falls back to a seeded random sample
    so each (seed, subset_size) combination is still reproducible.
    ``subset_size <= 0`` or ``>= n_tasks`` returns all task ids.
    """
    if task_suite_name == "libero_10" and subset_size == 2 and seed in _LIBERO_10_SEED_PAIRS:
        return list(_LIBERO_10_SEED_PAIRS[seed])

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
        all_initial_states: np.ndarray = task_suite.get_task_init_states(
            task_id
        )  # n_initial_states state_dim

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
