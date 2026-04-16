from dataclasses import dataclass
import numpy as np
from libero.libero import benchmark
from typing import List, Dict, Type
from jaxtyping import Float
import random


@dataclass
class Episode:
    """A single episode: one task, one initial state."""

    idx: int  # 1-indexed
    task_suite_name: str
    task_id: int
    task: benchmark.Task
    initial_state: np.ndarray

    def __str__(self) -> str:
        return f"Episode(task_suite_name={self.task_suite_name}, task_id={self.task_id}, task={self.task.language})"


def create_episodes(task_suite_name: str, num_trials_per_task: int) -> List[Episode]:
    benchmark_dict: Dict[str, Type[benchmark.Benchmark]] = (
        benchmark.get_benchmark_dict()
    )
    task_suite: benchmark.Benchmark = benchmark_dict[task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks

    episodes: List[Episode] = []
    for task_id in range(num_tasks_in_suite):
        task: benchmark.Task = task_suite.get_task(task_id)
        all_initial_states: Float[np.ndarray, "n_initial_states state_dim"] = (
            task_suite.get_task_init_states(task_id)
        )

        if len(all_initial_states) < num_trials_per_task:
            raise ValueError(
                f"Task {task_id} has less initial states than trials per task"
            )

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
