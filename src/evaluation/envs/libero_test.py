import pytest

from evaluation.envs.libero import sample_task_ids


def test_sample_task_ids_is_seeded_and_without_replacement() -> None:
    task_ids = sample_task_ids(num_tasks=10, num_robots=4, seed=17)

    assert task_ids == sample_task_ids(num_tasks=10, num_robots=4, seed=17)
    assert len(task_ids) == len(set(task_ids)) == 4


def test_sample_task_ids_rejects_more_robots_than_tasks() -> None:
    with pytest.raises(ValueError, match="without replacement"):
        sample_task_ids(num_tasks=2, num_robots=3, seed=17)
