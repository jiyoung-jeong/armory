from evaluation.envs.libero import sample_task_ids


def test_sample_task_ids_is_seeded_and_without_replacement() -> None:
    task_ids = sample_task_ids(num_tasks=10, num_robots=4, seed=17)

    assert task_ids == sample_task_ids(num_tasks=10, num_robots=4, seed=17)
    assert len(task_ids) == len(set(task_ids)) == 4


def test_sample_task_ids_reshuffles_after_each_complete_pass() -> None:
    task_ids = sample_task_ids(num_tasks=2, num_robots=5, seed=17)

    assert task_ids == sample_task_ids(num_tasks=2, num_robots=5, seed=17)
    assert sorted(task_ids[:2]) == [0, 1]
    assert sorted(task_ids[2:4]) == [0, 1]
    assert task_ids[4] in {0, 1}
