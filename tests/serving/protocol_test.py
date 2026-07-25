from armory.serving.protocol import SchedulerConfig


def test_scheduler_kwargs_use_exact_algorithm_names() -> None:
    multipliers = {1: 0.5, 4: 1.0}

    assert SchedulerConfig(
        scheduling_algorithm="lookahead-actions",
        action_horizon_multipliers=multipliers,
    ).to_scheduler_kwargs() == {"action_horizon_multipliers": multipliers}
    assert (
        SchedulerConfig(
            scheduling_algorithm="ahead",
            action_horizon_multipliers=multipliers,
        ).to_scheduler_kwargs()
        is None
    )


def test_dynamic_action_scheduler_kwargs_are_unchanged() -> None:
    assert SchedulerConfig(
        scheduling_algorithm="dynamic-action", alpha=0.25
    ).to_scheduler_kwargs() == {"alpha": 0.25}
