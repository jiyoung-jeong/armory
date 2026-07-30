from armory.serving.protocol import SchedulerConfig


def test_reconfigure_body_omits_the_boot_only_alpha() -> None:
    body = SchedulerConfig(
        scheduling_algorithm="lookahead-actions",
        alpha=0.25,
        action_horizon_multipliers={1: 0.5, 4: 1.0},
    ).to_reconfigure_body()

    assert body == {
        "scheduling_algorithm": "lookahead-actions",
        "action_horizon_multipliers": {"1": 0.5, "4": 1.0},
    }
