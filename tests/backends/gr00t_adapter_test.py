"""Pure conversion tests for the GR00T serving adapter."""

from __future__ import annotations

import numpy as np

from armory.serving.schemas import InternalRequest
from armory_client.messages import InferType
from gr00t_adapter.policy_adapter import (
    LANGUAGE_KEY,
    _convert_gripper,
    _groot_action_to_armory,
    _obs_to_groot,
)


def _request(
    robot_id: str,
    *,
    state: np.ndarray,
    image_value: int,
    wrist_value: int,
    prompt: str,
) -> InternalRequest:
    return InternalRequest(
        robot_id=robot_id,
        observation={
            "state": state,
            "image": np.full((2, 3, 3), image_value, dtype=np.uint8),
            "wrist_image": np.full((2, 3, 3), wrist_value, dtype=np.uint8),
            "prompt": prompt,
        },
        observation_step=0,
        action_index_start=0,
        request_timestamp=1.0,
        deadline=2.0,
        min_execution_horizon=0,
        max_execution_horizon=0,
        infer_type=InferType.SYNC,
    )


def test_obs_to_groot_preserves_batch_values_and_expected_shapes() -> None:
    requests = [
        _request(
            "robot-0",
            state=np.arange(8, dtype=np.float32),
            image_value=10,
            wrist_value=20,
            prompt="first task",
        ),
        _request(
            "robot-1",
            state=np.arange(10, 18, dtype=np.float32),
            image_value=30,
            wrist_value=40,
            prompt="second task",
        ),
    ]

    converted = _obs_to_groot(requests)

    assert converted["video"]["image"].shape == (2, 1, 2, 3, 3)
    assert converted["video"]["wrist_image"].shape == (2, 1, 2, 3, 3)
    np.testing.assert_array_equal(converted["video"]["image"][:, 0, 0, 0, 0], [10, 30])
    np.testing.assert_array_equal(converted["video"]["wrist_image"][:, 0, 0, 0, 0], [20, 40])

    scalar_keys = ("x", "y", "z", "roll", "pitch", "yaw")
    for index, key in enumerate(scalar_keys):
        assert converted["state"][key].shape == (2, 1, 1)
        np.testing.assert_array_equal(
            converted["state"][key][:, 0, 0],
            np.asarray([index, index + 10], dtype=np.float32),
        )

    assert converted["state"]["gripper"].shape == (2, 1, 2)
    np.testing.assert_array_equal(
        converted["state"]["gripper"][:, 0],
        np.asarray([[6, 7], [16, 17]], dtype=np.float32),
    )
    assert converted["language"][LANGUAGE_KEY] == [["first task"], ["second task"]]


def test_convert_gripper_uses_robosuite_polarity_without_mutating_input() -> None:
    actions = np.zeros((5, 7), dtype=np.float32)
    actions[:, -1] = [0.0, 0.49, 0.5, 0.51, 1.0]
    original = actions.copy()

    converted = _convert_gripper(actions)

    np.testing.assert_array_equal(actions, original)
    np.testing.assert_array_equal(converted[:, -1], [1.0, 1.0, 0.0, -1.0, -1.0])
    np.testing.assert_array_equal(converted[:, :-1], np.zeros((5, 6), dtype=np.float32))


def test_groot_action_to_armory_concatenates_keys_in_robot_action_order() -> None:
    batch_size = 2
    horizon = 3
    action_dict = {
        key: np.full((batch_size, horizon, 1), value, dtype=np.float32)
        for key, value in zip(
            ("x", "y", "z", "roll", "pitch", "yaw"),
            (1, 2, 3, 4, 5, 6),
            strict=True,
        )
    }
    action_dict["gripper"] = np.stack(
        [
            np.zeros((horizon, 1), dtype=np.float32),
            np.ones((horizon, 1), dtype=np.float32),
        ]
    )

    results = _groot_action_to_armory(action_dict, batch_size)

    assert len(results) == 2
    np.testing.assert_array_equal(
        results[0]["actions"],
        np.tile(np.asarray([1, 2, 3, 4, 5, 6, 1], dtype=np.float32), (horizon, 1)),
    )
    np.testing.assert_array_equal(
        results[1]["actions"],
        np.tile(np.asarray([1, 2, 3, 4, 5, 6, -1], dtype=np.float32), (horizon, 1)),
    )
    assert results[0]["noise"] is None
    assert results[1]["noise"] is None
    assert results[0]["rtc_prev_actions"] is results[0]["actions"]
    assert results[1]["rtc_prev_actions"] is results[1]["actions"]
