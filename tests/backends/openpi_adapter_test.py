"""Characterization tests for pure OpenPI adapter behavior."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from armory.serving.schemas import InternalRequest
from armory_client.messages import InferType, RTCParams
from openpi_adapter.policy_adapter import OpenPiPolicyAdapter, _recursive_stack, _rename_keys


def _request(
    robot_id: str,
    infer_type: InferType,
    params: RTCParams | None = None,
) -> InternalRequest:
    return InternalRequest(
        robot_id=robot_id,
        observation={},
        observation_step=0,
        action_index_start=0,
        request_timestamp=1.0,
        deadline=2.0,
        min_execution_horizon=0,
        max_execution_horizon=4,
        infer_type=infer_type,
        params=params,
    )


def test_rename_keys_converts_client_observation_to_openpi_names() -> None:
    observation = {
        "state": np.asarray([1.0, 2.0], dtype=np.float32),
        "image": np.full((2, 2, 3), 10, dtype=np.uint8),
        "wrist_image": np.full((2, 2, 3), 20, dtype=np.uint8),
        "prompt": "pick up the block",
    }

    renamed = _rename_keys(observation)

    assert set(renamed) == {
        "observation/state",
        "observation/image",
        "observation/wrist_image",
        "prompt",
    }
    assert renamed["observation/state"] is observation["state"]
    assert renamed["observation/image"] is observation["image"]
    assert renamed["observation/wrist_image"] is observation["wrist_image"]
    assert renamed["prompt"] == "pick up the block"


def test_rename_keys_leaves_openpi_observation_unchanged() -> None:
    observation = {
        "observation/state": np.asarray([1.0], dtype=np.float32),
        "observation/image": np.zeros((1, 1, 3), dtype=np.uint8),
        "observation/wrist_image": np.zeros((1, 1, 3), dtype=np.uint8),
        "prompt": "task",
    }

    assert _rename_keys(observation) is observation


def test_recursive_stack_batches_nested_arrays_and_scalars() -> None:
    observations = [
        {
            "state": np.asarray([1.0, 2.0], dtype=np.float32),
            "nested": {"image": np.full((2, 2), 3, dtype=np.uint8)},
            "enabled": np.bool_(True),
        },
        {
            "state": np.asarray([4.0, 5.0], dtype=np.float32),
            "nested": {"image": np.full((2, 2), 6, dtype=np.uint8)},
            "enabled": np.bool_(False),
        },
    ]

    batched = _recursive_stack(observations)

    np.testing.assert_array_equal(
        batched["state"],
        np.asarray([[1.0, 2.0], [4.0, 5.0]], dtype=np.float32),
    )
    assert batched["nested"]["image"].shape == (2, 2, 2)
    np.testing.assert_array_equal(batched["nested"]["image"][:, 0, 0], [3, 6])
    np.testing.assert_array_equal(batched["enabled"], [True, False])


def test_infer_batch_splits_rtc_requests_and_restores_original_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wrapped_policy = SimpleNamespace(_is_pytorch_model=False)
    adapter = OpenPiPolicyAdapter(wrapped_policy, make_example_fn=lambda: {})
    rtc_params = RTCParams(prev_action=np.zeros((4, 7), dtype=np.float32), s_param=2, d_param=1)
    requests = [
        _request("rtc-0", InferType.INFERENCE_TIME_RTC, rtc_params),
        _request("sync-1", InferType.SYNC),
        # Existing behavior: RTC mode without RTCParams follows the non-RTC path.
        _request("rtc-without-params-2", InferType.INFERENCE_TIME_RTC),
        _request("rtc-3", InferType.INFERENCE_TIME_RTC, rtc_params),
    ]
    calls: list[tuple[bool, list[str]]] = []

    def fake_infer_batch_group(
        grouped_requests: list[InternalRequest],
        *,
        use_rtc: bool,
    ) -> list[dict]:
        calls.append((use_rtc, [request.robot_id for request in grouped_requests]))
        return [{"robot_id": request.robot_id, "used_rtc": use_rtc} for request in grouped_requests]

    monkeypatch.setattr(adapter, "_infer_batch_group", fake_infer_batch_group)

    results = adapter.infer_batch(requests)

    assert calls == [
        (False, ["sync-1", "rtc-without-params-2"]),
        (True, ["rtc-0", "rtc-3"]),
    ]
    assert results == [
        {"robot_id": "rtc-0", "used_rtc": True},
        {"robot_id": "sync-1", "used_rtc": False},
        {"robot_id": "rtc-without-params-2", "used_rtc": False},
        {"robot_id": "rtc-3", "used_rtc": True},
    ]


def test_infer_batch_with_no_requests_does_not_invoke_policy() -> None:
    adapter = OpenPiPolicyAdapter(SimpleNamespace(), make_example_fn=lambda: {})

    assert adapter.infer_batch([]) == []
