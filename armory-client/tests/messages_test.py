import numpy as np
import pytest

from armory_client.messages import InferRequest, InferType, RTCParams


def test_infer_type_only_exposes_implemented_modes() -> None:
    assert {infer_type.value for infer_type in InferType} == {"sync", "inference_time_rtc"}
    with pytest.raises(ValueError):
        InferType("train_time_rtc")
    with pytest.raises(ValueError):
        InferType("vlash")


def test_inference_time_rtc_wire_values_are_reconstructed() -> None:
    previous_actions = np.zeros((4, 7), dtype=np.float32)
    request = InferRequest(
        robot_id="robot-0",
        observation={},
        observation_step=1,
        action_index_start=2,
        request_timestamp=3.0,
        deadline=4.0,
        min_execution_horizon=1,
        max_execution_horizon=4,
        infer_type="inference_time_rtc",  # type: ignore[arg-type]
        params={  # type: ignore[arg-type]
            "prev_action": previous_actions,
            "s_param": 2,
            "d_param": 1,
        },
    )

    assert request.infer_type is InferType.INFERENCE_TIME_RTC
    assert isinstance(request.params, RTCParams)
    np.testing.assert_array_equal(request.params.prev_action, previous_actions)
