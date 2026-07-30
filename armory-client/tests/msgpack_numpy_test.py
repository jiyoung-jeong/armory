from dataclasses import dataclass

import numpy as np
import pytest

from armory_client import msgpack_numpy


@dataclass(frozen=True)
class NestedPayload:
    name: str
    values: np.ndarray


@dataclass(frozen=True)
class Payload:
    nested: NestedPayload
    items: list[NestedPayload]


def _check(expected, actual):
    if isinstance(expected, np.ndarray):
        assert expected.shape == actual.shape
        assert expected.dtype == actual.dtype
        assert np.array_equal(expected, actual, equal_nan=expected.dtype.kind == "f")
    else:
        assert expected == actual


def _check_tree(expected, actual):
    if isinstance(expected, dict):
        assert expected.keys() == actual.keys()
        for key in expected:
            _check_tree(expected[key], actual[key])
        return

    if isinstance(expected, list):
        assert len(expected) == len(actual)
        for expected_item, actual_item in zip(expected, actual):
            _check_tree(expected_item, actual_item)
        return

    _check(expected, actual)


@pytest.mark.parametrize(
    "data",
    [
        np.bool_(True),  # boolean scalar
        np.array([1, 2, 3])[0],  # int scalar
        np.str_("asdf"),  # string scalar
        np.array(1.0),  # 0D array
        np.array(["asdf", "qwer"]),  # string array
        np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int16),  # 3D integer array
        np.array([np.nan, np.inf, -np.inf]),  # special float values
        {
            "arr": np.array([1, 2, 3]),
            "nested": {"arr": np.array([4, 5, 6])},
        },  # nested dict with arrays
        Payload(
            nested=NestedPayload("a", np.array([1, 2, 3])),
            items=[NestedPayload("b", np.array([4, 5, 6]))],
        ),  # nested dataclasses with arrays
    ],
)
def test_pack_unpack(data):
    packed = msgpack_numpy.packb(data)
    unpacked = msgpack_numpy.unpackb(packed)
    if isinstance(data, Payload):
        data = {
            "nested": {"name": "a", "values": np.array([1, 2, 3])},
            "items": [{"name": "b", "values": np.array([4, 5, 6])}],
        }
    _check_tree(data, unpacked)
