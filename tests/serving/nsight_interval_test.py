"""Device intervals overlap; summing their lengths would overcount activity."""

import pytest
from scripts.analyze_nsight_followup import clipped, merge_intervals, union_length


def test_nested_overlapping_and_touching_intervals_are_counted_once():
    intervals = [(5, 8), (1, 6), (2, 3), (8, 10), (12, 15)]
    assert merge_intervals(intervals) == [(1, 10), (12, 15)]
    assert union_length(intervals) == 12
    assert sum(b - a for a, b in intervals) > union_length(intervals)


def test_activity_is_clipped_to_inference_window():
    intervals = [(0, 5), (4, 8), (10, 13)]
    assert union_length(clipped(intervals, 2, 11)) == pytest.approx(7)
    assert union_length(clipped(intervals, 8, 10)) == 0


def test_empty_and_reversed_intervals_do_not_add_time():
    assert merge_intervals([(1, 1), (3, 2)]) == []
    assert union_length([]) == 0
