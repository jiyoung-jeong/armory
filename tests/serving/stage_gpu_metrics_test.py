"""Guard against overlap double counting and sample-bin alignment mistakes."""

import numpy as np
import pytest
from scripts.analyze_stage_gpu_metrics import integrate, merge_intervals


def test_gpu_overlap_counts_once_and_gaps_are_excluded():
    intervals = merge_intervals([(8, 12), (2, 6), (5, 9), (16, 19)])
    assert intervals.tolist() == [[2, 12], [16, 19]]
    # 8 ns at 20%, then (2 + 3) ns at 80%; gap [12,16] contributes nothing.
    result = integrate(np.array([0, 10, 20]), np.array([999, 20, 80]), intervals)
    assert result == pytest.approx((8 * 20 + 5 * 80) / 13)


def test_nonuniform_bins_and_boundary_are_integrated_by_duration():
    result = integrate(np.array([0, 4, 10]), np.array([999, 25, 75]), np.array([[0, 10]]))
    assert result == pytest.approx(55)


def test_alignment_shift_and_out_of_range_are_explicit():
    ts, values = np.array([0, 10, 20, 30]), np.array([999, 0, 100, 0])
    intervals = np.array([[12, 18]])
    assert integrate(ts, values, intervals) == 100
    assert integrate(ts, values, intervals, shift=10) == 0
    with pytest.raises(AssertionError):
        integrate(ts, values, np.array([[-1, 5]]))
