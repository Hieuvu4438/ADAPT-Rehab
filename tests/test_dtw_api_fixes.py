"""
Tests for the 4 API signature fixes (Phase 1).

Verifies:
1. ``constrained_dtw`` works on both 1D and 2D row-vector inputs.
2. ``weighted_constrained_dtw`` works with no weights argument.
3. ``compute_weighted_dtw`` works with no weights argument and str keys.
4. ``create_arm_raise_exercise`` / ``create_squat_exercise`` /
   ``create_elbow_flex_exercise`` work with no arguments.

Run with: pytest tests/test_dtw_api_fixes.py -v
"""

import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestConstrainedDTW:
    def test_1d_identical_sequences(self):
        from core.dtw_constrained import constrained_dtw
        s = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        d, path = constrained_dtw(s, s)
        assert d == 0.0
        assert len(path) == len(s)

    def test_1d_shifted_sequences(self):
        from core.dtw_constrained import constrained_dtw
        a = np.array([0.0, 1.0, 2.0, 3.0])
        b = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        d, path = constrained_dtw(a, b)
        assert d > 0
        assert len(path) > 0

    def test_2d_sequences_no_crash(self):
        """Phase 1 fix: 2D row-vector input must not crash."""
        from core.dtw_constrained import constrained_dtw
        rng = np.random.default_rng(42)
        a = rng.random((20, 3))
        b = rng.random((25, 3))
        d, path = constrained_dtw(a, b, window_percent=0.1)
        assert d >= 0
        assert len(path) > 0
        # Path entries should be valid (i, j) coordinates
        for i, j in path:
            assert 0 <= i < len(a)
            assert 0 <= j < len(b)

    def test_2d_identical_zero_distance(self):
        from core.dtw_constrained import constrained_dtw
        a = np.array([[1.0, 2.0, 3.0]] * 10)
        d, path = constrained_dtw(a, a)
        assert d < 1e-9

    def test_empty_sequences(self):
        from core.dtw_constrained import constrained_dtw
        d, path = constrained_dtw(np.array([]), np.array([1, 2, 3]))
        assert d == 0.0
        assert path == []


class TestWeightedConstrainedDTW:
    def test_no_weights_argument(self):
        """Phase 1 fix: weights=None must be accepted."""
        from core.dtw_constrained import weighted_constrained_dtw
        user = {"shoulder": [0, 10, 20, 30], "elbow": [0, 5, 10, 15]}
        ref = {"shoulder": [0, 10, 20, 30], "elbow": [0, 5, 10, 15]}
        result = weighted_constrained_dtw(user, ref)
        # Returns 3-tuple: (similarity, total_distance, details)
        assert len(result) == 3
        similarity, total, details = result
        assert 0.0 <= similarity <= 100.0
        assert "shoulder" in details and "elbow" in details

    def test_with_weights_argument(self):
        from core.dtw_constrained import weighted_constrained_dtw
        user = {"shoulder": [0, 10, 20, 30]}
        ref = {"shoulder": [0, 10, 20, 30]}
        weights = {"shoulder": 2.0}
        sim, total, details = weighted_constrained_dtw(user, ref, weights=weights)
        assert details["shoulder"]["weight"] == 2.0

    def test_identical_sequences_high_similarity(self):
        from core.dtw_constrained import weighted_constrained_dtw
        user = {"a": [0, 1, 2, 3, 4], "b": [10, 11, 12, 13, 14]}
        ref = {"a": [0, 1, 2, 3, 4], "b": [10, 11, 12, 13, 14]}
        sim, _, _ = weighted_constrained_dtw(user, ref)
        assert sim > 99.0


class TestComputeWeightedDTW:
    def test_no_weights_str_keys(self):
        """Phase 1 fix: weights=None + str keys must work."""
        from core.dtw_analysis import compute_weighted_dtw
        user = {"left_shoulder": [0, 30, 60, 90, 60, 30, 0]}
        ref = {"left_shoulder": [0, 30, 60, 90, 60, 30, 0]}
        result = compute_weighted_dtw(user, ref)
        assert result.similarity_score > 99.0
        # details is nested: result.details["joints"]["left_shoulder"]
        assert "left_shoulder" in result.details.get("joints", {})  # str keys, not enum

    def test_with_weights_argument(self):
        from core.dtw_analysis import compute_weighted_dtw
        user = {"a": [0, 1, 2, 3], "b": [10, 11, 12, 13]}
        ref = {"a": [0, 1, 2, 3], "b": [10, 11, 12, 13]}
        weights = {"a": 1.0, "b": 1.0}
        result = compute_weighted_dtw(user, ref, weights=weights)
        assert result.similarity_score > 99.0

    def test_empty_input(self):
        from core.dtw_analysis import compute_weighted_dtw
        result = compute_weighted_dtw({}, {})
        assert result.distance == 0.0


class TestExerciseFactories:
    def test_arm_raise_no_args(self):
        """Phase 1 fix: total_frames default must be 300."""
        from core.synchronizer import create_arm_raise_exercise
        ex = create_arm_raise_exercise()
        assert ex.name  # has a name
        assert len(ex.checkpoints) > 0  # has checkpoints

    def test_squat_no_args(self):
        from core.synchronizer import create_squat_exercise
        ex = create_squat_exercise()
        assert ex.name
        assert len(ex.checkpoints) > 0

    def test_elbow_flex_no_args(self):
        from core.synchronizer import create_elbow_flex_exercise
        ex = create_elbow_flex_exercise()
        assert ex.name
        assert len(ex.checkpoints) > 0

    def test_arm_raise_with_total_frames(self):
        from core.synchronizer import create_arm_raise_exercise
        ex = create_arm_raise_exercise(total_frames=600, fps=60.0)
        assert ex.total_frames == 600
