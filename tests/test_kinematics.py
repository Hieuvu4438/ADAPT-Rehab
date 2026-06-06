"""
Tests for kinematics modules.

Run with: pytest tests/test_kinematics.py -v
"""

import pytest
import numpy as np
from core.kinematics_quaternion import QuaternionKinematics
from core.smoothness import SmoothnessAnalyzer


class TestQuaternionKinematics:
    """Test quaternion-based joint angle computation."""

    def test_90_degree_angle(self):
        """Test 90-degree angle computation."""
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([0.0, 0.0, 0.0])
        c = np.array([0.0, 1.0, 0.0])
        angle = QuaternionKinematics.compute_angle(a, b, c)
        assert abs(angle - 90.0) < 1.0

    def test_180_degree_angle(self):
        """Test 180-degree angle (straight line)."""
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([0.0, 0.0, 0.0])
        c = np.array([-1.0, 0.0, 0.0])
        angle = QuaternionKinematics.compute_angle(a, b, c)
        assert abs(angle - 180.0) < 1.0

    def test_0_degree_angle(self):
        """Test 0-degree angle (parallel vectors)."""
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([0.0, 0.0, 0.0])
        c = np.array([2.0, 0.0, 0.0])
        angle = QuaternionKinematics.compute_angle(a, b, c)
        assert angle < 1.0

    def test_45_degree_angle(self):
        """Test 45-degree angle."""
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([0.0, 0.0, 0.0])
        c = np.array([1.0, 1.0, 0.0])
        angle = QuaternionKinematics.compute_angle(a, b, c)
        assert abs(angle - 45.0) < 2.0

    def test_3d_angle(self):
        """Test angle in 3D space (not just XY plane)."""
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([0.0, 0.0, 0.0])
        c = np.array([0.0, 0.0, 1.0])
        angle = QuaternionKinematics.compute_angle(a, b, c)
        assert abs(angle - 90.0) < 1.0

    def test_compute_angle_with_axis(self):
        """Test angle and rotation axis computation."""
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([0.0, 0.0, 0.0])
        c = np.array([0.0, 1.0, 0.0])
        angle, axis = QuaternionKinematics.compute_angle_with_axis(a, b, c)
        assert abs(angle - 90.0) < 1.0
        assert np.linalg.norm(axis) > 0.99  # Unit vector

    def test_slerp_identity(self):
        """Test SLERP at t=0 returns q1."""
        q1 = np.array([1.0, 0.0, 0.0, 0.0])
        q2 = np.array([0.0, 0.0, 0.0, 1.0])
        result = QuaternionKinematics.slerp(q1, q2, 0.0)
        assert np.allclose(result, q1, atol=0.01)

    def test_slerp_midpoint(self):
        """Test SLERP midpoint produces unit quaternion."""
        q1 = np.array([1.0, 0.0, 0.0, 0.0])
        q2 = np.array([0.0, 0.0, 0.0, 1.0])
        result = QuaternionKinematics.slerp(q1, q2, 0.5)
        assert abs(np.linalg.norm(result) - 1.0) < 0.01

    def test_compute_all(self):
        """Test batch angle computation."""
        keypoints = np.array([
            [0, 0, 0],    # 0: pelvis
            [-0.5, 0, 0], # 1: left_hip
            [0.5, 0, 0],  # 2: right_hip
            [0, 0.5, 0],  # 3: spine
            [-0.5, -0.5, 0],  # 4: left_knee
            [0.5, -0.5, 0],   # 5: right_knee
        ])
        joint_defs = {
            "left_knee": (1, 4, 0),  # Simple test
        }
        angles = QuaternionKinematics.compute_all(keypoints, joint_defs)
        assert "left_knee" in angles


class TestSmoothnessAnalyzer:
    """Test SPARC smoothness metric."""

    def test_smooth_sine_wave(self):
        """Test that smooth sine wave gets reasonable score."""
        t = np.linspace(0, 2 * np.pi, 100)
        angles = np.sin(t) * 90
        analyzer = SmoothnessAnalyzer(fs=30)
        result = analyzer.analyze(angles, t)
        assert result.is_valid
        assert result.smoothness_score >= 40  # Smooth sine should score reasonably

    def test_jerky_movement(self):
        """Test that random noise gets low score."""
        np.random.seed(42)
        angles = np.random.randn(100) * 20 + 90
        timestamps = np.linspace(0, 3.33, 100)
        analyzer = SmoothnessAnalyzer(fs=30)
        result = analyzer.analyze(angles, timestamps)
        assert result.is_valid
        assert result.smoothness_score < 80

    def test_short_sequence_invalid(self):
        """Test that short sequence returns invalid."""
        angles = np.array([0, 10, 20])
        analyzer = SmoothnessAnalyzer(fs=30)
        result = analyzer.analyze(angles)
        assert not result.is_valid

    def test_sparc_range(self):
        """Test SPARC is in expected range."""
        t = np.linspace(0, 2 * np.pi, 100)
        angles = np.sin(t) * 90
        analyzer = SmoothnessAnalyzer(fs=30)
        result = analyzer.analyze(angles, t)
        assert result.is_valid
        # SPARC should be negative (more negative = smoother)
        assert result.sparc < 0

    def test_nvp_counting(self):
        """Test velocity peak counting."""
        # Smooth: few peaks
        t = np.linspace(0, 2 * np.pi, 100)
        smooth = np.sin(t) * 90
        result_smooth = SmoothnessAnalyzer(30).analyze(smooth, t)

        # Jerky: many peaks
        np.random.seed(42)
        jerky = np.random.randn(100) * 20 + 90
        result_jerky = SmoothnessAnalyzer(30).analyze(jerky, t)

        assert result_smooth.is_valid and result_jerky.is_valid
        # Jerky should have more peaks
        assert result_jerky.nvp >= result_smooth.nvp
