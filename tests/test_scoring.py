"""
Tests for scoring, compensation, and fatigue modules.

Run with: pytest tests/test_scoring.py -v
"""

import sys
import os
import pytest
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Direct imports to avoid triggering modules/__init__.py
import importlib
compensation_mod = importlib.import_module("modules.compensation")
fatigue_mod = importlib.import_module("modules.fatigue")
scoring_mod = importlib.import_module("modules.scoring_v2")

CompensationDetector = compensation_mod.CompensationDetector
CompensationType = compensation_mod.CompensationType
FatigueAnalyzer = fatigue_mod.FatigueAnalyzer
FatigueLevel = fatigue_mod.FatigueLevel
EnhancedScorer = scoring_mod.EnhancedScorer


class TestCompensationDetector:
    """Test temporal compensation detection."""

    def _make_balanced_pose(self, n=10):
        """Create balanced pose sequence (no compensation)."""
        poses = []
        for _ in range(n):
            pose = np.zeros((33, 3))
            pose[5] = [0.3, 0.5, 0]   # Left shoulder (COCO idx 5)
            pose[6] = [0.7, 0.5, 0]   # Right shoulder (COCO idx 6, same height)
            pose[11] = [0.35, 0.7, 0]  # Left hip (COCO idx 11)
            pose[12] = [0.65, 0.7, 0]  # Right hip (COCO idx 12, same height)
            poses.append(pose)
        return poses

    def _make_shoulder_hiking_pose(self, n=10):
        """Create pose with shoulder hiking."""
        poses = []
        for _ in range(n):
            pose = np.zeros((33, 3))
            pose[5] = [0.3, 0.35, 0]   # Left shoulder (much higher)
            pose[6] = [0.7, 0.65, 0]   # Right shoulder
            pose[11] = [0.35, 0.7, 0]
            pose[12] = [0.65, 0.7, 0]
            poses.append(pose)
        return poses

    def _make_trunk_lean_pose(self, n=10):
        """Create pose with trunk lean (asymmetric hip shift)."""
        poses = []
        for _ in range(n):
            pose = np.zeros((33, 3))
            pose[5] = [0.4, 0.5, 0]    # Left shoulder
            pose[6] = [0.6, 0.5, 0]    # Right shoulder
            pose[11] = [0.2, 0.7, 0]   # Left hip shifted left (asymmetric)
            pose[12] = [0.6, 0.7, 0]   # Right hip normal
            poses.append(pose)
        return poses

    def test_balanced_no_compensation(self):
        """Test that balanced pose has no compensation."""
        poses = self._make_balanced_pose()
        detector = CompensationDetector()
        result = detector.analyze(poses)
        assert result.is_valid
        assert result.score > 90
        assert len(result.detected_types) == 0

    def test_shoulder_hiking_detected(self):
        """Test shoulder hiking detection."""
        poses = self._make_shoulder_hiking_pose()
        detector = CompensationDetector()
        result = detector.analyze(poses)
        assert result.is_valid
        assert result.score < 90
        assert any("Vai" in t for t in result.detected_types)

    def test_trunk_lean_detected(self):
        """Test trunk lean detection."""
        poses = self._make_trunk_lean_pose()
        detector = CompensationDetector()
        result = detector.analyze(poses)
        assert result.is_valid
        assert any("thân" in t.lower() or "nghiêng" in t.lower() for t in result.detected_types)

    def test_too_few_frames(self):
        """Test handling of too few frames."""
        poses = [np.zeros((33, 3))] * 2
        detector = CompensationDetector()
        result = detector.analyze(poses)
        assert not result.is_valid

    def test_add_frame_accumulation(self):
        """Test frame-by-frame accumulation."""
        detector = CompensationDetector()
        poses = self._make_balanced_pose()
        for pose in poses:
            detector.add_frame(pose)
        result = detector.analyze()
        assert result.is_valid
        assert result.score > 90

    def test_reset(self):
        """Test reset clears state."""
        detector = CompensationDetector()
        poses = self._make_shoulder_hiking_pose()
        for pose in poses:
            detector.add_frame(pose)
        detector.reset()
        result = detector.analyze()
        assert not result.is_valid  # No data after reset


class TestFatigueAnalyzer:
    """Test multi-indicator fatigue analysis."""

    def test_fresh_state(self):
        """Test fresh state with no fatigue."""
        analyzer = FatigueAnalyzer()
        analyzer.set_baseline({
            "jerk_value": 100.0,
            "max_angle": 90.0,
            "mean_velocity": 50.0,
            "angle_std": 3.0,
        })
        result = analyzer.analyze({
            "jerk_value": 100.0,
            "max_angle": 90.0,
            "mean_velocity": 50.0,
            "angle_std": 3.0,
        })
        assert result.is_valid
        assert result.level == FatigueLevel.FRESH
        assert result.composite_score < 20

    def test_light_fatigue(self):
        """Test light fatigue detection."""
        analyzer = FatigueAnalyzer()
        analyzer.set_baseline({"jerk_value": 100, "max_angle": 90, "mean_velocity": 50})
        result = analyzer.analyze({"jerk_value": 160, "max_angle": 85, "mean_velocity": 45})
        assert result.is_valid
        assert result.level.value >= FatigueLevel.LIGHT.value

    def test_heavy_fatigue(self):
        """Test heavy fatigue detection."""
        analyzer = FatigueAnalyzer()
        analyzer.set_baseline({"jerk_value": 100, "max_angle": 90, "mean_velocity": 50})
        result = analyzer.analyze({"jerk_value": 350, "max_angle": 60, "mean_velocity": 30})
        assert result.is_valid
        assert result.level == FatigueLevel.HEAVY

    def test_rom_degradation(self):
        """Test ROM degradation computation."""
        analyzer = FatigueAnalyzer()
        analyzer.set_baseline({"jerk_value": 100, "max_angle": 100, "mean_velocity": 50})
        result = analyzer.analyze({"jerk_value": 100, "max_angle": 70, "mean_velocity": 50})
        assert result.is_valid
        assert result.rom_degradation == pytest.approx(30.0, abs=1.0)

    def test_velocity_decline(self):
        """Test velocity decline computation."""
        analyzer = FatigueAnalyzer()
        analyzer.set_baseline({"jerk_value": 100, "max_angle": 90, "mean_velocity": 100})
        result = analyzer.analyze({"jerk_value": 100, "max_angle": 90, "mean_velocity": 60})
        assert result.is_valid
        assert result.velocity_decline == pytest.approx(40.0, abs=1.0)

    def test_recommendation_vietnamese(self):
        """Test Vietnamese recommendation text."""
        analyzer = FatigueAnalyzer()
        analyzer.set_baseline({"jerk_value": 100, "max_angle": 90, "mean_velocity": 50})
        result = analyzer.analyze({"jerk_value": 100, "max_angle": 90, "mean_velocity": 50})
        assert result.is_valid
        assert "Bác" in result.recommendation

    def test_trend_analysis(self):
        """Test fatigue trend across multiple reps."""
        analyzer = FatigueAnalyzer()
        analyzer.set_baseline({"jerk_value": 100, "max_angle": 90, "mean_velocity": 50})

        # Simulate increasing fatigue
        for i in range(6):
            analyzer.analyze({
                "jerk_value": 100 + i * 30,
                "max_angle": 90 - i * 3,
                "mean_velocity": 50 - i * 3,
            })

        trend = analyzer.get_trend()
        assert trend["trend"] in ["increasing", "increasing_fast"]
        assert trend["reps_analyzed"] == 7  # baseline + 6 reps

    def test_reset(self):
        """Test reset clears state."""
        analyzer = FatigueAnalyzer()
        analyzer.set_baseline({"jerk_value": 100, "max_angle": 90, "mean_velocity": 50})
        analyzer.analyze({"jerk_value": 300, "max_angle": 60, "mean_velocity": 30})
        analyzer.reset()
        assert analyzer._baseline_jerk is None
        assert len(analyzer._rep_history) == 0


class TestEnhancedScorer:
    """Test 6-dimension scoring system."""

    def _make_rep_data(self, n=30, target=150):
        """Create sample rep data."""
        # Smooth sine wave reaching target
        t = np.linspace(0, 2 * np.pi, n)
        angles = (np.sin(t) * 0.5 + 0.5) * target
        timestamps = np.linspace(0, 1.0, n)
        return angles, timestamps

    def test_score_rep_basic(self):
        """Test basic rep scoring."""
        scorer = EnhancedScorer()
        scorer.start_session("test_exercise")
        angles, timestamps = self._make_rep_data()
        score = scorer.score_rep(angles, timestamps, target_angle=150)
        assert score.rep_number == 1
        assert 0 <= score.total_score <= 100
        assert score.fatigue == FatigueLevel.FRESH

    def test_score_rep_rom(self):
        """Test ROM score computation."""
        scorer = EnhancedScorer()
        scorer.start_session("test")

        # Angles reaching 90% of target
        angles = np.linspace(0, 135, 30)
        timestamps = np.linspace(0, 1, 30)
        score = scorer.score_rep(angles, timestamps, target_angle=150)
        assert score.rom_score == pytest.approx(90.0, abs=5.0)

    def test_session_report(self):
        """Test session report generation."""
        scorer = EnhancedScorer()
        scorer.start_session("arm_raise", "test_session")

        for _ in range(3):
            angles, timestamps = self._make_rep_data()
            scorer.score_rep(angles, timestamps, target_angle=150)

        report = scorer.get_session_report()
        assert report.total_reps == 3
        assert "rom" in report.average_scores
        assert "smoothness" in report.average_scores
        assert len(report.rep_scores) == 3

    def test_with_symmetry(self):
        """Test scoring with symmetry data."""
        scorer = EnhancedScorer()
        scorer.start_session("test")

        angles, timestamps = self._make_rep_data()
        left = np.sin(np.linspace(0, 2 * np.pi, 30)) * 90
        right = np.sin(np.linspace(0, 2 * np.pi, 30)) * 85  # Slight asymmetry

        score = scorer.score_rep(angles, timestamps, target_angle=150,
                                  left_angles=left, right_angles=right)
        assert score.symmetry_score < 100  # Should detect asymmetry

    def test_with_compensation(self):
        """Test scoring with compensation data."""
        scorer = EnhancedScorer()
        scorer.start_session("test")

        angles, timestamps = self._make_rep_data()

        # Create pose sequence with shoulder hiking
        poses = []
        for _ in range(30):
            pose = np.zeros((33, 3))
            pose[11] = [0.3, 0.3, 0]  # High left shoulder
            pose[12] = [0.7, 0.7, 0]
            pose[23] = [0.35, 0.7, 0]
            pose[24] = [0.65, 0.7, 0]
            poses.append(pose)

        score = scorer.score_rep(angles, timestamps, target_angle=150,
                                  pose_sequence=poses)
        assert score.compensation_score < 100
        assert len(score.compensation_types) > 0

    def test_multiple_reps_fatigue(self):
        """Test fatigue detection across multiple reps."""
        scorer = EnhancedScorer()
        scorer.start_session("test")

        # First rep: good
        angles1, ts1 = self._make_rep_data()
        scorer.score_rep(angles1, ts1, target_angle=150)

        # Later reps: degrading (simulated by higher jerk)
        for i in range(5):
            t = np.linspace(0, 2 * np.pi, 30)
            angles = (np.sin(t) * 0.5 + 0.5) * 150 + np.random.randn(30) * (i * 5)
            ts = np.linspace(0, 1, 30)
            scorer.score_rep(angles, ts, target_angle=150)

        report = scorer.get_session_report()
        assert report.total_reps == 6
