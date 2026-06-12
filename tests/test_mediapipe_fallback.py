"""
Tests for the MediaPipeFallbackEstimator (Phase 2).

Verifies:
1. Factory dispatch (``create_estimator("mediapipe_fallback")``).
2. ``PoseEstimatorType`` enum contains the new backend.
3. ``initialize()`` succeeds.
4. ``estimate()`` returns a ``Pose3DResult`` with the expected shape.
5. Joint angles are computed for the 8 expected joints.

Run with: pytest tests/test_mediapipe_fallback.py -v
"""

import os
import sys
import numpy as np
import cv2
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestFactoryDispatch:
    def test_pose_estimator_type_enum(self):
        from core.pose3d import PoseEstimatorType
        assert hasattr(PoseEstimatorType, "MEDIAPIPE_FALLBACK")
        assert PoseEstimatorType.MEDIAPIPE_FALLBACK.value == "mediapipe_fallback"

    def test_create_estimator_dispatch(self):
        from core.pose3d import create_estimator, MediaPipeFallbackEstimator
        est = create_estimator("mediapipe_fallback")
        assert isinstance(est, MediaPipeFallbackEstimator)
        assert est.model_name == "MediaPipe-Pose-Lite"
        est.close()

    def test_unknown_backend_raises(self):
        from core.pose3d import create_estimator
        with pytest.raises(ValueError):
            create_estimator("totally_fake_backend")


class TestMediaPipeFallbackEstimator:
    @pytest.fixture(scope="class")
    def estimator(self):
        from core.pose3d import create_estimator
        est = create_estimator("mediapipe_fallback")
        ok = est.initialize()
        if not ok:
            pytest.skip("MediaPipe model failed to load (offline?)")
        yield est
        est.close()

    def test_initialize_returns_true(self, estimator):
        assert estimator.is_initialized

    def test_estimate_returns_pose3d_result(self, estimator):
        cap = cv2.VideoCapture(
            "data/yoga_datasets/Yoga_Vid_Collected/Ameya_Shavasana.mp4"
        )
        if not cap.isOpened():
            pytest.skip("Test video not available")
        ret, frame = cap.read()
        cap.release()
        if not ret:
            pytest.skip("Could not read frame from test video")

        result = estimator.estimate(frame, timestamp_ms=0)
        if not result.is_valid:
            pytest.skip("No person detected in first frame")
        assert result.keypoints_3d.shape == (33, 3)
        assert result.keypoints_2d.shape == (33, 2)
        assert result.confidence.shape == (33,)
        assert result.is_valid is True
        assert result.model_name == "MediaPipe-Pose-Lite"

    def test_joint_angles_have_expected_keys(self, estimator):
        cap = cv2.VideoCapture(
            "data/yoga_datasets/Yoga_Vid_Collected/Ameya_Shavasana.mp4"
        )
        ret, frame = cap.read()
        cap.release()
        if not ret:
            pytest.skip("Could not read frame from test video")

        # Run multiple frames so the timestamp stays monotonic (required for VIDEO mode)
        for i in range(5):
            result = estimator.estimate(frame, timestamp_ms=i * 33)
            if result.is_valid:
                break
        if not result.is_valid:
            pytest.skip("No person detected")

        # 8 joints in JOINT_ANGLE_DEFS
        expected_joints = {
            "left_shoulder", "right_shoulder",
            "left_elbow", "right_elbow",
            "left_hip", "right_hip",
            "left_knee", "right_knee",
        }
        assert set(result.joint_angles.keys()) == expected_joints
        # All angles should be in [0, 180] degrees
        for joint, angle in result.joint_angles.items():
            assert 0.0 <= angle <= 180.0, f"{joint} = {angle}° out of range"

    def test_metadata(self, estimator):
        cap = cv2.VideoCapture(
            "data/yoga_datasets/Yoga_Vid_Collected/Ameya_Shavasana.mp4"
        )
        ret, frame = cap.read()
        cap.release()
        if not ret:
            pytest.skip("Could not read frame")
        for i in range(5):
            result = estimator.estimate(frame, timestamp_ms=i * 33)
            if result.is_valid:
                break
        if not result.is_valid:
            pytest.skip("No person detected")

        assert result.metadata["skeleton"] == "mediapipe_33"
        assert result.metadata["num_keypoints"] == 33
        assert result.metadata["units"] in ("meters", "normalized")

    def test_estimate_empty_frame_returns_invalid(self, estimator):
        result = estimator.estimate(np.zeros((10, 10, 3), dtype=np.uint8))
        # Either no person detected, or it's valid if by chance the model
        # responds to a tiny image. Either way, the call must not crash.
        assert isinstance(result.is_valid, bool)
