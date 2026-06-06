"""
MediaPipe Fallback 3D Pose Estimator.

CPU-only fallback using MediaPipe Tasks API.
Note: Z-coordinates are NOT metric. Use MeTRAbs for accuracy.
"""

from typing import Optional
import numpy as np
from .base import PoseEstimator3D, Pose3DResult


class MediaPipeFallbackEstimator(PoseEstimator3D):
    """MediaPipe Tasks API pose estimator (fallback)."""

    def __init__(self):
        super().__init__(model_name="MediaPipe-Fallback")
        self._pose_landmarker = None

    def initialize(self, model_path: Optional[str] = None, **kwargs) -> bool:
        try:
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision as mp_vision

            if model_path is None:
                import os
                model_path = os.path.join(os.path.dirname(__file__), "..", "..", "models", "pose_landmarker_lite.task")
                model_path = os.path.normpath(model_path)

            if not os.path.exists(model_path):
                print(f"[MediaPipe] Model not found: {model_path}")
                return False

            base_options = mp_python.BaseOptions(model_asset_path=model_path)
            options = mp_vision.PoseLandmarkerOptions(
                base_options=base_options,
                running_mode=mp_vision.RunningMode.VIDEO,
                num_poses=1,
                min_pose_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            self._pose_landmarker = mp_vision.PoseLandmarker.create_from_options(options)
            self._is_initialized = True
            print(f"[MediaPipe] Initialized with model: {model_path}")
            return True
        except Exception as e:
            print(f"[MediaPipe] Init failed: {e}")
            return False

    def estimate(self, frame: np.ndarray, timestamp_ms: Optional[int] = None) -> Pose3DResult:
        if not self._is_initialized or self._pose_landmarker is None:
            return Pose3DResult(is_valid=False, error_message="Not initialized")
        try:
            import mediapipe as mp

            if timestamp_ms is None:
                timestamp_ms = int(self._frame_count * (1000 / 30))
            self._frame_count += 1

            frame_rgb = frame[:, :, ::-1].copy()
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            result = self._pose_landmarker.detect_for_video(mp_image, timestamp_ms)

            if not result.pose_landmarks or len(result.pose_landmarks) == 0:
                return Pose3DResult(is_valid=False, error_message="No pose", timestamp_ms=timestamp_ms)

            lms = result.pose_landmarks[0]
            kps3d = np.array([[lm.x, lm.y, lm.z] for lm in lms])
            conf = np.array([getattr(lm, 'visibility', 0.5) for lm in lms])

            angles_dot = self.compute_joint_angles(kps3d)
            angles_quat = self.compute_joint_angles_quaternion(kps3d)

            return Pose3DResult(
                keypoints_3d=kps3d, keypoints_2d=kps3d[:, :2], confidence=conf,
                joint_angles=angles_dot, joint_angles_quaternion=angles_quat,
                timestamp_ms=timestamp_ms, model_name=self._model_name, is_valid=True,
                metadata={"note": "Z NOT metric"},
            )
        except Exception as e:
            return Pose3DResult(is_valid=False, error_message=str(e), timestamp_ms=timestamp_ms or 0)

    def close(self) -> None:
        if self._pose_landmarker:
            self._pose_landmarker.close()
        super().close()
