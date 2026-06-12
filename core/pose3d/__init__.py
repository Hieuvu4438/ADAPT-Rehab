"""
3D Pose Estimation module.

Provides direct image-to-3D pose estimation.

Available backends:
- ``rtmw3d`` (default): RTMW3D via MMPose, 133 keypoints whole-body. Requires
  the full mmpose/mmcv/mmdet stack.
- ``mediapipe_fallback``: MediaPipe Pose Landmarker (Tasks API), 33 keypoints.
  Lightweight, no extra dependencies beyond ``mediapipe``.

Use :func:`create_estimator` to get an instance by name.
"""

from .base import PoseEstimator3D, Pose3DResult, PoseEstimatorType, create_estimator
from .rtmw3d import RTMW3DEstimator
from .mediapipe_fallback import MediaPipeFallbackEstimator

__all__ = [
    "PoseEstimator3D",
    "Pose3DResult",
    "PoseEstimatorType",
    "create_estimator",
    "RTMW3DEstimator",
    "MediaPipeFallbackEstimator",
]
