"""
3D Pose Estimation Backends.

Direct image-to-3D pose estimation for rehabilitation.
Replaces or supplements MediaPipe for better accuracy.

Backends:
    - MeTRAbs: Metric-scale 3D pose (recommended)
    - HybrIK: Physically plausible joint rotations
    - MediaPipeFallback: CPU-only fallback

Usage:
    from core.pose3d import create_estimator

    estimator = create_estimator("metrab")
    estimator.initialize()
    result = estimator.estimate(frame)
"""

from .base import PoseEstimator3D, Pose3DResult, create_estimator
from .metrab import MeTRAbsEstimator
from .hybrik import HybrIKEstimator
from .mediapipe_fallback import MediaPipeFallbackEstimator

__all__ = [
    "PoseEstimator3D",
    "Pose3DResult",
    "create_estimator",
    "MeTRAbsEstimator",
    "HybrIKEstimator",
    "MediaPipeFallbackEstimator",
]
