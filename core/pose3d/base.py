"""
Abstract base class for 3D pose estimation.

Defines the interface that all 3D pose estimators must implement.
This allows swapping between MeTRAbs, HybrIK, or fallback implementations.

Author: ADAPT-Rehab Team
Version: 3.0.0
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Dict, List
from enum import Enum
import numpy as np


class PoseEstimatorType(Enum):
    """Available pose estimator backends."""
    METRABS = "metrab"
    HYBRIK = "hybrik"
    MEDIAPIPE_FALLBACK = "mediapipe_fallback"


@dataclass
class Pose3DResult:
    """
    Result of 3D pose estimation.

    All coordinates are in metric scale (centimeters) with origin
    at the hip center when using MeTRAbs/HybrIK.

    Attributes:
        keypoints_3d: 3D joint positions, shape (N, 3).
        keypoints_2d: 2D joint positions (pixels), shape (N, 2).
        confidence: Per-joint confidence scores, shape (N,).
        joint_angles: Computed joint angles (dot product), degrees.
        joint_angles_quaternion: Joint angles via quaternions, degrees.
        timestamp_ms: Frame timestamp in milliseconds.
        model_name: Name of the estimator model used.
        is_valid: Whether the result is valid.
        error_message: Error message if invalid.
    """
    keypoints_3d: Optional[np.ndarray] = None
    keypoints_2d: Optional[np.ndarray] = None
    confidence: Optional[np.ndarray] = None
    joint_angles: Dict[str, float] = field(default_factory=dict)
    joint_angles_quaternion: Dict[str, float] = field(default_factory=dict)
    timestamp_ms: int = 0
    model_name: str = ""
    is_valid: bool = False
    error_message: str = ""
    metadata: Dict = field(default_factory=dict)

    @property
    def num_joints(self) -> int:
        if self.keypoints_3d is not None:
            return len(self.keypoints_3d)
        return 0

    def get_joint_angle(self, joint_name: str) -> Optional[float]:
        return self.joint_angles.get(joint_name)

    def to_dict(self) -> dict:
        return {
            "num_joints": self.num_joints,
            "joint_angles": self.joint_angles,
            "joint_angles_quaternion": self.joint_angles_quaternion,
            "timestamp_ms": self.timestamp_ms,
            "model_name": self.model_name,
            "is_valid": self.is_valid,
            "error_message": self.error_message,
        }


class PoseEstimator3D(ABC):
    """
    Abstract base class for 3D pose estimators.

    Example:
        >>> estimator = MeTRAbsEstimator()
        >>> estimator.initialize(model_path="models/metrib_384.bin")
        >>> result = estimator.estimate(frame)
        >>> if result.is_valid:
        ...     print(f"Shoulder: {result.joint_angles['left_shoulder']:.1f}deg")
    """

    # Joint angle definitions: (proximal_idx, vertex_idx, distal_idx)
    JOINT_ANGLE_DEFS = {
        "left_shoulder": (23, 11, 13),
        "right_shoulder": (24, 12, 14),
        "left_elbow": (11, 13, 15),
        "right_elbow": (12, 14, 16),
        "left_hip": (11, 23, 25),
        "right_hip": (12, 24, 26),
        "left_knee": (23, 25, 27),
        "right_knee": (24, 26, 28),
    }

    def __init__(self, model_name: str = "unknown"):
        self._model_name = model_name
        self._frame_count = 0
        self._is_initialized = False

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def is_initialized(self) -> bool:
        return self._is_initialized

    @abstractmethod
    def initialize(self, model_path: Optional[str] = None, **kwargs) -> bool:
        """Load and initialize the model."""
        pass

    @abstractmethod
    def estimate(self, frame: np.ndarray, timestamp_ms: Optional[int] = None) -> Pose3DResult:
        """Estimate 3D pose from a single frame."""
        pass

    def compute_joint_angles(self, keypoints_3d: np.ndarray) -> Dict[str, float]:
        """Compute joint angles using dot product method."""
        angles = {}
        for joint_name, (p_idx, v_idx, d_idx) in self.JOINT_ANGLE_DEFS.items():
            try:
                if max(p_idx, v_idx, d_idx) >= len(keypoints_3d):
                    continue
                a, b, c = keypoints_3d[p_idx], keypoints_3d[v_idx], keypoints_3d[d_idx]
                ba, bc = a - b, c - b
                cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
                angles[joint_name] = float(np.degrees(np.arccos(np.clip(cos_angle, -1, 1))))
            except (IndexError, ValueError):
                continue
        return angles

    def compute_joint_angles_quaternion(self, keypoints_3d: np.ndarray) -> Dict[str, float]:
        """Compute joint angles using quaternion rotation (no gimbal lock)."""
        angles = {}
        for joint_name, (p_idx, v_idx, d_idx) in self.JOINT_ANGLE_DEFS.items():
            try:
                if max(p_idx, v_idx, d_idx) >= len(keypoints_3d):
                    continue
                a, b, c = keypoints_3d[p_idx], keypoints_3d[v_idx], keypoints_3d[d_idx]
                v1, v2 = a - b, c - b
                v1_n = v1 / (np.linalg.norm(v1) + 1e-8)
                v2_n = v2 / (np.linalg.norm(v2) + 1e-8)
                dot = np.clip(np.dot(v1_n, v2_n), -1, 1)
                w = np.sqrt(max(0, (1 + dot) / 2))
                angles[joint_name] = float(2 * np.degrees(np.arccos(np.clip(w, 0, 1))))
            except (IndexError, ValueError):
                continue
        return angles

    def reset(self) -> None:
        self._frame_count = 0

    def close(self) -> None:
        self._is_initialized = False

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
        return False


def create_estimator(estimator_type: str = "metrab", **kwargs) -> PoseEstimator3D:
    """Factory function to create a pose estimator."""
    estimators = {
        "metrab": MeTRAbsEstimator,
        "hybrik": HybrIKEstimator,
        "mediapipe_fallback": MediaPipeFallbackEstimator,
    }
    estimator_type = estimator_type.lower()
    if estimator_type not in estimators:
        raise ValueError(f"Unknown: {estimator_type}. Available: {list(estimators.keys())}")
    return estimators[estimator_type](**kwargs)
