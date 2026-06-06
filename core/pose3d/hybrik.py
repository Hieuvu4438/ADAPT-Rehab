"""
HybrIK 3D Pose Estimator.

Hybrid analytical-neural inverse kinematics for physically plausible poses.
Reference: Li et al., CVPR 2021.
"""

from typing import Optional, List
import numpy as np
from .base import PoseEstimator3D, Pose3DResult


class HybrIKEstimator(PoseEstimator3D):
    """HybrIK-based 3D pose estimator."""

    def __init__(self, backbone: str = "hrnet"):
        super().__init__(model_name=f"HybrIK-{backbone}")
        self._model = None

    def initialize(self, model_path: Optional[str] = None, **kwargs) -> bool:
        try:
            import torch
            if model_path is None:
                print("[HybrIK] Model path required. Download from: https://github.com/Jeff-sjtu/HybrIK")
                return False
            self._is_initialized = True
            return True
        except ImportError:
            print("[HybrIK] Install: pip install torch")
            return False

    def estimate(self, frame: np.ndarray, timestamp_ms: Optional[int] = None) -> Pose3DResult:
        return Pose3DResult(
            is_valid=False,
            error_message="HybrIK requires HybrIK package installation",
            timestamp_ms=timestamp_ms or 0,
        )
