"""
RTMW3D 3D Pose Estimator.

Direct image-to-3D whole-body pose estimation using RTMW3D from MMPose.
Produces 133 keypoints: body (33) + hands (42) + face (58).

Reference: Jiang et al., "RTMW: Real-Time Multi-Person 2D and 3D
Whole-body Pose Estimation," arXiv 2024.

Usage:
    estimator = RTMW3DEstimator()
    estimator.initialize()
    result = estimator.estimate(frame)
    print(result.keypoints_3d.shape)  # (133, 3)
"""

import os
import sys
from typing import Optional, Dict, List
import numpy as np
import torch
from .base import PoseEstimator3D, Pose3DResult


# Body joint indices (COCO format, 0-indexed)
BODY_JOINTS = {
    "nose": 0, "left_eye": 1, "right_eye": 2,
    "left_ear": 3, "right_ear": 4,
    "left_shoulder": 5, "right_shoulder": 6,
    "left_elbow": 7, "right_elbow": 8,
    "left_wrist": 9, "right_wrist": 10,
    "left_hip": 11, "right_hip": 12,
    "left_knee": 13, "right_knee": 14,
    "left_ankle": 15, "right_ankle": 16,
}

# Joint angle definitions for RTMW3D (body only)
RTMW3D_ANGLE_DEFS = {
    "left_shoulder": ("left_hip", "left_shoulder", "left_elbow"),
    "right_shoulder": ("right_hip", "right_shoulder", "right_elbow"),
    "left_elbow": ("left_shoulder", "left_elbow", "left_wrist"),
    "right_elbow": ("right_shoulder", "right_elbow", "right_wrist"),
    "left_hip": ("left_shoulder", "left_hip", "left_knee"),
    "right_hip": ("right_shoulder", "right_hip", "right_knee"),
    "left_knee": ("left_hip", "left_knee", "left_ankle"),
    "right_knee": ("right_hip", "right_knee", "right_ankle"),
}


class RTMW3DEstimator(PoseEstimator3D):
    """
    RTMW3D-based 3D pose estimator.

    Direct image-to-3D whole-body pose estimation.
    Produces 133 keypoints in relative 3D coordinates.

    Example:
        >>> estimator = RTMW3DEstimator()
        >>> estimator.initialize()
        >>> result = estimator.estimate(frame)
        >>> for joint, angle in result.joint_angles.items():
        ...     print(f"{joint}: {angle:.1f} degrees")
    """

    def __init__(self, model_variant: str = "rtmw3d-l"):
        super().__init__(model_name=f"RTMW3D-{model_variant}")
        self._model_variant = model_variant
        self._model = None
        self._device = "cuda:0"

        # Find model files (local to project)
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self._model_dir = os.path.join(base_dir, "models", "rtmw3d")

        # Find MMPose repo: env var > common paths > pip install
        self._mmpose_path = self._find_mmpose_path()
        self._project_path = os.path.join(self._mmpose_path, "projects", "rtmpose3d") if self._mmpose_path else ""

        # Config: prefer local copy in models/rtmw3d/
        local_config = os.path.join(self._model_dir, "rtmw3d-l_8xb64_cocktail14-384x288.py")
        if os.path.exists(local_config):
            self._config_path = local_config
        elif self._project_path:
            self._config_path = os.path.join(
                self._project_path, "configs",
                "rtmw3d-l_8xb64_cocktail14-384x288.py"
            )
        else:
            self._config_path = local_config  # Will fail with clear message

        self._checkpoint_path = os.path.join(
            self._model_dir,
            "rtmw3d-l_8xb64_cocktail14-384x288-794dbc78_20240626.pth"
        )

    def _find_mmpose_path(self) -> Optional[str]:
        """Find MMPose repo path by checking env vars and common locations."""
        # 1. Environment variable
        env_path = os.environ.get("MMPOSE_PATH") or os.environ.get("MMPose_PATH")
        if env_path and os.path.isdir(env_path):
            return env_path

        # 2. Try importing mmpose (pip-installed)
        try:
            import mmpose
            mmpose_dir = os.path.dirname(os.path.dirname(mmpose.__file__))
            if os.path.isdir(os.path.join(mmpose_dir, "projects", "rtmpose3d")):
                return mmpose_dir
        except ImportError:
            pass

        # 3. Common locations
        candidates = [
            "/tmp/mmpose",
            os.path.expanduser("~/mmpose"),
            os.path.join(os.path.dirname(self._model_dir), "mmpose"),
        ]
        for path in candidates:
            if os.path.isdir(os.path.join(path, "projects", "rtmpose3d")):
                return path

        return None

    def initialize(self, model_path: Optional[str] = None, **kwargs) -> bool:
        """
        Load RTMW3D model.

        Returns:
            bool: True if model loaded successfully.
        """
        try:
            # Check GPU
            if not torch.cuda.is_available():
                print("[RTMW3D] Warning: CUDA not available, using CPU")
                self._device = "cpu"

            self._device = kwargs.get("device", self._device)

            # Verify files
            for path, name in [
                (self._config_path, "Config"),
                (self._checkpoint_path, "Model weights"),
            ]:
                if not os.path.exists(path):
                    print(f"[RTMW3D] {name} not found: {path}")
                    return False

            if not self._mmpose_path or not os.path.isdir(self._mmpose_path):
                print("[RTMW3D] MMPose repo not found. Set MMPOSE_PATH env var or install to /tmp/mmpose")
                return False

            # Add paths
            sys.path.insert(0, self._mmpose_path)
            sys.path.insert(0, self._project_path)

            # Patch mmengine to use weights_only=False (PyTorch 2.6+ compatibility)
            self._patch_mmengine()

            # Load model
            from mmpose.apis import init_model

            # Create modified config (fix base path)
            config_path = self._create_modified_config()

            print(f"[RTMW3D] Loading model from: {self._checkpoint_path}")
            self._model = init_model(
                config_path,
                self._checkpoint_path,
                device=self._device
            )

            self._is_initialized = True
            print(f"[RTMW3D] Model loaded successfully on {self._device}")
            return True

        except Exception as e:
            print(f"[RTMW3D] Initialization failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _patch_mmengine(self):
        """Patch mmengine to use weights_only=False for PyTorch 2.6+."""
        import mmengine.runner

        if hasattr(mmengine.runner, '_rtmw3d_patched'):
            return  # Already patched

        original_load_checkpoint = mmengine.runner.load_checkpoint

        def patched_load_checkpoint(model, filename, map_location='cpu', strict=False,
                                    logger=None, revise_keys=[(r'^module\.', '')]):
            checkpoint = torch.load(filename, map_location=map_location, weights_only=False)
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            if revise_keys:
                import re
                for pattern, replacement in revise_keys:
                    state_dict = {re.sub(pattern, replacement, k): v for k, v in state_dict.items()}
            model.load_state_dict(state_dict, strict=strict)
            return checkpoint

        mmengine.runner.load_checkpoint = patched_load_checkpoint
        mmengine.runner._rtmw3d_patched = True

    def _create_modified_config(self):
        """Create a modified config file with absolute base path."""
        import tempfile

        # Read original config
        with open(self._config_path, 'r') as f:
            content = f.read()

        # Replace relative base path with absolute
        content = content.replace(
            "_base_ = ['mmpose::_base_/default_runtime.py']",
            f"_base_ = ['{self._mmpose_path}/configs/_base_/default_runtime.py']"
        )

        # Write to temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(content)
            return f.name

    def estimate(self, frame: np.ndarray, timestamp_ms: Optional[int] = None) -> Pose3DResult:
        """
        Estimate 3D pose from a single frame.

        Args:
            frame: BGR image from OpenCV, shape (H, W, 3).
            timestamp_ms: Frame timestamp in milliseconds.

        Returns:
            Pose3DResult with 3D keypoints (133 joints).
        """
        if not self._is_initialized or self._model is None:
            return Pose3DResult(
                is_valid=False,
                error_message="RTMW3D model not initialized"
            )

        if frame is None or frame.size == 0:
            return Pose3DResult(
                is_valid=False,
                error_message="Invalid input frame"
            )

        try:
            # Auto timestamp
            if timestamp_ms is None:
                timestamp_ms = int(self._frame_count * (1000 / 30))
            self._frame_count += 1

            # Run inference
            from mmpose.apis import inference_topdown
            results = inference_topdown(self._model, frame)

            if not results or len(results) == 0:
                return Pose3DResult(
                    is_valid=False,
                    error_message="No person detected",
                    timestamp_ms=timestamp_ms,
                )

            # Extract keypoints from first person
            result = results[0]
            pred = result.pred_instances

            if not hasattr(pred, 'keypoints'):
                return Pose3DResult(
                    is_valid=False,
                    error_message="No keypoints in result",
                    timestamp_ms=timestamp_ms,
                )

            # Get 3D keypoints (133, 3)
            kps3d = pred.keypoints[0] if pred.keypoints.ndim == 3 else pred.keypoints
            kps3d = np.array(kps3d, dtype=np.float32)

            # Get confidence scores
            if hasattr(pred, 'keypoint_scores'):
                conf = pred.keypoint_scores[0] if pred.keypoint_scores.ndim == 2 else pred.keypoint_scores
                conf = np.array(conf, dtype=np.float32)
            else:
                conf = np.ones(len(kps3d), dtype=np.float32)

            # 2D keypoints (x, y)
            kps2d = kps3d[:, :2].copy()

            # Compute joint angles (body only)
            angles_dot = self._compute_angles(kps3d)
            angles_quat = self._compute_angles_quaternion(kps3d)

            return Pose3DResult(
                keypoints_3d=kps3d,
                keypoints_2d=kps2d,
                confidence=conf,
                joint_angles=angles_dot,
                joint_angles_quaternion=angles_quat,
                timestamp_ms=timestamp_ms,
                model_name=self._model_name,
                is_valid=True,
                metadata={
                    "skeleton": "rtmw3d_133",
                    "num_keypoints": len(kps3d),
                    "has_hands": True,
                    "has_face": True,
                }
            )

        except Exception as e:
            return Pose3DResult(
                is_valid=False,
                error_message=f"RTMW3D inference error: {str(e)}",
                timestamp_ms=timestamp_ms or 0,
            )

    def _compute_angles(self, keypoints_3d: np.ndarray) -> Dict[str, float]:
        """Compute joint angles using dot product method."""
        angles = {}
        for joint_name, (prox_name, vert_name, dist_name) in RTMW3D_ANGLE_DEFS.items():
            try:
                if prox_name not in BODY_JOINTS or vert_name not in BODY_JOINTS or dist_name not in BODY_JOINTS:
                    continue

                p_idx = BODY_JOINTS[prox_name]
                v_idx = BODY_JOINTS[vert_name]
                d_idx = BODY_JOINTS[dist_name]

                if max(p_idx, v_idx, d_idx) >= len(keypoints_3d):
                    continue

                a, b, c = keypoints_3d[p_idx], keypoints_3d[v_idx], keypoints_3d[d_idx]
                ba, bc = a - b, c - b
                norm_ba, norm_bc = np.linalg.norm(ba), np.linalg.norm(bc)

                if norm_ba < 1e-10 or norm_bc < 1e-10:
                    continue

                cos_angle = np.dot(ba, bc) / (norm_ba * norm_bc)
                cos_angle = np.clip(cos_angle, -1.0, 1.0)
                angles[joint_name] = float(np.degrees(np.arccos(cos_angle)))
            except (IndexError, ValueError):
                continue
        return angles

    def _compute_angles_quaternion(self, keypoints_3d: np.ndarray) -> Dict[str, float]:
        """Compute joint angles using proper quaternion rotation.

        Forms quaternion from cross product (axis) and dot product (angle):
        q = [cos(θ/2), sin(θ/2) * axis]
        """
        angles = {}
        for joint_name, (prox_name, vert_name, dist_name) in RTMW3D_ANGLE_DEFS.items():
            try:
                if prox_name not in BODY_JOINTS or vert_name not in BODY_JOINTS or dist_name not in BODY_JOINTS:
                    continue

                p_idx = BODY_JOINTS[prox_name]
                v_idx = BODY_JOINTS[vert_name]
                d_idx = BODY_JOINTS[dist_name]

                if max(p_idx, v_idx, d_idx) >= len(keypoints_3d):
                    continue

                a, b, c = keypoints_3d[p_idx], keypoints_3d[v_idx], keypoints_3d[d_idx]
                v1, v2 = a - b, c - b
                n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)

                if n1 < 1e-10 or n2 < 1e-10:
                    continue

                v1, v2 = v1 / n1, v2 / n2

                # Compute dot and cross product
                dot = np.clip(np.dot(v1, v2), -1.0, 1.0)
                cross = np.cross(v1, v2)
                cross_norm = np.linalg.norm(cross)

                # Form quaternion: q = [w, x, y, z]
                half_angle = np.arccos(dot) / 2.0
                w = np.cos(half_angle)

                # Extract angle: θ = 2 * arccos(w)
                w_clamped = np.clip(w, -1.0, 1.0)
                angle_rad = 2.0 * np.arccos(w_clamped)
                angles[joint_name] = float(np.degrees(angle_rad))
            except (IndexError, ValueError):
                continue
        return angles

    def close(self) -> None:
        """Release model resources."""
        self._model = None
        self._is_initialized = False
        print("[RTMW3D] Model released")
