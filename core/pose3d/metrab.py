"""
MeTRAbs 3D Pose Estimator.

Produces metric-scale 3D pose (real centimeters) directly from images.
Best accuracy for rehabilitation joint angle measurement.

Reference: Sarandi et al., "Metric-Scale Truncated-Barrel Human Body Shape
Estimation," WACV 2021.

Usage:
    estimator = MeTRAbsEstimator()
    estimator.initialize()  # Downloads model from TF Hub
    result = estimator.estimate(frame)
    print(result.joint_angles["left_shoulder"])
"""

from typing import Optional, List
import numpy as np
from .base import PoseEstimator3D, Pose3DResult


class MeTRAbsEstimator(PoseEstimator3D):
    """
    MeTRAbs-based 3D pose estimator.

    Produces metric-scale keypoints in centimeters.
    Joint angles computed via both dot-product and quaternion methods.

    Model variants:
        - metrib_256: Fast, lower accuracy
        - metrib_384: Balanced (recommended)
        - metrib_512: Higher accuracy, slower

    Example:
        >>> estimator = MeTRAbsEstimator(model_variant="metrib_384")
        >>> estimator.initialize()
        >>> result = estimator.estimate(frame)
        >>> for joint, angle in result.joint_angles.items():
        ...     print(f"{joint}: {angle:.1f} degrees")
    """

    # SMPL-24 joint indices (MeTRAbs default skeleton)
    SMPL_JOINTS = {
        "pelvis": 0, "left_hip": 1, "right_hip": 2,
        "spine1": 3, "left_knee": 4, "right_knee": 5,
        "spine2": 6, "left_ankle": 7, "right_ankle": 8,
        "spine3": 9, "left_foot": 10, "right_foot": 11,
        "neck": 12, "left_collar": 13, "right_collar": 14,
        "head": 15, "left_shoulder": 16, "right_shoulder": 17,
        "left_elbow": 18, "right_elbow": 19,
        "left_wrist": 20, "right_wrist": 21,
        "left_hand": 22, "right_hand": 23,
    }

    # Joint angle definitions using SMPL indices
    SMPL_ANGLE_DEFS = {
        "left_shoulder": ("pelvis", "left_shoulder", "left_elbow"),
        "right_shoulder": ("pelvis", "right_shoulder", "right_elbow"),
        "left_elbow": ("left_shoulder", "left_elbow", "left_wrist"),
        "right_elbow": ("right_shoulder", "right_elbow", "right_wrist"),
        "left_hip": ("spine1", "left_hip", "left_knee"),
        "right_hip": ("spine1", "right_hip", "right_knee"),
        "left_knee": ("left_hip", "left_knee", "left_ankle"),
        "right_knee": ("right_hip", "right_knee", "right_ankle"),
    }

    def __init__(self, model_variant: str = "metrib_384"):
        super().__init__(model_name=f"MeTRAbs-{model_variant}")
        self._model_variant = model_variant
        self._model = None

    def initialize(self, model_path: Optional[str] = None, **kwargs) -> bool:
        """
        Load MeTRAbs model.

        Args:
            model_path: Path to saved model directory. If None, downloads from TF Hub.
            **kwargs: Additional params:
                - device: "cpu" or "gpu" (default: auto-detect)

        Returns:
            bool: True if initialization successful.
        """
        try:
            import tensorflow as tf

            # Check GPU availability
            gpus = tf.config.list_physical_devices("GPU")
            if gpus:
                try:
                    for gpu in gpus:
                        tf.config.experimental.set_memory_growth(gpu, True)
                    print(f"[MeTRAbs] GPU available: {len(gpus)} device(s)")
                except RuntimeError:
                    pass

            if model_path:
                # Load from local directory
                print(f"[MeTRAbs] Loading model from: {model_path}")
                self._model = tf.saved_model.load(model_path)
            else:
                # Download from TensorFlow Hub
                try:
                    import tensorflow_hub as hub
                    model_url = f"https://tfhub.dev/google/metrabs/{self._model_variant}/1"
                    print(f"[MeTRAbs] Downloading model: {model_url}")
                    self._model = hub.load(model_url)
                except ImportError:
                    print("[MeTRAbs] tensorflow-hub not installed. Install: pip install tensorflow-hub")
                    return False

            self._is_initialized = True
            print(f"[MeTRAbs] Model loaded successfully: {self._model_variant}")
            return True

        except ImportError:
            print("[MeTRAbs] TensorFlow not installed. Install: pip install tensorflow tensorflow-hub")
            return False
        except Exception as e:
            print(f"[MeTRAbs] Initialization failed: {e}")
            return False

    def estimate(self, frame: np.ndarray, timestamp_ms: Optional[int] = None) -> Pose3DResult:
        """
        Estimate 3D pose from a single frame.

        Args:
            frame: BGR image from OpenCV, shape (H, W, 3).
            timestamp_ms: Frame timestamp in milliseconds.

        Returns:
            Pose3DResult with metric-scale 3D keypoints (centimeters).
        """
        if not self._is_initialized or self._model is None:
            return Pose3DResult(
                is_valid=False,
                error_message="MeTRAbs model not initialized"
            )

        if frame is None or frame.size == 0:
            return Pose3DResult(
                is_valid=False,
                error_message="Invalid input frame"
            )

        try:
            import tensorflow as tf

            # Auto timestamp
            if timestamp_ms is None:
                timestamp_ms = int(self._frame_count * (1000 / 30))
            self._frame_count += 1

            # Preprocess: BGR to RGB
            frame_rgb = frame[:, :, ::-1].copy()
            frame_tensor = tf.convert_to_tensor(frame_rgb, dtype=tf.uint8)

            # Run MeTRAbs inference
            # detect_poses returns: poses3d, poses2d, scores
            result = self._model.detect_poses(
                frame_tensor,
                skeleton="smpl_24"  # Standard 24-joint SMPL skeleton
            )

            # Extract results for first person
            poses3d = result["poses3d"].numpy()
            poses2d = result["poses2d"].numpy()
            scores = result["scores"].numpy()

            if len(poses3d) == 0:
                return Pose3DResult(
                    is_valid=False,
                    error_message="No person detected",
                    timestamp_ms=timestamp_ms,
                )

            # First person only
            kps3d = poses3d[0]  # (24, 3) in centimeters
            kps2d = poses2d[0]  # (24, 2) in pixels
            conf = scores[0]    # (24,) confidence

            # Compute joint angles using both methods
            angles_dot = self.compute_joint_angles_smpl(kps3d)
            angles_quat = self.compute_joint_angles_quaternion_smpl(kps3d)

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
                    "scale": "centimeters",
                    "skeleton": "smpl_24",
                    "num_persons": len(poses3d),
                }
            )

        except Exception as e:
            return Pose3DResult(
                is_valid=False,
                error_message=f"MeTRAbs inference error: {str(e)}",
                timestamp_ms=timestamp_ms or 0,
            )

    def compute_joint_angles_smpl(self, keypoints_3d: np.ndarray) -> dict:
        """Compute joint angles using dot product (SMPL skeleton)."""
        angles = {}
        for joint_name, (prox_name, vert_name, dist_name) in self.SMPL_ANGLE_DEFS.items():
            try:
                p_idx = self.SMPL_JOINTS[prox_name]
                v_idx = self.SMPL_JOINTS[vert_name]
                d_idx = self.SMPL_JOINTS[dist_name]

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

    def compute_joint_angles_quaternion_smpl(self, keypoints_3d: np.ndarray) -> dict:
        """Compute joint angles using quaternion rotation (SMPL skeleton)."""
        angles = {}
        for joint_name, (prox_name, vert_name, dist_name) in self.SMPL_ANGLE_DEFS.items():
            try:
                p_idx = self.SMPL_JOINTS[prox_name]
                v_idx = self.SMPL_JOINTS[vert_name]
                d_idx = self.SMPL_JOINTS[dist_name]

                if max(p_idx, v_idx, d_idx) >= len(keypoints_3d):
                    continue

                a, b, c = keypoints_3d[p_idx], keypoints_3d[v_idx], keypoints_3d[d_idx]
                v1, v2 = a - b, c - b
                n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)

                if n1 < 1e-10 or n2 < 1e-10:
                    continue

                v1, v2 = v1 / n1, v2 / n2
                dot = np.clip(np.dot(v1, v2), -1.0, 1.0)
                w = np.sqrt(max(0, (1 + dot) / 2))
                angle = 2 * np.degrees(np.arccos(np.clip(w, 0, 1)))
                angles[joint_name] = float(angle)
            except (IndexError, ValueError):
                continue
        return angles

    def close(self) -> None:
        """Release TensorFlow model resources."""
        self._model = None
        self._is_initialized = False
        print("[MeTRAbs] Model released")
