"""
MediaPipe Pose Fallback Estimator for ADAPT-Rehab.

This is the lightweight 3D pose backend used when RTMW3D (MMPose) is not
available. It wraps ``mediapipe.tasks.python.vision.PoseLandmarker`` (the
MediaPipe Tasks API) and exposes the same ``Pose3DResult`` interface as
``RTMW3DEstimator``.

Why this exists
---------------
RTMW3D requires the full ``mmpose`` + ``mmcv-full`` + ``mmdet`` stack which
is not always installable (e.g. on Python 3.13, or in CI sandboxes). MediaPipe
is a pip-installable, much smaller dependency. Its Pose Landmarker produces
33 whole-body keypoints in normalized image coords *and* in real-world
meters (``pose_world_landmarks``); we use the world landmarks directly as
``keypoints_3d`` (no separate 2D-to-3D lifting model is needed).

Joint angle indices match the base class's MediaPipe-style
``JOINT_ANGLE_DEFS`` (``left_shoulder=(23,11,13)`` etc.), so the inherited
``compute_joint_angles()`` and ``compute_joint_angles_quaternion()`` work
without remapping.

The model file ``pose_landmarker_lite.task`` (~5 MB) is auto-downloaded from
Google's public CDN on first run if not present on disk.

Usage
-----
>>> from core.pose3d import create_estimator
>>> est = create_estimator("mediapipe_fallback")
>>> est.initialize()
>>> import cv2
>>> frame = cv2.imread("photo.jpg")
>>> result = est.estimate(frame)
>>> result.joint_angles["left_elbow"]
"""

from __future__ import annotations

import os
import urllib.request
from typing import Optional

import numpy as np

from .base import Pose3DResult, PoseEstimator3D


# Default model download URL (Google's public MediaPipe model registry).
# The "lite" variant is small (~5 MB) and suitable for real-time use.
_POSE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_lite/float16/latest/"
    "pose_landmarker_lite.task"
)
_POSE_MODEL_FILENAME = "pose_landmarker_lite.task"

# Project root is two levels up from this file: core/pose3d/mediapipe_fallback.py
_PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
_DEFAULT_MODEL_DIR = os.path.join(_PROJECT_ROOT, "models", "mediapipe")


def _ensure_mediapipe():
    """Lazily import MediaPipe so that other estimators don't need it."""
    try:
        import mediapipe as mp
        from mediapipe.tasks import python as mp_tasks
        from mediapipe.tasks.python import vision as mp_vision
    except ImportError as e:
        raise ImportError(
            "MediaPipe is required for the MediaPipeFallbackEstimator. "
            "Install with: pip install mediapipe"
        ) from e
    return mp, mp_tasks, mp_vision


def _download_file(url: str, dest: str) -> None:
    """Download ``url`` to ``dest`` with a small progress message."""
    print(f"[MediaPipeFallback] Downloading {os.path.basename(dest)} ...")
    urllib.request.urlretrieve(url, dest)
    print(f"[MediaPipeFallback] Saved to {dest}")


def _ensure_model(model_path: str) -> str:
    """Return a valid path to ``pose_landmarker_lite.task``, downloading if needed.

    Args:
        model_path: User-provided path, or empty string to use the default
            location under ``<project_root>/models/mediapipe/``.

    Returns:
        Absolute path to an existing ``.task`` file.

    Raises:
        RuntimeError: If the model file is missing and cannot be downloaded
            (e.g. offline environment).
    """
    if model_path and os.path.exists(model_path):
        return model_path

    target_dir = os.path.dirname(model_path) if model_path else _DEFAULT_MODEL_DIR
    os.makedirs(target_dir, exist_ok=True)
    target = os.path.join(target_dir, _POSE_MODEL_FILENAME)

    if os.path.exists(target):
        return target

    if model_path:
        # Caller asked for a specific path that does not exist — don't auto-
        # download to a different location, fail loudly instead.
        raise RuntimeError(
            f"MediaPipe pose model not found at: {model_path}. "
            f"Either provide a valid path or pass model_path='' to allow "
            f"auto-download to {target}."
        )

    try:
        _download_file(_POSE_MODEL_URL, target)
    except Exception as e:
        raise RuntimeError(
            f"Failed to auto-download MediaPipe pose model: {e}. "
            f"Please download manually from {_POSE_MODEL_URL} and place it "
            f"at {target}."
        ) from e

    return target


class MediaPipeFallbackEstimator(PoseEstimator3D):
    """3D pose estimator backed by MediaPipe Pose Landmarker (Tasks API).

    Provides the same interface as ``RTMW3DEstimator`` but with the much
    lighter MediaPipe dependency. 33 keypoints are returned in
    MediaPipe-native ordering (face: 0-10, arms: 11-22, legs: 23-32);
    this matches the base class's ``JOINT_ANGLE_DEFS`` so joint angles
    are computed with no remapping.

    Attributes:
        model_name: Always ``"MediaPipe-Pose-Lite"``.
    """

    # Override angle defs to match the base class convention (MediaPipe indices).
    # Inherited JOINT_ANGLE_DEFS already uses MediaPipe-style indices.

    def __init__(self, model_path: str = "", device: str = "cpu"):
        super().__init__(model_name="MediaPipe-Pose-Lite")
        self._user_model_path = model_path
        self._device = device  # MediaPipe runs on CPU; this is recorded for parity.
        self._landmarker = None
        self._mp = None
        self._mp_tasks = None
        self._mp_vision = None
        self._frame_count = 0
        # Re-use the inherited pose3d/pose3d count. The base class already
        # tracks self._frame_count and self._is_initialized.

    def initialize(self, model_path: Optional[str] = None, **kwargs) -> bool:
        """Load the MediaPipe Pose Landmarker model.

        Args:
            model_path: Optional path to a ``.task`` file. If empty, the
                default location ``<project>/models/mediapipe/`` is used
                and the model is auto-downloaded if missing.
            **kwargs: Ignored (kept for signature parity with RTMW3D).

        Returns:
            True on success, False on failure (with a printed error).
        """
        try:
            self._mp, self._mp_tasks, self._mp_vision = _ensure_mediapipe()
        except ImportError as e:
            print(f"[MediaPipeFallback] {e}")
            return False

        requested = model_path if model_path else self._user_model_path
        try:
            resolved = _ensure_model(requested or "")
        except RuntimeError as e:
            print(f"[MediaPipeFallback] {e}")
            return False

        try:
            base_options = self._mp_tasks.BaseOptions(model_asset_path=resolved)
            options = self._mp_vision.PoseLandmarkerOptions(
                base_options=base_options,
                running_mode=self._mp_vision.RunningMode.VIDEO,
                num_poses=1,
                min_pose_detection_confidence=0.5,
                min_pose_presence_confidence=0.5,
                min_tracking_confidence=0.5,
                output_segmentation_masks=False,
            )
            self._landmarker = self._mp_vision.PoseLandmarker.create_from_options(options)
        except Exception as e:
            print(f"[MediaPipeFallback] Failed to load model: {e}")
            return False

        self._is_initialized = True
        self._frame_count = 0
        print(f"[MediaPipeFallback] Model loaded from {resolved}")
        return True

    def estimate(
        self,
        frame: np.ndarray,
        timestamp_ms: Optional[int] = None,
    ) -> Pose3DResult:
        """Run pose estimation on a single BGR frame.

        Args:
            frame: BGR image as a ``numpy.ndarray`` of shape ``(H, W, 3)``.
            timestamp_ms: Optional timestamp in milliseconds (required for
                ``RunningMode.VIDEO``). If None, computed from the
                monotonic frame counter assuming 30 FPS.

        Returns:
            ``Pose3DResult`` with MediaPipe-style 33 keypoints. ``is_valid``
            is False if no person was detected or the frame is empty.
        """
        if not self._is_initialized or self._landmarker is None:
            return Pose3DResult(
                is_valid=False,
                error_message="MediaPipeFallback model not initialized",
                model_name=self.model_name,
            )

        if frame is None or frame.size == 0 or frame.ndim != 3:
            return Pose3DResult(
                is_valid=False,
                error_message="Invalid input frame",
                model_name=self.model_name,
            )

        try:
            # MediaPipe Tasks API requires monotonically increasing timestamps
            # in VIDEO mode. Use a counter.
            if timestamp_ms is None:
                timestamp_ms = int(self._frame_count * (1000 / 30))
            self._frame_count += 1

            # BGR (OpenCV) → RGB
            rgb = frame[:, :, ::-1].copy()
            mp_image = self._mp.Image(
                image_format=self._mp.ImageFormat.SRGB, data=rgb
            )

            mp_result = self._landmarker.detect_for_video(mp_image, timestamp_ms)

            if (
                not mp_result.pose_landmarks
                or len(mp_result.pose_landmarks) == 0
            ):
                return Pose3DResult(
                    is_valid=False,
                    error_message="No person detected",
                    timestamp_ms=timestamp_ms,
                    model_name=self.model_name,
                )

            landmarks_2d = mp_result.pose_landmarks[0]  # normalized landmarks
            world_landmarks = (
                mp_result.pose_world_landmarks[0]
                if mp_result.pose_world_landmarks
                else None
            )

            h, w = frame.shape[:2]
            kp2d_pixels = np.array(
                [[lm.x * w, lm.y * h] for lm in landmarks_2d], dtype=np.float32
            )
            visibility = np.array(
                [getattr(lm, "visibility", 0.0) or 0.0 for lm in landmarks_2d],
                dtype=np.float32,
            )

            if world_landmarks is not None and len(world_landmarks) == len(landmarks_2d):
                # Real-world coords in meters, MediaPipe's origin at hip midpoint.
                kp3d = np.array(
                    [[lm.x, lm.y, lm.z] for lm in world_landmarks],
                    dtype=np.float32,
                )
                units = "meters"
            else:
                # Fallback: synthesize a tiny 3D coord from normalized 2D + z.
                kp3d = np.array(
                    [[lm.x, lm.y, getattr(lm, "z", 0.0) or 0.0] for lm in landmarks_2d],
                    dtype=np.float32,
                )
                units = "normalized"

            # Compute joint angles via the base class helpers (MediaPipe indices).
            joint_angles = self.compute_joint_angles(kp3d)
            joint_angles_quat = self.compute_joint_angles_quaternion(kp3d)

            return Pose3DResult(
                keypoints_3d=kp3d,
                keypoints_2d=kp2d_pixels,
                confidence=visibility,
                joint_angles=joint_angles,
                joint_angles_quaternion=joint_angles_quat,
                timestamp_ms=timestamp_ms,
                model_name=self.model_name,
                is_valid=True,
                error_message="",
                metadata={
                    "skeleton": "mediapipe_33",
                    "num_keypoints": len(kp3d),
                    "units": units,
                    "device": self._device,
                },
            )
        except Exception as e:
            return Pose3DResult(
                is_valid=False,
                error_message=f"MediaPipeFallback estimate error: {e}",
                timestamp_ms=timestamp_ms if timestamp_ms is not None else 0,
                model_name=self.model_name,
            )

    def close(self) -> None:
        """Release the underlying MediaPipe landmarker."""
        if self._landmarker is not None:
            try:
                self._landmarker.close()
            except Exception:
                pass
        self._landmarker = None
        self._is_initialized = False

    # The base class already provides compute_joint_angles() and
    # compute_joint_angles_quaternion(); inherit them as-is.

    def __repr__(self) -> str:
        status = "initialized" if self._is_initialized else "not initialized"
        return f"<MediaPipeFallbackEstimator {status}>"
