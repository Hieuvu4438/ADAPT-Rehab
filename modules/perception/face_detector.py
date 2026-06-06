"""
Face Detector using MediaPipe Face Mesh (Tasks API).

Extracts 468 face landmarks for downstream AU detection
and emotion classification.

Usage:
    detector = FaceDetector()
    detector.initialize()
    result = detector.detect(frame)
    if result.is_valid:
        print(f"Landmarks: {result.landmarks.shape}")
        print(f"BBox: {result.bbox}")
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class FaceResult:
    """Result of face detection."""
    landmarks: Optional[np.ndarray] = None  # (468, 3) in pixel coordinates
    bbox: Optional[np.ndarray] = None       # (4,) x1, y1, x2, y2
    confidence: float = 0.0
    is_valid: bool = False
    error_message: str = ""


# Key landmark indices for face analysis
class FaceLandmarkIndex:
    """MediaPipe Face Mesh landmark indices for FACS analysis."""
    # Eyebrows
    LEFT_EYEBROW_INNER = 107
    LEFT_EYEBROW_MIDDLE = 66
    LEFT_EYEBROW_OUTER = 105
    RIGHT_EYEBROW_INNER = 336
    RIGHT_EYEBROW_MIDDLE = 296
    RIGHT_EYEBROW_OUTER = 334

    # Eyes
    LEFT_EYE_TOP = 159
    LEFT_EYE_BOTTOM = 145
    LEFT_EYE_INNER = 133
    LEFT_EYE_OUTER = 33
    RIGHT_EYE_TOP = 386
    RIGHT_EYE_BOTTOM = 374
    RIGHT_EYE_INNER = 362
    RIGHT_EYE_OUTER = 263

    # Nose
    NOSE_TIP = 1
    NOSE_BRIDGE = 6
    NOSE_LEFT = 129
    NOSE_RIGHT = 358

    # Mouth
    UPPER_LIP_TOP = 0
    UPPER_LIP_BOTTOM = 13
    LOWER_LIP_TOP = 14
    LOWER_LIP_BOTTOM = 17
    MOUTH_LEFT = 61
    MOUTH_RIGHT = 291

    # Face outline
    FACE_TOP = 10
    FACE_BOTTOM = 152
    FACE_LEFT = 234
    FACE_RIGHT = 454


class FaceDetector:
    """
    MediaPipe Face Mesh detector using Tasks API.

    Extracts 468 face landmarks from video frames.

    Example:
        >>> detector = FaceDetector()
        >>> detector.initialize()
        >>> result = detector.detect(frame)
        >>> if result.is_valid:
        ...     print(f"Detected {len(result.landmarks)} landmarks")
    """

    def __init__(self):
        self._face_landmarker = None
        self._is_initialized = False
        self._frame_count = 0

    def initialize(self, model_path: Optional[str] = None, **kwargs) -> bool:
        """
        Initialize MediaPipe Face Landmarker.

        Args:
            model_path: Path to face_landmarker.task model. If None, uses default.
            **kwargs: Additional parameters (unused).

        Returns:
            bool: True if initialization successful.
        """
        try:
            import os
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision as mp_vision

            if model_path is None:
                # Try default path
                model_path = os.path.join(
                    os.path.dirname(__file__), "..", "..", "models", "face_landmarker.task"
                )
                model_path = os.path.normpath(model_path)

            if not os.path.exists(model_path):
                print(f"[FaceDetector] Model not found: {model_path}")
                print("[FaceDetector] Download from: https://developers.google.com/mediapipe/solutions/vision/face_landmarker")
                return False

            base_options = mp_python.BaseOptions(model_asset_path=model_path)
            options = mp_vision.FaceLandmarkerOptions(
                base_options=base_options,
                running_mode=mp_vision.RunningMode.VIDEO,
                num_faces=1,
                min_face_detection_confidence=kwargs.get("min_detection_confidence", 0.5),
                min_tracking_confidence=kwargs.get("min_tracking_confidence", 0.5),
                output_face_blendshapes=False,
                output_facial_transformation_matrixes=False,
            )
            self._face_landmarker = mp_vision.FaceLandmarker.create_from_options(options)
            self._is_initialized = True
            print(f"[FaceDetector] Initialized with model: {model_path}")
            return True

        except Exception as e:
            print(f"[FaceDetector] Init failed: {e}")
            return False

    def detect(self, frame: np.ndarray, timestamp_ms: Optional[int] = None) -> FaceResult:
        """
        Detect face and extract landmarks from a frame.

        Args:
            frame: BGR image from OpenCV, shape (H, W, 3).
            timestamp_ms: Frame timestamp in milliseconds.

        Returns:
            FaceResult with landmarks and bounding box.
        """
        if not self._is_initialized or self._face_landmarker is None:
            return FaceResult(error_message="Not initialized")

        try:
            import mediapipe as mp

            if timestamp_ms is None:
                timestamp_ms = int(self._frame_count * (1000 / 30))
            self._frame_count += 1

            # BGR to RGB
            frame_rgb = frame[:, :, ::-1].copy()
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

            # Detect
            result = self._face_landmarker.detect_for_video(mp_image, timestamp_ms)

            if not result.face_landmarks or len(result.face_landmarks) == 0:
                return FaceResult(error_message="No face detected", is_valid=False)

            # Extract landmarks
            face = result.face_landmarks[0]
            h, w = frame.shape[:2]
            landmarks = np.array([[lm.x * w, lm.y * h, lm.z * w] for lm in face.landmark])

            # Compute bounding box
            x_min, y_min = landmarks[:, 0].min(), landmarks[:, 1].min()
            x_max, y_max = landmarks[:, 0].max(), landmarks[:, 1].max()
            bbox = np.array([x_min, y_min, x_max, y_max])

            return FaceResult(
                landmarks=landmarks,
                bbox=bbox,
                confidence=0.9,
                is_valid=True,
            )

        except Exception as e:
            return FaceResult(error_message=str(e), is_valid=False)

    def close(self) -> None:
        """Release resources."""
        if self._face_landmarker:
            self._face_landmarker.close()
            self._face_landmarker = None
        self._is_initialized = False
