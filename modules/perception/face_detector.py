"""
Face Detector using MediaPipe Face Mesh.

Extracts 468 face landmarks for AU detection and emotion classification.
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class FaceResult:
    landmarks: Optional[np.ndarray] = None  # (468, 3)
    bbox: Optional[np.ndarray] = None       # (4,) x1, y1, x2, y2
    confidence: float = 0.0
    is_valid: bool = False
    error_message: str = ""


class FaceDetector:
    """MediaPipe Face Mesh detector (468 landmarks)."""

    def __init__(self):
        self._face_mesh = None
        self._is_initialized = False

    def initialize(self, **kwargs) -> bool:
        try:
            import mediapipe as mp
            self._face_mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=False, max_num_faces=1, refine_landmarks=True,
                min_detection_confidence=kwargs.get("min_detection_confidence", 0.5),
                min_tracking_confidence=kwargs.get("min_tracking_confidence", 0.5),
            )
            self._is_initialized = True
            return True
        except Exception as e:
            print(f"[FaceDetector] Init failed: {e}")
            return False

    def detect(self, frame: np.ndarray) -> FaceResult:
        if not self._is_initialized:
            return FaceResult(error_message="Not initialized")
        try:
            import mediapipe as mp
            results = self._face_mesh.process(frame[:, :, ::-1].copy())
            if not results.multi_face_landmarks:
                return FaceResult(error_message="No face detected")
            face = results.multi_face_landmarks[0]
            h, w = frame.shape[:2]
            lms = np.array([[lm.x * w, lm.y * h, lm.z * w] for lm in face.landmark])
            return FaceResult(
                landmarks=lms,
                bbox=np.array([lms[:, 0].min(), lms[:, 1].min(), lms[:, 0].max(), lms[:, 1].max()]),
                confidence=0.9, is_valid=True,
            )
        except Exception as e:
            return FaceResult(error_message=str(e))

    def close(self) -> None:
        if self._face_mesh:
            self._face_mesh.close()
        self._is_initialized = False
