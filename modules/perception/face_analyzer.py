"""
Combined Face Analyzer.

Integrates face detection, AU detection, and emotion classification
into a single easy-to-use interface.

Usage:
    analyzer = FaceAnalyzer()
    analyzer.initialize()
    result = analyzer.analyze(frame)
    if result.is_valid:
        print(f"Emotion: {result.emotion.value}")
        print(f"Pain: {result.pain_level} (PSPI={result.pain_score:.1f})")
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, List
import numpy as np

from .face_detector import FaceDetector, FaceResult
from .au_detector import ActionUnitDetector, AUResult
from .emotion_classifier import EmotionClassifier, EmotionResult, Emotion


@dataclass
class FaceAnalysisResult:
    """Combined result of face analysis."""
    # Face detection
    face_detected: bool = False
    face_bbox: Optional[np.ndarray] = None
    face_landmarks: Optional[np.ndarray] = None

    # Emotion
    emotion: Emotion = Emotion.NEUTRAL
    emotion_confidence: float = 0.0
    emotion_probabilities: Dict[str, float] = field(default_factory=dict)

    # Pain
    pain_score: float = 0.0  # PSPI (0-16)
    pain_level: str = "NONE"  # NONE, MILD, MODERATE, SEVERE
    au_activations: Dict[str, float] = field(default_factory=dict)

    # Meta
    is_valid: bool = False
    error_message: str = ""

    def to_dict(self) -> dict:
        return {
            "face_detected": self.face_detected,
            "emotion": self.emotion.value,
            "emotion_confidence": round(self.emotion_confidence, 3),
            "pain_score": round(self.pain_score, 2),
            "pain_level": self.pain_level,
            "au_activations": {k: round(v, 3) for k, v in self.au_activations.items()},
        }


class FaceAnalyzer:
    """
    Combined face analyzer.

    Integrates:
    - Face detection (MediaPipe Face Mesh)
    - AU detection (landmark-based FACS)
    - Emotion classification (geometric features)

    Example:
        >>> analyzer = FaceAnalyzer()
        >>> analyzer.initialize()
        >>> result = analyzer.analyze(frame)
        >>> if result.is_valid:
        ...     print(f"Emotion: {result.emotion.value}")
        ...     print(f"Pain: {result.pain_level}")
    """

    def __init__(self, use_baseline: bool = True):
        """
        Initialize face analyzer.

        Args:
            use_baseline: If True, calibrate AU detector from neutral face.
        """
        self._face_detector = FaceDetector()
        self._au_detector = ActionUnitDetector(use_baseline=use_baseline)
        self._emotion_classifier = EmotionClassifier()
        self._is_initialized = False
        self._frame_count = 0

    def initialize(self, face_model_path: Optional[str] = None, **kwargs) -> bool:
        """
        Initialize all components.

        Args:
            face_model_path: Path to face_landmarker.task model.
            **kwargs: Additional parameters.

        Returns:
            bool: True if all components initialized.
        """
        # Face detector
        if not self._face_detector.initialize(model_path=face_model_path, **kwargs):
            print("[FaceAnalyzer] Face detector initialization failed")
            return False

        # Emotion classifier (always works, no model needed)
        self._emotion_classifier.initialize()

        self._is_initialized = True
        print("[FaceAnalyzer] All components initialized")
        return True

    def calibrate_baseline(self, frame: np.ndarray) -> bool:
        """
        Calibrate AU detector baseline from a neutral face.

        Args:
            frame: BGR image with a neutral face.

        Returns:
            bool: True if calibration successful.
        """
        result = self._face_detector.detect(frame)
        if result.is_valid and result.landmarks is not None:
            self._au_detector.set_baseline(result.landmarks)
            return True
        return False

    def analyze(self, frame: np.ndarray, timestamp_ms: Optional[int] = None) -> FaceAnalysisResult:
        """
        Analyze face in a frame.

        Args:
            frame: BGR image from OpenCV.
            timestamp_ms: Frame timestamp in milliseconds.

        Returns:
            FaceAnalysisResult with emotion and pain analysis.
        """
        if not self._is_initialized:
            return FaceAnalysisResult(error_message="Not initialized")

        self._frame_count += 1

        # Step 1: Detect face
        face_result = self._face_detector.detect(frame, timestamp_ms)
        if not face_result.is_valid or face_result.landmarks is None:
            return FaceAnalysisResult(
                error_message=face_result.error_message or "No face detected"
            )

        landmarks = face_result.landmarks

        # Step 2: Detect AU and pain (pass frame for deep learning backends)
        au_result = self._au_detector.detect(landmarks, face_image=frame)

        # Step 3: Classify emotion (pass frame for deep learning backends)
        emotion_result = self._emotion_classifier.classify(landmarks, face_image=frame)

        # Combine results
        return FaceAnalysisResult(
            face_detected=True,
            face_bbox=face_result.bbox,
            face_landmarks=landmarks,
            emotion=emotion_result.emotion if emotion_result.is_valid else Emotion.NEUTRAL,
            emotion_confidence=emotion_result.confidence if emotion_result.is_valid else 0.0,
            emotion_probabilities=emotion_result.probabilities if emotion_result.is_valid else {},
            pain_score=au_result.pain_score if au_result.is_valid else 0.0,
            pain_level=au_result.pain_level if au_result.is_valid else "NONE",
            au_activations=au_result.au_activations if au_result.is_valid else {},
            is_valid=True,
        )

    def get_supported_emotions(self) -> List[str]:
        """Get list of supported emotions."""
        return [e.value for e in Emotion]

    def reset(self) -> None:
        """Reset analyzer state."""
        self._au_detector.reset()
        self._frame_count = 0

    def close(self) -> None:
        """Release resources."""
        self._face_detector.close()
        self._emotion_classifier.close()
        self._is_initialized = False
