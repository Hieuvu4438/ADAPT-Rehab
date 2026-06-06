"""
Emotion Classifier using Face Mesh landmarks.

Classifies facial emotions from MediaPipe Face Mesh landmarks
using geometric features (no deep learning model required).

7 basic emotions: angry, disgust, fear, happy, neutral, sad, surprise

Usage:
    classifier = EmotionClassifier()
    result = classifier.classify(face_landmarks)
    if result.is_valid:
        print(f"Emotion: {result.emotion.value} ({result.confidence:.1%})")
"""

from dataclasses import dataclass
from typing import Optional, Dict
from enum import Enum
import numpy as np

from .face_detector import FaceLandmarkIndex


class Emotion(Enum):
    """Basic emotions."""
    NEUTRAL = "neutral"
    HAPPY = "happy"
    SAD = "sad"
    ANGRY = "angry"
    FEAR = "fear"
    SURPRISE = "surprise"
    DISGUST = "disgust"


@dataclass
class EmotionResult:
    """Result of emotion classification."""
    emotion: Emotion = Emotion.NEUTRAL
    confidence: float = 0.0
    probabilities: Dict[str, float] = None
    is_valid: bool = False
    error_message: str = ""

    def __post_init__(self):
        if self.probabilities is None:
            self.probabilities = {}


class EmotionClassifier:
    """
    Geometric feature-based emotion classifier.

    Uses Face Mesh landmarks to compute facial geometry features
    and classify emotions using rule-based scoring.

    This is a rule-based classifier that doesn't require training.
    For better accuracy, use a deep learning model (MobileNetV3).

    Features used:
        - Eye openness (EAR)
        - Eyebrow position
        - Mouth shape (smile = happy, open = surprise)
        - Nose wrinkle (disgust)
        - Overall facial tension

    Example:
        >>> classifier = EmotionClassifier()
        >>> result = classifier.classify(face_landmarks)
        >>> print(f"Emotion: {result.emotion.value}")
    """

    def __init__(self):
        """Initialize emotion classifier."""
        self._is_initialized = True

    def initialize(self, **kwargs) -> bool:
        """Initialize classifier (no model loading needed for rule-based)."""
        self._is_initialized = True
        return True

    def classify(self, landmarks: np.ndarray) -> EmotionResult:
        """
        Classify emotion from face landmarks.

        Args:
            landmarks: Face landmarks array, shape (468, 3).

        Returns:
            EmotionResult with predicted emotion and confidence.
        """
        if landmarks is None or len(landmarks) < 468:
            return EmotionResult(error_message="Invalid landmarks", is_valid=False)

        try:
            # Extract geometric features
            features = self._extract_features(landmarks)

            # Score each emotion
            scores = self._score_emotions(features)

            # Get best emotion
            best_emotion = max(scores, key=scores.get)
            best_score = scores[best_emotion]

            # Normalize scores to probabilities
            total = sum(scores.values())
            if total > 0:
                probs = {k: v / total for k, v in scores.items()}
            else:
                probs = {k: 1.0 / len(scores) for k in scores}

            return EmotionResult(
                emotion=Emotion(best_emotion),
                confidence=best_score,
                probabilities=probs,
                is_valid=True,
            )

        except Exception as e:
            return EmotionResult(error_message=str(e), is_valid=False)

    def _extract_features(self, lms: np.ndarray) -> Dict[str, float]:
        """Extract geometric features from landmarks."""
        features = {}

        # Face height for normalization
        face_top = lms[FaceLandmarkIndex.FACE_TOP]
        face_bottom = lms[FaceLandmarkIndex.FACE_BOTTOM]
        face_height = np.linalg.norm(face_top[:2] - face_bottom[:2])
        if face_height < 1e-6:
            face_height = 1.0

        # Eye Aspect Ratio (EAR)
        features["ear"] = self._eye_aspect_ratio(lms)

        # Eyebrow position (normalized)
        features["brow_position"] = self._brow_position(lms, face_height)

        # Mouth openness
        features["mouth_openness"] = self._mouth_openness(lms, face_height)

        # Smile ratio (mouth width vs height)
        features["smile_ratio"] = self._smile_ratio(lms)

        # Nose wrinkle
        features["nose_wrinkle"] = self._nose_wrinkle(lms, face_height)

        # Lip corner position (for smile detection)
        features["lip_corner"] = self._lip_corner_position(lms, face_height)

        return features

    def _eye_aspect_ratio(self, lms: np.ndarray) -> float:
        """Compute average EAR."""
        def ear(side):
            if side == "left":
                top, bottom = lms[159], lms[145]
                inner, outer = lms[133], lms[33]
            else:
                top, bottom = lms[386], lms[374]
                inner, outer = lms[362], lms[263]
            v = np.linalg.norm(top[:2] - bottom[:2])
            h = np.linalg.norm(inner[:2] - outer[:2])
            return v / h if h > 1e-6 else 0.3

        return (ear("left") + ear("right")) / 2

    def _brow_position(self, lms: np.ndarray, fh: float) -> float:
        """Compute eyebrow position (lower = angry)."""
        left_brow = np.linalg.norm(lms[66][:2] - lms[159][:2]) / fh
        right_brow = np.linalg.norm(lms[296][:2] - lms[386][:2]) / fh
        return (left_brow + right_brow) / 2

    def _mouth_openness(self, lms: np.ndarray, fh: float) -> float:
        """Compute mouth openness (high = surprise)."""
        top = lms[13]  # Upper lip
        bottom = lms[14]  # Lower lip
        vertical = np.linalg.norm(top[:2] - bottom[:2])
        return vertical / fh

    def _smile_ratio(self, lms: np.ndarray) -> float:
        """Compute smile ratio (width/height, high = smile)."""
        left = lms[61]   # Mouth left corner
        right = lms[291]  # Mouth right corner
        top = lms[0]     # Upper lip
        bottom = lms[17]  # Lower lip

        width = np.linalg.norm(left[:2] - right[:2])
        height = np.linalg.norm(top[:2] - bottom[:2])

        return width / height if height > 1e-6 else 3.0

    def _nose_wrinkle(self, lms: np.ndarray, fh: float) -> float:
        """Compute nose wrinkle ratio."""
        tip = lms[1]
        bridge = lms[6]
        return np.linalg.norm(tip[:2] - bridge[:2]) / fh

    def _lip_corner_position(self, lms: np.ndarray, fh: float) -> float:
        """Compute lip corner position relative to center (positive = smile)."""
        left = lms[61]
        right = lms[291]
        center = lms[13]  # Upper lip center

        corner_y = (left[1] + right[1]) / 2
        return (center[1] - corner_y) / fh  # Positive = corners up = smile

    def _score_emotions(self, f: Dict[str, float]) -> Dict[str, float]:
        """
        Score each emotion based on geometric features.

        Returns raw scores (not normalized).
        """
        scores = {}

        # NEUTRAL: moderate features, no extremes
        scores["neutral"] = (
            3.0
            - abs(f["ear"] - 0.28) * 10
            - abs(f["brow_position"] - 0.08) * 10
            - abs(f["mouth_openness"] - 0.02) * 20
        )

        # HAPPY: high smile ratio, lip corners up, moderate eye
        scores["happy"] = (
            f["smile_ratio"] * 2
            + f["lip_corner"] * 15
            + (1 if f["ear"] > 0.22 else 0)
        )

        # SAD: low brow, low mouth corners, slightly closed eyes
        scores["sad"] = (
            (1 if f["brow_position"] < 0.06 else 0) * 2
            + (1 if f["lip_corner"] < -0.01 else 0) * 2
            + (1 if f["ear"] < 0.25 else 0)
        )

        # ANGRY: low brows, tight mouth, narrow eyes
        scores["angry"] = (
            (1 if f["brow_position"] < 0.05 else 0) * 3
            + (1 if f["ear"] < 0.22 else 0) * 2
            + (1 if f["mouth_openness"] < 0.02 else 0)
        )

        # FEAR: wide eyes, high brows, open mouth
        scores["fear"] = (
            (1 if f["ear"] > 0.32 else 0) * 2
            + (1 if f["brow_position"] > 0.10 else 0) * 2
            + (1 if f["mouth_openness"] > 0.04 else 0)
        )

        # SURPRISE: wide eyes, high brows, very open mouth
        scores["surprise"] = (
            (1 if f["ear"] > 0.30 else 0) * 2
            + (1 if f["brow_position"] > 0.09 else 0) * 2
            + f["mouth_openness"] * 20
        )

        # DISGUST: nose wrinkle, lip curl, squinted eyes
        scores["disgust"] = (
            f["nose_wrinkle"] * 10
            + (1 if f["ear"] < 0.24 else 0)
            + (1 if f["lip_corner"] < 0 else 0)
        )

        # Ensure all scores are positive
        min_score = min(scores.values())
        if min_score < 0:
            scores = {k: v - min_score + 0.1 for k, v in scores.items()}

        return scores

    def close(self) -> None:
        """Release resources."""
        self._is_initialized = False
