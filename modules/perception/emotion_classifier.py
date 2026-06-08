"""
Emotion Classification Pipeline.

Multi-backend facial emotion recognition:
    1. Py-Feat (preferred) — pretrained deep learning models
    2. MobileNetV3-Large — custom lightweight model
    3. Geometric rules — always available, zero dependencies

7 basic emotions: angry, disgust, fear, happy, neutral, sad, surprise

Usage:
    classifier = EmotionClassifier()
    result = classifier.classify(face_landmarks)
    if result.is_valid:
        print(f"Emotion: {result.emotion.value} ({result.confidence:.1%})")
"""

import logging
import os
from dataclasses import dataclass
from typing import Dict, Optional
from enum import Enum

import numpy as np

from .face_detector import FaceLandmarkIndex

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Safe deep learning imports
# ---------------------------------------------------------------------------
_PYFEAT_AVAILABLE = False
_TORCH_AVAILABLE = False

try:
    from feat import Detector as FeatDetector
    _PYFEAT_AVAILABLE = True
except ImportError:
    logger.debug("py-feat not available for emotion classification.")

torch = None
nn = None
try:
    import torch
    import torch.nn as _nn
    nn = _nn
    _TORCH_AVAILABLE = True
except ImportError:
    logger.debug("PyTorch not available for emotion classification.")


# ---------------------------------------------------------------------------
# Data structures (backward compatible)
# ---------------------------------------------------------------------------
class Emotion(Enum):
    """Basic emotions."""
    NEUTRAL = "neutral"
    HAPPY = "happy"
    SAD = "sad"
    ANGRY = "angry"
    FEAR = "fear"
    SURPRISE = "surprise"
    DISGUST = "disgust"


EMOTION_LABELS = [
    "angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"
]

EMOTION_TO_ENUM = {e.value: e for e in Emotion}


@dataclass
class EmotionResult:
    """Result of emotion classification.

    Attributes:
        emotion: Predicted emotion.
        confidence: Confidence score (0-1).
        probabilities: Per-emotion probability distribution.
        is_valid: Whether classification succeeded.
        error_message: Error details if failed.
        backend: Detection backend used ('pyfeat', 'mobilenet', 'geometric').
    """
    emotion: Emotion = Emotion.NEUTRAL
    confidence: float = 0.0
    probabilities: Dict[str, float] = None
    is_valid: bool = False
    error_message: str = ""
    backend: str = "geometric"

    def __post_init__(self):
        if self.probabilities is None:
            self.probabilities = {}


# ---------------------------------------------------------------------------
# Py-Feat Backend
# ---------------------------------------------------------------------------
class _PyFeatEmotionBackend:
    """Emotion classification using py-feat library."""

    def __init__(self):
        if not _PYFEAT_AVAILABLE:
            raise ImportError("py-feat is not installed.")

        self._detector = FeatDetector(
            face_model="retinaface",
            landmark_model="mobilenet",
            emotion_model="fer",
        )
        logger.info("PyFeatEmotionBackend initialized.")

    def classify(
        self,
        face_image: Optional[np.ndarray] = None,
        face_landmarks: Optional[np.ndarray] = None,
    ) -> EmotionResult:
        """Classify emotion from face image using py-feat."""
        if face_image is None:
            return EmotionResult(error_message="No face image", backend="pyfeat")

        try:
            import cv2

            if len(face_image.shape) == 3 and face_image.shape[2] == 3:
                rgb_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
            else:
                rgb_image = face_image

            result = self._detector.detect_image(rgb_image)

            if result is None or len(result) == 0:
                return EmotionResult(error_message="No face detected", backend="pyfeat")

            # Extract emotion columns
            emotion_cols = [col for col in result.columns if col.startswith("emotion_")]
            if not emotion_cols:
                # Try alternative column names
                emotion_cols = [col for col in result.columns if col.lower() in EMOTION_LABELS]

            if not emotion_cols:
                return EmotionResult(error_message="No emotion columns found", backend="pyfeat")

            # Build probability dict
            probs = {}
            for col in emotion_cols:
                emotion_name = col.replace("emotion_", "").lower()
                if emotion_name in EMOTION_LABELS:
                    probs[emotion_name] = float(result[col].iloc[0])

            # Normalize
            total = sum(probs.values())
            if total > 0:
                probs = {k: v / total for k, v in probs.items()}
            else:
                probs = {k: 1.0 / len(EMOTION_LABELS) for k in EMOTION_LABELS}

            best = max(probs, key=probs.get)

            return EmotionResult(
                emotion=EMOTION_TO_ENUM.get(best, Emotion.NEUTRAL),
                confidence=probs[best],
                probabilities=probs,
                is_valid=True,
                backend="pyfeat",
            )

        except Exception as e:
            logger.warning(f"Py-Feat emotion detection failed: {e}")
            return EmotionResult(error_message=str(e), backend="pyfeat")


# ---------------------------------------------------------------------------
# MobileNetV3-Large Backend
# ---------------------------------------------------------------------------
if _TORCH_AVAILABLE:

    class _MobileNetV3Emotion(nn.Module):
        """
        MobileNetV3-Large for emotion classification.

        Input:  (B, 3, 224, 224) — cropped & resized face
        Output: (B, 7) — softmax probabilities for 7 emotions
        """

        def __init__(self, pretrained_backbone: bool = True):
            super().__init__()

            try:
                from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights
                weights = MobileNet_V3_Large_Weights.DEFAULT if pretrained_backbone else None
                backbone = mobilenet_v3_large(weights=weights)
            except ImportError:
                from torchvision.models import mobilenet_v3_large
                backbone = mobilenet_v3_large(pretrained=pretrained_backbone)

            self.features = backbone.features
            self.pool = nn.AdaptiveAvgPool2d(1)
            feat_dim = 960

            self.classifier = nn.Sequential(
                nn.Linear(feat_dim, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(inplace=True),
                nn.Dropout(0.4),
                nn.Linear(512, 128),
                nn.ReLU(inplace=True),
                nn.Dropout(0.2),
                nn.Linear(128, 7),
            )

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            """Returns (B, 7) raw logits. Apply softmax externally."""
            feat = self.features(x)
            feat = self.pool(feat).flatten(1)
            return self.classifier(feat)


class _MobileNetEmotionBackend:
    """MobileNetV3-based emotion classification backend."""

    def __init__(self, model_path: Optional[str] = None, device: str = "auto"):
        if not _TORCH_AVAILABLE:
            raise ImportError("PyTorch required.")

        self._device = self._resolve_device(device)
        self._model = None
        self._initialized = False

        if model_path is None:
            model_path = os.path.join(
                os.path.dirname(__file__),
                "..", "..", "models", "emotion_mobilenetv3.pth",
            )

        self._load_model(model_path)

    def _resolve_device(self, device: str) -> str:
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device

    def _load_model(self, model_path: str) -> bool:
        try:
            model_path = os.path.normpath(model_path)
            if not os.path.exists(model_path):
                logger.info(f"Emotion MobileNetV3 weights not found: {model_path}")
                return False

            model = _MobileNetV3Emotion(pretrained_backbone=False)
            state_dict = torch.load(model_path, map_location=self._device)
            model.load_state_dict(state_dict)
            model.to(self._device)
            model.eval()

            self._model = model
            self._initialized = True
            logger.info(f"Emotion MobileNetV3 loaded from {model_path}")
            return True

        except Exception as e:
            logger.warning(f"Failed to load emotion MobileNetV3: {e}")
            return False

    @property
    def is_ready(self) -> bool:
        return self._initialized and self._model is not None

    def classify(
        self,
        face_image: Optional[np.ndarray] = None,
        face_landmarks: Optional[np.ndarray] = None,
    ) -> EmotionResult:
        """Classify emotion using MobileNetV3."""
        if not self.is_ready:
            return EmotionResult(error_message="Model not loaded", backend="mobilenet")

        if face_image is None:
            return EmotionResult(error_message="No face image", backend="mobilenet")

        try:
            face_crop = self._crop_face(face_image, face_landmarks)
            if face_crop is None:
                return EmotionResult(error_message="Face crop failed", backend="mobilenet")

            tensor = self._preprocess(face_crop)

            with torch.no_grad():
                logits = self._model(tensor)
                probs = torch.softmax(logits, dim=1).cpu().numpy().flatten()

            probs_dict = {label: float(probs[i]) for i, label in enumerate(EMOTION_LABELS)}
            best_idx = int(np.argmax(probs))

            return EmotionResult(
                emotion=EMOTION_TO_ENUM.get(EMOTION_LABELS[best_idx], Emotion.NEUTRAL),
                confidence=float(probs[best_idx]),
                probabilities=probs_dict,
                is_valid=True,
                backend="mobilenet",
            )

        except Exception as e:
            logger.warning(f"MobileNetV3 emotion detection failed: {e}")
            return EmotionResult(error_message=str(e), backend="mobilenet")

    def _crop_face(
        self, image: np.ndarray, landmarks: Optional[np.ndarray]
    ) -> Optional[np.ndarray]:
        """Crop face region from image."""
        h, w = image.shape[:2]

        if landmarks is not None and len(landmarks) >= 468:
            pts = np.asarray(landmarks, dtype=np.float32)
            if pts.ndim == 2 and pts.shape[1] >= 2:
                x_min, y_min = pts[:, :2].min(axis=0)
                x_max, y_max = pts[:, :2].max(axis=0)

                pad_x = (x_max - x_min) * 0.2
                pad_y = (y_max - y_min) * 0.2
                x_min = max(0, int(x_min - pad_x))
                y_min = max(0, int(y_min - pad_y))
                x_max = min(w, int(x_max + pad_x))
                y_max = min(h, int(y_max + pad_y))

                if x_max > x_min and y_max > y_min:
                    return image[y_min:y_max, x_min:x_max].copy()

        crop_size = min(h, w) // 2
        cx, cy = w // 2, h // 2
        return image[
            max(0, cy - crop_size):min(h, cy + crop_size),
            max(0, cx - crop_size):min(w, cx + crop_size),
        ].copy()

    def _preprocess(self, face_crop: np.ndarray) -> "torch.Tensor":
        """Preprocess face crop for MobileNetV3."""
        try:
            import cv2
        except ImportError:
            from PIL import Image
            pil = Image.fromarray(face_crop[:, :, ::-1] if face_crop.shape[2] == 3 else face_crop)
            pil = pil.resize((224, 224))
            arr = np.array(pil).astype(np.float32) / 255.0
            tensor = torch.from_numpy(arr).permute(2, 0, 1)
        else:
            resized = cv2.resize(face_crop, (224, 224), interpolation=cv2.INTER_LINEAR)
            if len(resized.shape) == 3 and resized.shape[2] == 3:
                resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            tensor = torch.from_numpy(resized).float() / 255.0
            tensor = tensor.permute(2, 0, 1)

        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        tensor = (tensor - mean) / std

        return tensor.unsqueeze(0).to(self._device)


# ---------------------------------------------------------------------------
# Geometric Backend (Fallback — always available)
# ---------------------------------------------------------------------------
class _GeometricEmotionBackend:
    """
    Rule-based emotion classification from Face Mesh landmarks.

    Uses geometric features (EAR, brow position, mouth shape) to score
    each emotion. Always available — no external dependencies.
    """

    def classify(
        self,
        face_image: Optional[np.ndarray] = None,
        face_landmarks: Optional[np.ndarray] = None,
    ) -> EmotionResult:
        """Classify emotion from face landmarks using geometric rules."""
        if face_landmarks is None or len(face_landmarks) < 468:
            return EmotionResult(error_message="Invalid landmarks", backend="geometric")

        try:
            lms = np.asarray(face_landmarks, dtype=np.float32)
            if lms.ndim == 1:
                lms = lms.reshape(-1, 3)

            features = self._extract_features(lms)
            scores = self._score_emotions(features)

            # Normalize to probabilities
            total = sum(scores.values())
            if total > 0:
                probs = {k: v / total for k, v in scores.items()}
            else:
                probs = {k: 1.0 / len(scores) for k in scores}

            best = max(probs, key=probs.get)

            return EmotionResult(
                emotion=EMOTION_TO_ENUM.get(best, Emotion.NEUTRAL),
                confidence=probs[best],
                probabilities=probs,
                is_valid=True,
                backend="geometric",
            )

        except Exception as e:
            logger.warning(f"Geometric emotion classification failed: {e}")
            return EmotionResult(error_message=str(e), backend="geometric")

    def _extract_features(self, lms: np.ndarray) -> Dict[str, float]:
        """Extract geometric features from landmarks."""
        features = {}

        face_top = lms[FaceLandmarkIndex.FACE_TOP]
        face_bottom = lms[FaceLandmarkIndex.FACE_BOTTOM]
        face_height = np.linalg.norm(face_top[:2] - face_bottom[:2])
        if face_height < 1e-6:
            face_height = 1.0

        features["ear"] = self._ear(lms)
        features["brow_position"] = self._brow_position(lms, face_height)
        features["mouth_openness"] = self._mouth_openness(lms, face_height)
        features["smile_ratio"] = self._smile_ratio(lms)
        features["nose_wrinkle"] = self._nose_wrinkle(lms, face_height)
        features["lip_corner"] = self._lip_corner_position(lms, face_height)

        return features

    def _ear(self, lms: np.ndarray) -> float:
        """Average Eye Aspect Ratio."""
        def _ear_side(side):
            if side == "left":
                top, bottom = lms[159], lms[145]
                inner, outer = lms[133], lms[33]
            else:
                top, bottom = lms[386], lms[374]
                inner, outer = lms[362], lms[263]
            v = np.linalg.norm(top[:2] - bottom[:2])
            h = np.linalg.norm(inner[:2] - outer[:2])
            return v / h if h > 1e-6 else 0.3
        return (_ear_side("left") + _ear_side("right")) / 2

    def _brow_position(self, lms: np.ndarray, fh: float) -> float:
        left = np.linalg.norm(lms[66][:2] - lms[159][:2]) / fh
        right = np.linalg.norm(lms[296][:2] - lms[386][:2]) / fh
        return (left + right) / 2

    def _mouth_openness(self, lms: np.ndarray, fh: float) -> float:
        top = lms[13]
        bottom = lms[14]
        return np.linalg.norm(top[:2] - bottom[:2]) / fh

    def _smile_ratio(self, lms: np.ndarray) -> float:
        left = lms[61]
        right = lms[291]
        top = lms[0]
        bottom = lms[17]
        width = np.linalg.norm(left[:2] - right[:2])
        height = np.linalg.norm(top[:2] - bottom[:2])
        return width / height if height > 1e-6 else 3.0

    def _nose_wrinkle(self, lms: np.ndarray, fh: float) -> float:
        tip = lms[1]
        bridge = lms[6]
        return np.linalg.norm(tip[:2] - bridge[:2]) / fh

    def _lip_corner_position(self, lms: np.ndarray, fh: float) -> float:
        left = lms[61]
        right = lms[291]
        center = lms[13]
        corner_y = (left[1] + right[1]) / 2
        return (center[1] - corner_y) / fh

    def _score_emotions(self, f: Dict[str, float]) -> Dict[str, float]:
        """Score each emotion based on geometric features."""
        scores = {}

        scores["neutral"] = (
            3.0
            - abs(f["ear"] - 0.28) * 10
            - abs(f["brow_position"] - 0.08) * 10
            - abs(f["mouth_openness"] - 0.02) * 20
        )

        scores["happy"] = (
            f["smile_ratio"] * 2
            + f["lip_corner"] * 15
            + (1 if f["ear"] > 0.22 else 0)
        )

        scores["sad"] = (
            (1 if f["brow_position"] < 0.06 else 0) * 2
            + (1 if f["lip_corner"] < -0.01 else 0) * 2
            + (1 if f["ear"] < 0.25 else 0)
        )

        scores["angry"] = (
            (1 if f["brow_position"] < 0.05 else 0) * 3
            + (1 if f["ear"] < 0.22 else 0) * 2
            + (1 if f["mouth_openness"] < 0.02 else 0)
        )

        scores["fear"] = (
            (1 if f["ear"] > 0.32 else 0) * 2
            + (1 if f["brow_position"] > 0.10 else 0) * 2
            + (1 if f["mouth_openness"] > 0.04 else 0)
        )

        scores["surprise"] = (
            (1 if f["ear"] > 0.30 else 0) * 2
            + (1 if f["brow_position"] > 0.09 else 0) * 2
            + f["mouth_openness"] * 20
        )

        scores["disgust"] = (
            f["nose_wrinkle"] * 10
            + (1 if f["ear"] < 0.24 else 0)
            + (1 if f["lip_corner"] < 0 else 0)
        )

        # Ensure positive
        min_score = min(scores.values())
        if min_score < 0:
            scores = {k: v - min_score + 0.1 for k, v in scores.items()}

        return scores


# ---------------------------------------------------------------------------
# Main Pipeline (backward compatible class name)
# ---------------------------------------------------------------------------
class EmotionClassifier:
    """
    Multi-backend emotion classification pipeline with automatic fallback.

    Backend priority:
        1. Py-Feat (deep learning, best accuracy)
        2. MobileNetV3-Large (lightweight deep learning)
        3. Geometric rules (always available)

    Args:
        backend: Force 'pyfeat', 'mobilenet', or 'geometric'. None = auto.

    Example:
        >>> classifier = EmotionClassifier()
        >>> result = classifier.classify(face_landmarks)
        >>> print(f"Emotion: {result.emotion.value} ({result.backend})")
    """

    def __init__(self, backend: Optional[str] = None):
        self._active_backend = "none"
        self._pyfeat: Optional[_PyFeatEmotionBackend] = None
        self._mobilenet: Optional[_MobileNetEmotionBackend] = None
        self._geometric = _GeometricEmotionBackend()

        if backend == "geometric":
            self._active_backend = "geometric"
            return

        # Try Py-Feat
        if backend in (None, "pyfeat") and _PYFEAT_AVAILABLE:
            try:
                self._pyfeat = _PyFeatEmotionBackend()
                self._active_backend = "pyfeat"
                logger.info("EmotionClassifier: Using Py-Feat backend.")
                return
            except Exception as e:
                logger.info(f"Py-Feat emotion init failed: {e}")

        # Try MobileNetV3
        if backend in (None, "mobilenet") and _TORCH_AVAILABLE:
            try:
                self._mobilenet = _MobileNetEmotionBackend()
                if self._mobilenet.is_ready:
                    self._active_backend = "mobilenet"
                    logger.info("EmotionClassifier: Using MobileNetV3 backend.")
                    return
            except Exception as e:
                logger.info(f"MobileNetV3 emotion init failed: {e}")

        # Geometric fallback
        self._active_backend = "geometric"
        logger.info("EmotionClassifier: Using geometric fallback.")

    @property
    def backend(self) -> str:
        """Currently active backend."""
        return self._active_backend

    def initialize(self, **kwargs) -> bool:
        """Initialize classifier (compatibility method)."""
        return True

    def classify(
        self,
        landmarks: np.ndarray,
        face_image: Optional[np.ndarray] = None,
    ) -> EmotionResult:
        """
        Classify emotion from face landmarks (and optionally image).

        Args:
            landmarks: Face landmarks array, shape (468, 3).
            face_image: BGR image (H, W, 3). Required for deep learning backends.

        Returns:
            EmotionResult with predicted emotion and confidence.
        """
        result = None

        # Try Py-Feat
        if self._active_backend == "pyfeat" and self._pyfeat is not None:
            result = self._pyfeat.classify(face_image, landmarks)
            if result.is_valid:
                return result
            logger.info("Py-Feat emotion failed, trying next backend...")

        # Try MobileNetV3
        if self._active_backend in ("pyfeat", "mobilenet") and self._mobilenet is not None:
            result = self._mobilenet.classify(face_image, landmarks)
            if result.is_valid:
                return result
            logger.info("MobileNetV3 emotion failed, trying next backend...")

        # Geometric fallback
        return self._geometric.classify(face_image, landmarks)

    def close(self) -> None:
        """Release resources."""
        self._pyfeat = None
        self._mobilenet = None
