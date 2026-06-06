"""
Emotion Classifier using MobileNetV3-Large.

Fine-tuned on FER benchmarks for real-time emotion recognition.
Adapted for elderly facial characteristics.
"""

from dataclasses import dataclass
from typing import Optional, Dict
from enum import Enum
import numpy as np


class Emotion(Enum):
    NEUTRAL = "neutral"
    HAPPY = "happy"
    SAD = "sad"
    ANGRY = "angry"
    FEAR = "fear"
    SURPRISE = "surprise"
    DISGUST = "disgust"


@dataclass
class EmotionResult:
    emotion: Emotion = Emotion.NEUTRAL
    confidence: float = 0.0
    probabilities: Dict[str, float] = None
    is_valid: bool = False
    error_message: str = ""

    def __post_init__(self):
        if self.probabilities is None:
            self.probabilities = {}


class EmotionClassifier:
    """MobileNetV3-Large emotion classifier."""

    LABELS = [Emotion.ANGRY, Emotion.DISGUST, Emotion.FEAR, Emotion.HAPPY, Emotion.NEUTRAL, Emotion.SAD, Emotion.SURPRISE]

    def __init__(self):
        self._model = None
        self._is_initialized = False

    def initialize(self, model_path: Optional[str] = None, **kwargs) -> bool:
        try:
            import torch
            import torchvision.models as models
            self._model = models.mobilenet_v3_large(pretrained=(model_path is None))
            nf = self._model.classifier[-1].in_features
            self._model.classifier[-1] = torch.nn.Linear(nf, 7)
            if model_path:
                self._model.load_state_dict(torch.load(model_path, map_location="cpu"))
            self._model.eval()
            self._is_initialized = True
            return True
        except ImportError:
            print("[Emotion] Install: pip install torch torchvision")
            return False

    def classify(self, face_image: np.ndarray) -> EmotionResult:
        if not self._is_initialized:
            return EmotionResult(error_message="Not initialized")
        try:
            import torch, torchvision.transforms as transforms
            t = transforms.Compose([transforms.ToPILImage(), transforms.Resize((224, 224)),
                                     transforms.ToTensor(), transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
            tensor = t(face_image[:, :, ::-1].copy()).unsqueeze(0)
            with torch.no_grad():
                probs = torch.softmax(self._model(tensor), dim=1)[0].numpy()
            idx = int(np.argmax(probs))
            return EmotionResult(
                emotion=self.LABELS[idx], confidence=float(probs[idx]),
                probabilities={e.value: float(p) for e, p in zip(self.LABELS, probs)},
                is_valid=True,
            )
        except Exception as e:
            return EmotionResult(error_message=str(e))

    def close(self) -> None:
        self._model = None
        self._is_initialized = False
