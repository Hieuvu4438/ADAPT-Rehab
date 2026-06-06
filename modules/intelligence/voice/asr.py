"""Whisper-based speech recognition for elderly Vietnamese users."""

from dataclasses import dataclass
import numpy as np


@dataclass
class TranscriptionResult:
    text: str = ""
    language: str = ""
    confidence: float = 0.0
    is_valid: bool = False


class SpeechRecognizer:
    def __init__(self, model_size: str = "base"):
        self._model_size = model_size
        self._model = None

    def initialize(self) -> bool:
        try:
            import whisper
            self._model = whisper.load_model(self._model_size)
            return True
        except ImportError:
            print("[ASR] Install: pip install openai-whisper")
            return False

    def transcribe(self, audio: np.ndarray, language: str = "vi") -> TranscriptionResult:
        if not self._model:
            return TranscriptionResult()
        try:
            r = self._model.transcribe(audio, language=language, fp16=False)
            return TranscriptionResult(text=r["text"].strip(), language=r.get("language", language), confidence=0.8, is_valid=True)
        except Exception:
            return TranscriptionResult()

    def close(self):
        self._model = None
