"""
Whisper-based Speech Recognition for elderly Vietnamese users.

Uses OpenAI Whisper for automatic speech recognition.
Default model: large-v3 (best accuracy for Vietnamese).

Usage:
    recognizer = SpeechRecognizer(model_size="large-v3")
    recognizer.initialize()
    result = recognizer.transcribe(audio_array, language="vi")
    print(result.text)
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class TranscriptionResult:
    """Result of speech transcription."""
    text: str = ""
    language: str = ""
    confidence: float = 0.0
    segments: list = None
    is_valid: bool = False
    error_message: str = ""


class SpeechRecognizer:
    """Whisper-based speech recognition.

    Supports multiple model sizes:
    - tiny: fastest, lowest accuracy
    - base: fast, reasonable accuracy
    - small: balanced
    - medium: good accuracy
    - large-v3: best accuracy (recommended for Vietnamese)
    """

    def __init__(self, model_size: str = "large-v3"):
        """Initialize speech recognizer.

        Args:
            model_size: Whisper model size ("tiny", "base", "small", "medium", "large-v3")
        """
        self._model_size = model_size
        self._model = None
        self._is_initialized = False

    def initialize(self) -> bool:
        """Load Whisper model.

        Returns:
            True if model loaded successfully.
        """
        try:
            import whisper
            print(f"[ASR] Loading Whisper {self._model_size}...")
            self._model = whisper.load_model(self._model_size)
            self._is_initialized = True
            print(f"[ASR] Whisper {self._model_size} loaded")
            return True
        except ImportError:
            print("[ASR] Install: pip install openai-whisper")
            return False
        except Exception as e:
            print(f"[ASR] Load failed: {e}")
            return False

    def transcribe(
        self,
        audio: np.ndarray,
        language: str = "vi",
        sample_rate: int = 16000,
    ) -> TranscriptionResult:
        """Transcribe audio array to text.

        Args:
            audio: Audio samples as numpy array (16kHz mono)
            language: Language code ("vi" for Vietnamese)
            sample_rate: Audio sample rate (default 16kHz)

        Returns:
            TranscriptionResult with text, confidence, etc.
        """
        if not self._is_initialized or self._model is None:
            return TranscriptionResult(error_message="Model not initialized")

        try:
            # Whisper expects float32 audio normalized to [-1, 1]
            if audio.dtype != np.float32:
                audio = audio.astype(np.float32)
            if np.max(np.abs(audio)) > 1.0:
                audio = audio / 32768.0  # Assume int16 input

            # Transcribe
            result = self._model.transcribe(
                audio,
                language=language,
                fp16=False,
                task="transcribe",
            )

            # Extract text and confidence
            text = result["text"].strip()
            detected_lang = result.get("language", language)

            # Compute average confidence from segments
            segments = result.get("segments", [])
            if segments:
                avg_conf = np.mean([s.get("no_speech_prob", 0) for s in segments])
                confidence = 1.0 - avg_conf
            else:
                confidence = 0.5

            return TranscriptionResult(
                text=text,
                language=detected_lang,
                confidence=float(confidence),
                segments=segments,
                is_valid=True,
            )

        except Exception as e:
            return TranscriptionResult(error_message=str(e))

    def close(self):
        """Release model resources."""
        self._model = None
        self._is_initialized = False
        print("[ASR] Model released")
