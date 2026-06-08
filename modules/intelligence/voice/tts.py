"""
Edge-TTS for Vietnamese voice synthesis.

Uses Microsoft Edge-TTS for high-quality Vietnamese voice output.
Supports both male and female voices.

Usage:
    tts = TextToSpeech(voice="vi-VN-HoaiMyNeural")
    tts.initialize()
    tts.synthesize("Xin chào! Hãy bắt đầu bài tập.")
"""

import asyncio
import os
import tempfile


class TextToSpeech:
    """Edge-TTS based text-to-speech for Vietnamese.

    Voices:
    - vi-VN-HoaiMyNeural: Female (default)
    - vi-VN-NamMinhNeural: Male
    """

    VOICES = {
        "female": "vi-VN-HoaiMyNeural",
        "male": "vi-VN-NamMinhNeural",
    }

    def __init__(self, voice: str = "vi-VN-HoaiMyNeural"):
        """Initialize TTS.

        Args:
            voice: Voice name (default: Vietnamese female)
        """
        self._voice = voice
        self._ready = False

    def initialize(self) -> bool:
        """Check if edge-tts is available.

        Returns:
            True if edge-tts is installed.
        """
        try:
            import edge_tts
            self._ready = True
            print(f"[TTS] Ready: {self._voice}")
            return True
        except ImportError:
            print("[TTS] Install: pip install edge-tts")
            return False

    def synthesize(self, text: str, output_path: str = None) -> str:
        """Synthesize text to speech.

        Args:
            text: Text to synthesize (Vietnamese)
            output_path: Output file path (default: auto-generated temp file)

        Returns:
            Path to generated audio file, or empty string on failure.
        """
        if not self._ready:
            return ""

        if output_path is None:
            # Use unique temp file
            fd, output_path = tempfile.mkstemp(suffix=".mp3", prefix="tts_")
            os.close(fd)

        try:
            import edge_tts
            asyncio.run(edge_tts.Communicate(text, self._voice).save(output_path))
            return output_path
        except Exception as e:
            print(f"[TTS] Error: {e}")
            return ""

    def close(self):
        """Release resources."""
        self._ready = False
