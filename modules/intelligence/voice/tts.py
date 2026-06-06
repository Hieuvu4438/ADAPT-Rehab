"""Edge-TTS for Vietnamese voice synthesis."""

import asyncio


class TextToSpeech:
    VOICES = {"female": "vi-VN-HoaiMyNeural", "male": "vi-VN-NamMinhNeural"}

    def __init__(self, voice: str = "vi-VN-HoaiMyNeural"):
        self._voice = voice
        self._ready = False

    def initialize(self) -> bool:
        try:
            import edge_tts
            self._ready = True
            return True
        except ImportError:
            print("[TTS] Install: pip install edge-tts")
            return False

    def synthesize(self, text: str, output_path: str = "/tmp/tts.mp3") -> str:
        if not self._ready:
            return ""
        try:
            import edge_tts
            asyncio.run(edge_tts.Communicate(text, self._voice).save(output_path))
            return output_path
        except Exception:
            return ""

    def close(self):
        self._ready = False
