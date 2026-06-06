"""
Intelligence Modules.

LLM-based coaching, voice interaction, and safety guardrails.

Version: 3.0.0
"""

from .llm.client import LLMClient, LLMResponse
from .llm.prompts import PromptTemplates
from .llm.rag import RAGPipeline
from .llm.safety import SafetyGuardrails
from .voice.asr import SpeechRecognizer
from .voice.tts import TextToSpeech
from .coach.rehab_coach import RehabCoach

__all__ = [
    "LLMClient", "LLMResponse", "PromptTemplates", "RAGPipeline",
    "SafetyGuardrails", "SpeechRecognizer", "TextToSpeech", "RehabCoach",
]
