"""
LLM API Client.

Supports multiple LLM providers:
- Gemini (Google)
- OpenAI (GPT-4o)
- MiMo (Xiaomi)

Usage:
    # Gemini
    client = LLMClient(provider="gemini", api_key="AQ.Ab8R...")
    client.initialize()
    response = client.chat("What exercises for shoulder pain?")

    # OpenAI
    client = LLMClient(provider="openai", api_key="sk-...")

    # MiMo
    client = LLMClient(provider="mimo", api_key="...")
"""

from dataclasses import dataclass
from typing import Optional, List, Dict
from enum import Enum
import time


class LLMProvider(Enum):
    GEMINI = "gemini"
    OPENAI = "openai"
    MIMO = "mimo"


@dataclass
class LLMResponse:
    """Response from LLM API."""
    content: str = ""
    model: str = ""
    tokens_used: int = 0
    latency_ms: float = 0.0
    is_valid: bool = False
    error_message: str = ""


class LLMClient:
    """Unified LLM API client.

    Supports:
    - Google Gemini (gemini-2.0-flash, gemini-1.5-pro, etc.)
    - OpenAI (GPT-4o, GPT-4, etc.)
    - MiMo (Xiaomi MiMo, OpenAI-compatible)
    """

    def __init__(self, provider: str = "gemini", api_key: Optional[str] = None, model: Optional[str] = None):
        """Initialize LLM client.

        Args:
            provider: LLM provider name ("gemini", "openai", "mimo")
            api_key: API key
            model: Model name (default depends on provider)
        """
        self._provider = LLMProvider(provider.lower())
        self._api_key = api_key

        # Default models per provider
        defaults = {
            LLMProvider.GEMINI: "gemini-2.0-flash",
            LLMProvider.OPENAI: "gpt-4o",
            LLMProvider.MIMO: "mimo-v2.5-pro",
        }
        self._model = model or defaults.get(self._provider, "gemini-2.0-flash")
        self._client = None
        self._is_initialized = False

    def initialize(self, **kwargs) -> bool:
        """Initialize the LLM client.

        Returns:
            True if initialization successful.
        """
        try:
            if self._provider == LLMProvider.GEMINI:
                # Gemini uses REST API directly, no SDK needed
                # Just verify API key format
                if not self._api_key or len(self._api_key) < 10:
                    print("[LLM] Invalid Gemini API key")
                    return False
                self._client = "gemini_rest"  # Marker for REST API

            elif self._provider in (LLMProvider.OPENAI, LLMProvider.MIMO):
                from openai import OpenAI

                if self._provider == LLMProvider.MIMO:
                    self._client = OpenAI(
                        api_key=self._api_key,
                        base_url="https://api.xiaomimimo.com/v1",
                    )
                else:
                    self._client = OpenAI(api_key=self._api_key)

            self._is_initialized = True
            print(f"[LLM] Initialized: {self._provider.value} / {self._model}")
            return True

        except ImportError as e:
            print(f"[LLM] Missing package: {e}")
            if self._provider == LLMProvider.GEMINI:
                print("[LLM] Install: pip install google-generativeai")
            else:
                print("[LLM] Install: pip install openai")
            return False
        except Exception as e:
            print(f"[LLM] Init failed: {e}")
            return False

    def chat(
        self,
        message: str,
        system_prompt: Optional[str] = None,
        history: Optional[List[Dict]] = None,
        temperature: float = 0.7,
        max_tokens: int = 500,
    ) -> LLMResponse:
        """Send a chat message to the LLM.

        Args:
            message: User message
            system_prompt: System prompt (optional)
            history: Previous conversation history (optional)
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response

        Returns:
            LLMResponse with content, model, tokens_used, etc.
        """
        if not self._is_initialized:
            return LLMResponse(error_message="Not initialized")

        try:
            start = time.time()

            if self._provider == LLMProvider.GEMINI:
                resp = self._chat_gemini(message, system_prompt, history, temperature, max_tokens)
            elif self._provider in (LLMProvider.OPENAI, LLMProvider.MIMO):
                resp = self._chat_openai(message, system_prompt, history, temperature, max_tokens)
            else:
                return LLMResponse(error_message=f"Unknown provider: {self._provider}")

            resp.latency_ms = (time.time() - start) * 1000
            return resp

        except Exception as e:
            return LLMResponse(error_message=str(e))

    def _chat_gemini(self, msg, sys, hist, temp, max_t) -> LLMResponse:
        """Chat using Google Gemini REST API (no SDK dependency)."""
        import json
        import urllib.request

        # Build contents array
        contents = []

        # Add system instruction if provided
        system_instruction = None
        if sys:
            system_instruction = {"parts": [{"text": sys}]}

        # Add history
        if hist:
            for h in hist:
                role = h.get("role", "user")
                content = h.get("content", "")
                contents.append({
                    "role": "user" if role == "user" else "model",
                    "parts": [{"text": content}]
                })

        # Add current message
        contents.append({"role": "user", "parts": [{"text": msg}]})

        # Build request body
        body = {
            "contents": contents,
            "generationConfig": {
                "temperature": temp,
                "maxOutputTokens": max_t,
            }
        }
        if system_instruction:
            body["systemInstruction"] = system_instruction

        # Make REST API call
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self._model}:generateContent?key={self._api_key}"

        req = urllib.request.Request(
            url,
            data=json.dumps(body).encode('utf-8'),
            headers={"Content-Type": "application/json"},
        )

        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                result = json.loads(resp.read().decode('utf-8'))

            # Extract response text
            content = ""
            if "candidates" in result and len(result["candidates"]) > 0:
                candidate = result["candidates"][0]
                if "content" in candidate and "parts" in candidate["content"]:
                    parts = candidate["content"]["parts"]
                    content = "".join(p.get("text", "") for p in parts)

            # Extract token usage
            tokens_used = 0
            if "usageMetadata" in result:
                usage = result["usageMetadata"]
                tokens_used = usage.get("totalTokenCount", 0)

            return LLMResponse(
                content=content,
                model=self._model,
                tokens_used=tokens_used,
                is_valid=True,
            )

        except Exception as e:
            return LLMResponse(error_message=str(e))

    def _chat_openai(self, msg, sys, hist, temp, max_t) -> LLMResponse:
        """Chat using OpenAI-compatible API (OpenAI, MiMo, etc.)."""
        msgs = []
        if sys:
            msgs.append({"role": "system", "content": sys})
        if hist:
            msgs.extend(hist)
        msgs.append({"role": "user", "content": msg})

        r = self._client.chat.completions.create(
            model=self._model,
            messages=msgs,
            temperature=temp,
            max_tokens=max_t,
        )

        return LLMResponse(
            content=r.choices[0].message.content,
            model=r.model,
            tokens_used=r.usage.total_tokens if r.usage else 0,
            is_valid=True,
        )

    def close(self):
        """Release resources."""
        self._client = None
        self._is_initialized = False
