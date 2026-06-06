"""
LLM API Client.

Uses GPT-4o or Claude API (not self-hosted).
Standard practice for research papers.

Usage:
    client = LLMClient(provider="openai", api_key="sk-...")
    client.initialize()
    response = client.chat("What exercises for shoulder pain?")
"""

from dataclasses import dataclass
from typing import Optional, List, Dict
from enum import Enum
import time


class LLMProvider(Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"


@dataclass
class LLMResponse:
    content: str = ""
    model: str = ""
    tokens_used: int = 0
    latency_ms: float = 0.0
    is_valid: bool = False
    error_message: str = ""


class LLMClient:
    """Unified LLM API client for GPT-4o and Claude."""

    def __init__(self, provider: str = "openai", api_key: Optional[str] = None, model: Optional[str] = None):
        self._provider = LLMProvider(provider.lower())
        self._api_key = api_key
        self._model = model or ("gpt-4o" if self._provider == LLMProvider.OPENAI else "claude-sonnet-4-20250514")
        self._client = None
        self._is_initialized = False

    def initialize(self, **kwargs) -> bool:
        try:
            if self._provider == LLMProvider.OPENAI:
                from openai import OpenAI
                self._client = OpenAI(api_key=self._api_key)
            else:
                from anthropic import Anthropic
                self._client = Anthropic(api_key=self._api_key)
            self._is_initialized = True
            return True
        except ImportError:
            pkg = "openai" if self._provider == LLMProvider.OPENAI else "anthropic"
            print(f"[LLM] Install: pip install {pkg}")
            return False

    def chat(self, message: str, system_prompt: Optional[str] = None,
             history: Optional[List[Dict]] = None, temperature: float = 0.7, max_tokens: int = 500) -> LLMResponse:
        if not self._is_initialized:
            return LLMResponse(error_message="Not initialized")
        try:
            start = time.time()
            if self._provider == LLMProvider.OPENAI:
                resp = self._chat_openai(message, system_prompt, history, temperature, max_tokens)
            else:
                resp = self._chat_anthropic(message, system_prompt, history, temperature, max_tokens)
            resp.latency_ms = (time.time() - start) * 1000
            return resp
        except Exception as e:
            return LLMResponse(error_message=str(e))

    def _chat_openai(self, msg, sys, hist, temp, max_t) -> LLMResponse:
        msgs = []
        if sys: msgs.append({"role": "system", "content": sys})
        if hist: msgs.extend(hist)
        msgs.append({"role": "user", "content": msg})
        r = self._client.chat.completions.create(model=self._model, messages=msgs, temperature=temp, max_tokens=max_t)
        return LLMResponse(content=r.choices[0].message.content, model=r.model, tokens_used=r.usage.total_tokens, is_valid=True)

    def _chat_anthropic(self, msg, sys, hist, temp, max_t) -> LLMResponse:
        kw = {"model": self._model, "max_tokens": max_t, "temperature": temp, "messages": []}
        if sys: kw["system"] = sys
        if hist: kw["messages"].extend(hist)
        kw["messages"].append({"role": "user", "content": msg})
        r = self._client.messages.create(**kw)
        return LLMResponse(content=r.content[0].text, model=r.model, tokens_used=r.usage.input_tokens + r.usage.output_tokens, is_valid=True)

    def close(self):
        self._client = None
        self._is_initialized = False
