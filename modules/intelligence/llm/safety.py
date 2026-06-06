"""Safety guardrails for LLM rehabilitation advice."""

from dataclasses import dataclass, field
from typing import List


@dataclass
class SafetyCheck:
    is_safe: bool = True
    warnings: List[str] = field(default_factory=list)
    blocked_reason: str = ""


class SafetyGuardrails:
    """Filters LLM responses to prevent harmful advice."""

    BLOCKED = ["ignore pain", "push through", "take medication", "diagnosis", "you need surgery"]
    ELDERLY_RISKY = ["heavy lifting", "high impact", "jumping", "deep squat"]

    def validate(self, response: str, user_age: int = 70, pain_level: str = "NONE") -> SafetyCheck:
        lower = response.lower()
        for kw in self.BLOCKED:
            if kw in lower:
                return SafetyCheck(is_safe=False, blocked_reason=f"Blocked keyword: {kw}")
        warnings = []
        if user_age >= 60:
            for kw in self.ELDERLY_RISKY:
                if kw in lower:
                    warnings.append(f"Risky for elderly: {kw}")
        if pain_level in ["MODERATE", "SEVERE"] and any(w in lower for w in ["continue", "keep going"]):
            return SafetyCheck(is_safe=False, blocked_reason="Continue during pain")
        return SafetyCheck(is_safe=True, warnings=warnings)

    def filter(self, response: str) -> str:
        lower = response.lower()
        for kw in self.BLOCKED:
            if kw in lower:
                return "Bác ơi, mình nghỉ ngơi một chút nhé!"
        return response
