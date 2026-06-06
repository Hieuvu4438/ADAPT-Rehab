"""
Multi-Indicator Fatigue Analysis.

Detects fatigue using multiple kinematic indicators:
- Jerk ratio: current jerk vs baseline (primary indicator)
- ROM degradation: decrease in range of motion across reps
- Velocity decline: slower movement speed
- Movement variability: increased jitter/unsteadiness

Multi-indicator approach is more robust than single-metric (jerk only).

Usage:
    analyzer = FatigueAnalyzer()
    analyzer.set_baseline(first_rep_data)
    for rep_data in reps:
        result = analyzer.analyze(rep_data)
        if result.level.value >= FatigueLevel.MODERATE.value:
            print("Warning: fatigue detected")
"""

from dataclasses import dataclass
from typing import Dict, Optional, List
from enum import Enum
import numpy as np


class FatigueLevel(Enum):
    """Fatigue severity levels."""
    FRESH = 0       # No fatigue
    LIGHT = 1       # Slight fatigue, can continue
    MODERATE = 2    # Noticeable fatigue, consider rest
    HEAVY = 3       # Significant fatigue, should rest


@dataclass
class FatigueResult:
    """Result of fatigue analysis."""
    level: FatigueLevel = FatigueLevel.FRESH
    jerk_ratio: float = 1.0          # Current jerk / baseline jerk
    rom_degradation: float = 0.0     # ROM decrease percentage
    velocity_decline: float = 0.0    # Velocity decrease percentage
    variability_increase: float = 0.0  # Variability increase percentage
    composite_score: float = 0.0     # 0 = fresh, 100 = exhausted
    recommendation: str = ""         # Vietnamese recommendation
    is_valid: bool = False


class FatigueAnalyzer:
    """
    Multi-indicator fatigue analyzer.

    Combines multiple kinematic indicators for robust fatigue detection.
    Each indicator captures a different aspect of fatigue:
    - Jerk: movement quality degradation
    - ROM: strength/endurance decline
    - Velocity: speed decline
    - Variability: control degradation

    Example:
        >>> analyzer = FatigueAnalyzer()
        >>> # Set baseline from first rep
        >>> analyzer.set_baseline({
        ...     "jerk_value": 150.0,
        ...     "max_angle": 90.0,
        ...     "mean_velocity": 45.0,
        ...     "angle_std": 3.0
        ... })
        >>> # Analyze subsequent reps
        >>> for rep in reps:
        ...     result = analyzer.analyze(rep)
        ...     print(f"Fatigue: {result.level.name} ({result.composite_score:.0f})")
    """

    # Jerk ratio thresholds
    JERK_THRESHOLDS = {
        FatigueLevel.LIGHT: 1.5,     # 50% increase
        FatigueLevel.MODERATE: 2.0,  # 100% increase
        FatigueLevel.HEAVY: 3.0,     # 200% increase
    }

    # Vietnamese recommendations
    RECOMMENDATIONS = {
        FatigueLevel.FRESH: "Bác tập tốt lắm! Tiếp tục nhé!",
        FatigueLevel.LIGHT: "Bác hơi mệt rồi. Cố thêm vài rep nữa nhé!",
        FatigueLevel.MODERATE: "Bác nên nghỉ ngơi một chút rồi tiếp tục.",
        FatigueLevel.HEAVY: "Bác đã mệt nhiều rồi. Mình nghỉ ngơi nhé!",
    }

    def __init__(self):
        """Initialize fatigue analyzer."""
        self._baseline_jerk: Optional[float] = None
        self._baseline_rom: Optional[float] = None
        self._baseline_velocity: Optional[float] = None
        self._baseline_variability: Optional[float] = None
        self._rep_history: List[Dict] = []

    def set_baseline(self, rep_data: Dict) -> None:
        """
        Set baseline from first rep (or average of first few reps).

        Args:
            rep_data: Dictionary with keys:
                - jerk_value: Squared jerk value
                - max_angle: Maximum angle achieved (degrees)
                - mean_velocity: Mean angular velocity (degrees/sec)
                - angle_std: Standard deviation of angles
        """
        self._baseline_jerk = rep_data.get("jerk_value", 0.0)
        self._baseline_rom = rep_data.get("max_angle", 0.0)
        self._baseline_velocity = rep_data.get("mean_velocity", 0.0)
        self._baseline_variability = rep_data.get("angle_std", 0.0)
        self._rep_history = [rep_data]

    def analyze(self, rep_data: Dict) -> FatigueResult:
        """
        Analyze fatigue for a single rep.

        Args:
            rep_data: Dictionary with same keys as set_baseline().

        Returns:
            FatigueResult with fatigue indicators and level.
        """
        self._rep_history.append(rep_data)

        # Compute indicators
        jerk_ratio = self._compute_jerk_ratio(rep_data)
        rom_degradation = self._compute_rom_degradation(rep_data)
        velocity_decline = self._compute_velocity_decline(rep_data)
        variability_increase = self._compute_variability_increase(rep_data)

        # Classify fatigue level (jerk is primary indicator)
        level = self._classify_from_jerk(jerk_ratio)

        # Upgrade level if other indicators are severe
        if level == FatigueLevel.FRESH:
            if rom_degradation > 30 or velocity_decline > 40:
                level = FatigueLevel.MODERATE
            elif rom_degradation > 15 or velocity_decline > 20:
                level = FatigueLevel.LIGHT

        # Composite score (weighted combination)
        composite = (
            0.40 * min(100, max(0, (jerk_ratio - 1) * 50)) +
            0.30 * min(100, rom_degradation) +
            0.20 * min(100, velocity_decline) +
            0.10 * min(100, variability_increase)
        )

        return FatigueResult(
            level=level,
            jerk_ratio=jerk_ratio,
            rom_degradation=rom_degradation,
            velocity_decline=velocity_decline,
            variability_increase=variability_increase,
            composite_score=max(0, min(100, composite)),
            recommendation=self.RECOMMENDATIONS[level],
            is_valid=True,
        )

    def _compute_jerk_ratio(self, rep_data: Dict) -> float:
        """Compute jerk ratio: current / baseline."""
        if self._baseline_jerk is None or self._baseline_jerk < 1e-6:
            return 1.0
        return rep_data.get("jerk_value", 0.0) / self._baseline_jerk

    def _compute_rom_degradation(self, rep_data: Dict) -> float:
        """Compute ROM degradation percentage."""
        if self._baseline_rom is None or self._baseline_rom < 1e-6:
            return 0.0
        current = rep_data.get("max_angle", 0.0)
        degradation = (self._baseline_rom - current) / self._baseline_rom * 100
        return max(0, degradation)

    def _compute_velocity_decline(self, rep_data: Dict) -> float:
        """Compute velocity decline percentage."""
        if self._baseline_velocity is None or self._baseline_velocity < 1e-6:
            return 0.0
        current = rep_data.get("mean_velocity", 0.0)
        decline = (self._baseline_velocity - current) / self._baseline_velocity * 100
        return max(0, decline)

    def _compute_variability_increase(self, rep_data: Dict) -> float:
        """Compute movement variability increase percentage."""
        if self._baseline_variability is None or self._baseline_variability < 1e-6:
            return 0.0
        current = rep_data.get("angle_std", 0.0)
        increase = (current - self._baseline_variability) / self._baseline_variability * 100
        return max(0, increase)

    def _classify_from_jerk(self, jerk_ratio: float) -> FatigueLevel:
        """Classify fatigue level from jerk ratio."""
        if jerk_ratio >= self.JERK_THRESHOLDS[FatigueLevel.HEAVY]:
            return FatigueLevel.HEAVY
        elif jerk_ratio >= self.JERK_THRESHOLDS[FatigueLevel.MODERATE]:
            return FatigueLevel.MODERATE
        elif jerk_ratio >= self.JERK_THRESHOLDS[FatigueLevel.LIGHT]:
            return FatigueLevel.LIGHT
        return FatigueLevel.FRESH

    def get_trend(self) -> Dict:
        """
        Analyze fatigue trend across all reps.

        Returns:
            Dictionary with trend information.
        """
        if len(self._rep_history) < 2:
            return {"trend": "stable", "reps_analyzed": len(self._rep_history)}

        # Compare first half vs second half
        mid = len(self._rep_history) // 2
        first_half_jerk = np.mean([r.get("jerk_value", 0) for r in self._rep_history[:mid]])
        second_half_jerk = np.mean([r.get("jerk_value", 0) for r in self._rep_history[mid:]])

        if first_half_jerk < 1e-6:
            return {"trend": "stable", "reps_analyzed": len(self._rep_history)}

        increase_pct = ((second_half_jerk - first_half_jerk) / first_half_jerk) * 100

        if increase_pct > 100:
            trend = "increasing_fast"
        elif increase_pct > 30:
            trend = "increasing"
        elif increase_pct < -20:
            trend = "improving"
        else:
            trend = "stable"

        return {
            "trend": trend,
            "jerk_increase_percent": round(increase_pct, 1),
            "reps_analyzed": len(self._rep_history),
        }

    def reset(self) -> None:
        """Reset analyzer state."""
        self._baseline_jerk = None
        self._baseline_rom = None
        self._baseline_velocity = None
        self._baseline_variability = None
        self._rep_history = []
