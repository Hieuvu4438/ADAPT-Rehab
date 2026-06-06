"""
Movement Smoothness Metrics.

SPARC (Spectral Arc Length) - duration-independent, clinically validated.
LDLJ (Log-Dimensionless Jerk) - commonly used in stroke rehab.

Reference: Balasubramanian et al., Journal of Biomechanics, 2012.
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class SmoothnessResult:
    sparc: float = 0.0         # Spectral Arc Length (more negative = smoother)
    ldjl: float = 0.0          # Log-Dimensionless Jerk
    nvp: int = 0               # Number of Velocity Peaks
    smoothness_score: float = 0.0  # Normalized 0-100
    is_valid: bool = False


class SmoothnessAnalyzer:
    """Movement smoothness analyzer using SPARC and LDLJ."""

    def __init__(self, fs: float = 30.0):
        self._fs = fs

    def analyze(self, angles: np.ndarray, timestamps: Optional[np.ndarray] = None) -> SmoothnessResult:
        if len(angles) < 10:
            return SmoothnessResult(is_valid=False)
        try:
            if timestamps is not None and len(timestamps) == len(angles):
                dt = np.diff(timestamps)
                dt = np.where(dt < 1e-6, 1e-6, dt)
                velocity = np.diff(angles) / dt
                fs = 1.0 / np.mean(dt)
            else:
                velocity = np.diff(angles) * self._fs
                fs = self._fs

            sparc = self._sparc(velocity, fs)
            ldjl = self._ldlj(angles, timestamps)
            nvp = self._nvp(velocity)

            sparc_score = max(0, min(100, (sparc + 2) / 2 * 100))
            ldjl_score = max(0, min(100, (ldjl + 10) / 8 * 100))
            score = 0.6 * sparc_score + 0.4 * ldjl_score

            return SmoothnessResult(sparc=sparc, ldjl=ldjl, nvp=nvp, smoothness_score=score, is_valid=True)
        except Exception:
            return SmoothnessResult(is_valid=False)

    def _sparc(self, velocity: np.ndarray, fs: float) -> float:
        n = len(velocity)
        freq = np.fft.rfftfreq(n, d=1.0 / fs)
        mag = np.abs(np.fft.rfft(velocity))
        if np.max(mag) < 1e-10:
            return 0.0
        mag_norm = mag / np.max(mag)
        return float(-np.sum(np.sqrt(np.diff(freq)**2 + np.diff(mag_norm)**2)))

    def _ldlj(self, angles: np.ndarray, timestamps: Optional[np.ndarray] = None) -> float:
        n = len(angles)
        if n < 4:
            return -10.0
        dt = np.diff(timestamps) if timestamps is not None else np.ones(n - 1) / self._fs
        dt = np.where(dt < 1e-6, 1e-6, dt)
        v = np.diff(angles) / dt
        a = np.diff(v) / dt[:-1]
        j = np.diff(a) / dt[:-2]
        T, A = np.sum(dt), np.max(angles) - np.min(angles)
        if A < 1e-6:
            return -10.0
        dlj = (T**5 / A**2) * np.sum(j**2) * np.mean(dt[:-2])
        return float(np.log(dlj)) if dlj > 0 else -10.0

    def _nvp(self, velocity: np.ndarray) -> int:
        return sum(1 for i in range(1, len(velocity) - 1) if velocity[i] > velocity[i-1] and velocity[i] > velocity[i+1])
