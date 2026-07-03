"""
Movement Smoothness Metrics.

SPARC (Spectral Arc Length) - duration-independent, clinically validated.
LDLJ (Log-Dimensionless Jerk) - commonly used in stroke rehab.

References:
    - Balasubramanian, S., Melendez-Calderon, A., & Burdet, E. (2012).
      "A robust and sensitive metric for quantifying movement smoothness."
      IEEE Trans. Biomed. Eng., 59(8), 2126-2136.
    - Rohrer, B., et al. (2002). "Movement smoothness changes in stroke
      hemiparesis." Brain, 125(6), 1225-1239.
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
    """Movement smoothness analyzer using SPARC and LDLJ.

    Args:
        fs: Sampling frequency in Hz (default: 30.0)
        sparc_weight: Weight for SPARC in combined score (default: 0.6)
        ldjl_weight: Weight for LDLJ in combined score (default: 0.4)
        sparc_threshold: Frequency magnitude threshold for cutoff (default: 0.05)
    """

    def __init__(self, fs: float = 30.0, sparc_weight: float = 0.6,
                 ldjl_weight: float = 0.4, sparc_threshold: float = 0.05):
        self._fs = fs
        self._sparc_weight = sparc_weight
        self._ldjl_weight = ldjl_weight
        self._sparc_threshold = sparc_threshold

    def analyze(self, angles: np.ndarray, timestamps: Optional[np.ndarray] = None) -> SmoothnessResult:
        # Minimum 30 data points for reliable SPARC (Balasubramanian et al., 2021)
        if len(angles) < 30:
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

            # SPARC normalization: empirical range [-6, 0] for rehabilitation
            # -6: very impaired, 0: theoretically perfect
            # Balasubramanian et al. (2012) Fig. 4 shows clinical values to -6+
            sparc_score = max(0.0, min(100.0, (sparc + 6.0) / 6.0 * 100.0))
            # LDLJ normalization: empirical range [-40, 0] for rehabilitation trajectories
            # Observed values are -32 to -34 (not the textbook [-10, 0] range)
            # Using wider range maps -40 → 0, 0 → 100
            ldjl_score = max(0.0, min(100.0, (ldjl + 40.0) / 40.0 * 100.0))
            score = self._sparc_weight * sparc_score + self._ldjl_weight * ldjl_score

            return SmoothnessResult(sparc=sparc, ldjl=ldjl, nvp=nvp, smoothness_score=score, is_valid=True)
        except Exception:
            return SmoothnessResult(is_valid=False)

    def _sparc(self, velocity: np.ndarray, fs: float) -> float:
        """Spectral Arc Length (SPARC) per Balasubramanian et al. (2012).

        Exact formula (Eq. 4 in the paper):
            SPARC = -∫₀^ωc √((1/ωc)² + (dM̂(ω)/dω)²) dω

        Where:
            M̂(ω) = |V(ω)| / max(|V(ω)|)  (normalized magnitude spectrum)
            ωc = cutoff frequency where M̂(ω) <= threshold (0.05)

        Returns a value in [-6, 0] where MORE NEGATIVE = LESS SMOOTH.
        """
        n = len(velocity)
        freq = np.fft.rfftfreq(n, d=1.0 / fs)
        mag = np.abs(np.fft.rfft(velocity))
        if np.max(mag) < 1e-10:
            return 0.0

        # Normalize magnitude by its max (Balasubramanian Eq. 3)
        mag_norm = mag / np.max(mag)

        # Find cutoff frequency: last index where normalized magnitude > threshold
        above_threshold = np.where(mag_norm > self._sparc_threshold)[0]
        if len(above_threshold) == 0:
            return 0.0

        fc_idx = above_threshold[-1]
        omega_c = freq[fc_idx]  # Cutoff frequency in Hz

        if omega_c < 1e-10:
            return 0.0

        # Crop to cutoff frequency
        freq_crop = freq[:fc_idx + 1]
        mag_crop = mag_norm[:fc_idx + 1]

        # Normalize frequency axis by cutoff frequency (Balasubramanian Eq. 4)
        # This is the key: Δω̄ = Δω / ωc, NOT arc_length / freq_range
        omega_bar = freq_crop / omega_c  # Normalized frequency [0, 1]
        d_omega_bar = np.diff(omega_bar)
        d_mag = np.diff(mag_crop)

        # Spectral arc length integral
        arc_length = np.sum(np.sqrt(d_omega_bar ** 2 + d_mag ** 2))

        # SPARC = -arc_length (already normalized by ωc in the frequency axis)
        sparc = -arc_length

        # Clip to empirically valid range for human movements
        return float(np.clip(sparc, -6.0, 0.0))

    def _ldlj(self, angles: np.ndarray, timestamps: Optional[np.ndarray] = None) -> float:
        """Log-Dimensionless Jerk (LDLJ) per Rohrer et al. (2002).

        Exact formula:
            DJ = -(T⁵/D²) ∫₀ᵀ |j(t)|² dt
            LDLJ = -ln(|DJ|)

        Where:
            T = movement duration
            D = movement amplitude (max - min angle)
            j(t) = jerk signal (third derivative of position)

        Higher LDLJ = smoother movement.
        """
        n = len(angles)
        if n < 4:
            return -10.0
        dt = np.diff(timestamps) if timestamps is not None else np.ones(n - 1) / self._fs
        dt = np.where(dt < 1e-6, 1e-6, dt)
        v = np.diff(angles) / dt
        a = np.diff(v) / dt[:-1]
        j = np.diff(a) / dt[:-2]
        T = np.sum(dt)
        A = np.max(angles) - np.min(angles)
        if A < 1e-6:
            return -10.0
        # Trapezoidal integration of jerk squared (Rohrer et al., 2002)
        integral_jerk_sq = np.trapezoid(j ** 2, dx=np.mean(dt[:-2]))
        dlj = -(T ** 5 / A ** 2) * integral_jerk_sq
        # LDLJ = -ln(|DJ|), higher = smoother
        return float(-np.log(abs(dlj))) if abs(dlj) > 1e-10 else -10.0

    def _nvp(self, velocity: np.ndarray) -> int:
        return sum(1 for i in range(1, len(velocity) - 1) if velocity[i] > velocity[i-1] and velocity[i] > velocity[i+1])
