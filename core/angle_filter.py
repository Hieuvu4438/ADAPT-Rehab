"""
Butterworth low-pass filter for joint angle sequences.

Raw pose estimation output (e.g., from RTMW3D) contains high-frequency noise
that corrupts smoothness, flow, and stability metrics. A 4th-order Butterworth
filter with a 6 Hz cutoff (appropriate for human movement at ~1-2 Hz) removes
tracking jitter while preserving true movement dynamics.

Reference:
    Winter, D.A. (2009). "Biomechanics and Motor Control of Human Movement."
    4th ed., Wiley. Chapter 2: Signal processing.

Usage:
    filter = AngleFilter(cutoff_hz=6.0, fs=30.0)
    smoothed = filter.filter(raw_angles)
"""

import numpy as np
from scipy.signal import butter, filtfilt


class AngleFilter:
    """Butterworth low-pass filter for angle time series.

    Args:
        cutoff_hz: Cutoff frequency in Hz (default: 6.0).
            Human voluntary movement is typically 0.5-3 Hz.
            6 Hz preserves movement while removing tracking noise.
        fs: Sampling frequency in Hz (default: 30.0).
        order: Filter order (default: 4). Higher = sharper cutoff.
    """

    def __init__(self, cutoff_hz: float = 6.0, fs: float = 30.0, order: int = 4):
        self._cutoff_hz = cutoff_hz
        self._fs = fs
        self._order = order
        self._nyquist = fs / 2.0
        self._normalized_cutoff = cutoff_hz / self._nyquist

        # Validate
        if self._normalized_cutoff >= 1.0:
            # Cutoff too high, effectively no filtering
            self._b = None
            self._a = None
        else:
            self._b, self._a = butter(order, self._normalized_cutoff, btype='low')

    def filter(self, angles: np.ndarray) -> np.ndarray:
        """Apply zero-phase Butterworth filter to angle sequence.

        Uses filtfilt (forward-backward) for zero phase distortion,
        which is critical for timing-sensitive metrics like SPARC.

        Args:
            angles: Raw angle sequence, shape (N,).

        Returns:
            Filtered angle sequence, same length as input.
        """
        angles = np.asarray(angles, dtype=np.float64)

        if len(angles) < 10:
            return angles

        if self._b is None:
            return angles  # Cutoff too high, return as-is

        # filtfilt needs at least 3 * max(len(a), len(b)) data points
        min_len = max(len(self._b), len(self._a)) * 3 + 1
        if len(angles) < min_len:
            return angles

        try:
            return filtfilt(self._b, self._a, angles)
        except Exception:
            return angles

    def filter_dict(self, angles_dict: dict) -> dict:
        """Filter all angle sequences in a dict.

        Args:
            angles_dict: {joint_name: [angle_values]} or {joint_name: np.ndarray}.

        Returns:
            Same dict with filtered values.
        """
        return {k: self.filter(np.array(v)).tolist() for k, v in angles_dict.items()}
