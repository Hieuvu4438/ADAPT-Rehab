"""
Temporal Compensation Detection.

Detects compensatory movements during rehabilitation exercises:
- Shoulder hiking: elevating shoulder to compensate for limited ROM
- Trunk lean: leaning body instead of using target joint
- Hip shift: shifting hips for balance compensation

Uses temporal analysis over pose sequences for robustness,
rather than frame-by-frame threshold methods.

Usage:
    detector = CompensationDetector()
    result = detector.analyze(pose_sequence)
    if result.score < 80:
        print(f"Compensation: {result.detected_types}")
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from enum import Enum
import numpy as np


class CompensationType(Enum):
    """Types of compensatory movements."""
    SHOULDER_HIKING = "shoulder_hiking"
    TRUNK_LEAN = "trunk_lean"
    HIP_SHIFT = "hip_shift"


@dataclass
class CompensationEvent:
    """A single compensation event."""
    type: CompensationType
    severity: float  # 0-1 (0 = none, 1 = severe)
    details: str = ""


@dataclass
class CompensationResult:
    """Result of compensation analysis.

    Attributes:
        events: List of detected compensation events.
        score: Overall score (100 = no compensation, 0 = severe).
        detected_types: Human-readable Vietnamese labels.
        shoulder_diff_avg: Average shoulder height difference.
        trunk_tilt_avg: Average trunk tilt angle (degrees).
        hip_diff_avg: Average hip horizontal difference.
        is_valid: Whether analysis was performed successfully.
    """
    events: List[CompensationEvent] = field(default_factory=list)
    score: float = 100.0  # 100 = no compensation
    detected_types: List[str] = field(default_factory=list)
    shoulder_diff_avg: float = 0.0
    trunk_tilt_avg: float = 0.0
    hip_diff_avg: float = 0.0
    is_valid: bool = False


class CompensationDetector:
    """
    Temporal compensation detector.

    Analyzes pose sequences over time to detect compensatory movements.
    More robust than frame-by-frame thresholds because it considers
    the temporal pattern of movement.

    Thresholds (can be tuned):
        - SHOULDER_THRESHOLD: 0.05 (5% of frame height)
        - TRUNK_THRESHOLD: 15.0 degrees
        - HIP_THRESHOLD: 0.06 (6% of frame height)

    Example:
        >>> detector = CompensationDetector()
        >>> # Collect poses over time
        >>> for frame in video:
        ...     pose = detect_pose(frame)
        ...     detector.add_frame(pose)
        >>> result = detector.get_result()
        >>> print(f"Score: {result.score}")
    """

    # Tunable thresholds
    SHOULDER_THRESHOLD = 0.05  # 5% of frame height
    TRUNK_THRESHOLD = 15.0     # degrees
    HIP_THRESHOLD = 0.06       # 6% of frame height

    # MediaPipe landmark indices
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_HIP = 23
    RIGHT_HIP = 24

    def __init__(self, window_size: int = 30):
        """
        Initialize compensation detector.

        Args:
            window_size: Number of frames for temporal analysis.
        """
        self._window_size = window_size
        self._shoulder_history: List[float] = []
        self._trunk_history: List[float] = []
        self._hip_history: List[float] = []
        self._all_poses: List[np.ndarray] = []

    def add_frame(self, pose_landmarks: np.ndarray) -> None:
        """
        Add a single frame's pose data.

        Args:
            pose_landmarks: Pose landmarks array, shape (33, 3) or similar.
        """
        if len(pose_landmarks) < 25:
            return

        self._all_poses.append(pose_landmarks.copy())

        # Shoulder height difference
        ls = pose_landmarks[self.LEFT_SHOULDER]
        rs = pose_landmarks[self.RIGHT_SHOULDER]
        self._shoulder_history.append(abs(float(ls[1] - rs[1])))

        # Trunk tilt
        mid_shoulder = (ls[:2] + rs[:2]) / 2
        mid_hip = (pose_landmarks[self.LEFT_HIP][:2] + pose_landmarks[self.RIGHT_HIP][:2]) / 2
        dx = mid_hip[0] - mid_shoulder[0]
        dy = mid_hip[1] - mid_shoulder[1]
        if abs(dy) > 1e-6:
            tilt = abs(np.degrees(np.arctan2(dx, dy)))
        else:
            tilt = 0.0
        self._trunk_history.append(tilt)

        # Hip shift (vertical height difference between hips)
        lh = pose_landmarks[self.LEFT_HIP]
        rh = pose_landmarks[self.RIGHT_HIP]
        self._hip_history.append(abs(float(lh[1] - rh[1])))

        # Keep window size
        if len(self._shoulder_history) > self._window_size:
            self._shoulder_history.pop(0)
            self._trunk_history.pop(0)
            self._hip_history.pop(0)

    def analyze(self, pose_sequence: List[np.ndarray] = None) -> CompensationResult:
        """
        Analyze a sequence of poses for compensation.

        Args:
            pose_sequence: Optional list of pose landmarks. If None, uses accumulated history.

        Returns:
            CompensationResult with detected compensations.
        """
        if pose_sequence is not None:
            # Process full sequence
            self._shoulder_history = []
            self._trunk_history = []
            self._hip_history = []
            for pose in pose_sequence:
                self.add_frame(pose)

        if len(self._shoulder_history) < 5:
            return CompensationResult(is_valid=False)

        events = []
        types = []

        # Compute statistics
        shoulder_arr = np.array(self._shoulder_history)
        trunk_arr = np.array(self._trunk_history)
        hip_arr = np.array(self._hip_history)

        shoulder_avg = float(np.mean(shoulder_arr))
        shoulder_max = float(np.max(shoulder_arr))
        trunk_avg = float(np.mean(trunk_arr))
        trunk_max = float(np.max(trunk_arr))
        hip_avg = float(np.mean(hip_arr))
        hip_max = float(np.max(hip_arr))

        # Detect shoulder hiking
        if shoulder_max > self.SHOULDER_THRESHOLD * 1.5:
            severity = min(1.0, shoulder_max / (self.SHOULDER_THRESHOLD * 3))
            events.append(CompensationEvent(
                type=CompensationType.SHOULDER_HIKING,
                severity=severity,
                details=f"Max shoulder diff: {shoulder_max:.3f} (threshold: {self.SHOULDER_THRESHOLD:.3f})"
            ))
            types.append("Vai không đều (nặng)")
        elif shoulder_max > self.SHOULDER_THRESHOLD:
            severity = min(1.0, shoulder_max / (self.SHOULDER_THRESHOLD * 2))
            events.append(CompensationEvent(
                type=CompensationType.SHOULDER_HIKING,
                severity=severity,
                details=f"Max shoulder diff: {shoulder_max:.3f}"
            ))
            types.append("Vai không đều")

        # Detect trunk lean
        if trunk_max > self.TRUNK_THRESHOLD * 1.5:
            severity = min(1.0, trunk_max / (self.TRUNK_THRESHOLD * 2))
            events.append(CompensationEvent(
                type=CompensationType.TRUNK_LEAN,
                severity=severity,
                details=f"Max trunk tilt: {trunk_max:.1f} deg (threshold: {self.TRUNK_THRESHOLD:.1f})"
            ))
            types.append("Nghiêng thân nhiều")
        elif trunk_max > self.TRUNK_THRESHOLD:
            severity = min(1.0, trunk_max / (self.TRUNK_THRESHOLD * 1.5))
            events.append(CompensationEvent(
                type=CompensationType.TRUNK_LEAN,
                severity=severity,
                details=f"Max trunk tilt: {trunk_max:.1f} deg"
            ))
            types.append("Nghiêng thân")

        # Detect hip shift
        if hip_max > self.HIP_THRESHOLD * 1.5:
            severity = min(1.0, hip_max / (self.HIP_THRESHOLD * 2))
            events.append(CompensationEvent(
                type=CompensationType.HIP_SHIFT,
                severity=severity,
                details=f"Max hip diff: {hip_max:.3f} (threshold: {self.HIP_THRESHOLD:.3f})"
            ))
            types.append("Hông không cân bằng")
        elif hip_max > self.HIP_THRESHOLD:
            severity = min(1.0, hip_max / (self.HIP_THRESHOLD * 1.5))
            events.append(CompensationEvent(
                type=CompensationType.HIP_SHIFT,
                severity=severity,
                details=f"Max hip diff: {hip_max:.3f}"
            ))
            types.append("Hông lệch nhẹ")

        # Compute score (100 = no compensation)
        total_penalty = sum(e.severity * 40 for e in events)
        score = max(0, 100 - total_penalty)

        return CompensationResult(
            events=events,
            score=score,
            detected_types=types,
            shoulder_diff_avg=shoulder_avg,
            trunk_tilt_avg=trunk_avg,
            hip_diff_avg=hip_avg,
            is_valid=True,
        )

    def reset(self) -> None:
        """Reset detector state."""
        self._shoulder_history = []
        self._trunk_history = []
        self._hip_history = []
        self._all_poses = []
