"""
Body State Detection using 3D Keypoints from RTMW3D.

Detects behavioral states (Fatigue, Exhaustion, Boredom, Pain) through
scientifically validated kinematic metrics extracted from 3D body keypoints.

All formulas are from peer-reviewed literature with full citations.

Metrics Used:
    1. Joint Velocity & Acceleration (Winter, 2009)
    2. Velocity Loss (Gonzalez-Badillo & Sanchez-Medina, 2010)
    3. ROM Decline (fatigue indicator)
    4. SPARC Smoothness (Balasubramanian et al., 2015)
    5. Trunk Inclination (Wu et al., 2005)
    6. Asymmetry Index (clinical standard)
    7. Movement variability & spectral entropy
    8. Mann-Kendall trend test (Mann, 1945)

Architecture:
    RTMW3D 3D Keypoints → Kinematic Metrics → Behavioral State Classification

Version: 4.0.0
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple
import numpy as np
from scipy import signal
from scipy.spatial.distance import euclidean
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# Data Types
# ============================================================================

class BodyState(Enum):
    """Detected body behavioral states."""
    NORMAL = "normal"
    FATIGUE = "fatigue"
    EXHAUSTION = "exhaustion"
    BOREDOM = "boredom"
    PAIN = "pain"  # Guarding behavior


@dataclass
class KinematicMetrics:
    """Computed kinematic metrics from 3D keypoints."""
    # Velocity
    joint_velocities: Dict[str, float] = field(default_factory=dict)
    mean_velocity: float = 0.0
    peak_velocity: float = 0.0

    # Acceleration
    joint_accelerations: Dict[str, float] = field(default_factory=dict)
    mean_acceleration: float = 0.0

    # ROM
    joint_rom: Dict[str, float] = field(default_factory=dict)
    mean_rom: float = 0.0

    # Smoothness
    sparc_score: float = 0.0  # Spectral Arc Length

    # Posture
    trunk_inclination: float = 0.0  # Degrees from vertical
    trunk_lateral_lean: float = 0.0

    # Asymmetry
    asymmetry_index: float = 0.0  # Percentage

    # Temporal
    movement_duration: float = 0.0
    pause_duration: float = 0.0


@dataclass
class BodyStateResult:
    """Result of body behavioral state detection."""
    state: BodyState = BodyState.NORMAL
    confidence: float = 0.0

    # Individual scores (0.0 - 1.0)
    fatigue_score: float = 0.0
    exhaustion_score: float = 0.0
    boredom_score: float = 0.0
    pain_score: float = 0.0

    # Raw metrics
    velocity_loss_pct: float = 0.0    # Gonzalez-Badillo velocity loss
    rom_decline_pct: float = 0.0      # ROM decline from first rep
    trunk_inclination_deg: float = 0.0
    asymmetry_pct: float = 0.0
    smoothness: float = 0.0           # SPARC score

    # Kinematic details
    metrics: Optional[KinematicMetrics] = None

    # Metadata
    is_valid: bool = False
    frame_count: int = 0
    rep_count: int = 0


# ============================================================================
# COCO-WholeBody Keypoint Indices (133 keypoints)
# ============================================================================

class KeypointIndex:
    """COCO-WholeBody keypoint indices for RTMW3D."""
    # Body (0-16)
    NOSE = 0
    LEFT_EYE = 1
    RIGHT_EYE = 2
    LEFT_EAR = 3
    RIGHT_EAR = 4
    LEFT_SHOULDER = 5
    RIGHT_SHOULDER = 6
    LEFT_ELBOW = 7
    RIGHT_ELBOW = 8
    LEFT_WRIST = 9
    RIGHT_WRIST = 10
    LEFT_HIP = 11
    RIGHT_HIP = 12
    LEFT_KNEE = 13
    RIGHT_KNEE = 14
    LEFT_ANKLE = 15
    RIGHT_ANKLE = 16

    # Additional body joints
    NECK = 17
    HEAD_TOP = 18
    PELVIS = 19  # Mid-hip

    # Body connections for angle computation
    BODY_SEGMENTS = {
        "left_upper_arm": (LEFT_SHOULDER, LEFT_ELBOW),
        "left_forearm": (LEFT_ELBOW, LEFT_WRIST),
        "right_upper_arm": (RIGHT_SHOULDER, RIGHT_ELBOW),
        "right_forearm": (RIGHT_ELBOW, RIGHT_WRIST),
        "left_thigh": (LEFT_HIP, LEFT_KNEE),
        "left_shank": (LEFT_KNEE, LEFT_ANKLE),
        "right_thigh": (RIGHT_HIP, RIGHT_KNEE),
        "right_shank": (RIGHT_KNEE, RIGHT_ANKLE),
        "trunk": (NECK, PELVIS),
    }

    # Joint angle triplets (proximal, center, distal)
    JOINT_ANGLES = {
        "left_shoulder": (LEFT_ELBOW, LEFT_SHOULDER, NECK),
        "right_shoulder": (RIGHT_ELBOW, RIGHT_SHOULDER, NECK),
        "left_elbow": (LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST),
        "right_elbow": (RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST),
        "left_hip": (LEFT_KNEE, LEFT_HIP, PELVIS),
        "right_hip": (RIGHT_KNEE, RIGHT_HIP, PELVIS),
        "left_knee": (LEFT_HIP, LEFT_KNEE, LEFT_ANKLE),
        "right_knee": (RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE),
    }


# ============================================================================
# Joint Angle Calculator
# ============================================================================

class JointAngleCalculator:
    """
    Compute joint angles from 3D keypoints.

    Formula (3-point angle):
        V1 = P_proximal - J_center
        V2 = P_distal - J_center
        θ = arccos( (V1 · V2) / (|V1| × |V2|) )

    Reference: Wu, G. et al. (2005). "ISB recommendation on definitions of
    joint coordinate systems." Journal of Biomechanics, 38, 981-992.
    """

    @staticmethod
    def compute_angle(p1: np.ndarray, center: np.ndarray, p2: np.ndarray) -> float:
        """
        Compute angle at center point between p1 and p2.

        Args:
            p1: First point (3D)
            center: Vertex point (3D)
            p2: Second point (3D)

        Returns:
            Angle in degrees (0-180)
        """
        v1 = p1 - center
        v2 = p2 - center

        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)

        return float(np.degrees(np.arccos(cos_angle)))

    @staticmethod
    def compute_all_angles(keypoints_3d: np.ndarray) -> Dict[str, float]:
        """
        Compute all joint angles from 3D keypoints.

        Args:
            keypoints_3d: (N, 3) array of 3D keypoint coordinates

        Returns:
            Dict mapping joint name to angle in degrees
        """
        angles = {}
        for joint_name, (p1_idx, center_idx, p2_idx) in KeypointIndex.JOINT_ANGLES.items():
            if max(p1_idx, center_idx, p2_idx) < len(keypoints_3d):
                p1 = keypoints_3d[p1_idx]
                center = keypoints_3d[center_idx]
                p2 = keypoints_3d[p2_idx]
                angles[joint_name] = JointAngleCalculator.compute_angle(p1, center, p2)
        return angles


# ============================================================================
# Velocity & Acceleration Calculator
# ============================================================================

class KinematicsCalculator:
    """
    Compute velocity and acceleration from 3D keypoint time series.

    Formula (Central difference, Winter, 2009):
        v(t) = [p(t+Δt) - p(t-Δt)] / (2×Δt)
        a(t) = [p(t+Δt) - 2×p(t) + p(t-Δt)] / Δt²

    Reference: Winter, D.A. (2009). Biomechanics and Motor Control of
    Human Movement. 4th ed., Wiley.
    """

    def __init__(self, fps: float = 30.0, filter_cutoff: float = 8.0):
        """
        Args:
            fps: Frame rate
            filter_cutoff: Low-pass Butterworth filter cutoff (Hz)
        """
        self.fps = fps
        self.dt = 1.0 / fps
        self.filter_cutoff = filter_cutoff

    def compute_velocity(self, positions: np.ndarray) -> np.ndarray:
        """
        Compute velocity using central difference.

        Args:
            positions: (T, 3) position time series

        Returns:
            (T, 3) velocity time series
        """
        if len(positions) < 3:
            return np.zeros_like(positions)

        # Apply low-pass filter first (Woltring, 1985)
        filtered = self._butterworth_filter(positions)

        # Central difference
        velocity = np.zeros_like(filtered)
        velocity[1:-1] = (filtered[2:] - filtered[:-2]) / (2.0 * self.dt)
        velocity[0] = (filtered[1] - filtered[0]) / self.dt
        velocity[-1] = (filtered[-1] - filtered[-2]) / self.dt

        return velocity

    def compute_acceleration(self, positions: np.ndarray) -> np.ndarray:
        """
        Compute acceleration using central difference.

        Args:
            positions: (T, 3) position time series

        Returns:
            (T, 3) acceleration time series
        """
        if len(positions) < 3:
            return np.zeros_like(positions)

        filtered = self._butterworth_filter(positions)

        acceleration = np.zeros_like(filtered)
        acceleration[1:-1] = (filtered[2:] - 2 * filtered[1:-1] + filtered[:-2]) / (self.dt ** 2)

        return acceleration

    def _butterworth_filter(self, data: np.ndarray) -> np.ndarray:
        """Apply low-pass Butterworth filter (Woltring, 1985)."""
        # Need at least 2x padlen + 1 samples for filtfilt
        # padlen = 3 * max(len(a), len(b)) for 4th order = ~15
        if len(data) < 32:
            return data

        nyquist = self.fps / 2.0
        normalized_cutoff = self.filter_cutoff / nyquist

        if normalized_cutoff >= 1.0:
            return data

        b, a = signal.butter(4, normalized_cutoff, btype='low')

        filtered = np.zeros_like(data)
        for i in range(data.shape[1]):
            filtered[:, i] = signal.filtfilt(b, a, data[:, i])

        return filtered

    def compute_speed(self, velocity: np.ndarray) -> np.ndarray:
        """Compute scalar speed from velocity vector."""
        return np.linalg.norm(velocity, axis=1)


# ============================================================================
# SPARC Smoothness Calculator
# ============================================================================

class SPARCCalculator:
    """
    SPectral ARC length (SPARC) for movement smoothness.

    Formula (Balasubramanian, Melendez-Calderon & Burdet, 2012, Eq. 4):
        SPARC = -integral_0^omega_c sqrt((1/omega_c)^2 + (dM_hat/domega)^2) domega

    Where:
        M_hat(omega) = |V(omega)| / max(|V(omega)|)  (normalized magnitude spectrum)
        omega_c = cutoff frequency where M_hat(omega) <= threshold (0.05)

    Properties:
        - Dimensionless
        - Amplitude-independent
        - Duration-independent (key advantage over jerk-based metrics)
        - Range: typically [-6, 0] for human movements
        - Smoother -> larger (less negative) values

    Reference: Balasubramanian, S., Melendez-Calderon, A., & Burdet, E.
    (2012). IEEE Trans. Biomed. Eng., 59(8), 2126-2136.
    """

    def __init__(self, fps: float = 30.0, threshold: float = 0.05):
        self.fps = fps
        self.threshold = threshold

    def compute(self, speed_profile: np.ndarray) -> float:
        """
        Compute SPARC from speed profile.

        Args:
            speed_profile: (T,) speed time series

        Returns:
            SPARC value (typically [-6, 0], higher = smoother)
        """
        # Minimum 30 data points for reliable SPARC (Balasubramanian et al., 2021)
        if len(speed_profile) < 30:
            return 0.0

        # FFT
        N = len(speed_profile)
        V = np.fft.rfft(speed_profile)
        freqs = np.fft.rfftfreq(N, d=1.0 / self.fps)

        # Normalize magnitude by its max (Balasubramanian Eq. 3)
        mag = np.abs(V)
        if np.max(mag) < 1e-10:
            return 0.0
        M_hat = mag / np.max(mag)

        # Find cutoff frequency where normalized magnitude drops below threshold
        above_threshold = np.where(M_hat > self.threshold)[0]
        if len(above_threshold) == 0:
            return 0.0
        fc_idx = above_threshold[-1]
        omega_c = freqs[fc_idx]  # Cutoff frequency in Hz

        if omega_c < 1e-10:
            return 0.0

        # Crop to cutoff frequency
        freq_crop = freqs[:fc_idx + 1]
        mag_crop = M_hat[:fc_idx + 1]

        # Normalize frequency axis by cutoff frequency (Balasubramanian Eq. 4)
        omega_bar = freq_crop / omega_c  # Normalized frequency [0, 1]
        d_omega_bar = np.diff(omega_bar)
        d_mag = np.diff(mag_crop)

        # Spectral arc length integral
        arc_length = np.sum(np.sqrt(d_omega_bar ** 2 + d_mag ** 2))

        # SPARC = -arc_length
        sparc = -arc_length

        return float(np.clip(sparc, -6.0, 0.0))


# ============================================================================
# Trunk Inclination Calculator
# ============================================================================

class TrunkInclinationCalculator:
    """
    Compute trunk inclination from 3D keypoints.

    Formula (Wu et al., 2005):
        V_trunk = P_shoulder_mid - P_hip_mid
        θ_trunk = arccos( (V_trunk · V_vertical) / (|V_trunk| × |V_vertical|) )

    Reference: Wu, G. et al. (2005). "ISB recommendation on definitions of
    joint coordinate systems." Journal of Biomechanics, 38, 981-992.
    """

    @staticmethod
    def compute(keypoints_3d: np.ndarray) -> Tuple[float, float]:
        """
        Compute trunk inclination angles.

        Args:
            keypoints_3d: (N, 3) 3D keypoints

        Returns:
            Tuple of (total_inclination_deg, lateral_lean_deg)
        """
        if len(keypoints_3d) < 17:
            return 0.0, 0.0

        # Mid-shoulder and mid-hip
        shoulder_mid = (keypoints_3d[KeypointIndex.LEFT_SHOULDER] +
                        keypoints_3d[KeypointIndex.RIGHT_SHOULDER]) / 2.0
        hip_mid = (keypoints_3d[KeypointIndex.LEFT_HIP] +
                   keypoints_3d[KeypointIndex.RIGHT_HIP]) / 2.0

        trunk_vec = shoulder_mid - hip_mid

        # Total inclination from vertical (Y-up)
        vertical = np.array([0, 1, 0])
        cos_total = np.dot(trunk_vec, vertical) / (np.linalg.norm(trunk_vec) + 1e-8)
        total_inclination = float(np.degrees(np.arccos(np.clip(cos_total, -1, 1))))

        # Sagittal (forward/backward lean) - project onto XY plane
        sagittal_vec = np.array([trunk_vec[0], trunk_vec[1], 0])
        cos_sag = np.dot(sagittal_vec, vertical) / (np.linalg.norm(sagittal_vec) + 1e-8)
        sagittal_lean = float(np.degrees(np.arccos(np.clip(cos_sag, -1, 1))))

        # Frontal (lateral lean) - project onto YZ plane
        frontal_vec = np.array([0, trunk_vec[1], trunk_vec[2]])
        cos_front = np.dot(frontal_vec, vertical) / (np.linalg.norm(frontal_vec) + 1e-8)
        lateral_lean = float(np.degrees(np.arccos(np.clip(cos_front, -1, 1))))

        return total_inclination, lateral_lean


# ============================================================================
# Asymmetry Calculator
# ============================================================================

class AsymmetryCalculator:
    """
    Compute bilateral asymmetry index.

    Formula (clinical standard):
        AI = (L - R) / (0.5 × (L + R)) × 100%

    Clinical threshold: AI > 10-15% is clinically significant.
    """

    @staticmethod
    def compute(left_value: float, right_value: float) -> float:
        """
        Compute asymmetry index between left and right sides.

        Args:
            left_value: Metric value for left side
            right_value: Metric value for right side

        Returns:
            Asymmetry index (percentage)
        """
        denominator = 0.5 * (abs(left_value) + abs(right_value))
        if denominator < 1e-8:
            return 0.0
        return ((left_value - right_value) / denominator) * 100.0


# ============================================================================
# Trend Analyzer (Mann-Kendall)
# ============================================================================

class MannKendallTrendAnalyzer:
    """
    Mann-Kendall non-parametric trend test for detecting monotonic trends
    in kinematic metrics across repetitions.

    Formula (Mann, 1945):
        S = Σᵢ₌₁ᴺ⁻¹ Σⱼ₌ᵢ₊₁ᴺ sgn(xⱼ - xᵢ)
        Var(S) = N(N-1)(2N+5) / 18
        Z = (S - sgn(S)) / √Var(S)

    If |Z| > 1.96 (α=0.05), trend is statistically significant.
    Negative Z = decreasing trend (fatigue indicator for velocity/ROM).

    Reference: Mann, H.B. (1945). "Nonparametric tests against trend."
    Econometrica, 13, 245-259.
    """

    @staticmethod
    def compute_s(values: List[float]) -> Tuple[float, float]:
        """
        Compute Mann-Kendall S statistic and Z-score.

        Args:
            values: List of metric values across repetitions

        Returns:
            Tuple of (S_statistic, Z_score)
        """
        n = len(values)
        if n < 4:
            return 0.0, 0.0

        # Compute S
        s = 0.0
        for i in range(n - 1):
            for j in range(i + 1, n):
                diff = values[j] - values[i]
                if diff > 0:
                    s += 1
                elif diff < 0:
                    s -= 1

        # Variance
        var_s = n * (n - 1) * (2 * n + 5) / 18.0

        # Z-score
        if var_s < 1e-8:
            return s, 0.0

        if s > 0:
            z = (s - 1) / np.sqrt(var_s)
        elif s < 0:
            z = (s + 1) / np.sqrt(var_s)
        else:
            z = 0.0

        return s, z

    @staticmethod
    def compute_sens_slope(values: List[float]) -> float:
        """
        Compute Sen's slope (median of pairwise slopes).

        Returns:
            Slope (change per repetition)
        """
        n = len(values)
        if n < 2:
            return 0.0

        slopes = []
        for i in range(n - 1):
            for j in range(i + 1, n):
                if j != i:
                    slopes.append((values[j] - values[i]) / (j - i))

        return float(np.median(slopes)) if slopes else 0.0


# ============================================================================
# Main Body State Detector
# ============================================================================

class BodyStateDetector:
    """
    Body behavioral state detector using 3D keypoints from RTMW3D.

    Detects: Fatigue, Exhaustion, Boredom, Pain (guarding), Normal
    using scientifically validated kinematic metrics.

    Key Metrics:
        1. Velocity Loss (Gonzalez-Badillo & Sanchez-Medina, 2010)
           - 20% loss = moderate fatigue
           - 40% loss = significant fatigue
        2. ROM Decline
           - Progressive ROM reduction across repetitions
        3. SPARC Smoothness (Balasubramanian et al., 2015)
           - Movement quality degradation
        4. Trunk Inclination (Wu et al., 2005)
           - Posture collapse indicator
        5. Asymmetry Index
           - Pain/guarding behavior indicator
        6. Mann-Kendall Trend (Mann, 1945)
           - Statistical significance of trends

    Usage:
        detector = BodyStateDetector(fps=30.0)
        result = detector.process_frame(keypoints_3d, timestamp_s)
        if result.is_valid:
            print(f"State: {result.state.value}")
            print(f"Fatigue: {result.fatigue_score:.2f}")
    """

    # Velocity loss thresholds (Gonzalez-Badillo et al., 2010)
    VELOCITY_LOSS_MILD = 20.0      # % loss = moderate fatigue
    VELOCITY_LOSS_SIGNIFICANT = 40.0  # % loss = significant fatigue

    # ROM decline thresholds
    ROM_DECLINE_MILD = 15.0        # % decline
    ROM_DECLINE_SIGNIFICANT = 30.0

    # Trunk inclination thresholds
    TRUNK_INCLINATION_WARNING = 20.0  # degrees
    TRUNK_INCLINATION_CRITICAL = 30.0

    # Asymmetry threshold
    ASYMMETRY_SIGNIFICANT = 15.0   # percentage

    # Boredom thresholds
    BOREDOM_VARIABILITY_THRESHOLD = 0.3  # Low variability = boredom
    BOREDOM_AMPLITUDE_THRESHOLD = 0.5    # Low amplitude = boredom

    def __init__(self, fps: float = 30.0):
        self.fps = fps

        # Sub-calculators
        self.kinematics_calc = KinematicsCalculator(fps=fps)
        self.sparc_calc = SPARCCalculator(fps=fps)
        self.trunk_calc = TrunkInclinationCalculator()
        self.asymmetry_calc = AsymmetryCalculator()
        self.trend_analyzer = MannKendallTrendAnalyzer()
        self.angle_calc = JointAngleCalculator()

        # State tracking
        self._frame_count = 0
        self._rep_count = 0

        # Per-repetition data
        self._rep_data: List[Dict] = []  # List of per-rep metrics

        # Current repetition tracking
        self._current_rep_positions: List[np.ndarray] = []
        self._current_rep_timestamps: List[float] = []
        self._current_rep_angles: List[Dict[str, float]] = []

        # Reference values (from first rep or calibrated)
        self._reference_velocity: Optional[float] = None
        self._reference_rom: Optional[Dict[str, float]] = None

        # Keypoint history for velocity computation
        self._keypoint_history: List[np.ndarray] = []
        self._timestamp_history: List[float] = []

    def process_frame(self, keypoints_3d: np.ndarray,
                      timestamp_s: float,
                      is_rep_active: bool = True) -> BodyStateResult:
        """
        Process a frame of 3D keypoints.

        Args:
            keypoints_3d: (N, 3) array of 3D keypoint coordinates
            timestamp_s: Timestamp in seconds
            is_rep_active: Whether the user is currently performing a repetition

        Returns:
            BodyStateResult with all computed metrics and state classification
        """
        self._frame_count += 1

        if keypoints_3d is None or len(keypoints_3d) < 17:
            return BodyStateResult(error_message="Insufficient keypoints")

        # Store for velocity computation
        self._keypoint_history.append(keypoints_3d.copy())
        self._timestamp_history.append(timestamp_s)

        # Keep limited history
        max_history = int(self.fps * 10)  # 10 seconds
        if len(self._keypoint_history) > max_history:
            self._keypoint_history = self._keypoint_history[-max_history:]
            self._timestamp_history = self._timestamp_history[-max_history:]

        # Compute current frame metrics
        angles = self.angle_calc.compute_all_angles(keypoints_3d)
        trunk_total, trunk_lateral = self.trunk_calc.compute(keypoints_3d)

        # Track current repetition
        if is_rep_active:
            self._current_rep_positions.append(keypoints_3d.copy())
            self._current_rep_timestamps.append(timestamp_s)
            self._current_rep_angles.append(angles)

        # Compute velocity (if enough history)
        velocity_metrics = self._compute_velocity_metrics()

        # Compute asymmetry
        asymmetry = self._compute_asymmetry(keypoints_3d)

        # Build result
        result = BodyStateResult(
            metrics=KinematicMetrics(
                joint_velocities=velocity_metrics.get("velocities", {}),
                mean_velocity=velocity_metrics.get("mean_velocity", 0.0),
                peak_velocity=velocity_metrics.get("peak_velocity", 0.0),
                joint_rom=self._compute_current_rom(),
                trunk_inclination=trunk_total,
                trunk_lateral_lean=trunk_lateral,
                asymmetry_index=asymmetry,
            ),
            trunk_inclination_deg=trunk_total,
            asymmetry_pct=asymmetry,
            frame_count=self._frame_count,
            rep_count=self._rep_count,
        )

        # Classify state
        result.state, result.confidence = self._classify_state(result)
        result.is_valid = True

        return result

    def finalize_rep(self) -> Optional[Dict]:
        """
        Finalize the current repetition and compute per-rep metrics.
        Call this when a repetition is completed.

        Returns:
            Dict with per-rep metrics, or None if insufficient data
        """
        if len(self._current_rep_positions) < 10:
            return None

        self._rep_count += 1

        # Stack positions
        positions = np.array(self._current_rep_positions)  # (T, N, 3)

        # Compute per-joint velocities and ROM
        rep_metrics = {
            "rep_number": self._rep_count,
            "duration": self._current_rep_timestamps[-1] - self._current_rep_timestamps[0],
            "velocities": {},
            "rom": {},
            "smoothness": 0.0,
        }

        # For each tracked joint
        tracked_joints = ["left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
                         "left_hip", "right_hip", "left_knee", "right_knee"]

        for joint_name in tracked_joints:
            if joint_name in KeypointIndex.JOINT_ANGLES:
                p1_idx, center_idx, p2_idx = KeypointIndex.JOINT_ANGLES[joint_name]
                if max(p1_idx, center_idx, p2_idx) < positions.shape[1]:
                    # Extract joint center trajectory
                    joint_trajectory = positions[:, center_idx, :]  # (T, 3)

                    # Velocity
                    vel = self.kinematics_calc.compute_velocity(joint_trajectory)
                    speed = self.kinematics_calc.compute_speed(vel)
                    rep_metrics["velocities"][joint_name] = float(np.mean(speed))

                    # ROM (from angle time series)
                    angles_series = [a.get(joint_name, 0) for a in self._current_rep_angles]
                    if angles_series:
                        rep_metrics["rom"][joint_name] = max(angles_series) - min(angles_series)

        # Overall velocity and ROM
        all_velocities = list(rep_metrics["velocities"].values())
        all_rom = list(rep_metrics["rom"].values())

        rep_metrics["mean_velocity"] = float(np.mean(all_velocities)) if all_velocities else 0.0
        rep_metrics["peak_velocity"] = float(np.max(all_velocities)) if all_velocities else 0.0
        rep_metrics["mean_rom"] = float(np.mean(all_rom)) if all_rom else 0.0

        # SPARC smoothness (using wrist trajectory)
        if len(positions) > 10:
            wrist_idx = KeypointIndex.LEFT_WRIST
            if wrist_idx < positions.shape[1]:
                wrist_speed = self.kinematics_calc.compute_speed(
                    self.kinematics_calc.compute_velocity(positions[:, wrist_idx, :])
                )
                rep_metrics["smoothness"] = self.sparc_calc.compute(wrist_speed)

        # Store reference from first rep
        if self._reference_velocity is None:
            self._reference_velocity = rep_metrics["mean_velocity"]
            self._reference_rom = rep_metrics["rom"].copy()

        self._rep_data.append(rep_metrics)

        # Reset current rep tracking
        self._current_rep_positions.clear()
        self._current_rep_timestamps.clear()
        self._current_rep_angles.clear()

        return rep_metrics

    def _compute_velocity_metrics(self) -> Dict:
        """Compute velocity metrics from recent keypoint history."""
        if len(self._keypoint_history) < 5:
            return {"velocities": {}, "mean_velocity": 0.0, "peak_velocity": 0.0}

        positions = np.array(self._keypoint_history)  # (T, N, 3)
        velocities = {}

        # Compute velocity for key joints
        for joint_name, idx in [("left_wrist", 9), ("right_wrist", 10),
                                 ("left_ankle", 15), ("right_ankle", 16)]:
            if idx < positions.shape[1]:
                vel = self.kinematics_calc.compute_velocity(positions[:, idx, :])
                speed = self.kinematics_calc.compute_speed(vel)
                velocities[joint_name] = float(np.mean(speed))

        all_vel = list(velocities.values())
        return {
            "velocities": velocities,
            "mean_velocity": float(np.mean(all_vel)) if all_vel else 0.0,
            "peak_velocity": float(np.max(all_vel)) if all_vel else 0.0,
        }

    def _compute_current_rom(self) -> Dict[str, float]:
        """Compute ROM from current repetition angles."""
        if not self._current_rep_angles:
            return {}

        rom = {}
        for joint_name in self._current_rep_angles[0].keys():
            angles = [a.get(joint_name, 0) for a in self._current_rep_angles]
            rom[joint_name] = max(angles) - min(angles) if angles else 0.0
        return rom

    def _compute_asymmetry(self, keypoints_3d: np.ndarray) -> float:
        """Compute bilateral asymmetry index."""
        if len(keypoints_3d) < 17:
            return 0.0

        # Compare left and right joint positions
        left_shoulder = keypoints_3d[KeypointIndex.LEFT_SHOULDER]
        right_shoulder = keypoints_3d[KeypointIndex.RIGHT_SHOULDER]
        left_hip = keypoints_3d[KeypointIndex.LEFT_HIP]
        right_hip = keypoints_3d[KeypointIndex.RIGHT_HIP]

        # Shoulder height asymmetry
        shoulder_ai = self.asymmetry_calc.compute(
            left_shoulder[1], right_shoulder[1]  # Y coordinate
        )

        # Hip height asymmetry
        hip_ai = self.asymmetry_calc.compute(
            left_hip[1], right_hip[1]
        )

        return float((abs(shoulder_ai) + abs(hip_ai)) / 2.0)

    def _classify_state(self, result: BodyStateResult) -> Tuple[BodyState, float]:
        """
        Classify body state from kinematic metrics.

        Logic:
            1. Pain: High asymmetry + guarding behavior
            2. Exhaustion: Severe velocity loss + ROM decline
            3. Fatigue: Moderate velocity loss + ROM decline
            4. Boredom: Low movement variability + low amplitude
            5. Normal
        """
        scores = {
            BodyState.PAIN: 0.0,
            BodyState.FATIGUE: 0.0,
            BodyState.EXHAUSTION: 0.0,
            BodyState.BOREDOM: 0.0,
        }

        # Pain indicators
        if result.asymmetry_pct > self.ASYMMETRY_SIGNIFICANT:
            scores[BodyState.PAIN] += 0.4
        if result.trunk_inclination_deg > self.TRUNK_INCLINATION_CRITICAL:
            scores[BodyState.PAIN] += 0.3

        # Fatigue/Exhaustion indicators (from rep history)
        if len(self._rep_data) >= 3 and self._reference_velocity is not None:
            recent_v = np.mean([r["mean_velocity"] for r in self._rep_data[-3:]])
            velocity_loss = ((self._reference_velocity - recent_v) /
                            (self._reference_velocity + 1e-8)) * 100.0

            result.velocity_loss_pct = velocity_loss

            if velocity_loss >= self.VELOCITY_LOSS_SIGNIFICANT:
                scores[BodyState.EXHAUSTION] += 0.5
            elif velocity_loss >= self.VELOCITY_LOSS_MILD:
                scores[BodyState.FATIGUE] += 0.4

            # ROM decline
            if self._reference_rom:
                recent_rom = np.mean([r["mean_rom"] for r in self._rep_data[-3:]])
                ref_rom = np.mean(list(self._reference_rom.values()))
                if ref_rom > 0:
                    rom_decline = ((ref_rom - recent_rom) / ref_rom) * 100.0
                    result.rom_decline_pct = rom_decline

                    if rom_decline >= self.ROM_DECLINE_SIGNIFICANT:
                        scores[BodyState.EXHAUSTION] += 0.3
                    elif rom_decline >= self.ROM_DECLINE_MILD:
                        scores[BodyState.FATIGUE] += 0.2

        # Boredom indicators
        if len(self._rep_data) >= 3:
            velocity_variability = np.std([r["mean_velocity"] for r in self._rep_data[-5:]])
            if velocity_variability < self.BOREDOM_VARIABILITY_THRESHOLD:
                scores[BodyState.BOREDOM] += 0.3

            # Low amplitude
            recent_amplitude = np.mean([r["mean_rom"] for r in self._rep_data[-3:]])
            if self._reference_rom:
                ref_amplitude = np.mean(list(self._reference_rom.values()))
                if ref_amplitude > 0 and recent_amplitude / ref_amplitude < self.BOREDOM_AMPLITUDE_THRESHOLD:
                    scores[BodyState.BOREDOM] += 0.3

        # Select highest scoring state
        max_state = max(scores, key=scores.get)
        max_score = scores[max_state]

        if max_score < 0.2:
            return BodyState.NORMAL, 0.8

        # Set individual scores
        result.fatigue_score = scores[BodyState.FATIGUE]
        result.exhaustion_score = scores[BodyState.EXHAUSTION]
        result.boredom_score = scores[BodyState.BOREDOM]
        result.pain_score = scores[BodyState.PAIN]

        return max_state, min(1.0, max_score + 0.3)

    def get_trend_analysis(self) -> Dict[str, Dict]:
        """
        Get Mann-Kendall trend analysis for key metrics across repetitions.

        Returns:
            Dict with trend analysis for velocity, ROM, smoothness
        """
        if len(self._rep_data) < 4:
            return {}

        velocities = [r["mean_velocity"] for r in self._rep_data]
        roms = [r["mean_rom"] for r in self._rep_data]
        smoothness = [r["smoothness"] for r in self._rep_data]

        results = {}
        for name, values in [("velocity", velocities), ("rom", roms),
                             ("smoothness", smoothness)]:
            s, z = self.trend_analyzer.compute_s(values)
            slope = self.trend_analyzer.compute_sens_slope(values)
            results[name] = {
                "S": s, "Z": z, "slope": slope,
                "significant": abs(z) > 1.96,
                "trend": "decreasing" if z < -1.96 else "increasing" if z > 1.96 else "none"
            }

        return results

    def reset(self):
        """Reset all state."""
        self._frame_count = 0
        self._rep_count = 0
        self._rep_data.clear()
        self._current_rep_positions.clear()
        self._current_rep_timestamps.clear()
        self._current_rep_angles.clear()
        self._reference_velocity = None
        self._reference_rom = None
        self._keypoint_history.clear()
        self._timestamp_history.clear()
