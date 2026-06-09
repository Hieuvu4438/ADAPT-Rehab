"""
Enhanced Scoring System v2.

6-dimension scoring with SPARC smoothness metric:

| Dimension     | Weight | Metric                          |
|---------------|--------|---------------------------------|
| ROM           | 25%    | Max angle vs target             |
| Stability     | 15%    | Angle variability in HOLD phase |
| Flow          | 20%    | DTW similarity or velocity      |
| Symmetry      | 15%    | Left-right angle difference     |
| Compensation  | 15%    | Compensatory movement detection |
| Smoothness    | 10%    | SPARC metric (NEW)              |

Usage:
    scorer = EnhancedScorer()
    scorer.start_session("arm_raise")
    for rep in reps:
        score = scorer.score_rep(angles, timestamps, target_angle=150)
        print(f"Total: {score.total_score:.1f}")
    report = scorer.get_session_report()
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np

from core.smoothness import SmoothnessAnalyzer, SmoothnessResult
from modules.compensation import CompensationDetector, CompensationResult
from modules.fatigue import FatigueAnalyzer, FatigueResult, FatigueLevel


@dataclass
class RepScoreV2:
    """Score for a single repetition."""
    rep_number: int = 0
    rom_score: float = 0.0
    stability_score: float = 0.0
    flow_score: float = 0.0
    symmetry_score: float = 0.0
    compensation_score: float = 100.0
    smoothness_score: float = 0.0
    total_score: float = 0.0
    fatigue: FatigueLevel = FatigueLevel.FRESH
    fatigue_score: float = 0.0
    compensation_types: List[str] = field(default_factory=list)
    notes: str = ""

    def to_dict(self) -> dict:
        return {
            "rep_number": self.rep_number,
            "rom_score": round(self.rom_score, 1),
            "stability_score": round(self.stability_score, 1),
            "flow_score": round(self.flow_score, 1),
            "symmetry_score": round(self.symmetry_score, 1),
            "compensation_score": round(self.compensation_score, 1),
            "smoothness_score": round(self.smoothness_score, 1),
            "total_score": round(self.total_score, 1),
            "fatigue": self.fatigue.name,
            "compensation_types": self.compensation_types,
            "notes": self.notes,
        }


@dataclass
class SessionReportV2:
    """Summary report for an exercise session."""
    session_id: str = ""
    exercise_name: str = ""
    total_reps: int = 0
    average_scores: Dict[str, float] = field(default_factory=dict)
    rep_scores: List[dict] = field(default_factory=list)
    fatigue_trend: Dict = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)


class EnhancedScorer:
    """
    Enhanced 6-dimension scorer.

    Integrates SPARC smoothness metric alongside traditional
    ROM, stability, flow, symmetry, and compensation scores.

    Example:
        >>> scorer = EnhancedScorer()
        >>> scorer.start_session("arm_raise")
        >>> # For each rep:
        >>> score = scorer.score_rep(
        ...     angles=np.array([10, 30, 60, 90, 120, 150, 120, 90, 60, 30]),
        ...     timestamps=np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]),
        ...     target_angle=150.0,
        ... )
        >>> print(f"Rep {score.rep_number}: {score.total_score:.1f}/100")
    """

    DEFAULT_WEIGHTS = {
        "rom": 0.25,
        "stability": 0.15,
        "flow": 0.20,
        "symmetry": 0.15,
        "compensation": 0.15,
        "smoothness": 0.10,
    }

    def __init__(self, weights: Optional[Dict[str, float]] = None):
        """Initialize enhanced scorer.

        Args:
            weights: Optional custom weights dict. If None, uses DEFAULT_WEIGHTS.
                     Weights should sum to 1.0.
        """
        self.WEIGHTS = weights if weights is not None else self.DEFAULT_WEIGHTS.copy()
        self._smoothness_analyzer = SmoothnessAnalyzer()
        self._compensation_detector = CompensationDetector()
        self._fatigue_analyzer = FatigueAnalyzer()
        self._rep_scores: List[RepScoreV2] = []
        self._exercise_name: str = ""
        self._session_id: str = ""

    def start_session(self, exercise_name: str, session_id: str = "") -> None:
        """Start a new scoring session."""
        self._exercise_name = exercise_name
        self._session_id = session_id or f"session_{len(self._rep_scores)}"
        self._rep_scores = []
        self._fatigue_analyzer.reset()

    def score_rep(
        self,
        angles: np.ndarray,
        timestamps: np.ndarray,
        target_angle: float,
        left_angles: Optional[np.ndarray] = None,
        right_angles: Optional[np.ndarray] = None,
        pose_sequence: Optional[List[np.ndarray]] = None,
        dtw_score: Optional[float] = None,
        hold_phase_indices: Optional[np.ndarray] = None,
    ) -> RepScoreV2:
        """
        Score a single repetition.

        Args:
            angles: Joint angle sequence for the rep (degrees).
            timestamps: Timestamp sequence (seconds).
            target_angle: Target angle for the exercise (degrees).
            left_angles: Left-side angles for symmetry computation.
            right_angles: Right-side angles for symmetry computation.
            pose_sequence: Pose landmarks for compensation detection.
            dtw_score: Pre-computed DTW similarity score (0-100).
            hold_phase_indices: Indices of frames in HOLD phase.

        Returns:
            RepScoreV2 with all dimension scores.
        """
        rep_num = len(self._rep_scores) + 1

        # 1. ROM Score
        rom_score = self._compute_rom(angles, target_angle)

        # 2. Stability Score
        stability_score = self._compute_stability(angles, hold_phase_indices)

        # 3. Flow Score
        if dtw_score is not None:
            flow_score = dtw_score
        else:
            flow_score = self._compute_flow(angles, timestamps)

        # 4. Symmetry Score
        symmetry_score = self._compute_symmetry(left_angles, right_angles)

        # 5. Compensation Score
        compensation_score, comp_types = self._compute_compensation(pose_sequence)

        # 6. Smoothness Score (SPARC)
        smoothness_result = self._smoothness_analyzer.analyze(angles, timestamps)
        smoothness_score = smoothness_result.smoothness_score

        # 7. Fatigue Analysis
        rep_data = {
            "jerk_value": self._compute_jerk(angles, timestamps),
            "max_angle": float(np.max(angles)),
            "mean_velocity": float(np.mean(np.abs(np.diff(angles)))) if len(angles) > 1 else 0,
            "angle_std": float(np.std(angles)),
        }

        if rep_num == 1:
            self._fatigue_analyzer.set_baseline(rep_data)

        fatigue_result = self._fatigue_analyzer.analyze(rep_data)

        # Total Score (weighted combination)
        total = (
            self.WEIGHTS["rom"] * rom_score +
            self.WEIGHTS["stability"] * stability_score +
            self.WEIGHTS["flow"] * flow_score +
            self.WEIGHTS["symmetry"] * symmetry_score +
            self.WEIGHTS["compensation"] * compensation_score +
            self.WEIGHTS["smoothness"] * smoothness_score
        )

        # Generate notes
        notes = []
        if fatigue_result.level.value >= FatigueLevel.MODERATE.value:
            notes.append(f"Mệt mỏi: {fatigue_result.level.name}")
        if comp_types:
            notes.append(f"Bù trừ: {', '.join(comp_types)}")

        score = RepScoreV2(
            rep_number=rep_num,
            rom_score=rom_score,
            stability_score=stability_score,
            flow_score=flow_score,
            symmetry_score=symmetry_score,
            compensation_score=compensation_score,
            smoothness_score=smoothness_score,
            total_score=total,
            fatigue=fatigue_result.level,
            fatigue_score=fatigue_result.composite_score,
            compensation_types=comp_types,
            notes="; ".join(notes),
        )

        self._rep_scores.append(score)
        return score

    def get_session_report(self) -> SessionReportV2:
        """Generate session summary report."""
        if not self._rep_scores:
            return SessionReportV2()

        avg_scores = {}
        for dim in ["rom", "stability", "flow", "symmetry", "compensation", "smoothness"]:
            scores = [getattr(s, f"{dim}_score") for s in self._rep_scores]
            avg_scores[dim] = float(np.mean(scores))
        avg_scores["total"] = float(np.mean([s.total_score for s in self._rep_scores]))

        recommendations = self._generate_recommendations(avg_scores)

        return SessionReportV2(
            session_id=self._session_id,
            exercise_name=self._exercise_name,
            total_reps=len(self._rep_scores),
            average_scores=avg_scores,
            rep_scores=[s.to_dict() for s in self._rep_scores],
            fatigue_trend=self._fatigue_analyzer.get_trend(),
            recommendations=recommendations,
        )

    def _compute_rom(self, angles: np.ndarray, target: float) -> float:
        """ROM Score with 3 sub-components (matching v1 logic).

        40% max angle achievement + 30% hold time near target + 30% peak quality.
        """
        if target <= 0:
            return 100.0
        if len(angles) < 5:
            return 0.0

        # 1. Max angle score (40%)
        max_achieved = float(np.max(angles))
        max_score = min(100.0, (max_achieved / target) * 100)

        # 2. Hold time score — frames above 80% of target (30%)
        threshold = target * 0.8
        frames_above = int(np.sum(angles >= threshold))
        min_frames_required = max(3, len(angles) * 0.1)
        hold_ratio = min(1.0, frames_above / min_frames_required)
        hold_score = hold_ratio * 100

        # 3. Peak quality score — stability around the peak (30%)
        peak_idx = int(np.argmax(angles))
        window = max(3, len(angles) // 10)
        start_idx = max(0, peak_idx - window)
        end_idx = min(len(angles), peak_idx + window + 1)
        peak_region = angles[start_idx:end_idx]

        if len(peak_region) >= 3:
            peak_std = float(np.std(peak_region))
            # Use divisor of 3 instead of 5 for elderly-friendly scoring
            # A std of 33 degrees gives score 0 (vs 20 degrees with divisor 5)
            peak_quality_score = max(0.0, 100.0 - peak_std * 3)
        else:
            peak_quality_score = 50.0

        return min(100.0, max(0.0,
            0.40 * max_score + 0.30 * hold_score + 0.30 * peak_quality_score
        ))

    def _compute_stability(self, angles: np.ndarray, hold_indices: Optional[np.ndarray] = None) -> float:
        """Stability Score with 3 sub-components (matching v1 logic).

        50% std + 30% oscillation count + 20% drift detection.
        """
        if hold_indices is not None and len(hold_indices) > 0:
            hold_angles = angles[hold_indices]
        else:
            # Use middle 40% as approximation of HOLD phase
            n = len(angles)
            start, end = int(n * 0.3), int(n * 0.7)
            hold_angles = angles[start:end]

        if len(hold_angles) < 3:
            return 80.0

        hold_arr = np.asarray(hold_angles, dtype=np.float64)

        # 1. Standard deviation score (50%)
        std = float(np.std(hold_arr))
        std_score = max(0.0, 100.0 - std * 10)

        # 2. Oscillation count — frames exceeding 3° from HOLD mean (30%)
        mean_angle = float(np.mean(hold_arr))
        oscillation_threshold = 3.0
        deviations = np.abs(hold_arr - mean_angle)
        crossings = int(np.sum(deviations > oscillation_threshold))
        max_allowed = max(1, len(hold_arr) * 0.2)
        oscillation_ratio = min(1.0, crossings / max_allowed)
        oscillation_score = (1 - oscillation_ratio) * 100

        # 3. Drift score — angle drop between first and second half (20%)
        if len(hold_arr) >= 3:
            mid = len(hold_arr) // 2
            first_half_mean = float(np.mean(hold_arr[:mid]))
            second_half_mean = float(np.mean(hold_arr[mid:]))
            drift = first_half_mean - second_half_mean  # positive = angle dropping
            drift_penalty = min(1.0, max(0.0, drift) / 5.0)
            drift_score = (1 - drift_penalty) * 100
        else:
            drift_score = 100.0

        return min(100.0, max(0.0,
            0.50 * std_score + 0.30 * oscillation_score + 0.20 * drift_score
        ))

    def _compute_flow(self, angles: np.ndarray, timestamps: np.ndarray) -> float:
        """Flow Score with 3 sub-components (matching v1 logic).

        40% velocity smoothness + 30% continuity + 30% direction consistency.
        """
        if len(angles) < 5:
            return 70.0

        dt = np.diff(timestamps)
        dt = np.where(dt < 1e-6, 1e-6, dt)
        velocity = np.diff(angles) / dt

        if len(velocity) < 3:
            return 70.0

        # 1. Velocity smoothness — acceleration std (40%)
        acceleration = np.diff(velocity) / dt[:-1]
        accel_std = float(np.std(acceleration))
        smoothness_score = max(0.0, 100.0 - accel_std * 0.2)

        # 2. Continuity — no sudden jumps > 15°/frame (30%)
        angle_diffs = np.abs(np.diff(angles))
        max_allowed_jump = 15.0
        jumps = int(np.sum(angle_diffs > max_allowed_jump))
        jump_ratio = jumps / len(angle_diffs) if len(angle_diffs) > 0 else 0
        continuity_score = (1 - min(1.0, jump_ratio * 5)) * 100

        # 3. Direction consistency — sign-change ratio in velocity (30%)
        if len(velocity) >= 5:
            sign_changes = int(np.sum(np.abs(np.diff(np.sign(velocity))) > 0))
            max_changes = len(velocity) * 0.3
            direction_ratio = min(1.0, sign_changes / max(1, max_changes))
            direction_score = (1 - direction_ratio) * 100
        else:
            direction_score = 70.0

        return min(100.0, max(0.0,
            0.40 * smoothness_score + 0.30 * continuity_score + 0.30 * direction_score
        ))

    def _compute_symmetry(self, left: Optional[np.ndarray], right: Optional[np.ndarray]) -> float:
        if left is None or right is None:
            return 85.0  # Default when no symmetry data

        min_len = min(len(left), len(right))
        if min_len < 3:
            return 85.0

        diff = float(np.mean(np.abs(left[:min_len] - right[:min_len])))
        return max(0, min(100, 100 - diff * 4))

    def _compute_compensation(self, pose_sequence: Optional[List[np.ndarray]]) -> Tuple[float, List[str]]:
        if pose_sequence is None:
            return 100.0, []

        result = self._compensation_detector.analyze(pose_sequence)
        return result.score, result.detected_types

    def _compute_jerk(self, angles: np.ndarray, timestamps: np.ndarray) -> float:
        if len(angles) < 4:
            return 0.0

        dt = np.diff(timestamps)
        dt = np.where(dt < 1e-6, 1e-6, dt)

        velocity = np.diff(angles) / dt
        acceleration = np.diff(velocity) / dt[:-1]
        jerk = np.diff(acceleration) / dt[:-2]

        total_time = float(timestamps[-1] - timestamps[0])
        if total_time < 1e-6:
            return 0.0

        return float(np.sum(jerk ** 2) / total_time)

    def _generate_recommendations(self, avg_scores: Dict[str, float]) -> List[str]:
        recommendations = []

        if avg_scores.get("rom", 100) < 70:
            recommendations.append("Bác chưa đạt góc mục tiêu. Cố gắng thêm một chút nhé!")
        elif avg_scores.get("rom", 0) >= 95:
            recommendations.append("Tuyệt vời! Bác đạt góc mục tiêu rất tốt!")

        if avg_scores.get("stability", 100) < 60:
            recommendations.append("Khi giữ tư thế, bác cố giữ yên hơn nhé. Thở đều và tập trung.")

        if avg_scores.get("compensation", 100) < 70:
            recommendations.append("Bác có xu hướng bù trừ khi tập. Cố gắng giữ tư thế chuẩn hơn nhé!")

        if avg_scores.get("smoothness", 100) < 50:
            recommendations.append("Chuyển động chưa mượt. Bác hãy tập chậm hơn và đều hơn nhé!")

        if not recommendations:
            recommendations.append("Buổi tập tốt! Hẹn gặp lại bác vào buổi tập sau nhé!")

        return recommendations
