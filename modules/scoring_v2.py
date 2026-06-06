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

    WEIGHTS = {
        "rom": 0.25,
        "stability": 0.15,
        "flow": 0.20,
        "symmetry": 0.15,
        "compensation": 0.15,
        "smoothness": 0.10,
    }

    def __init__(self):
        """Initialize enhanced scorer."""
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
        if target <= 0:
            return 100.0
        max_achieved = float(np.max(angles))
        return min(100.0, (max_achieved / target) * 100)

    def _compute_stability(self, angles: np.ndarray, hold_indices: Optional[np.ndarray] = None) -> float:
        if hold_indices is not None and len(hold_indices) > 0:
            hold_angles = angles[hold_indices]
        else:
            # Use middle 40% as approximation of HOLD phase
            n = len(angles)
            start, end = int(n * 0.3), int(n * 0.7)
            hold_angles = angles[start:end]

        if len(hold_angles) < 3:
            return 80.0

        std = float(np.std(hold_angles))
        return max(0, min(100, 100 - std * 5))

    def _compute_flow(self, angles: np.ndarray, timestamps: np.ndarray) -> float:
        if len(angles) < 5:
            return 70.0

        dt = np.diff(timestamps)
        dt = np.where(dt < 1e-6, 1e-6, dt)
        velocity = np.diff(angles) / dt

        if len(velocity) < 3:
            return 70.0

        acceleration = np.diff(velocity) / dt[:-1]
        accel_std = float(np.std(acceleration))

        return max(0, min(100, 100 - accel_std * 0.1))

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
