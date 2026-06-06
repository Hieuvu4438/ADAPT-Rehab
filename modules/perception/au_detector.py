"""
Action Unit Detector using Face Mesh landmarks.

Computes FACS Action Units from MediaPipe Face Mesh landmarks
without requiring external AU detection libraries (py-feat).

Computes PSPI (Prkachin-Solomon Pain Intensity) score:
    PSPI = AU4 + max(AU6, AU7) + max(AU9, AU10) + max(AU20, AU25, AU26)

Usage:
    detector = ActionUnitDetector()
    result = detector.detect(face_landmarks)
    if result.is_valid:
        print(f"Pain score: {result.pain_score:.1f}")
        print(f"AU4 (brow lowerer): {result.au_activations['AU4']:.2f}")
"""

from dataclasses import dataclass, field
from typing import Dict, Optional
import numpy as np

from .face_detector import FaceLandmarkIndex


@dataclass
class AUResult:
    """Result of Action Unit detection."""
    au_activations: Dict[str, float] = field(default_factory=dict)
    pain_score: float = 0.0  # PSPI score (0-16)
    pain_level: str = "NONE"  # NONE, MILD, MODERATE, SEVERE
    is_valid: bool = False
    error_message: str = ""


class ActionUnitDetector:
    """
    Action Unit detector using Face Mesh landmarks.

    Computes FACS Action Units from landmark distances and ratios.
    No external AU detection library required.

    AU mappings:
        AU4:  Brow Lowerer (frowning)
        AU6:  Cheek Raiser (orbital tightening)
        AU7:  Lid Tightener (eye squinting)
        AU9:  Nose Wrinkler
        AU10: Upper Lip Raiser
        AU43: Eye Closure

    Example:
        >>> detector = ActionUnitDetector()
        >>> result = detector.detect(face_landmarks)
        >>> if result.is_valid:
        ...     print(f"PSPI: {result.pain_score:.1f}")
    """

    # AU detection thresholds (tuned for general population)
    AU_THRESHOLDS = {
        "AU4": 0.15,    # Brow lowering threshold
        "AU6": 0.12,    # Cheek raising threshold
        "AU7": 0.15,    # Lid tightening threshold
        "AU9": 0.10,    # Nose wrinkling threshold
        "AU10": 0.12,   # Upper lip raising threshold
        "AU43": 0.40,   # Eye closure threshold
    }

    # PSPI pain level thresholds
    PAIN_THRESHOLDS = {
        "NONE": 0,
        "MILD": 4,
        "MODERATE": 8,
        "SEVERE": 12,
    }

    def __init__(self, use_baseline: bool = True):
        """
        Initialize AU detector.

        Args:
            use_baseline: If True, calibrate from neutral face first.
        """
        self._use_baseline = use_baseline
        self._baseline: Optional[Dict[str, float]] = None
        self._is_calibrated = False

    def set_baseline(self, landmarks: np.ndarray) -> None:
        """
        Set baseline measurements from neutral face.

        Args:
            landmarks: Face landmarks (468, 3) from neutral expression.
        """
        self._baseline = self._compute_raw_measurements(landmarks)
        self._is_calibrated = True
        print("[AU] Baseline calibrated from neutral face")

    def detect(self, landmarks: np.ndarray) -> AUResult:
        """
        Detect Action Units from face landmarks.

        Args:
            landmarks: Face landmarks array, shape (468, 3).

        Returns:
            AUResult with AU activations and PSPI score.
        """
        if landmarks is None or len(landmarks) < 468:
            return AUResult(error_message="Invalid landmarks", is_valid=False)

        try:
            # Compute raw measurements
            raw = self._compute_raw_measurements(landmarks)

            # If no baseline, use defaults
            if self._baseline is None and self._use_baseline:
                self._baseline = self._get_default_baseline()

            # Compute AU activations
            aus = self._compute_au_activations(raw)

            # Compute PSPI
            pspi = self._compute_pspi(aus)

            # Classify pain level
            pain_level = self._classify_pain(pspi)

            return AUResult(
                au_activations=aus,
                pain_score=pspi,
                pain_level=pain_level,
                is_valid=True,
            )

        except Exception as e:
            return AUResult(error_message=str(e), is_valid=False)

    def _compute_raw_measurements(self, lms: np.ndarray) -> Dict[str, float]:
        """Compute raw facial measurements from landmarks."""
        measurements = {}

        try:
            # Face height for normalization
            face_top = lms[FaceLandmarkIndex.FACE_TOP]
            face_bottom = lms[FaceLandmarkIndex.FACE_BOTTOM]
            face_height = np.linalg.norm(face_top[:2] - face_bottom[:2])
            if face_height < 1e-6:
                face_height = 1.0

            # Eye Aspect Ratio (EAR) - for AU6, AU7, AU43
            measurements["left_ear"] = self._eye_aspect_ratio(lms, "left")
            measurements["right_ear"] = self._eye_aspect_ratio(lms, "right")
            measurements["eye_aspect_ratio"] = (measurements["left_ear"] + measurements["right_ear"]) / 2

            # Eyebrow position - for AU4
            measurements["left_brow"] = self._eyebrow_position(lms, "left", face_height)
            measurements["right_brow"] = self._eyebrow_position(lms, "right", face_height)
            measurements["eyebrow_position"] = (measurements["left_brow"] + measurements["right_brow"]) / 2

            # Nose wrinkle - for AU9
            measurements["nose_wrinkle"] = self._nose_wrinkle(lms, face_height)

            # Upper lip position - for AU10
            measurements["upper_lip_raise"] = self._upper_lip_position(lms, face_height)

            # Mouth aspect ratio
            measurements["mouth_aspect_ratio"] = self._mouth_aspect_ratio(lms)

        except (IndexError, ValueError):
            pass

        return measurements

    def _eye_aspect_ratio(self, lms: np.ndarray, side: str) -> float:
        """
        Compute Eye Aspect Ratio (EAR).

        EAR = (|p2-p6| + |p3-p5|) / (2 * |p1-p4|)
        Low EAR = eye closing/squinting
        """
        if side == "left":
            top = lms[FaceLandmarkIndex.LEFT_EYE_TOP]
            bottom = lms[FaceLandmarkIndex.LEFT_EYE_BOTTOM]
            inner = lms[FaceLandmarkIndex.LEFT_EYE_INNER]
            outer = lms[FaceLandmarkIndex.LEFT_EYE_OUTER]
        else:
            top = lms[FaceLandmarkIndex.RIGHT_EYE_TOP]
            bottom = lms[FaceLandmarkIndex.RIGHT_EYE_BOTTOM]
            inner = lms[FaceLandmarkIndex.RIGHT_EYE_INNER]
            outer = lms[FaceLandmarkIndex.RIGHT_EYE_OUTER]

        vertical = np.linalg.norm(top[:2] - bottom[:2])
        horizontal = np.linalg.norm(inner[:2] - outer[:2])

        if horizontal < 1e-6:
            return 0.3  # Default

        return vertical / horizontal

    def _eyebrow_position(self, lms: np.ndarray, side: str, face_height: float) -> float:
        """
        Compute eyebrow position relative to eye.

        Lower value = brow lowered (frowning) = AU4
        """
        if side == "left":
            brow = lms[FaceLandmarkIndex.LEFT_EYEBROW_MIDDLE]
            eye = lms[FaceLandmarkIndex.LEFT_EYE_TOP]
        else:
            brow = lms[FaceLandmarkIndex.RIGHT_EYEBROW_MIDDLE]
            eye = lms[FaceLandmarkIndex.RIGHT_EYE_TOP]

        dist = np.linalg.norm(brow[:2] - eye[:2])
        return dist / face_height

    def _nose_wrinkle(self, lms: np.ndarray, face_height: float) -> float:
        """Compute nose wrinkle ratio (AU9)."""
        nose_tip = lms[FaceLandmarkIndex.NOSE_TIP]
        nose_bridge = lms[FaceLandmarkIndex.NOSE_BRIDGE]
        nose_length = np.linalg.norm(nose_tip[:2] - nose_bridge[:2])
        return nose_length / face_height

    def _upper_lip_position(self, lms: np.ndarray, face_height: float) -> float:
        """Compute upper lip position (AU10)."""
        upper_lip = lms[FaceLandmarkIndex.UPPER_LIP_TOP]
        nose_tip = lms[FaceLandmarkIndex.NOSE_TIP]
        dist = np.linalg.norm(upper_lip[:2] - nose_tip[:2])
        return dist / face_height

    def _mouth_aspect_ratio(self, lms: np.ndarray) -> float:
        """Compute mouth aspect ratio."""
        top = lms[FaceLandmarkIndex.UPPER_LIP_BOTTOM]
        bottom = lms[FaceLandmarkIndex.LOWER_LIP_TOP]
        left = lms[FaceLandmarkIndex.MOUTH_LEFT]
        right = lms[FaceLandmarkIndex.MOUTH_RIGHT]

        vertical = np.linalg.norm(top[:2] - bottom[:2])
        horizontal = np.linalg.norm(left[:2] - right[:2])

        if horizontal < 1e-6:
            return 0.2

        return vertical / horizontal

    def _compute_au_activations(self, raw: Dict[str, float]) -> Dict[str, float]:
        """Compute AU activations from raw measurements vs baseline."""
        if not self._baseline:
            return {au: 0.0 for au in self.AU_THRESHOLDS}

        activations = {}

        # AU4: Brow Lowerer - decrease in eyebrow position
        base_brow = self._baseline.get("eyebrow_position", 0.08)
        curr_brow = raw.get("eyebrow_position", 0.08)
        if base_brow > 1e-6:
            au4 = max(0, (base_brow - curr_brow) / base_brow)
            activations["AU4"] = min(1.0, au4)
        else:
            activations["AU4"] = 0.0

        # AU6/AU7: Cheek Raiser / Lid Tightener - decrease in EAR
        base_ear = self._baseline.get("eye_aspect_ratio", 0.28)
        curr_ear = raw.get("eye_aspect_ratio", 0.28)
        if base_ear > 1e-6:
            eye_change = max(0, (base_ear - curr_ear) / base_ear)
            activations["AU6"] = min(1.0, eye_change * 0.8)
            activations["AU7"] = min(1.0, eye_change)
        else:
            activations["AU6"] = 0.0
            activations["AU7"] = 0.0

        # AU9: Nose Wrinkler
        base_nose = self._baseline.get("nose_wrinkle", 0.12)
        curr_nose = raw.get("nose_wrinkle", 0.12)
        if base_nose > 1e-6:
            activations["AU9"] = min(1.0, max(0, abs(base_nose - curr_nose) / base_nose))
        else:
            activations["AU9"] = 0.0

        # AU10: Upper Lip Raiser
        base_lip = self._baseline.get("upper_lip_raise", 0.10)
        curr_lip = raw.get("upper_lip_raise", 0.10)
        if base_lip > 1e-6:
            activations["AU10"] = min(1.0, max(0, (base_lip - curr_lip) / base_lip))
        else:
            activations["AU10"] = 0.0

        # AU43: Eye Closure
        if base_ear > 1e-6:
            activations["AU43"] = min(1.0, max(0, (base_ear - curr_ear) / base_ear))
        else:
            activations["AU43"] = 0.0

        return activations

    def _compute_pspi(self, aus: Dict[str, float]) -> float:
        """
        Compute Prkachin-Solomon Pain Intensity (PSPI) score.

        PSPI = AU4 + max(AU6, AU7) + max(AU9, AU10) + max(AU20, AU25, AU26)

        Since we don't detect AU20/25/26, we use a simplified version:
        PSPI = AU4 + max(AU6, AU7) + max(AU9, AU10) + AU43
        """
        au4 = aus.get("AU4", 0)
        au6 = aus.get("AU6", 0)
        au7 = aus.get("AU7", 0)
        au9 = aus.get("AU9", 0)
        au10 = aus.get("AU10", 0)
        au43 = aus.get("AU43", 0)

        pspi = au4 + max(au6, au7) + max(au9, au10) + au43
        return float(pspi)

    def _classify_pain(self, pspi: float) -> str:
        """Classify pain level from PSPI score."""
        if pspi >= self.PAIN_THRESHOLDS["SEVERE"]:
            return "SEVERE"
        elif pspi >= self.PAIN_THRESHOLDS["MODERATE"]:
            return "MODERATE"
        elif pspi >= self.PAIN_THRESHOLDS["MILD"]:
            return "MILD"
        return "NONE"

    def _get_default_baseline(self) -> Dict[str, float]:
        """Get default baseline measurements."""
        return {
            "eye_aspect_ratio": 0.28,
            "left_ear": 0.28,
            "right_ear": 0.28,
            "eyebrow_position": 0.08,
            "left_brow": 0.08,
            "right_brow": 0.08,
            "nose_wrinkle": 0.12,
            "upper_lip_raise": 0.10,
            "mouth_aspect_ratio": 0.15,
        }

    def reset(self) -> None:
        """Reset detector state."""
        self._baseline = None
        self._is_calibrated = False
