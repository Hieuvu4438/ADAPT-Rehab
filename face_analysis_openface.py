"""
OpenFace Analyzer for ADAPT-Rehab.

Provides Action Unit (AU) detection, emotion classification, and
gaze estimation for facial behavior analysis.

Wraps **OpenFace 3.0** (CMU MultiComp Lab) — see:
    "OpenFace 3.0: A Multitask Toolkit for Facial Behavior Analysis"
    arXiv:2506.02891, June 2025

Architecture:
    Frame → MediaPipe Face Mesh (468 landmarks, face detection)
          → OpenFace 3.0 MTL (EfficientNet-B0 + GNN, AU + emotion + gaze)
          → FacialStateDetector (clinical AU formulas)
          → OpenFaceResult

**Primary path**: OpenFace 3.0 MTL model (``openface.multitask_model.MultitaskPredictor``)
takes a face crop and outputs 8 AU scores (GNN), 8 emotion logits, and
gaze (yaw, pitch). Requires ``openface-test`` pip package and
``MTL_backbone.pth`` weights (auto-downloaded from HuggingFace).

**Fallback**: If OpenFace 3.0 is not installed or fails, falls back to
a calibrated geometric AU estimator using MediaPipe 468-landmark distances.
This is a rule-based heuristic, not a neural-network prediction.

The eight AUs produced (FACS codes):

- AU1: Inner Brow Raiser
- AU2: Outer Brow Raiser
- AU4: Brow Lowerer
- AU6: Cheek Raiser
- AU9: Nose Wrinkler
- AU12: Lip Corner Puller (smile)
- AU25: Lips Part
- AU26: Jaw Drop

These feed into ``FacialStateDetector`` for clinical state detection
(PSPI pain, PERCLOS fatigue, Engagement boredom, etc.).

Version: 5.0.0
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import logging
import os

from .face_state import (
    FacialStateDetector,
    FacialStateResult,
    FacialState,
    AUData,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Data Types (unchanged - public API preserved)
# ============================================================================

@dataclass
class OpenFaceResult:
    """Complete result from OpenFace analysis."""
    # Face detection
    face_bbox: Optional[np.ndarray] = None      # (4,) x1,y1,x2,y2
    face_confidence: float = 0.0

    # Landmarks
    landmarks_68: Optional[np.ndarray] = None    # (68, 2) 2D landmarks
    landmarks_468: Optional[np.ndarray] = None   # (468, 3) from MediaPipe

    # Action Units (8 AUs)
    au_intensities: Dict[str, float] = None      # AU1,AU2,AU4,AU6,AU9,AU12,AU25,AU26

    # Emotion (8 classes from AffectNet)
    emotion_logits: Optional[np.ndarray] = None  # (8,) raw logits
    emotion_label: str = "neutral"               # Predicted label
    emotion_confidence: float = 0.0

    # Gaze
    gaze_yaw: float = 0.0       # Horizontal angle (degrees)
    gaze_pitch: float = 0.0     # Vertical angle (degrees)

    # State detection
    state_result: Optional[FacialStateResult] = None

    # Metadata
    is_valid: bool = False
    error_message: str = ""

    @property
    def emotion_probabilities(self) -> Optional[np.ndarray]:
        """Softmax probabilities from emotion logits."""
        if self.emotion_logits is None:
            return None
        exp = np.exp(self.emotion_logits - np.max(self.emotion_logits))
        return exp / exp.sum()


# Emotion labels (AffectNet 8 classes)
EMOTION_LABELS = [
    "neutral", "happy", "sad", "surprise",
    "fear", "disgust", "anger", "contempt"
]


# ============================================================================
# MediaPipe Face Mesh landmark indices (canonical FACS mapping)
# Reference: Ekman & Friesen FACS, MediaPipe Face Mesh topology.
# ============================================================================

# Inner brow (between brows) → use the left inner brow as reference
# (MediaPipe indices 107 and 336 are the canonical "inner brow" points).
_INNER_BROW_LEFT = 107
_INNER_BROW_RIGHT = 336
# Outer brow: 105 (right) and 334 (left)
_OUTER_BROW_RIGHT = 105
_OUTER_BROW_LEFT = 334
# Eye aperture landmarks (right eye)
_EYE_TOP_RIGHT = 159
_EYE_BOTTOM_RIGHT = 145
_EYE_TOP_LEFT = 386
_EYE_BOTTOM_LEFT = 374
# Nose wings for AU9 (nose wrinkler)
_NOSE_WING_RIGHT = 129
_NOSE_WING_LEFT = 358
# Mouth corners for AU12 (lip corner puller)
_MOUTH_RIGHT = 61
_MOUTH_LEFT = 291
# Upper/lower inner lip for AU25 (lips part)
_LIP_TOP = 13
_LIP_BOTTOM = 14
# Chin & forehead for face height normalization
_FOREHEAD = 10
_CHIN = 152
# Reference nose tip for jaw-drop normalization
_NOSE_TIP = 1


# Population default baselines (used if calibration is incomplete)
# These are normalized by face height (||p[10] - p[152]||).
_POPULATION_DEFAULTS = {
    "brow_eye":        0.16,    # inner brow to eye
    "outer_brow_eye":  0.10,    # outer brow to eye
    "eye_open":        0.030,   # vertical eye aperture
    "nose_w":          0.34,    # nose wings
    "mouth_w":         0.45,    # mouth corners
    "mouth_v":         0.025,   # lip aperture
    "jaw":             0.85,    # chin to nose tip
}

# Slopes map a unitless relative change to an AU intensity in [0, 5].
# Tuned empirically so that a 2x change in a geometric feature gives
# roughly 2.5/5 AU intensity, which is "moderate" on the FACS scale.
_AU_SLOPES = {
    "AU1":  30.0,   # inner brow raise
    "AU2":  30.0,   # outer brow raise
    "AU4":  30.0,   # brow lowerer (inverse of AU1)
    "AU6":   4.0,   # cheek raise (eye aperture narrows)
    "AU9":   6.0,   # nose wrinkle
    "AU12":  4.0,   # lip corner pull
    "AU25": 20.0,   # lips part
    "AU26":  6.0,   # jaw drop
}

# The 8 AUs that FacialStateDetector expects.
_REQUIRED_AUS = ["AU1", "AU2", "AU4", "AU6", "AU9", "AU12", "AU25", "AU26"]


# ============================================================================
# OpenFace 3.0 MTL constants
# ============================================================================
# The MTL model outputs 8 AU cosine-similarity scores and 8 emotion logits.
# Neither the source code nor the model card specifies the index-to-label
# mapping.  The mapping below follows the standard BP4D/DISFA AU subset
# ordering used by JAANet-style models.  If it's wrong for this particular
# checkpoint, change only this dict — everything else derives from it.
#
# OpenFace 3.0 AU output indices → our FACS AU names.
# Where the OF3 AU doesn't exist in our 8, the closest equivalent is used.
_OF3_AU_INDEX_MAP = {
    0: "AU1",    # Inner Brow Raise
    1: "AU2",    # Outer Brow Raiser
    2: "AU4",    # Brow Lowerer
    3: "AU6",    # Cheek Raise
    4: "AU6",    # AU7 (Lid Tightener) → mapped to AU6 (closest)
    5: "AU9",    # Nose Wrinkler
    6: "AU12",   # Lip Corner Puller
    7: "AU26",   # AU20 (Lip Stretcher) → mapped to AU26 (closest jaw-related)
}
# Note: AU25 (Lips Part) and AU26 (Jaw Drop) are not directly in the
# standard BP4D 8-AU set.  Index 7 (AU20 / Lip Stretcher) is mapped to
# AU26.  AU25 defaults to 0.0 from OF3 — the geometric fallback or the
# MediaPipe EAR-based approximation in FacialStateDetector covers it.

# OpenFace 3.0 emotion output indices → label names.
# Standard AffectNet 8-class ordering used by EfficientNet-B0 variants.
_OF3_EMOTION_LABELS = [
    "neutral", "anger", "contempt", "disgust",
    "fear", "happy", "sad", "surprise",
]


# ============================================================================
# OpenFace Analyzer
# ============================================================================

class OpenFaceAnalyzer:
    """
    OpenFace analyzer for ADAPT-Rehab (MediaPipe-Face-Mesh-backed).

    Provides AU intensities, emotion labels, and facial state for
    downstream state detection (pain / fatigue / exhaustion / boredom /
    normal). See module docstring for the implementation strategy.

    Usage:
        analyzer = OpenFaceAnalyzer()
        analyzer.initialize()
        result = analyzer.analyze(frame, timestamp_ms)
        if result.is_valid:
            print(f"State: {result.state_result.state.value}")
            print(f"Emotion: {result.emotion_label}")
    """

    # Number of consecutive frames to use for the neutral-face baseline.
    # Small enough to feel responsive, large enough to average out
    # blinks and micro-expressions.
    _CALIBRATION_FRAMES = 30

    def __init__(self, device: str = "cpu", model_dir: Optional[str] = None):
        """
        Args:
            device: "cuda" or "cpu". Passed to OpenFace 3.0 MTL when
                available; MediaPipe always runs on CPU regardless.
            model_dir: Directory containing ``MTL_backbone.pth``.
                Defaults to ``models/openface3/`` relative to project root.
        """
        self.device = device
        self.model_dir = model_dir or os.path.join(
            os.path.dirname(__file__), "..", "..", "models", "openface3"
        )

        # Underlying MediaPipe face detector (lazy import)
        self._face_detector = None

        # OpenFace 3.0 MTL predictor (lazy, set in initialize())
        self._of3_predictor = None
        self._of3_available = False

        # State detector
        self._state_detector = FacialStateDetector(fps=30.0)

        # Calibration state (used by geometric fallback only)
        self._is_initialized = False
        self._frame_count = 0
        self._calibration_buffer: List[Dict[str, float]] = []
        self._baseline: Optional[Dict[str, float]] = None

    def initialize(self) -> bool:
        """
        Initialize face detector and (optionally) OpenFace 3.0 MTL model.

        The face detector (MediaPipe Face Mesh) is always required.
        OpenFace 3.0 is optional — if it fails to load, the geometric
        fallback is used silently.

        Returns:
            True if the face detector initialized successfully.
        """
        # --- MediaPipe face detector (required) ---
        try:
            from .face_detector import FaceDetector
        except ImportError as e:
            logger.error(f"[OpenFace] FaceDetector import failed: {e}")
            return False

        try:
            self._face_detector = FaceDetector()
            if not self._face_detector.initialize():
                logger.error("[OpenFace] FaceDetector.initialize() returned False")
                return False
        except Exception as e:
            logger.error(f"[OpenFace] FaceDetector init error: {e}")
            return False

        # --- OpenFace 3.0 MTL model (optional) ---
        self._of3_available = False
        self._of3_predictor = None
        try:
            from openface.multitask_model import MultitaskPredictor  # type: ignore
            mt_path = self._find_mtl_weights()
            if mt_path is not None:
                dev = self.device if self.device.startswith("cuda") else "cpu"
                self._of3_predictor = MultitaskPredictor(
                    model_path=mt_path, device=dev,
                )
                self._of3_available = True
                logger.info(f"[OpenFace] OpenFace 3.0 MTL loaded on {dev}")
            else:
                logger.info("[OpenFace] MTL_backbone.pth not found — using geometric fallback")
        except ImportError:
            logger.info("[OpenFace] openface-test not installed — using geometric fallback")
        except Exception as e:
            logger.warning(f"[OpenFace] OF3 MTL init failed ({e}) — using geometric fallback")

        self._is_initialized = True
        self._frame_count = 0
        self._calibration_buffer = []
        self._baseline = None

        backend = "OpenFace 3.0 MTL" if self._of3_available else "geometric fallback"
        logger.info(f"[OpenFace] Initialized ({backend})")
        return True

    def analyze(
        self,
        frame: np.ndarray,
        timestamp_ms: int = 0,
        face_landmarks: Optional[np.ndarray] = None,
    ) -> OpenFaceResult:
        """Analyze a frame for AUs, emotion, and behavioral state.

        Uses OpenFace 3.0 MTL (GNN) when available, otherwise falls back
        to calibrated geometric estimation from MediaPipe landmarks.

        Args:
            frame: BGR image from OpenCV, shape (H, W, 3).
            timestamp_ms: Frame timestamp in milliseconds.
            face_landmarks: Optional pre-computed (468, 3) face landmarks
                (skip face detection).

        Returns:
            ``OpenFaceResult`` with all analysis results. ``is_valid``
            is False if no face is detected or the analyzer is not
            initialized.
        """
        if not self._is_initialized:
            return OpenFaceResult(error_message="Not initialized")

        self._frame_count += 1
        result = OpenFaceResult()

        try:
            # ---- 1) Face detection (or use provided landmarks) ----
            landmarks_468 = face_landmarks
            face_bbox = None
            confidence = 0.0

            if landmarks_468 is None and self._face_detector is not None:
                detected = self._face_detector.detect(frame)
                if detected is None or not getattr(detected, "is_valid", False):
                    return OpenFaceResult(error_message="No face detected")
                landmarks_468 = getattr(detected, "landmarks", None)
                face_bbox = getattr(detected, "bbox", None)
                confidence = float(getattr(detected, "confidence", 0.7) or 0.7)

            if landmarks_468 is None or len(landmarks_468) < 468:
                return OpenFaceResult(error_message="No face landmarks available")

            landmarks_468 = np.asarray(landmarks_468, dtype=np.float32)
            result.landmarks_468 = landmarks_468
            result.face_bbox = (
                face_bbox if face_bbox is not None
                else self._bbox_from_landmarks(landmarks_468)
            )
            result.face_confidence = confidence if confidence > 0 else 0.7

            # ---- 2) AU + Emotion (OpenFace 3.0 primary, geometric fallback) ----
            of3_used = False
            if self._of3_available and self._of3_predictor is not None:
                face_crop = self._crop_face(frame, result.face_bbox)
                if face_crop is not None:
                    try:
                        au, emo_label, emo_conf, emo_logits, gaze_y, gaze_p = \
                            self._analyze_with_of3(face_crop)
                        result.au_intensities = au
                        result.emotion_label = emo_label
                        result.emotion_confidence = emo_conf
                        result.emotion_logits = emo_logits
                        result.gaze_yaw = gaze_y
                        result.gaze_pitch = gaze_p
                        of3_used = True
                    except Exception as e:
                        logger.warning(
                            f"[OpenFace] OF3 predict failed, "
                            f"falling back to geometric: {e}"
                        )

            if not of3_used:
                # Geometric fallback path (calibration-based)
                result = self._analyze_geometric(
                    frame, result, landmarks_468,
                )
                if result is None:
                    return OpenFaceResult(error_message="Calibrating")
                # gaze stays at 0.0 for geometric path

            # ---- 3) State detection (Pain / Fatigue / etc.) ----
            result.state_result = self._state_detector.process_frame(
                au_raw=result.au_intensities,
                face_landmarks=landmarks_468,
            )

            result.is_valid = True
            return result

        except Exception as e:
            logger.error(f"[OpenFace] Analysis error: {e}")
            return OpenFaceResult(error_message=str(e))

    # ---------------------------------------------------------------------
    # OpenFace 3.0 MTL helpers
    # ---------------------------------------------------------------------

    def _find_mtl_weights(self) -> Optional[str]:
        """Locate MTL_backbone.pth in model_dir or via HuggingFace download."""
        mt_path = os.path.join(self.model_dir, "MTL_backbone.pth")
        if os.path.exists(mt_path):
            return mt_path

        # Try downloading from HuggingFace Hub
        try:
            from huggingface_hub import hf_hub_download  # type: ignore
            logger.info("[OpenFace] Downloading MTL_backbone.pth from HuggingFace...")
            downloaded = hf_hub_download(
                repo_id="nutPace/openface_weights",
                filename="MTL_backbone.pth",
                local_dir=self.model_dir,
                repo_type="model",
            )
            if os.path.exists(downloaded):
                return downloaded
        except Exception as e:
            logger.debug(f"[OpenFace] HF download failed: {e}")

        return None

    @staticmethod
    def _crop_face(
        frame: np.ndarray, bbox: np.ndarray, padding: float = 0.2,
    ) -> Optional[np.ndarray]:
        """Crop face region from frame using a bounding box with padding.

        Args:
            frame: BGR image, shape (H, W, 3).
            bbox: (4,) array ``[x1, y1, x2, y2]``.
            padding: Fractional padding around the bbox (0.2 = 20%).

        Returns:
            Cropped face BGR array, or ``None`` if the crop is empty.
        """
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = [float(v) for v in bbox]
        bw, bh = x2 - x1, y2 - y1
        if bw < 1 or bh < 1:
            return None
        pad_x, pad_y = bw * padding, bh * padding
        cx1 = max(0, int(x1 - pad_x))
        cy1 = max(0, int(y1 - pad_y))
        cx2 = min(w, int(x2 + pad_x))
        cy2 = min(h, int(y2 + pad_y))
        crop = frame[cy1:cy2, cx1:cx2]
        if crop.size == 0:
            return None
        return crop

    def _analyze_with_of3(
        self, face_crop: np.ndarray,
    ) -> Tuple[Dict[str, float], str, float, np.ndarray, float, float]:
        """Run OpenFace 3.0 MTL on a face crop.

        Args:
            face_crop: BGR face crop from ``_crop_face()``.

        Returns:
            ``(au_intensities, emotion_label, emotion_confidence,
            emotion_logits, gaze_yaw, gaze_pitch)``
        """
        emotion_tensor, gaze_tensor, au_tensor = \
            self._of3_predictor.predict(face_crop)  # type: ignore

        # --- AU: cosine similarities → FACS [0, 5] ---
        au_raw = au_tensor.squeeze(0).detach().cpu().numpy()  # (8,)
        au_intensities: Dict[str, float] = {au: 0.0 for au in _REQUIRED_AUS}
        for idx, val in enumerate(au_raw):
            au_name = _OF3_AU_INDEX_MAP.get(idx)
            if au_name is not None:
                # Map cosine similarity [-1, 1] → [0, 5] FACS
                facs = max(0.0, (float(val) + 1.0) / 2.0) * 5.0
                # If multiple indices map to same AU, take the max
                au_intensities[au_name] = max(au_intensities[au_name], facs)

        # --- Emotion: argmax + softmax ---
        import torch  # already imported at module level by of3
        emotion_probs = torch.softmax(
            emotion_tensor, dim=1,
        ).squeeze(0).detach().cpu().numpy()
        emotion_idx = int(torch.argmax(emotion_tensor, dim=1).item())
        emotion_label = _OF3_EMOTION_LABELS[emotion_idx]
        emotion_confidence = float(emotion_probs[emotion_idx])
        emotion_logits = emotion_tensor.squeeze(0).detach().cpu().numpy()

        # --- Gaze ---
        gaze_yaw = float(gaze_tensor[0][0])
        gaze_pitch = float(gaze_tensor[0][1])

        return au_intensities, emotion_label, emotion_confidence, \
            emotion_logits, gaze_yaw, gaze_pitch

    def _analyze_geometric(
        self,
        frame: np.ndarray,
        result: OpenFaceResult,
        landmarks_468: np.ndarray,
    ) -> Optional[OpenFaceResult]:
        """Geometric fallback: calibration-based AU + rule-based emotion.

        Returns the populated result, or ``None`` during calibration
        (in which case the caller should return a ``calibrating`` result).
        """
        features = self._extract_features(landmarks_468)

        # Calibrate baseline on the first N frames
        if self._baseline is None:
            self._calibration_buffer.append(features)
            if len(self._calibration_buffer) >= self._CALIBRATION_FRAMES:
                self._baseline = self._build_baseline(self._calibration_buffer)
            # During calibration, return zero AUs but a valid result so
            # downstream state detection can run.
            result.au_intensities = {k: 0.0 for k in _REQUIRED_AUS}
            result.emotion_label = "neutral"
            result.emotion_confidence = 0.0
            result.is_valid = True
            result.error_message = "calibrating"
            return None  # signal to caller

        # Compute AU intensities from relative changes vs baseline
        result.au_intensities = self._compute_au_intensities(
            features, self._baseline,
        )

        # Emotion from rule-based mapping
        emo_label, emo_conf, emo_logits = self._emotion_from_aus(
            result.au_intensities,
        )
        result.emotion_label = emo_label
        result.emotion_confidence = float(emo_conf)
        result.emotion_logits = emo_logits

        return result

    # ---------------------------------------------------------------------
    # Feature extraction (scale-normalized)
    # ---------------------------------------------------------------------

    @staticmethod
    def _extract_features(landmarks: np.ndarray) -> Dict[str, float]:
        """Compute scale-normalized facial features from 468 landmarks.

        All linear distances are normalized by face height
        ``||p[10] - p[152]||`` so the features are scale-invariant.
        """
        face_height = max(float(np.linalg.norm(landmarks[_FOREHEAD] - landmarks[_CHIN])), 1e-6)
        norm = 1.0 / face_height

        # AU1: inner brow raise — distance from inner brow to inner eye,
        # averaged across both sides.
        d_au1 = float(
            (np.linalg.norm(landmarks[_INNER_BROW_LEFT] - landmarks[_EYE_TOP_LEFT]) +
             np.linalg.norm(landmarks[_INNER_BROW_RIGHT] - landmarks[_EYE_TOP_RIGHT])) / 2.0
        ) * norm
        # AU2: outer brow raise — distance from outer brow to outer eye
        d_au2 = float(
            (np.linalg.norm(landmarks[_OUTER_BROW_LEFT] - landmarks[_EYE_TOP_LEFT]) +
             np.linalg.norm(landmarks[_OUTER_BROW_RIGHT] - landmarks[_EYE_TOP_RIGHT])) / 2.0
        ) * norm
        # AU6: cheek raiser / eye aperture — vertical eye opening
        eye_open = float(
            (np.linalg.norm(landmarks[_EYE_TOP_RIGHT] - landmarks[_EYE_BOTTOM_RIGHT]) +
             np.linalg.norm(landmarks[_EYE_TOP_LEFT] - landmarks[_EYE_BOTTOM_LEFT])) / 2.0
        ) * norm
        # AU9: nose wrinkler — nose wings spread
        nose_w = float(np.linalg.norm(landmarks[_NOSE_WING_RIGHT] - landmarks[_NOSE_WING_LEFT])) * norm
        # AU12: lip corner puller — mouth width
        mouth_w = float(np.linalg.norm(landmarks[_MOUTH_RIGHT] - landmarks[_MOUTH_LEFT])) * norm
        # AU25: lips part — vertical mouth aperture
        mouth_v = float(np.linalg.norm(landmarks[_LIP_TOP] - landmarks[_LIP_BOTTOM])) * norm
        # AU26: jaw drop — chin to nose tip
        jaw = float(np.linalg.norm(landmarks[_NOSE_TIP] - landmarks[_CHIN])) * norm

        return {
            "brow_eye": d_au1,
            "outer_brow_eye": d_au2,
            "eye_open": eye_open,
            "nose_w": nose_w,
            "mouth_w": mouth_w,
            "mouth_v": mouth_v,
            "jaw": jaw,
        }

    @staticmethod
    def _build_baseline(samples: List[Dict[str, float]]) -> Dict[str, float]:
        """Build a robust neutral baseline from the calibration buffer.

        Uses the median per feature (robust to blinks and small
        micro-movements), then falls back to population defaults for
        any feature whose calibration samples are degenerate.
        """
        if not samples:
            return dict(_POPULATION_DEFAULTS)

        keys = samples[0].keys()
        baseline: Dict[str, float] = {}
        for k in keys:
            vals = np.array([s[k] for s in samples], dtype=np.float32)
            med = float(np.median(vals))
            # Guard against zero / near-zero baselines (would divide by zero)
            baseline[k] = med if med > 1e-4 else _POPULATION_DEFAULTS.get(k, med)
        return baseline

    @staticmethod
    def _compute_au_intensities(
        features: Dict[str, float],
        baseline: Dict[str, float],
    ) -> Dict[str, float]:
        """Convert relative feature changes into AU intensities in [0, 5].

        All changes are computed as ``(current - baseline) / baseline``,
        which is the fractional change. Multiplying by the per-AU slope
        gives an intensity; the result is clipped to ``[0, 5]`` to match
        the FACS A-B-C-D-E scale (0=absent, 5=maximum).
        """
        def rel(name: str) -> float:
            b = baseline.get(name, 1e-4)
            return (features.get(name, b) - b) / b

        # AU1: positive when inner brow moves up
        au1 = 5.0 * max(0.0, rel("brow_eye")) * _AU_SLOPES["AU1"] / 5.0
        # AU2: positive when outer brow moves up
        au2 = 5.0 * max(0.0, rel("outer_brow_eye")) * _AU_SLOPES["AU2"] / 5.0
        # AU4: positive when inner brow moves DOWN (inverse of AU1)
        au4 = 5.0 * max(0.0, -rel("brow_eye")) * _AU_SLOPES["AU4"] / 5.0
        # AU6: positive when eye aperture narrows (cheek raise)
        eye_rel = rel("eye_open")
        au6 = 5.0 * max(0.0, -eye_rel) * _AU_SLOPES["AU6"] / 5.0
        # AU9: positive when nose wings contract (wrinkle)
        au9 = 5.0 * max(0.0, -rel("nose_w")) * _AU_SLOPES["AU9"] / 5.0
        # AU12: positive when mouth widens
        au12 = 5.0 * max(0.0, rel("mouth_w")) * _AU_SLOPES["AU12"] / 5.0
        # AU25: positive when lips part vertically
        au25 = 5.0 * max(0.0, rel("mouth_v")) * _AU_SLOPES["AU25"] / 5.0
        # AU26: positive when jaw drops
        au26 = 5.0 * max(0.0, rel("jaw")) * _AU_SLOPES["AU26"] / 5.0

        return {
            "AU1":  float(np.clip(au1, 0.0, 5.0)),
            "AU2":  float(np.clip(au2, 0.0, 5.0)),
            "AU4":  float(np.clip(au4, 0.0, 5.0)),
            "AU6":  float(np.clip(au6, 0.0, 5.0)),
            "AU9":  float(np.clip(au9, 0.0, 5.0)),
            "AU12": float(np.clip(au12, 0.0, 5.0)),
            "AU25": float(np.clip(au25, 0.0, 5.0)),
            "AU26": float(np.clip(au26, 0.0, 5.0)),
        }

    # ---------------------------------------------------------------------
    # Emotion
    # ---------------------------------------------------------------------

    @staticmethod
    def _emotion_from_aus(au: Dict[str, float]) -> Tuple[str, float, np.ndarray]:
        """Rule-based emotion mapping from the 8 AU intensities.

        Returns:
            (label, confidence, logits) tuple. ``logits`` is a
            length-8 vector aligned with ``EMOTION_LABELS`` so the
            existing ``emotion_probabilities`` property keeps working.
        """
        # Initialize logits with small prior on "neutral"
        logits = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)

        # Strong AU12 (smile) + moderate AU6 (cheek raise) → happy
        logits[1] += 2.0 * au.get("AU12", 0) + 1.0 * au.get("AU6", 0)
        # AU4 (brow lowerer) → anger or contempt
        logits[6] += 1.5 * au.get("AU4", 0)
        # AU1+AU4 → sad
        logits[2] += 1.0 * au.get("AU1", 0) + 1.2 * au.get("AU4", 0)
        # AU1+AU2 (brow raise) → surprise
        logits[3] += 1.5 * (au.get("AU1", 0) + au.get("AU2", 0)) / 2.0
        # AU9+AU4 (nose wrinkle + brow lower) → disgust
        logits[5] += 1.5 * au.get("AU9", 0) + 0.5 * au.get("AU4", 0)
        # AU1+AU2+AU4 → fear
        logits[4] += 0.8 * au.get("AU1", 0) + 0.8 * au.get("AU2", 0) + 0.4 * au.get("AU4", 0)
        # AU12 unilateral → contempt (rough heuristic: smile without AU6)
        if au.get("AU12", 0) > 1.0 and au.get("AU6", 0) < 0.5:
            logits[7] += 0.8 * au.get("AU12", 0)

        # Pick the highest
        idx = int(np.argmax(logits))
        label = EMOTION_LABELS[idx]

        # Softmax for confidence
        exp = np.exp(logits - np.max(logits))
        probs = exp / exp.sum()
        confidence = float(probs[idx])

        return label, confidence, logits

    # ---------------------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------------------

    @staticmethod
    def _bbox_from_landmarks(landmarks: np.ndarray) -> np.ndarray:
        """Compute a (4,) bbox (x1, y1, x2, y2) from face landmarks."""
        xy = landmarks[:, :2]
        return np.array(
            [xy[:, 0].min(), xy[:, 1].min(), xy[:, 0].max(), xy[:, 1].max()],
            dtype=np.float32,
        )

    def reset(self):
        """Reset all state (calibration + frame counter)."""
        self._frame_count = 0
        self._calibration_buffer = []
        self._baseline = None
        if self._state_detector is not None and hasattr(self._state_detector, "reset"):
            self._state_detector.reset()

    def close(self):
        """Release resources."""
        if self._face_detector is not None:
            try:
                self._face_detector.close()
            except Exception:
                pass
        self._face_detector = None
        self._of3_predictor = None
        self._of3_available = False
        self._is_initialized = False
        self._calibration_buffer = []
        self._baseline = None