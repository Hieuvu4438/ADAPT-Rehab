"""
OpenFace 3.0 Analyzer for ADAPT-Rehab.

Wraps OpenFace 3.0 (CMU MultiComp Lab) for:
- Action Unit (AU) detection (8 AUs)
- Emotion classification (8 classes)
- Gaze estimation
- Face detection (RetinaFace)

Then feeds AU data to FacialStateDetector for scientifically validated
state classification (Pain/Fatigue/Exhaustion/Boredom/Normal).

OpenFace 3.0 Paper:
    "OpenFace 3.0: A Multitask Toolkit for Facial Behavior Analysis"
    arXiv:2506.02891, June 2025

Architecture:
    Frame → OpenFace 3.0 (face detect + landmark + AU + emotion + gaze)
          → FacialStateDetector (AU-based formulas)
          → FacialStateResult (pain/fatigue/exhaustion/boredom/normal)

Version: 4.0.0
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import logging
import os

from .facial_state_detector import (
    FacialStateDetector,
    FacialStateResult,
    FacialState,
    AUData,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Data Types
# ============================================================================

@dataclass
class OpenFaceResult:
    """Complete result from OpenFace 3.0 analysis."""
    # Face detection
    face_bbox: Optional[np.ndarray] = None      # (4,) x1,y1,x2,y2
    face_confidence: float = 0.0

    # Landmarks
    landmarks_68: Optional[np.ndarray] = None    # (68, 2) 2D landmarks
    landmarks_468: Optional[np.ndarray] = None   # (468, 3) from MediaPipe (if used)

    # Action Units (8 AUs from OpenFace 3.0)
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
# OpenFace 3.0 Analyzer
# ============================================================================

class OpenFaceAnalyzer:
    """
    OpenFace 3.0 wrapper for ADAPT-Rehab.

    Integrates OpenFace 3.0's multitask model (AU + Emotion + Gaze)
    with the FacialStateDetector for rehabilitation state monitoring.

    OpenFace 3.0 detects 8 Action Units using a graph neural network:
    - AU1: Inner Brow Raiser
    - AU2: Outer Brow Raiser
    - AU4: Brow Lowerer
    - AU6: Cheek Raiser
    - AU9: Nose Wrinkler
    - AU12: Lip Corner Puller
    - AU25: Lips Part
    - AU26: Jaw Drop

    The AU head uses Attentional Feature Generation (AFG) + Fine-Grained
    Graph (GNN) module for inter-AU relationship modeling.

    Usage:
        analyzer = OpenFaceAnalyzer()
        analyzer.initialize()
        result = analyzer.analyze(frame, timestamp_ms)
        if result.is_valid:
            print(f"State: {result.state_result.state.value}")
            print(f"Emotion: {result.emotion_label}")
    """

    def __init__(self, device: str = "cuda", model_dir: Optional[str] = None):
        """
        Args:
            device: "cuda" or "cpu"
            model_dir: Directory containing OpenFace 3.0 model weights.
                       If None, uses models/openface3/
        """
        self.device = device
        self.model_dir = model_dir or os.path.join(
            os.path.dirname(__file__), "..", "..", "models", "openface3"
        )

        # OpenFace components
        self._face_detector = None
        self._landmark_detector = None
        self._multitask_predictor = None

        # State detector
        self._state_detector = FacialStateDetector(fps=30.0)

        # State
        self._is_initialized = False
        self._frame_count = 0
        self._use_mediapipe_fallback = False

    def initialize(self) -> bool:
        """
        Initialize OpenFace 3.0 models.

        Returns:
            True if initialization successful.
        """
        try:
            # Try importing openface
            try:
                from openface.face_detection import FaceDetector
                from openface.landmark_detection import LandmarkDetector
                from openface.multitask_model import MultitaskPredictor

                logger.info("[OpenFace] Loading OpenFace 3.0 models...")

                # Face detector
                face_model_path = os.path.join(self.model_dir, "RetinaFace.pth")
                if os.path.exists(face_model_path):
                    self._face_detector = FaceDetector(
                        model_path=face_model_path,
                        device=self.device,
                        confidence_threshold=0.5,
                        nms_threshold=0.4,
                    )
                    logger.info("[OpenFace] Face detector loaded")
                else:
                    logger.warning(f"[OpenFace] RetinaFace model not found: {face_model_path}")

                # Landmark detector
                landmark_model_path = os.path.join(self.model_dir, "Landmark_98.pkl")
                if os.path.exists(landmark_model_path):
                    self._landmark_detector = LandmarkDetector(
                        model_path=landmark_model_path,
                        device=self.device,
                    )
                    logger.info("[OpenFace] Landmark detector loaded")

                # Multitask predictor (AU + Emotion + Gaze)
                mt_model_path = os.path.join(self.model_dir, "stage2_epoch_7.pth")
                if os.path.exists(mt_model_path):
                    self._multitask_predictor = MultitaskPredictor(
                        model_path=mt_model_path,
                        device=self.device,
                    )
                    logger.info("[OpenFace] Multitask predictor loaded")
                else:
                    logger.warning(f"[OpenFace] Multitask model not found: {mt_model_path}")

                if self._multitask_predictor is not None:
                    self._is_initialized = True
                    logger.info("[OpenFace] Initialized successfully")
                    return True

            except ImportError as e:
                logger.warning(f"[OpenFace] openface package not available: {e}")

            # Fallback: Use MediaPipe for face + landmarks, skip AU
            logger.info("[OpenFace] Falling back to MediaPipe face detection only")
            self._use_mediapipe_fallback = True
            self._is_initialized = True
            return True

        except Exception as e:
            logger.error(f"[OpenFace] Initialization failed: {e}")
            return False

    def analyze(self, frame: np.ndarray, timestamp_ms: int = 0,
                face_landmarks: Optional[np.ndarray] = None) -> OpenFaceResult:
        """
        Analyze a frame for AUs, emotion, gaze, and behavioral state.

        Args:
            frame: BGR image from OpenCV, shape (H, W, 3)
            timestamp_ms: Frame timestamp in milliseconds
            face_landmarks: Optional pre-computed face landmarks (468, 3)
                           from MediaPipe Face Mesh (used as fallback)

        Returns:
            OpenFaceResult with all analysis results
        """
        if not self._is_initialized:
            return OpenFaceResult(error_message="Not initialized")

        self._frame_count += 1
        result = OpenFaceResult()

        try:
            if self._use_mediapipe_fallback:
                return self._analyze_mediapipe_fallback(frame, face_landmarks)

            # === OpenFace 3.0 Pipeline ===

            # 1. Face detection
            if self._face_detector is not None:
                cropped_face, dets = self._face_detector.get_face(frame, resize=1)
                if dets is None or len(dets) == 0:
                    return OpenFaceResult(error_message="No face detected")
                result.face_bbox = dets[0][:4]  # x1, y1, x2, y2
                result.face_confidence = float(dets[0][4]) if len(dets[0]) > 4 else 0.9
            else:
                # Use provided landmarks or simple center crop
                cropped_face = self._center_crop_face(frame)
                result.face_confidence = 0.5

            # 2. Landmark detection
            if self._landmark_detector is not None:
                landmarks = self._landmark_detector.detect_landmarks(
                    frame, dets if self._face_detector else None
                )
                if landmarks is not None:
                    result.landmarks_68 = landmarks

            # 3. Multitask prediction (AU + Emotion + Gaze)
            if self._multitask_predictor is not None and cropped_face is not None:
                import torch
                with torch.no_grad():
                    emotion_logits, gaze_output, au_output = \
                        self._multitask_predictor.predict(cropped_face)

                # Parse AU output (8 values)
                au_values = au_output.cpu().numpy().flatten()
                result.au_intensities = {
                    "AU1": float(au_values[0]),
                    "AU2": float(au_values[1]),
                    "AU4": float(au_values[2]),
                    "AU6": float(au_values[3]),
                    "AU9": float(au_values[4]),
                    "AU12": float(au_values[5]),
                    "AU25": float(au_values[6]),
                    "AU26": float(au_values[7]),
                }

                # Parse emotion
                emo_logits = emotion_logits.cpu().numpy().flatten()
                result.emotion_logits = emo_logits
                emo_idx = int(np.argmax(emo_logits))
                result.emotion_label = EMOTION_LABELS[emo_idx]
                probs = result.emotion_probabilities
                result.emotion_confidence = float(probs[emo_idx]) if probs is not None else 0.0

                # Parse gaze
                gaze = gaze_output.cpu().numpy().flatten()
                result.gaze_yaw = float(gaze[0])
                result.gaze_pitch = float(gaze[1])

            # 4. State detection using AU formulas
            if result.au_intensities is not None:
                result.state_result = self._state_detector.process_frame(
                    au_raw=result.au_intensities,
                    face_landmarks=face_landmarks or result.landmarks_468,
                )

            result.is_valid = True
            return result

        except Exception as e:
            logger.error(f"[OpenFace] Analysis error: {e}")
            return OpenFaceResult(error_message=str(e))

    def _analyze_mediapipe_fallback(self, frame: np.ndarray,
                                     face_landmarks: Optional[np.ndarray]) -> OpenFaceResult:
        """
        Fallback analysis using MediaPipe landmarks only.

        When OpenFace 3.0 is not available, we can still:
        1. Compute EAR for eye closure (AU43 approximation)
        2. Estimate some AU-like features from landmark geometry
        3. Use the state detector with available data

        This is a degraded mode - AU intensities are approximated, not detected.
        """
        result = OpenFaceResult()

        if face_landmarks is None or len(face_landmarks) < 468:
            return OpenFaceResult(error_message="No face landmarks available")

        result.landmarks_468 = face_landmarks
        result.face_confidence = 0.7

        # Approximate AU-like features from landmark geometry
        # These are rough approximations, not real AU detection
        au_approx = self._approximate_aus_from_landmarks(face_landmarks)
        result.au_intensities = au_approx

        # State detection with approximated AUs
        result.state_result = self._state_detector.process_frame(
            au_raw=au_approx,
            face_landmarks=face_landmarks,
        )

        # Approximate emotion from AU patterns
        result.emotion_label = self._approximate_emotion(au_approx)
        result.emotion_confidence = 0.4  # Low confidence for approximation

        result.is_valid = True
        return result

    def _approximate_aus_from_landmarks(self, landmarks: np.ndarray) -> Dict[str, float]:
        """
        Approximate AU intensities from MediaPipe Face Mesh landmarks.

        This is a FALLBACK only - real AU detection requires OpenFace 3.0.
        These approximations are geometric estimates, not FACS-coded intensities.

        Note: These are NOT scientifically validated AU intensities.
        They provide rough signals for the state detector to work with.
        """
        au = {f"AU{i}": 0.0 for i in [1, 2, 4, 6, 9, 12, 25, 26]}

        if len(landmarks) < 468:
            return au

        # AU4 (Brow Lowerer): Distance between brow and eye
        # Lower brow = smaller distance = higher AU4
        left_brow_eye = np.linalg.norm(landmarks[66][:2] - landmarks[159][:2])
        right_brow_eye = np.linalg.norm(landmarks[296][:2] - landmarks[386][:2])
        brow_eye_dist = (left_brow_eye + right_brow_eye) / 2.0
        # Normalize: typical range 20-60 pixels at 640x480
        au["AU4"] = max(0.0, min(5.0, 5.0 * (1.0 - brow_eye_dist / 60.0)))

        # AU12 (Lip Corner Puller / Smile): Mouth width relative to height
        mouth_width = np.linalg.norm(landmarks[61][:2] - landmarks[291][:2])
        mouth_height = np.linalg.norm(landmarks[0][:2] - landmarks[17][:2])
        if mouth_height > 0:
            smile_ratio = mouth_width / mouth_height
            au["AU12"] = max(0.0, min(5.0, 5.0 * (smile_ratio - 2.0) / 3.0))

        # AU25 (Lips Part): Distance between upper and lower lip
        lip_dist = np.linalg.norm(landmarks[13][:2] - landmarks[14][:2])
        au["AU25"] = max(0.0, min(5.0, 5.0 * lip_dist / 20.0))

        # AU26 (Jaw Drop): Distance between chin and nose
        jaw_dist = np.linalg.norm(landmarks[152][:2] - landmarks[1][:2])
        au["AU26"] = max(0.0, min(5.0, 5.0 * (jaw_dist - 80.0) / 40.0))

        # AU1/AU2 (Brow Raiser): Opposite of AU4
        au["AU1"] = max(0.0, min(5.0, 5.0 * (brow_eye_dist / 60.0 - 0.5)))
        au["AU2"] = au["AU1"] * 0.8

        # AU6 (Cheek Raiser): Approximate from cheek-eye distance
        au["AU6"] = max(0.0, au["AU12"] * 0.5)  # Often co-occurs with smile

        # AU9 (Nose Wrinkler): Approximate from nose width
        nose_width = np.linalg.norm(landmarks[129][:2] - landmarks[358][:2])
        au["AU9"] = max(0.0, min(5.0, 5.0 * (nose_width - 30.0) / 20.0))

        return au

    def _approximate_emotion(self, au: Dict[str, float]) -> str:
        """Rough emotion approximation from AU patterns."""
        if au.get("AU12", 0) > 2.0:
            return "happy"
        elif au.get("AU4", 0) > 2.0 and au.get("AU1", 0) > 1.0:
            return "sad"
        elif au.get("AU4", 0) > 3.0:
            return "anger"
        elif au.get("AU1", 0) > 2.0 and au.get("AU2", 0) > 2.0:
            return "surprise"
        else:
            return "neutral"

    def _center_crop_face(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Simple center crop as face detection fallback."""
        h, w = frame.shape[:2]
        size = min(h, w) // 2
        cx, cy = w // 2, h // 2
        crop = frame[cy-size:cy+size, cx-size:cx+size]
        if crop.size == 0:
            return None
        return crop

    def reset(self):
        """Reset all state."""
        self._frame_count = 0
        self._state_detector.reset()

    def close(self):
        """Release resources."""
        self._face_detector = None
        self._landmark_detector = None
        self._multitask_predictor = None
        self._is_initialized = False
