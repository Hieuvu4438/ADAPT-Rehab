"""
Action Unit (AU) Detection Pipeline.

Multi-backend AU detection for facial action coding:
    1. Py-Feat (preferred) — pretrained deep learning models (JAANet/DRML)
    2. MobileNetV3-Large — custom lightweight model for edge deployment
    3. Geometric rules — always available, zero dependencies

Computes PSPI (Prkachin-Solomon Pain Intensity) from AU values:
    PSPI = AU4 + max(AU6, AU7) + max(AU9, AU10) + AU43

Usage:
    detector = ActionUnitDetector()
    result = detector.detect(face_landmarks)
    if result.is_valid:
        print(f"Pain: {result.pain_score:.1f}, Backend: {result.backend}")
"""

import logging
import os
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np

from .face_detector import FaceLandmarkIndex

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Safe deep learning imports
# ---------------------------------------------------------------------------
_PYFEAT_AVAILABLE = False
_TORCH_AVAILABLE = False

try:
    from feat import Detector as FeatDetector
    _PYFEAT_AVAILABLE = True
except ImportError:
    logger.debug("py-feat not available. Will try MobileNetV3 or geometric fallback.")

torch = None
nn = None
try:
    import torch
    import torch.nn as _nn
    nn = _nn
    _TORCH_AVAILABLE = True
except ImportError:
    logger.debug("PyTorch not available for AU detection.")


# ---------------------------------------------------------------------------
# Result dataclass (backward compatible)
# ---------------------------------------------------------------------------
@dataclass
class AUResult:
    """Result of Action Unit detection.

    Attributes:
        au_activations: Dictionary of AU values (0-1 or 0-5 depending on backend).
        pain_score: PSPI pain intensity score (0-16).
        pain_level: Categorical pain level (NONE/MILD/MODERATE/SEVERE).
        is_valid: Whether detection succeeded.
        error_message: Error details if failed.
        backend: Detection backend used ('pyfeat', 'mobilenet', 'geometric').
    """
    au_activations: Dict[str, float] = field(default_factory=dict)
    pain_score: float = 0.0
    pain_level: str = "NONE"
    is_valid: bool = False
    error_message: str = ""
    backend: str = "geometric"


# ---------------------------------------------------------------------------
# PSPI Computation
# ---------------------------------------------------------------------------
def compute_pspi(aus: Dict[str, float]) -> float:
    """
    Compute PSPI (Prkachin-Solomon Pain Intensity) score.

    PSPI = AU4 + max(AU6, AU7) + max(AU9, AU10) + AU43

    Range: 0-16 (0 = no pain, 16 = maximum pain)

    Args:
        aus: Dictionary of AU values.

    Returns:
        PSPI score (float).
    """
    au4 = aus.get("AU4", aus.get("AU04", 0.0))
    au6 = aus.get("AU6", aus.get("AU06", 0.0))
    au7 = aus.get("AU7", aus.get("AU07", 0.0))
    au9 = aus.get("AU9", aus.get("AU09", 0.0))
    au10 = aus.get("AU10", aus.get("AU10", 0.0))
    au43 = aus.get("AU43", aus.get("AU43", 0.0))

    return float(au4 + max(au6, au7) + max(au9, au10) + au43)


# ---------------------------------------------------------------------------
# Py-Feat Backend
# ---------------------------------------------------------------------------
class _PyFeatBackend:
    """
    AU detection using py-feat library.

    Uses pre-trained JAANet or DRML models for accurate AU detection.
    """

    def __init__(self, au_model: str = "jaanet"):
        if not _PYFEAT_AVAILABLE:
            raise ImportError("py-feat is not installed.")

        self._detector = FeatDetector(
            au_model=au_model,
            face_model="retinaface",
            landmark_model="mobilenet",
        )
        logger.info(f"PyFeatBackend initialized: au_model={au_model}")

    def detect(
        self,
        face_image: Optional[np.ndarray] = None,
        face_landmarks: Optional[np.ndarray] = None,
    ) -> AUResult:
        """Detect AUs from face image using py-feat."""
        if face_image is None:
            return AUResult(backend="pyfeat", error_message="No face image provided")

        try:
            import cv2

            # py-feat expects RGB
            if len(face_image.shape) == 3 and face_image.shape[2] == 3:
                rgb_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
            else:
                rgb_image = face_image

            result = self._detector.detect_image(rgb_image)

            if result is None or len(result) == 0:
                return AUResult(backend="pyfeat", error_message="No face detected by py-feat")

            # Extract AU values
            aus = {}
            au_columns = [col for col in result.columns if col.startswith("AU")]

            for col in au_columns:
                # Normalize column names to AU4, AU6, etc.
                au_name = col.upper().replace("AU0", "AU")
                if au_name in result.columns:
                    aus[au_name] = float(result[au_name].iloc[0])
                elif col in result.columns:
                    aus[au_name] = float(result[col].iloc[0])

            pain_score = compute_pspi(aus)
            pain_level = self._classify_pain(pain_score)

            return AUResult(
                au_activations=aus,
                pain_score=pain_score,
                pain_level=pain_level,
                is_valid=True,
                backend="pyfeat",
            )

        except Exception as e:
            logger.warning(f"Py-Feat detection failed: {e}")
            return AUResult(backend="pyfeat", error_message=str(e))

    @staticmethod
    def _classify_pain(pspi: float) -> str:
        if pspi >= 12:
            return "SEVERE"
        elif pspi >= 8:
            return "MODERATE"
        elif pspi >= 4:
            return "MILD"
        return "NONE"


# ---------------------------------------------------------------------------
# MobileNetV3-Large Backend
# ---------------------------------------------------------------------------
if _TORCH_AVAILABLE:

    class _MobileNetV3AU(nn.Module):
        """
        Multi-task MobileNetV3-Large for AU detection and pain estimation.

        Architecture:
            Backbone: MobileNetV3-Large (pretrained on ImageNet)
            → Global Average Pooling
            → Shared FC (512)
            → AU Head: 6 AUs × sigmoid (multi-label)
            → Pain Head: 1 × ReLU (PSPI regression)

        Input:  (B, 3, 224, 224) — cropped & resized face
        Output: (B, 7) — 6 AU values + 1 PSPI score
        """

        AU_NAMES = ["AU4", "AU6", "AU7", "AU9", "AU10", "AU43"]

        def __init__(self, pretrained_backbone: bool = True):
            super().__init__()

            try:
                from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights
                weights = MobileNet_V3_Large_Weights.DEFAULT if pretrained_backbone else None
                backbone = mobilenet_v3_large(weights=weights)
            except ImportError:
                from torchvision.models import mobilenet_v3_large
                backbone = mobilenet_v3_large(pretrained=pretrained_backbone)

            self.features = backbone.features
            self.pool = nn.AdaptiveAvgPool2d(1)
            feat_dim = 960  # MobileNetV3-Large feature dimension

            # Shared representation
            self.shared = nn.Sequential(
                nn.Linear(feat_dim, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
            )

            # AU classification head (multi-label, 0-5 intensity)
            self.au_head = nn.Sequential(
                nn.Linear(512, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.2),
                nn.Linear(256, len(self.AU_NAMES)),
                nn.Sigmoid(),
            )

            # Pain regression head (PSPI 0-16)
            self.pain_head = nn.Sequential(
                nn.Linear(512, 128),
                nn.ReLU(inplace=True),
                nn.Dropout(0.2),
                nn.Linear(128, 1),
                nn.ReLU(),
            )

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            """Returns (B, 7): 6 AU values (0-5) + 1 PSPI score."""
            feat = self.features(x)
            feat = self.pool(feat).flatten(1)
            shared = self.shared(feat)

            au_out = self.au_head(shared) * 5  # Scale to 0-5
            pain_out = self.pain_head(shared)

            return torch.cat([au_out, pain_out], dim=1)


class _MobileNetBackend:
    """
    MobileNetV3-based AU detection backend.

    Requires pre-trained weights file at models/au_mobilenetv3.pth.
    """

    AU_NAMES = ["AU4", "AU6", "AU7", "AU9", "AU10", "AU43"]

    def __init__(self, model_path: Optional[str] = None, device: str = "auto"):
        if not _TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for MobileNetBackend.")

        self._device = self._resolve_device(device)
        self._model = None
        self._initialized = False

        if model_path is None:
            model_path = os.path.join(
                os.path.dirname(__file__),
                "..", "..", "models", "au_mobilenetv3.pth",
            )

        self._load_model(model_path)

    def _resolve_device(self, device: str) -> str:
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device

    def _load_model(self, model_path: str) -> bool:
        try:
            model_path = os.path.normpath(model_path)
            if not os.path.exists(model_path):
                logger.info(f"MobileNetV3 AU weights not found: {model_path}")
                return False

            model = _MobileNetV3AU(pretrained_backbone=False)
            state_dict = torch.load(model_path, map_location=self._device)
            model.load_state_dict(state_dict)
            model.to(self._device)
            model.eval()

            self._model = model
            self._initialized = True
            logger.info(f"MobileNetV3 AU model loaded from {model_path}")
            return True

        except Exception as e:
            logger.warning(f"Failed to load MobileNetV3 AU model: {e}")
            return False

    @property
    def is_ready(self) -> bool:
        return self._initialized and self._model is not None

    def detect(
        self,
        face_image: Optional[np.ndarray] = None,
        face_landmarks: Optional[np.ndarray] = None,
    ) -> AUResult:
        """Detect AUs using MobileNetV3."""
        if not self.is_ready:
            return AUResult(backend="mobilenet", error_message="Model not loaded")

        if face_image is None:
            return AUResult(backend="mobilenet", error_message="No face image")

        try:
            face_crop = self._crop_face(face_image, face_landmarks)
            if face_crop is None:
                return AUResult(backend="mobilenet", error_message="Face crop failed")

            tensor = self._preprocess(face_crop)

            with torch.no_grad():
                output = self._model(tensor).cpu().numpy().flatten()

            aus = {name: float(np.clip(output[i], 0, 5)) for i, name in enumerate(self.AU_NAMES)}
            pain_score = float(np.clip(output[6], 0, 16))

            # Cross-validate PSPI from AUs
            pspi_au = compute_pspi(aus)
            pain_score = max(pain_score, pspi_au)

            pain_level = self._classify_pain(pain_score)

            return AUResult(
                au_activations=aus,
                pain_score=pain_score,
                pain_level=pain_level,
                is_valid=True,
                backend="mobilenet",
            )

        except Exception as e:
            logger.warning(f"MobileNetV3 AU detection failed: {e}")
            return AUResult(backend="mobilenet", error_message=str(e))

    def _crop_face(
        self, image: np.ndarray, landmarks: Optional[np.ndarray]
    ) -> Optional[np.ndarray]:
        """Crop face region from image."""
        h, w = image.shape[:2]

        if landmarks is not None and len(landmarks) >= 468:
            pts = np.asarray(landmarks, dtype=np.float32)
            if pts.ndim == 2 and pts.shape[1] >= 2:
                x_min, y_min = pts[:, :2].min(axis=0)
                x_max, y_max = pts[:, :2].max(axis=0)

                pad_x = (x_max - x_min) * 0.2
                pad_y = (y_max - y_min) * 0.2
                x_min = max(0, int(x_min - pad_x))
                y_min = max(0, int(y_min - pad_y))
                x_max = min(w, int(x_max + pad_x))
                y_max = min(h, int(y_max + pad_y))

                if x_max > x_min and y_max > y_min:
                    return image[y_min:y_max, x_min:x_max].copy()

        # Center crop fallback
        crop_size = min(h, w) // 2
        cx, cy = w // 2, h // 2
        return image[
            max(0, cy - crop_size):min(h, cy + crop_size),
            max(0, cx - crop_size):min(w, cx + crop_size),
        ].copy()

    def _preprocess(self, face_crop: np.ndarray) -> "torch.Tensor":
        """Preprocess face crop for MobileNetV3."""
        try:
            import cv2
        except ImportError:
            # Fallback: numpy resize
            from PIL import Image
            pil = Image.fromarray(face_crop[:, :, ::-1] if face_crop.shape[2] == 3 else face_crop)
            pil = pil.resize((224, 224))
            arr = np.array(pil).astype(np.float32) / 255.0
            tensor = torch.from_numpy(arr).permute(2, 0, 1)
        else:
            resized = cv2.resize(face_crop, (224, 224), interpolation=cv2.INTER_LINEAR)
            if len(resized.shape) == 3 and resized.shape[2] == 3:
                resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            tensor = torch.from_numpy(resized).float() / 255.0
            tensor = tensor.permute(2, 0, 1)

        # ImageNet normalization
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        tensor = (tensor - mean) / std

        return tensor.unsqueeze(0).to(self._device)

    @staticmethod
    def _classify_pain(pspi: float) -> str:
        if pspi >= 12:
            return "SEVERE"
        elif pspi >= 8:
            return "MODERATE"
        elif pspi >= 4:
            return "MILD"
        return "NONE"


# ---------------------------------------------------------------------------
# Geometric Backend (Fallback — always available)
# ---------------------------------------------------------------------------
class _GeometricBackend:
    """
    Rule-based AU estimation from Face Mesh landmarks.

    Always available — no external dependencies beyond numpy.
    Uses geometric ratios and distances to approximate AU intensities.
    """

    # AU detection thresholds
    AU_THRESHOLDS = {
        "AU4": 0.15,
        "AU6": 0.12,
        "AU7": 0.15,
        "AU9": 0.10,
        "AU10": 0.12,
        "AU43": 0.40,
    }

    PAIN_THRESHOLDS = {"NONE": 0, "MILD": 4, "MODERATE": 8, "SEVERE": 12}

    def __init__(self, use_baseline: bool = True):
        self._use_baseline = use_baseline
        self._baseline: Optional[Dict[str, float]] = None
        self._is_calibrated = False

    def set_baseline(self, landmarks: np.ndarray) -> None:
        """Set baseline from neutral face."""
        self._baseline = self._compute_raw_measurements(landmarks)
        self._is_calibrated = True

    def detect(
        self,
        face_image: Optional[np.ndarray] = None,
        face_landmarks: Optional[np.ndarray] = None,
    ) -> AUResult:
        """Estimate AUs from face landmarks using geometric rules."""
        if face_landmarks is None or len(face_landmarks) < 468:
            return AUResult(backend="geometric", error_message="Invalid landmarks")

        try:
            lms = np.asarray(face_landmarks, dtype=np.float32)
            if lms.ndim == 1:
                lms = lms.reshape(-1, 3)

            raw = self._compute_raw_measurements(lms)

            if self._baseline is None and self._use_baseline:
                self._baseline = self._get_default_baseline()

            aus = self._compute_au_activations(raw)
            pspi = compute_pspi(aus)
            pain_level = self._classify_pain(pspi)

            return AUResult(
                au_activations=aus,
                pain_score=pspi,
                pain_level=pain_level,
                is_valid=True,
                backend="geometric",
            )

        except Exception as e:
            logger.warning(f"Geometric AU detection failed: {e}")
            return AUResult(backend="geometric", error_message=str(e))

    def _compute_raw_measurements(self, lms: np.ndarray) -> Dict[str, float]:
        """Compute raw facial measurements from landmarks."""
        measurements = {}

        try:
            face_top = lms[FaceLandmarkIndex.FACE_TOP]
            face_bottom = lms[FaceLandmarkIndex.FACE_BOTTOM]
            face_height = np.linalg.norm(face_top[:2] - face_bottom[:2])
            if face_height < 1e-6:
                face_height = 1.0

            measurements["left_ear"] = self._ear(lms, "left")
            measurements["right_ear"] = self._ear(lms, "right")
            measurements["eye_aspect_ratio"] = (
                measurements["left_ear"] + measurements["right_ear"]
            ) / 2

            measurements["left_brow"] = self._brow_position(lms, "left", face_height)
            measurements["right_brow"] = self._brow_position(lms, "right", face_height)
            measurements["eyebrow_position"] = (
                measurements["left_brow"] + measurements["right_brow"]
            ) / 2

            measurements["nose_wrinkle"] = self._nose_wrinkle(lms, face_height)
            measurements["upper_lip_raise"] = self._upper_lip_position(lms, face_height)
            measurements["mouth_aspect_ratio"] = self._mouth_aspect_ratio(lms)

        except (IndexError, ValueError):
            pass

        return measurements

    def _ear(self, lms: np.ndarray, side: str) -> float:
        """Eye Aspect Ratio."""
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

        v = np.linalg.norm(top[:2] - bottom[:2])
        h = np.linalg.norm(inner[:2] - outer[:2])
        return v / h if h > 1e-6 else 0.3

    def _brow_position(self, lms: np.ndarray, side: str, fh: float) -> float:
        """Eyebrow-to-eye distance (normalized)."""
        if side == "left":
            brow = lms[FaceLandmarkIndex.LEFT_EYEBROW_MIDDLE]
            eye = lms[FaceLandmarkIndex.LEFT_EYE_TOP]
        else:
            brow = lms[FaceLandmarkIndex.RIGHT_EYEBROW_MIDDLE]
            eye = lms[FaceLandmarkIndex.RIGHT_EYE_TOP]
        return np.linalg.norm(brow[:2] - eye[:2]) / fh

    def _nose_wrinkle(self, lms: np.ndarray, fh: float) -> float:
        tip = lms[FaceLandmarkIndex.NOSE_TIP]
        bridge = lms[FaceLandmarkIndex.NOSE_BRIDGE]
        return np.linalg.norm(tip[:2] - bridge[:2]) / fh

    def _upper_lip_position(self, lms: np.ndarray, fh: float) -> float:
        upper_lip = lms[FaceLandmarkIndex.UPPER_LIP_TOP]
        nose_tip = lms[FaceLandmarkIndex.NOSE_TIP]
        return np.linalg.norm(upper_lip[:2] - nose_tip[:2]) / fh

    def _mouth_aspect_ratio(self, lms: np.ndarray) -> float:
        top = lms[FaceLandmarkIndex.UPPER_LIP_BOTTOM]
        bottom = lms[FaceLandmarkIndex.LOWER_LIP_TOP]
        left = lms[FaceLandmarkIndex.MOUTH_LEFT]
        right = lms[FaceLandmarkIndex.MOUTH_RIGHT]

        v = np.linalg.norm(top[:2] - bottom[:2])
        h = np.linalg.norm(left[:2] - right[:2])
        return v / h if h > 1e-6 else 0.2

    def _compute_au_activations(self, raw: Dict[str, float]) -> Dict[str, float]:
        """Compute AU activations from raw measurements vs baseline.

        Values are scaled to 0-5 range to match PSPI expectations.
        """
        if not self._baseline:
            return {au: 0.0 for au in self.AU_THRESHOLDS}

        activations = {}

        base_brow = self._baseline.get("eyebrow_position", 0.08)
        curr_brow = raw.get("eyebrow_position", 0.08)
        if base_brow > 1e-6:
            # Scale to 0-5 range
            activations["AU4"] = min(5.0, max(0, (base_brow - curr_brow) / base_brow * 5.0))
        else:
            activations["AU4"] = 0.0

        base_ear = self._baseline.get("eye_aspect_ratio", 0.28)
        curr_ear = raw.get("eye_aspect_ratio", 0.28)
        if base_ear > 1e-6:
            eye_change = max(0, (base_ear - curr_ear) / base_ear)
            activations["AU6"] = min(5.0, eye_change * 4.0)
            activations["AU7"] = min(5.0, eye_change * 5.0)
            activations["AU43"] = min(5.0, eye_change * 5.0)
        else:
            activations["AU6"] = 0.0
            activations["AU7"] = 0.0
            activations["AU43"] = 0.0

        base_nose = self._baseline.get("nose_wrinkle", 0.12)
        curr_nose = raw.get("nose_wrinkle", 0.12)
        if base_nose > 1e-6:
            activations["AU9"] = min(5.0, max(0, abs(base_nose - curr_nose) / base_nose * 5.0))
        else:
            activations["AU9"] = 0.0

        base_lip = self._baseline.get("upper_lip_raise", 0.10)
        curr_lip = raw.get("upper_lip_raise", 0.10)
        if base_lip > 1e-6:
            activations["AU10"] = min(5.0, max(0, (base_lip - curr_lip) / base_lip * 5.0))
        else:
            activations["AU10"] = 0.0

        return activations

    def _get_default_baseline(self) -> Dict[str, float]:
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

    @staticmethod
    def _classify_pain(pspi: float) -> str:
        if pspi >= 12:
            return "SEVERE"
        elif pspi >= 8:
            return "MODERATE"
        elif pspi >= 4:
            return "MILD"
        return "NONE"


# ---------------------------------------------------------------------------
# Main Pipeline (backward compatible class name)
# ---------------------------------------------------------------------------
class ActionUnitDetector:
    """
    Multi-backend AU detection pipeline with automatic fallback.

    Backend priority:
        1. Py-Feat (deep learning, best accuracy)
        2. MobileNetV3-Large (lightweight deep learning)
        3. Geometric rules (always available)

    The fallback is seamless — no crash, no user intervention needed.

    Args:
        backend: Force 'pyfeat', 'mobilenet', or 'geometric'. None = auto.
        use_baseline: Calibrate from neutral face (geometric backend only).
        pyfeat_au_model: Py-Feat AU model name.
        mobilenet_model_path: Path to MobileNetV3 weights.

    Example:
        >>> detector = ActionUnitDetector()
        >>> result = detector.detect(face_landmarks)
        >>> print(f"PSPI: {result.pain_score:.1f} ({result.backend})")
    """

    def __init__(
        self,
        backend: Optional[str] = None,
        use_baseline: bool = True,
        pyfeat_au_model: str = "jaanet",
        mobilenet_model_path: Optional[str] = None,
    ):
        self._active_backend = "none"
        self._pyfeat: Optional[_PyFeatBackend] = None
        self._mobilenet: Optional[_MobileNetBackend] = None
        self._geometric = _GeometricBackend(use_baseline=use_baseline)

        if backend == "geometric":
            self._active_backend = "geometric"
            return

        # Try Py-Feat first
        if backend in (None, "pyfeat") and _PYFEAT_AVAILABLE:
            try:
                self._pyfeat = _PyFeatBackend(au_model=pyfeat_au_model)
                self._active_backend = "pyfeat"
                logger.info("ActionUnitDetector: Using Py-Feat backend.")
                return
            except Exception as e:
                logger.info(f"Py-Feat init failed: {e}")

        # Try MobileNetV3
        if backend in (None, "mobilenet") and _TORCH_AVAILABLE:
            try:
                self._mobilenet = _MobileNetBackend(model_path=mobilenet_model_path)
                if self._mobilenet.is_ready:
                    self._active_backend = "mobilenet"
                    logger.info("ActionUnitDetector: Using MobileNetV3 backend.")
                    return
            except Exception as e:
                logger.info(f"MobileNetV3 init failed: {e}")

        # Geometric fallback
        self._active_backend = "geometric"
        logger.info("ActionUnitDetector: Using geometric fallback.")

    @property
    def backend(self) -> str:
        """Currently active backend."""
        return self._active_backend

    def set_baseline(self, landmarks: np.ndarray) -> None:
        """Set baseline from neutral face (geometric backend only)."""
        self._geometric.set_baseline(landmarks)

    def detect(
        self,
        face_landmarks: np.ndarray,
        face_image: Optional[np.ndarray] = None,
    ) -> AUResult:
        """
        Detect Action Units and compute pain score.

        Args:
            face_landmarks: (468, 3) Face Mesh landmarks.
            face_image: BGR image (H, W, 3). Required for deep learning backends.

        Returns:
            AUResult with AU values, PSPI score, pain level, and backend info.
        """
        result = None

        # Try Py-Feat
        if self._active_backend == "pyfeat" and self._pyfeat is not None:
            result = self._pyfeat.detect(face_image, face_landmarks)
            if result.is_valid:
                return result
            logger.info("Py-Feat failed, trying next backend...")

        # Try MobileNetV3
        if self._active_backend in ("pyfeat", "mobilenet") and self._mobilenet is not None:
            result = self._mobilenet.detect(face_image, face_landmarks)
            if result.is_valid:
                return result
            logger.info("MobileNetV3 failed, trying next backend...")

        # Geometric fallback
        return self._geometric.detect(face_image, face_landmarks)

    def reset(self) -> None:
        """Reset detector state."""
        self._geometric._baseline = None
        self._geometric._is_calibrated = False
