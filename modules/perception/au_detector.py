"""
Action Unit Detector using py-feat.

Detects FACS Action Units for pain assessment.
Key AUs: AU4 (brow lowerer), AU6/7 (eye tightening), AU9/10 (grimace), AU43 (eye closure).
"""

from dataclasses import dataclass, field
from typing import Dict
import numpy as np


@dataclass
class AUResult:
    au_activations: Dict[str, float] = field(default_factory=dict)
    pain_score: float = 0.0  # PSPI-based
    is_valid: bool = False
    error_message: str = ""


class ActionUnitDetector:
    """AU detector using py-feat library."""

    def __init__(self, backend: str = "jaanet"):
        self._backend = backend
        self._detector = None
        self._is_initialized = False

    def initialize(self, **kwargs) -> bool:
        try:
            from feat import Detector
            self._detector = Detector(face_model=self._backend, landmark_model="mobilefacenet", au_model=self._backend)
            self._is_initialized = True
            return True
        except ImportError:
            print("[AU] Install: pip install py-feat")
            return False
        except Exception as e:
            print(f"[AU] Init failed: {e}")
            return False

    def detect(self, frame: np.ndarray) -> AUResult:
        if not self._is_initialized:
            return AUResult(error_message="Not initialized")
        try:
            result = self._detector.detect_image(frame)
            aus = result.aus.iloc[0].to_dict() if len(result.aus) > 0 else {}
            pspi = aus.get("AU4", 0) + max(aus.get("AU6", 0), aus.get("AU7", 0)) + \
                   max(aus.get("AU9", 0), aus.get("AU10", 0)) + \
                   max(aus.get("AU20", 0), aus.get("AU25", 0), aus.get("AU26", 0))
            return AUResult(au_activations=aus, pain_score=float(pspi), is_valid=True)
        except Exception as e:
            return AUResult(error_message=str(e))

    def close(self) -> None:
        self._is_initialized = False
