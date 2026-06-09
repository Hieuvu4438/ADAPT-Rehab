"""
Perception Modules.

Face analysis using OpenFace 3.0 for AU-based state detection
(Pain, Fatigue, Exhaustion, Boredom, Normal) during rehabilitation.

All detection formulas are from peer-reviewed literature:
- PSPI: Prkachin & Solomon (2008), Pain
- PERCLOS: Wierwille et al. (1994), NHTSA
- Engagement: Whitehill et al. (2014), IEEE TAFFC
- EAR: Soukupova & Cech (2016), CVWW

Version: 4.0.0
"""

from .face_detector import FaceDetector, FaceResult, FaceLandmarkIndex
from .openface_analyzer import OpenFaceAnalyzer, OpenFaceResult, EMOTION_LABELS
from .facial_state_detector import (
    FacialStateDetector,
    FacialStateResult,
    FacialState,
    AUData,
    PSPICalculator,
    PERCLOSCalculator,
    BlinkDetector,
    YawnDetector,
    EngagementCalculator,
    FatigueScoreCalculator,
    EARCalculator,
)

__all__ = [
    # Face detection (MediaPipe)
    "FaceDetector", "FaceResult", "FaceLandmarkIndex",
    # OpenFace 3.0 analyzer
    "OpenFaceAnalyzer", "OpenFaceResult", "EMOTION_LABELS",
    # AU-based state detection
    "FacialStateDetector", "FacialStateResult", "FacialState", "AUData",
    "PSPICalculator", "PERCLOSCalculator", "BlinkDetector", "YawnDetector",
    "EngagementCalculator", "FatigueScoreCalculator", "EARCalculator",
]
