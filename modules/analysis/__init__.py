"""
Analysis Modules.

Kinematic analysis and behavioral state detection from 3D body keypoints.

All formulas are from peer-reviewed literature:
- Velocity/Acceleration: Winter (2009), Biomechanics
- Velocity Loss: Gonzalez-Badillo & Sanchez-Medina (2010), IJSM
- SPARC: Balasubramanian et al. (2015), IEEE TBME
- Trunk Inclination: Wu et al. (2005), J Biomechanics
- Mann-Kendall: Mann (1945), Econometrica
- Asymmetry Index: Clinical standard

Version: 4.0.0
"""

from .body_state_detector import (
    BodyStateDetector,
    BodyStateResult,
    BodyState,
    KinematicMetrics,
    JointAngleCalculator,
    KinematicsCalculator,
    SPARCCalculator,
    TrunkInclinationCalculator,
    AsymmetryCalculator,
    MannKendallTrendAnalyzer,
    KeypointIndex,
)

__all__ = [
    "BodyStateDetector", "BodyStateResult", "BodyState", "KinematicMetrics",
    "JointAngleCalculator", "KinematicsCalculator", "SPARCCalculator",
    "TrunkInclinationCalculator", "AsymmetryCalculator",
    "MannKendallTrendAnalyzer", "KeypointIndex",
]
