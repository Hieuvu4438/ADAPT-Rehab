"""
Quaternion-Based Joint Angle Computation.

Replaces dot-product method with quaternion rotation for:
- No gimbal lock at 90 degrees
- Sequence-independent 3D rotation
- Better accuracy in frontal/transverse planes

Usage:
    from core.kinematics_quaternion import QuaternionKinematics

    qk = QuaternionKinematics()
    angle = qk.compute_angle(point_a, point_b, point_c)
"""

from typing import Dict, Tuple
import numpy as np


class QuaternionKinematics:
    """Quaternion-based joint angle computation."""

    @staticmethod
    def compute_angle(point_a: np.ndarray, point_b: np.ndarray, point_c: np.ndarray) -> float:
        """Compute joint angle using quaternion rotation. Returns degrees [0, 180]."""
        v1 = point_a - point_b
        v2 = point_c - point_b
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 < 1e-10 or n2 < 1e-10:
            return 0.0
        v1, v2 = v1 / n1, v2 / n2
        dot = np.clip(np.dot(v1, v2), -1.0, 1.0)
        w = np.sqrt(max(0, (1 + dot) / 2))
        return float(2 * np.degrees(np.arccos(np.clip(w, 0, 1))))

    @staticmethod
    def compute_angle_with_axis(point_a: np.ndarray, point_b: np.ndarray, point_c: np.ndarray) -> Tuple[float, np.ndarray]:
        """Compute angle and rotation axis."""
        v1 = point_a - point_b
        v2 = point_c - point_b
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 < 1e-10 or n2 < 1e-10:
            return 0.0, np.array([0, 0, 1])
        v1, v2 = v1 / n1, v2 / n2
        axis = np.cross(v1, v2)
        axis_norm = np.linalg.norm(axis)
        if axis_norm < 1e-10:
            return 0.0, np.array([0, 0, 1])
        axis = axis / axis_norm
        dot = np.clip(np.dot(v1, v2), -1, 1)
        return float(np.degrees(np.arccos(dot))), axis

    @staticmethod
    def compute_all(keypoints_3d: np.ndarray, joint_defs: Dict) -> Dict[str, float]:
        """Compute angles for all defined joints."""
        angles = {}
        for name, (p, v, d) in joint_defs.items():
            try:
                if max(p, v, d) >= len(keypoints_3d):
                    continue
                angles[name] = QuaternionKinematics.compute_angle(keypoints_3d[p], keypoints_3d[v], keypoints_3d[d])
            except (IndexError, ValueError):
                continue
        return angles

    @staticmethod
    def slerp(q1: np.ndarray, q2: np.ndarray, t: float) -> np.ndarray:
        """Spherical Linear Interpolation between quaternions."""
        dot = np.clip(np.dot(q1, q2), -1, 1)
        if dot < 0:
            q2, dot = -q2, -dot
        if dot > 0.9995:
            r = q1 + t * (q2 - q1)
            return r / np.linalg.norm(r)
        theta = np.arccos(dot)
        return (np.sin((1 - t) * theta) * q1 + np.sin(t * theta) * q2) / np.sin(theta)
