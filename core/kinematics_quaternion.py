"""
Quaternion-Based Joint Angle Computation.

Replaces dot-product method with quaternion rotation for:
- No gimbal lock at 90 degrees
- Sequence-independent 3D rotation
- Better accuracy in frontal/transverse planes

Reference: Aurand et al. (2024), "Euler Angles vs. Quaternions for Joint Angles
in Gait Analysis," IEEE.

Usage:
    from core.kinematics_quaternion import QuaternionKinematics

    qk = QuaternionKinematics()
    angle = qk.compute_angle(point_a, point_b, point_c)
"""

from typing import Dict, Tuple
import numpy as np


class QuaternionKinematics:
    """Quaternion-based joint angle computation.

    Uses proper quaternion rotation to compute angles between 3D vectors.
    The rotation quaternion is formed from the cross product (axis) and
    dot product (angle) between two vectors.
    """

    @staticmethod
    def compute_angle(point_a: np.ndarray, point_b: np.ndarray, point_c: np.ndarray) -> float:
        """Compute joint angle using quaternion rotation.

        Forms a proper quaternion from the rotation between two vectors:
        q = [cos(θ/2), sin(θ/2) * axis]

        Args:
            point_a: Proximal joint position (e.g., shoulder)
            point_b: Vertex joint position (e.g., elbow)
            point_c: Distal joint position (e.g., wrist)

        Returns:
            Joint angle in degrees [0, 180].
        """
        v1 = point_a - point_b
        v2 = point_c - point_b

        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)

        if n1 < 1e-10 or n2 < 1e-10:
            return 0.0

        v1 = v1 / n1
        v2 = v2 / n2

        # Compute dot product and cross product
        dot = np.clip(np.dot(v1, v2), -1.0, 1.0)
        cross = np.cross(v1, v2)
        cross_norm = np.linalg.norm(cross)

        # Form quaternion: q = [w, x, y, z]
        # w = cos(θ/2)
        # [x, y, z] = sin(θ/2) * axis
        half_angle = np.arccos(dot) / 2.0
        w = np.cos(half_angle)

        if cross_norm > 1e-10:
            axis = cross / cross_norm
            xyz = np.sin(half_angle) * axis
        else:
            # Vectors are parallel or anti-parallel
            xyz = np.zeros(3)

        # Extract angle from quaternion: θ = 2 * arccos(w)
        # Clamp w to [-1, 1] for numerical stability
        w_clamped = np.clip(w, -1.0, 1.0)
        angle_rad = 2.0 * np.arccos(w_clamped)

        return float(np.degrees(angle_rad))

    @staticmethod
    def compute_angle_with_axis(
        point_a: np.ndarray,
        point_b: np.ndarray,
        point_c: np.ndarray,
    ) -> Tuple[float, np.ndarray]:
        """Compute angle and rotation axis using quaternion.

        Args:
            point_a: Proximal joint position
            point_b: Vertex joint position
            point_c: Distal joint position

        Returns:
            Tuple of (angle_degrees, rotation_axis).
        """
        v1 = point_a - point_b
        v2 = point_c - point_b

        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)

        if n1 < 1e-10 or n2 < 1e-10:
            return 0.0, np.array([0, 0, 1])

        v1 = v1 / n1
        v2 = v2 / n2

        # Compute rotation axis from cross product
        cross = np.cross(v1, v2)
        cross_norm = np.linalg.norm(cross)

        if cross_norm < 1e-10:
            # Vectors are parallel
            return 0.0, np.array([0, 0, 1])

        axis = cross / cross_norm

        # Compute angle from dot product
        dot = np.clip(np.dot(v1, v2), -1.0, 1.0)
        angle_rad = np.arccos(dot)

        return float(np.degrees(angle_rad)), axis

    @staticmethod
    def compute_all(keypoints_3d: np.ndarray, joint_defs: Dict) -> Dict[str, float]:
        """Compute angles for all defined joints.

        Args:
            keypoints_3d: 3D keypoints array, shape (N, 3)
            joint_defs: Dict mapping joint name to (proximal, vertex, distal) indices

        Returns:
            Dict mapping joint name to angle in degrees.
        """
        angles = {}
        for name, (p, v, d) in joint_defs.items():
            try:
                if max(p, v, d) >= len(keypoints_3d):
                    continue
                angles[name] = QuaternionKinematics.compute_angle(
                    keypoints_3d[p], keypoints_3d[v], keypoints_3d[d]
                )
            except (IndexError, ValueError):
                continue
        return angles

    @staticmethod
    def slerp(q1: np.ndarray, q2: np.ndarray, t: float) -> np.ndarray:
        """Spherical Linear Interpolation between quaternions.

        Args:
            q1: First quaternion [w, x, y, z]
            q2: Second quaternion [w, x, y, z]
            t: Interpolation parameter [0, 1]

        Returns:
            Interpolated quaternion.
        """
        dot = np.clip(np.dot(q1, q2), -1, 1)
        if dot < 0:
            q2, dot = -q2, -dot
        if dot > 0.9995:
            r = q1 + t * (q2 - q1)
            return r / np.linalg.norm(r)
        theta = np.arccos(dot)
        return (np.sin((1 - t) * theta) * q1 + np.sin(t * theta) * q2) / np.sin(theta)

    @staticmethod
    def quaternion_to_angle(q: np.ndarray) -> float:
        """Extract angle from quaternion.

        Args:
            q: Quaternion [w, x, y, z]

        Returns:
            Angle in degrees.
        """
        w = np.clip(q[0], -1.0, 1.0)
        return float(2.0 * np.degrees(np.arccos(w)))
