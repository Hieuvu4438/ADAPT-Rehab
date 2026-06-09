"""
Quaternion-Based Joint Angle Computation.

Uses the robust Melax (1998) formulation for computing rotation quaternions
from two vectors. This avoids the numerical issues of arccos near 0 and pi.

References:
    - Melax, S. (1998). "The Shortest Arc Quaternion." Game Programming Gems.
    - Horn, B.K.P. (1987). "Closed-form solution of absolute orientation using
      unit quaternions." J Opt Soc Am A, 4(4), 629-642.
    - Wu, G. et al. (2005). "ISB recommendation on definitions of joint
      coordinate systems." J Biomech, 38(5), 981-992.

Usage:
    from core.kinematics_quaternion import QuaternionKinematics

    qk = QuaternionKinematics()
    angle = qk.compute_angle(point_a, point_b, point_c)
"""

from typing import Dict, Tuple
import numpy as np


class QuaternionKinematics:
    """Quaternion-based joint angle computation.

    Uses the robust Melax (1998) sqrt formulation instead of circular
    arccos-based quaternion construction. The cross product axis is
    properly used for the rotation, making this superior to the
    dot-product-only method.
    """

    @staticmethod
    def compute_angle(point_a: np.ndarray, point_b: np.ndarray, point_c: np.ndarray) -> float:
        """Compute joint angle using robust quaternion rotation (Melax, 1998).

        Uses sqrt((1+d)*2) formulation which avoids arccos numerical issues
        near 0 and pi degrees.

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

        # Dot product (clamped for numerical stability)
        d = np.clip(np.dot(v1, v2), -1.0, 1.0)

        # Anti-parallel edge case (180 degrees)
        if d < -0.999999:
            return 180.0

        # Robust Melax (1998) formulation
        # s = 2*cos(theta/2), avoids arccos entirely
        s = np.sqrt((1.0 + d) * 2.0)
        invs = 1.0 / s

        # Quaternion components
        w = s * 0.5  # cos(theta/2)
        xyz = np.cross(v1, v2) * invs  # sin(theta/2) * axis

        # Extract angle from quaternion: theta = 2 * arccos(w)
        # w is already cos(theta/2), so this recovers the angle
        w_clamped = np.clip(w, 0.0, 1.0)
        angle_rad = 2.0 * np.arccos(w_clamped)

        return float(np.degrees(angle_rad))

    @staticmethod
    def compute_angle_with_axis(
        point_a: np.ndarray,
        point_b: np.ndarray,
        point_c: np.ndarray,
    ) -> Tuple[float, np.ndarray]:
        """Compute angle and rotation axis using robust quaternion.

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

        d = np.clip(np.dot(v1, v2), -1.0, 1.0)

        # Anti-parallel edge case
        if d < -0.999999:
            # Find a perpendicular axis
            axis = np.cross(np.array([1.0, 0.0, 0.0]), v1)
            if np.linalg.norm(axis) < 1e-6:
                axis = np.cross(np.array([0.0, 1.0, 0.0]), v1)
            axis = axis / np.linalg.norm(axis)
            return 180.0, axis

        # Robust formulation
        s = np.sqrt((1.0 + d) * 2.0)
        invs = 1.0 / s

        cross = np.cross(v1, v2)
        xyz = cross * invs  # sin(theta/2) * axis

        # Extract axis from xyz component
        xyz_norm = np.linalg.norm(xyz)
        if xyz_norm > 1e-10:
            axis = xyz / xyz_norm
        else:
            # Vectors are nearly parallel, axis is ambiguous
            axis = np.array([0, 0, 1])

        # Extract angle
        w = s * 0.5
        w_clamped = np.clip(w, 0.0, 1.0)
        angle_rad = 2.0 * np.arccos(w_clamped)

        return float(np.degrees(angle_rad)), axis

    @staticmethod
    def compute_clinical_angle(point_a: np.ndarray, point_b: np.ndarray,
                               point_c: np.ndarray, joint_type: str = 'flexion') -> float:
        """Compute clinical joint angle (ISB convention).

        For flexion joints (knee, elbow): clinical_angle = 180 - included_angle
        This gives 0 degrees at full extension (anatomical zero).

        Args:
            point_a: Proximal joint position
            point_b: Vertex joint position
            point_c: Distal joint position
            joint_type: 'flexion' or 'extension' (default: 'flexion')

        Returns:
            Clinical angle in degrees.
        """
        included_angle = QuaternionKinematics.compute_angle(point_a, point_b, point_c)

        if joint_type == 'flexion':
            # For knee/elbow: 0 = full extension, positive = flexion
            return 180.0 - included_angle
        else:
            return included_angle

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
