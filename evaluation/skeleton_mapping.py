"""
Skeleton joint mapping between different pose estimation frameworks.

Provides mapping between:
- MediaPipe (33 landmarks)
- SMPL-24 (MeTRAbs default)
- Kinect v2 (25 joints, used in UI-PRMD dataset)

For evaluation, we use a common 14-joint subset that exists in all skeletons.

Reference:
- MediaPipe: https://google.github.io/mediapipe/solutions/pose.html
- SMPL-24: MeTRAbs paper (Sarandi et al., WACV 2021)
- Kinect v2: Microsoft Kinect SDK documentation
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from enum import Enum


class SkeletonType(Enum):
    MEDIAPIPE = "mediapipe"  # 33 landmarks
    SMPL24 = "smpl_24"       # 24 joints (MeTRAbs)
    KINECT_V2 = "kinect_v2"  # 25 joints (UI-PRMD)
    COMMON14 = "common14"    # 14-joint evaluation subset


# ─── Full skeleton definitions ───────────────────────────────────────────────

# MediaPipe 33 landmarks
MEDIAPIPE_JOINTS = {
    "nose": 0, "left_eye_inner": 1, "left_eye": 2, "left_eye_outer": 3,
    "right_eye_inner": 4, "right_eye": 5, "right_eye_outer": 6,
    "left_ear": 7, "right_ear": 8, "mouth_left": 9, "mouth_right": 10,
    "left_shoulder": 11, "right_shoulder": 12,
    "left_elbow": 13, "right_elbow": 14,
    "left_wrist": 15, "right_wrist": 16,
    "left_pinky": 17, "right_pinky": 18,
    "left_index": 19, "right_index": 20,
    "left_thumb": 21, "right_thumb": 22,
    "left_hip": 23, "right_hip": 24,
    "left_knee": 25, "right_knee": 26,
    "left_ankle": 27, "right_ankle": 28,
    "left_heel": 29, "right_heel": 30,
    "left_foot_index": 31, "right_foot_index": 32,
}

# SMPL-24 joints (MeTRAbs default)
SMPL24_JOINTS = {
    "pelvis": 0, "left_hip": 1, "right_hip": 2,
    "spine1": 3, "left_knee": 4, "right_knee": 5,
    "spine2": 6, "left_ankle": 7, "right_ankle": 8,
    "spine3": 9, "left_foot": 10, "right_foot": 11,
    "neck": 12, "left_collar": 13, "right_collar": 14,
    "head": 15, "left_shoulder": 16, "right_shoulder": 17,
    "left_elbow": 18, "right_elbow": 19,
    "left_wrist": 20, "right_wrist": 21,
    "left_hand": 22, "right_hand": 23,
}

# Kinect v2 (25 joints) — used in UI-PRMD dataset
KINECT_V2_JOINTS = {
    "spine_base": 0, "spine_mid": 1, "neck": 2, "head": 3,
    "shoulder_left": 4, "elbow_left": 5, "wrist_left": 6, "hand_left": 7,
    "shoulder_right": 8, "elbow_right": 9, "wrist_right": 10, "hand_right": 11,
    "hip_left": 12, "knee_left": 13, "ankle_left": 14, "foot_left": 15,
    "hip_right": 16, "knee_right": 17, "ankle_right": 18, "foot_right": 19,
    "spine_shoulder": 20, "hand_tip_left": 21, "thumb_left": 22,
    "hand_tip_right": 23, "thumb_right": 24,
}


# ─── Common 14-joint evaluation subset ───────────────────────────────────────

# Standard evaluation joints that exist in all three skeletons
# Naming follows H36M convention for compatibility with literature
COMMON14_NAMES = [
    "pelvis",       # root / hip center
    "spine",        # mid-spine / torso
    "neck",         # neck / throat
    "head",         # head top
    "left_shoulder",
    "left_elbow",
    "left_wrist",
    "right_shoulder",
    "right_elbow",
    "right_wrist",
    "left_hip",
    "left_knee",
    "left_ankle",
    "right_hip",
]

# Mapping: common14 index → source skeleton index
COMMON14_FROM_MEDIAPIPE = {
    "pelvis": None,  # need to average left_hip(23) and right_hip(24)
    "spine": None,    # approximate from shoulders midpoint or use 23/24 midpoint
    "neck": None,     # approximate from shoulder midpoint
    "head": 0,        # nose
    "left_shoulder": 11,
    "left_elbow": 13,
    "left_wrist": 15,
    "right_shoulder": 12,
    "right_elbow": 14,
    "right_wrist": 16,
    "left_hip": 23,
    "left_knee": 25,
    "left_ankle": 27,
    "right_hip": 24,
}

COMMON14_FROM_SMPL24 = {
    "pelvis": 0,
    "spine": 3,       # spine1
    "neck": 12,
    "head": 15,
    "left_shoulder": 16,
    "left_elbow": 18,
    "left_wrist": 20,
    "right_shoulder": 17,
    "right_elbow": 19,
    "right_wrist": 21,
    "left_hip": 1,
    "left_knee": 4,
    "left_ankle": 7,
    "right_hip": 2,
}

COMMON14_FROM_KINECT_V2 = {
    "pelvis": None,  # average hip_left(12) and hip_right(16)
    "spine": 1,       # spine_mid
    "neck": 2,
    "head": 3,
    "left_shoulder": 4,
    "left_elbow": 5,
    "left_wrist": 6,
    "right_shoulder": 8,
    "right_elbow": 9,
    "right_wrist": 10,
    "left_hip": 12,
    "left_knee": 13,
    "left_ankle": 14,
    "right_hip": 16,
}


# ─── H36M 17-joint skeleton (standard in literature) ────────────────────────

H36M_17_NAMES = [
    "pelvis", "right_hip", "right_knee", "right_ankle",
    "left_hip", "left_knee", "left_ankle",
    "spine", "neck", "head",
    "left_shoulder", "left_elbow", "left_wrist",
    "right_shoulder", "right_elbow", "right_wrist",
    "head_top",
]

H36M_17_FROM_SMPL24 = {
    "pelvis": 0, "right_hip": 2, "right_knee": 5, "right_ankle": 8,
    "left_hip": 1, "left_knee": 4, "left_ankle": 7,
    "spine": 3, "neck": 12, "head": 15,
    "left_shoulder": 16, "left_elbow": 18, "left_wrist": 20,
    "right_shoulder": 17, "right_elbow": 19, "right_wrist": 21,
    "head_top": 15,  # approximate with head
}

H36M_17_FROM_MEDIAPIPE = {
    "pelvis": None,  # average hips
    "right_hip": 24, "right_knee": 26, "right_ankle": 28,
    "left_hip": 23, "left_knee": 25, "left_ankle": 27,
    "spine": None,   # midpoint
    "neck": None,    # midpoint of shoulders
    "head": 0,       # nose
    "left_shoulder": 11, "left_elbow": 13, "left_wrist": 15,
    "right_shoulder": 12, "right_elbow": 14, "right_wrist": 16,
    "head_top": 0,   # approximate with nose
}

H36M_17_FROM_KINECT_V2 = {
    "pelvis": None,  # average hips
    "right_hip": 16, "right_knee": 17, "right_ankle": 18,
    "left_hip": 12, "left_knee": 13, "left_ankle": 14,
    "spine": 1, "neck": 2, "head": 3,
    "left_shoulder": 4, "left_elbow": 5, "left_wrist": 6,
    "right_shoulder": 8, "right_elbow": 9, "right_wrist": 10,
    "head_top": 3,
}


# ─── Remapping functions ────────────────────────────────────────────────────

def remap_keypoints(
    keypoints: np.ndarray,
    source: SkeletonType,
    target: SkeletonType,
) -> np.ndarray:
    """Remap keypoints from source skeleton to target skeleton.

    Args:
        keypoints: Source keypoints, shape (..., N_src, 3).
        source: Source skeleton type.
        target: Target skeleton type.

    Returns:
        Remapped keypoints, shape (..., N_tgt, 3).

    Raises:
        ValueError: If mapping is not supported.
    """
    if source == target:
        return keypoints

    # Select the appropriate mapping
    mapping = _get_mapping(source, target)

    # Build index array for advanced indexing
    result = np.zeros(keypoints.shape[:-2] + (len(mapping), 3), dtype=keypoints.dtype)

    for target_idx, (joint_name, source_idx) in enumerate(mapping.items()):
        if source_idx is not None:
            # Direct mapping
            result[..., target_idx, :] = keypoints[..., source_idx, :]
        else:
            # Derived joint (e.g., pelvis = average of hips)
            result[..., target_idx, :] = _derive_joint(
                keypoints, joint_name, source
            )

    return result


def remap_to_common14(
    keypoints: np.ndarray,
    source: SkeletonType,
) -> np.ndarray:
    """Convenience: remap any skeleton to common 14-joint evaluation set."""
    return remap_keypoints(keypoints, source, SkeletonType.COMMON14)


def remap_to_h36m17(
    keypoints: np.ndarray,
    source: SkeletonType,
) -> np.ndarray:
    """Convenience: remap any skeleton to H36M 17-joint evaluation set."""
    mapping_key = f"{source.value}_to_h36m17"
    if source == SkeletonType.SMPL24:
        mapping = H36M_17_FROM_SMPL24
    elif source == SkeletonType.MEDIAPIPE:
        mapping = H36M_17_FROM_MEDIAPIPE
    elif source == SkeletonType.KINECT_V2:
        mapping = H36M_17_FROM_KINECT_V2
    else:
        raise ValueError(f"Cannot remap from {source} to H36M-17")

    result = np.zeros(keypoints.shape[:-2] + (len(mapping), 3), dtype=keypoints.dtype)
    for target_idx, (joint_name, source_idx) in enumerate(mapping.items()):
        if source_idx is not None:
            result[..., target_idx, :] = keypoints[..., source_idx, :]
        else:
            result[..., target_idx, :] = _derive_joint(keypoints, joint_name, source)
    return result


def _get_mapping(source: SkeletonType, target: SkeletonType) -> Dict[str, Optional[int]]:
    """Get the joint index mapping from source to target."""
    if target == SkeletonType.COMMON14:
        if source == SkeletonType.MEDIAPIPE:
            return COMMON14_FROM_MEDIAPIPE
        elif source == SkeletonType.SMPL24:
            return COMMON14_FROM_SMPL24
        elif source == SkeletonType.KINECT_V2:
            return COMMON14_FROM_KINECT_V2
    elif target == SkeletonType.KINECT_V2 and source == SkeletonType.SMPL24:
        return _build_smpl_to_kinect()
    elif target == SkeletonType.SMPL24 and source == SkeletonType.KINECT_V2:
        return _build_kinect_to_smpl()

    raise ValueError(f"Unsupported mapping: {source.value} → {target.value}")


def _derive_joint(
    keypoints: np.ndarray,
    joint_name: str,
    source: SkeletonType,
) -> np.ndarray:
    """Derive a joint position from other joints (e.g., pelvis = avg of hips)."""
    if joint_name == "pelvis":
        if source == SkeletonType.MEDIAPIPE:
            return (keypoints[..., 23, :] + keypoints[..., 24, :]) / 2
        elif source == SkeletonType.KINECT_V2:
            return (keypoints[..., 12, :] + keypoints[..., 16, :]) / 2
    elif joint_name == "spine":
        if source == SkeletonType.MEDIAPIPE:
            # Approximate as midpoint between hips and shoulders
            hip_mid = (keypoints[..., 23, :] + keypoints[..., 24, :]) / 2
            shoulder_mid = (keypoints[..., 11, :] + keypoints[..., 12, :]) / 2
            return (hip_mid + shoulder_mid) / 2
    elif joint_name == "neck":
        if source == SkeletonType.MEDIAPIPE:
            return (keypoints[..., 11, :] + keypoints[..., 12, :]) / 2
    elif joint_name == "head_top":
        if source == SkeletonType.MEDIAPIPE:
            return keypoints[..., 0, :]  # nose as approximation

    # Fallback: return zeros
    return np.zeros(keypoints.shape[:-1] + (3,))


def _build_smpl_to_kinect() -> Dict[str, Optional[int]]:
    """Build SMPL-24 → Kinect v2 mapping."""
    return {
        "spine_base": 0,     # pelvis
        "spine_mid": 3,      # spine1
        "neck": 12,
        "head": 15,
        "shoulder_left": 16,
        "elbow_left": 18,
        "wrist_left": 20,
        "hand_left": 22,
        "shoulder_right": 17,
        "elbow_right": 19,
        "wrist_right": 21,
        "hand_right": 23,
        "hip_left": 1,
        "knee_left": 4,
        "ankle_left": 7,
        "foot_left": 10,
        "hip_right": 2,
        "knee_right": 5,
        "ankle_right": 8,
        "foot_right": 11,
        "spine_shoulder": 9,  # spine3
        "hand_tip_left": 22,  # same as hand_left
        "thumb_left": 22,
        "hand_tip_right": 23,
        "thumb_right": 23,
    }


def _build_kinect_to_smpl() -> Dict[str, Optional[int]]:
    """Build Kinect v2 → SMPL-24 mapping."""
    return {
        "pelvis": 0,
        "left_hip": 12,
        "right_hip": 16,
        "spine1": 1,
        "left_knee": 13,
        "right_knee": 17,
        "spine2": None,  # derive
        "left_ankle": 14,
        "right_ankle": 18,
        "spine3": 20,    # spine_shoulder
        "left_foot": 15,
        "right_foot": 19,
        "neck": 2,
        "left_collar": None,
        "right_collar": None,
        "head": 3,
        "left_shoulder": 4,
        "right_shoulder": 8,
        "left_elbow": 5,
        "right_elbow": 9,
        "left_wrist": 6,
        "right_wrist": 10,
        "left_hand": 7,
        "right_hand": 11,
    }


def get_joint_names(skeleton: SkeletonType) -> List[str]:
    """Get ordered list of joint names for a skeleton type."""
    if skeleton == SkeletonType.COMMON14:
        return COMMON14_NAMES
    elif skeleton == SkeletonType.MEDIAPIPE:
        return list(MEDIAPIPE_JOINTS.keys())
    elif skeleton == SkeletonType.SMPL24:
        return list(SMPL24_JOINTS.keys())
    elif skeleton == SkeletonType.KINECT_V2:
        return list(KINECT_V2_JOINTS.keys())
    else:
        raise ValueError(f"Unknown skeleton type: {skeleton}")


def compute_angle_from_keypoints(
    keypoints: np.ndarray,
    proximal_idx: int,
    vertex_idx: int,
    distal_idx: int,
) -> float:
    """Compute joint angle from 3D keypoints using dot product.

    Args:
        keypoints: 3D keypoints, shape (J, 3).
        proximal_idx: Index of proximal joint (e.g., shoulder for elbow angle).
        vertex_idx: Index of vertex joint (the joint being measured).
        distal_idx: Index of distal joint (e.g., wrist for elbow angle).

    Returns:
        Joint angle in degrees.
    """
    a = keypoints[proximal_idx]
    b = keypoints[vertex_idx]
    c = keypoints[distal_idx]

    ba = a - b
    bc = c - b

    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)

    if norm_ba < 1e-10 or norm_bc < 1e-10:
        return 0.0

    cos_angle = np.dot(ba, bc) / (norm_ba * norm_bc)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))
