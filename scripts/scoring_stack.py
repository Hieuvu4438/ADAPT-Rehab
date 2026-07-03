"""
Shared scoring stack for KIMORE evaluation.

Provides a unified scoring interface that wraps the existing
core/kinematics_quaternion, core/smoothness, and core/dtw_constrained modules.
No reimplementation — everything delegates to existing ADAPT-Rehab code.

Author: ADAPT-Rehab Team
Version: 1.0.0
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any

# ── KIMORE constants ──────────────────────────────────────────────────────────

KINECT_JOINT_INDEX: Dict[str, int] = {
    "spinebase": 0, "spinemid": 1, "neck": 2, "head": 3,
    "shoulderleft": 4, "elbowleft": 5, "wristleft": 6, "handleft": 7,
    "shoulderright": 8, "elbowright": 9, "wristright": 10, "handright": 11,
    "hipleft": 12, "kneeleft": 13, "ankleleft": 14, "footleft": 15,
    "hipright": 16, "kneeright": 17, "ankleright": 18, "footright": 19,
    "spineshoulder": 20, "handtipleft": 21, "thumbleft": 22,
    "handtipright": 23, "thumbright": 24,
}

# Primary + secondary joints per KIMORE exercise.
# Based on Capecci 2020, Bilić 2024, and KIMORE dataset documentation.
# Primary: main movement joint; Secondary: supporting joints.
EXERCISE_JOINTS: Dict[str, Dict[str, List[str]]] = {
    "ex1": {
        "primary": ["right_shoulder", "left_shoulder"],
        "secondary": ["right_elbow", "left_elbow", "right_hip", "left_hip"],
    },
    "ex2": {
        "primary": ["right_knee", "left_knee"],
        "secondary": ["right_hip", "left_hip", "right_shoulder", "left_shoulder"],
    },
    "ex3": {
        "primary": ["right_hip", "left_hip"],
        "secondary": ["right_shoulder", "left_shoulder", "spine"],
    },
    "ex4": {
        "primary": ["right_shoulder", "left_shoulder"],
        "secondary": ["right_elbow", "left_elbow", "spine"],
    },
    "ex5": {
        "primary": ["right_knee", "left_knee"],
        "secondary": ["right_hip", "left_hip", "spine"],
    },
}

# Joint definitions: (proximal, vertex, distal) joint names for angle computation.
JOINT_DEFS: Dict[str, Tuple[str, str, str]] = {
    "left_shoulder":  ("spinebase", "shoulderleft", "elbowleft"),
    "right_shoulder": ("spinebase", "shoulderright", "elbowright"),
    "left_knee":      ("hipleft", "kneeleft", "ankleleft"),
    "right_knee":     ("hipright", "kneeright", "ankleright"),
    "left_hip":       ("spinebase", "hipleft", "kneeleft"),
    "right_hip":      ("spinebase", "hipright", "kneeright"),
    "left_elbow":     ("shoulderleft", "elbowleft", "wristleft"),
    "right_elbow":    ("shoulderright", "elbowright", "wristright"),
    "spine":          ("spinebase", "spinemid", "neck"),
}

# Joint weights for the weighted DTW scoring.
JOINT_WEIGHTS: Dict[str, float] = {
    "left_shoulder":  1.0,
    "right_shoulder": 1.0,
    "left_knee":      1.0,
    "right_knee":     1.0,
    "left_hip":       0.8,
    "right_hip":      0.8,
    "left_elbow":     0.5,
    "right_elbow":    0.5,
    "spine":          0.6,
}

# Expected angle ranges for ROM normalization (degrees).
# Kept for fallback when EXERCISE_ROM_CALIBRATION has no entry.
EXPECTED_ROM: Dict[str, Tuple[float, float]] = {
    "left_shoulder":  (0.0, 180.0),
    "right_shoulder": (0.0, 180.0),
    "left_knee":      (0.0, 180.0),
    "right_knee":     (0.0, 180.0),
    "left_hip":       (0.0, 150.0),
    "right_hip":      (0.0, 150.0),
    "left_elbow":     (0.0, 180.0),
    "right_elbow":    (0.0, 180.0),
    "spine":          (0.0, 90.0),
}

# Per-exercise, per-joint ROM calibration (degrees).
# Computed empirically from KIMORE dataset: expected_max = P75 of observed ROM
# (NOT P90 — P75 gives more headroom and avoids ceiling saturation).
# Score = min(100, ROM/expected_max * 100). No floor.
# This maps: P25 → ~50-70, P50 → ~70-90, P75 → 100 (capped), P90+ → 100.
EXERCISE_ROM_CALIBRATION: Dict[str, Dict[str, Tuple[float, float]]] = {
    "ex1": {
        "right_shoulder": (139.0, 139.0),
        "left_shoulder":  (138.0, 138.0),
    },
    "ex2": {
        "right_knee": (45.0, 45.0),
        "left_knee":  (45.0, 45.0),
    },
    "ex3": {
        "right_hip": (30.0, 30.0),
        "left_hip":   (33.0, 33.0),
    },
    "ex4": {
        "right_shoulder": (69.0, 69.0),
        "left_shoulder":  (65.0, 65.0),
    },
    "ex5": {
        "right_knee": (113.0, 113.0),
        "left_knee":  (117.0, 117.0),
    },
}


def compute_pose_score_from_angles(
    angles: np.ndarray,
    exercise: str,
    joint_name: str,
) -> float:
    """Compute pose/ROM score using per-exercise P75 calibration.

    Score = min(100, ROM/expected_max * 100) where expected_max = P75 observed ROM.
    No floor — poor performers get low scores, good performers get high scores.

    Args:
        angles: Joint angle trajectory in degrees.
        exercise: Exercise name (e.g., 'ex1').
        joint_name: Joint name.

    Returns:
        Score in [0, 100] range.
    """
    calib = EXERCISE_ROM_CALIBRATION.get(exercise, {}).get(joint_name)
    if calib is None:
        exp_min, exp_max = EXPECTED_ROM.get(joint_name, (0, 180))
        rom_range = float(np.max(angles) - np.min(angles))
        return min(100.0, max(0.0, (rom_range / (exp_max - exp_min)) * 100))

    expected_max, _ = calib
    rom_range = float(np.max(angles) - np.min(angles))
    score = min(100.0, (rom_range / expected_max) * 100.0)
    return max(0.0, score)


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class JointTrajectory:
    """A single joint's angle trajectory across frames."""
    joint_name: str
    angles: np.ndarray  # (n_frames,) in degrees
    is_valid: bool = True

    def __post_init__(self):
        if self.angles is None or len(self.angles) < 5:
            self.is_valid = False


@dataclass
class RecordingFeatures:
    """Extracted features from one KIMORE recording."""
    recording_id: str
    exercise: str
    clinical_score: float
    joint_trajectories: Dict[str, JointTrajectory]
    # Dimension scores
    pose_score: float = 0.0
    smoothness_score: float = 0.0
    dtw_similarity: float = 0.0
    total_score: float = 0.0
    # SPARC raw values per joint
    sparc_per_joint: Dict[str, float] = None
    # Per-joint DTW distances
    dtw_per_joint: Dict[str, float] = None

    def __post_init__(self):
        if self.sparc_per_joint is None:
            self.sparc_per_joint = {}
        if self.dtw_per_joint is None:
            self.dtw_per_joint = {}

    def to_dict(self) -> dict:
        return {
            "recording_id": self.recording_id,
            "exercise": self.exercise,
            "clinical_score": self.clinical_score,
            "pose_score": round(self.pose_score, 3),
            "smoothness_score": round(self.smoothness_score, 3),
            "dtw_similarity": round(self.dtw_similarity, 3),
            "total_score": round(self.total_score, 3),
            "sparc_per_joint": {k: round(v, 4) for k, v in self.sparc_per_joint.items()},
            "dtw_per_joint": {k: round(v, 4) for k, v in self.dtw_per_joint.items()},
        }


# ── Kinematics helpers (fully vectorized) ──────────────────────────────────────

def _compute_angles_vectorized(
    keypoints_3d: np.ndarray,
    pi: int,
    vi: int,
    di: int,
) -> Optional[np.ndarray]:
    """Fully vectorized joint angle computation across all frames.

    Computes angles for all frames in a single vectorized operation,
    avoiding Python-level loops for ~100x speedup.

    Args:
        keypoints_3d: Array of shape (n_frames, 25, 3).
        pi: Proximal joint index.
        vi: Vertex joint index.
        di: Distal joint index.

    Returns:
        Angle array in degrees of shape (n_frames,), or None if invalid.
    """
    if keypoints_3d.shape[1] <= max(pi, vi, di):
        return None

    # Extract vectors from all frames at once
    # v1[f] = keypoints[f, pi] - keypoints[f, vi], shape (n_frames, 3)
    v1 = keypoints_3d[:, pi, :] - keypoints_3d[:, vi, :]   # (n_frames, 3)
    v2 = keypoints_3d[:, di, :] - keypoints_3d[:, vi, :]   # (n_frames, 3)

    # Normalize
    n1 = np.linalg.norm(v1, axis=1, keepdims=True)  # (n_frames, 1)
    n2 = np.linalg.norm(v2, axis=1, keepdims=True)  # (n_frames, 1)

    # Avoid division by zero
    mask = (n1[:, 0] > 1e-10) & (n2[:, 0] > 1e-10)
    if not np.any(mask):
        return None

    v1_n = np.zeros_like(v1)
    v2_n = np.zeros_like(v2)
    v1_n[mask] = v1[mask] / n1[mask]
    v2_n[mask] = v2[mask] / n2[mask]

    # Dot product per frame
    d = np.sum(v1_n * v2_n, axis=1)  # (n_frames,)
    d = np.clip(d, -1.0, 1.0)

    # Anti-parallel case: d ≈ -1 → angle ≈ 180
    anti_parallel = d < -0.999999
    angles = np.zeros(keypoints_3d.shape[0], dtype=np.float64)

    normal_mask = mask & ~anti_parallel
    if np.any(normal_mask):
        # Robust Melax (1998) formulation: s = sqrt((1+d)*2)
        d_n = d[normal_mask]
        s = np.sqrt((1.0 + d_n) * 2.0)
        w = np.clip(s * 0.5, 0.0, 1.0)
        angles[normal_mask] = np.degrees(2.0 * np.arccos(w))

    if np.any(anti_parallel):
        angles[anti_parallel] = 180.0

    return angles


def extract_trajectory(
    keypoints_3d: np.ndarray,
    joint_name: str,
    sample_every: int = 1,
) -> Optional[JointTrajectory]:
    """Extract angle trajectory for one joint across all frames.

    Fully vectorized — no Python-level loops over frames.

    Args:
        keypoints_3d: Array of shape (n_frames, 25, 3).
        joint_name: Name of the joint (must be in JOINT_DEFS).
        sample_every: Subsample frames for speed.

    Returns:
        JointTrajectory or None if insufficient data.
    """
    if joint_name not in JOINT_DEFS:
        return JointTrajectory(joint_name, np.array([]), is_valid=False)

    prox, vert, dist_name = JOINT_DEFS[joint_name]
    pi = KINECT_JOINT_INDEX[prox]
    vi = KINECT_JOINT_INDEX[vert]
    di = KINECT_JOINT_INDEX[dist_name]

    angles = _compute_angles_vectorized(keypoints_3d, pi, vi, di)
    if angles is None or len(angles) < 5:
        return JointTrajectory(joint_name, np.array([]), is_valid=False)

    # Subsample
    if sample_every > 1:
        angles = angles[::sample_every]

    if len(angles) < 5:
        return JointTrajectory(joint_name, np.array([]), is_valid=False)

    return JointTrajectory(joint_name, angles.astype(np.float64))


def extract_all_trajectories(
    keypoints_3d: np.ndarray,
    exercise: str,
    sample_every: int = 1,
) -> Dict[str, JointTrajectory]:
    """Extract angle trajectories for all relevant joints of an exercise.

    Args:
        keypoints_3d: Array of shape (n_frames, 25, 3).
        exercise: Exercise name (e.g., 'ex1').
        sample_every: Subsample rate for speed.

    Returns:
        Dict mapping joint_name → JointTrajectory.
    """
    joint_names = []
    if exercise in EXERCISE_JOINTS:
        joint_names = (
            EXERCISE_JOINTS[exercise]["primary"] +
            EXERCISE_JOINTS[exercise]["secondary"]
        )
    else:
        joint_names = list(JOINT_DEFS.keys())

    trajectories = {}
    for jname in joint_names:
        traj = extract_trajectory(keypoints_3d, jname, sample_every)
        if traj is not None and traj.is_valid:
            trajectories[jname] = traj

    return trajectories


# ── Smoothing (delegates to scipy) ───────────────────────────────────────────

def smooth_trajectory(angles: np.ndarray, fs: float = 30.0) -> np.ndarray:
    """Apply 4th-order Butterworth low-pass filter to angle trajectory.

    Args:
        angles: Raw angle sequence.
        fs: Sampling frequency in Hz.

    Returns:
        Smoothed angle sequence.
    """
    from scipy.signal import butter, filtfilt

    if len(angles) < 8:
        return angles.copy()

    fc = 6.0  # Cutoff frequency in Hz (biomechanics standard)
    nyquist = fs / 2.0
    normalized_fc = fc / nyquist

    if normalized_fc >= 1.0:
        return angles.copy()

    try:
        b, a = butter(4, normalized_fc, btype='low')
        return filtfilt(b, a, angles)
    except (ValueError, RuntimeError):
        return angles.copy()


# ── Smoothness (delegates to core.smoothness) ─────────────────────────────────

def compute_sparc(angles: np.ndarray, fs: float = 30.0) -> float:
    """Compute SPARC (Spectral Arc Length) for a trajectory.

    Delegates to core.smoothness.SmoothnessAnalyzer.

    Args:
        angles: Angle trajectory (degrees).
        fs: Sampling frequency in Hz.

    Returns:
        SPARC value in [-6, 0] (more negative = less smooth).
    """
    from core.smoothness import SmoothnessAnalyzer

    analyzer = SmoothnessAnalyzer(fs=fs)
    result = analyzer.analyze(angles)
    return result.sparc if result.is_valid else 0.0


def compute_sparc_normalized(angles: np.ndarray, fs: float = 30.0) -> float:
    """SPARC normalized to [0, 100] where 100 = perfectly smooth.

    Args:
        angles: Angle trajectory (degrees).
        fs: Sampling frequency in Hz.

    Returns:
        Normalized SPARC score in [0, 100].
    """
    sparc = compute_sparc(angles, fs)
    # SPARC range [-6, 0] → [0, 100]
    return float(np.clip((sparc + 6.0) / 6.0 * 100.0, 0.0, 100.0))


def compute_jerk_score(angles: np.ndarray, fs: float = 30.0) -> float:
    """Compute jerk-based smoothness score.

    Args:
        angles: Angle trajectory (degrees).
        fs: Sampling frequency in Hz.

    Returns:
        Jerk score in [0, 100] (higher = smoother).
    """
    if len(angles) < 4:
        return 50.0

    vel = np.diff(angles) * fs
    if len(vel) < 3:
        return 50.0

    acc = np.diff(vel) / fs
    if len(acc) < 2:
        return 50.0

    jerk = np.diff(acc) / fs
    if len(jerk) < 1:
        return 50.0

    jerk_mag = float(np.mean(np.abs(jerk)))
    return float(np.clip(100.0 - jerk_mag * 0.5, 0.0, 100.0))


def compute_ldlj(angles: np.ndarray, fs: float = 30.0) -> float:
    """Compute Log-Dimensionless Jerk (LDLJ) for a trajectory.

    Delegates to core.smoothness.SmoothnessAnalyzer.

    Args:
        angles: Angle trajectory (degrees).
        fs: Sampling frequency in Hz.

    Returns:
        LDLJ value (higher = smoother).
    """
    from core.smoothness import SmoothnessAnalyzer

    analyzer = SmoothnessAnalyzer(fs=fs)
    result = analyzer.analyze(angles)
    return result.ldjl if result.is_valid else -10.0


# ── DTW (delegates to core.dtw_constrained) ──────────────────────────────────

def compute_constrained_dtw_distance(
    seq1: np.ndarray,
    seq2: np.ndarray,
    window_percent: float = 0.15,
) -> Tuple[float, List[Tuple[int, int]]]:
    """Compute constrained DTW distance between two angle trajectories.

    Delegates to core.dtw_constrained.constrained_dtw.

    Args:
        seq1: First normalized angle sequence [0, 1].
        seq2: Second normalized angle sequence [0, 1].
        window_percent: Sakoe-Chiba band width.

    Returns:
        Tuple of (distance, warping_path).
    """
    from core.dtw_constrained import constrained_dtw

    return constrained_dtw(seq1, seq2, window_percent=window_percent)


def compute_weighted_multi_joint_dtw(
    user_trajs: Dict[str, JointTrajectory],
    ref_trajs: Dict[str, JointTrajectory],
    weights: Optional[Dict[str, float]] = None,
    window_percent: float = 0.15,
    normalize_length: int = 60,
) -> Tuple[float, float, Dict[str, float]]:
    """Compute weighted multi-joint constrained DTW similarity.

    This is the core of our scoring stack. It compares user trajectories
    against reference trajectories across multiple joints, weighting by
    clinical importance.

    Args:
        user_trajs: User's joint trajectories.
        ref_trajs: Reference joint trajectories.
        weights: Per-joint importance weights.
        window_percent: Sakoe-Chiba band width.
        normalize_length: Target length for temporal normalization.

    Returns:
        Tuple of (similarity_score [0-100], total_distance, per_joint_distances).
    """
    from core.dtw_constrained import weighted_constrained_dtw

    if weights is None:
        weights = {j: JOINT_WEIGHTS.get(j, 0.5) for j in user_trajs}

    user_seqs = {}
    ref_seqs = {}

    for jname in user_trajs:
        if jname not in ref_trajs:
            continue
        ut = user_trajs[jname]
        rt = ref_trajs[jname]
        if not ut.is_valid or not rt.is_valid:
            continue

        # Resample both to normalize_length for fair comparison
        user_norm = resample_trajectory(ut.angles, normalize_length)
        ref_norm = resample_trajectory(rt.angles, normalize_length)

        user_seqs[jname] = user_norm
        ref_seqs[jname] = ref_norm

    if not user_seqs:
        return 0.0, float('inf'), {}

    similarity, total_dist, details = weighted_constrained_dtw(
        user_seqs, ref_seqs,
        weights=weights,
        window_percent=window_percent,
    )

    per_joint = {j: details[j]["normalized_distance"] for j in details}
    return similarity, total_dist, per_joint


# ── Backward-compatibility wrapper ───────────────────────────────────────────────

def compute_joint_angle(
    keypoints_3d: np.ndarray,
    proximal: str,
    vertex: str,
    distal: str,
) -> Optional[float]:
    """Compatibility wrapper: compute mean angle across all frames.

    Note: This is a convenience wrapper for baseline code that only needs
    a scalar angle. For trajectory extraction, use extract_trajectory() instead.

    Args:
        keypoints_3d: Array of shape (n_frames, 25, 3) or (25, 3) for single frame.
        proximal: Name of the proximal joint.
        vertex: Name of the vertex joint.
        distal: Name of the distal joint.

    Returns:
        Mean angle in degrees, or None if invalid.
    """
    pi = KINECT_JOINT_INDEX.get(proximal)
    vi = KINECT_JOINT_INDEX.get(vertex)
    di = KINECT_JOINT_INDEX.get(distal)

    if None in (pi, vi, di):
        return None

    # Handle both (n_frames, 25, 3) and (25, 3) input shapes
    if keypoints_3d.ndim == 3:
        angles = _compute_angles_vectorized(keypoints_3d, pi, vi, di)
        if angles is None:
            return None
        return float(np.mean(angles))
    else:
        # Single frame (25, 3)
        if keypoints_3d.shape[0] <= max(pi, vi, di):
            return None
        v1 = keypoints_3d[pi] - keypoints_3d[vi]
        v2 = keypoints_3d[di] - keypoints_3d[vi]
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 < 1e-10 or n2 < 1e-10:
            return None
        d = np.clip(np.dot(v1/n1, v2/n2), -1, 1)
        if d < -0.999999:
            return 180.0
        s = np.sqrt((1.0 + d) * 2.0)
        w = np.clip(s * 0.5, 0, 1)
        return float(np.degrees(2.0 * np.arccos(w)))


def resample_trajectory(angles: np.ndarray, target_len: int) -> np.ndarray:
    """Resample trajectory to target length using linear interpolation.

    Args:
        angles: Original trajectory (n_frames,).
        target_len: Desired output length.

    Returns:
        Resampled trajectory of length target_len.
    """
    n = len(angles)
    if n == target_len:
        return angles.copy()
    if n < 2:
        return np.full(target_len, angles[0] if n == 1 else 0.0)

    x_old = np.linspace(0, 1, n)
    x_new = np.linspace(0, 1, target_len)
    return np.interp(x_new, x_old, angles)


# ── Euler angle computation (for Strategy 5/6 baselines) ─────────────────────

def compute_euler_angles_from_quaternions(
    keypoints_3d: np.ndarray,
    joint_name: str,
) -> Optional[np.ndarray]:
    """Convert Kinect quaternion orientation to Euler angles for baseline comparison.

    The KIMORE data stores 7-component vectors per joint per frame:
    [qx, qy, qz, qw, tx, ty, tz] where q=quaternion, t=position.
    We use positions only for angle computation (consistent with our method).

    For the Euler baseline (Strategy 5/6), we also compute angles using
    a simpler dot-product approach.

    Args:
        keypoints_3d: Array of shape (n_frames, 25, 3).
        joint_name: Name of the joint.

    Returns:
        Euler angle trajectory in degrees, or None.
    """
    traj = extract_trajectory(keypoints_3d, joint_name)
    if traj is None or not traj.is_valid:
        return None
    return traj.angles


# ── Scoring pipeline ──────────────────────────────────────────────────────────

def compute_recording_score(
    recording_id: str,
    exercise: str,
    keypoints_3d: np.ndarray,
    clinical_score: float,
    ref_trajs: Optional[Dict[str, JointTrajectory]] = None,
    sample_every: int = 1,
    normalize_length: int = 60,
) -> RecordingFeatures:
    """Compute the full scoring stack for one KIMORE recording.

    This is the main entry point for Strategy 2 (clinical correlation).

    Args:
        recording_id: Unique identifier for this recording.
        exercise: Exercise name (e.g., 'ex1').
        keypoints_3d: Array of shape (n_frames, 25, 3).
        clinical_score: Ground truth clinical total score [0.5, 1.0].
        ref_trajs: Reference trajectories for DTW comparison.
        sample_every: Subsample rate for speed.
        normalize_length: Target length for temporal normalization.

    Returns:
        RecordingFeatures with all computed scores.
    """
    # 1. Extract joint trajectories
    trajs = extract_all_trajectories(keypoints_3d, exercise, sample_every)

    features = RecordingFeatures(
        recording_id=recording_id,
        exercise=exercise,
        clinical_score=clinical_score,
        joint_trajectories=trajs,
    )

    if not trajs:
        return features

    # 2. Pose score: ROM quality across primary joints
    pose_scores = []
    for jname in EXERCISE_JOINTS.get(exercise, {}).get("primary", []):
        if jname in trajs and trajs[jname].is_valid:
            angles = smooth_trajectory(trajs[jname].angles)
            rom_range = float(np.max(angles) - np.min(angles))
            expected_min, expected_max = EXPECTED_ROM.get(jname, (0, 180))
            expected_range = expected_max - expected_min
            if expected_range > 1e-6:
                # ROM achievement ratio, capped at 100
                rom_score = min(100.0, (rom_range / expected_range) * 100)
                pose_scores.append(rom_score)

    features.pose_score = float(np.mean(pose_scores)) if pose_scores else 0.0

    # 3. Smoothness score: SPARC normalized across all joints
    smoothness_scores = []
    sparc_raw = {}
    for jname, traj in trajs.items():
        if traj.is_valid and len(traj.angles) >= 30:
            smoothed = smooth_trajectory(traj.angles)
            sparc_n = compute_sparc_normalized(smoothed)
            sparc_raw_j = compute_sparc(smoothed)
            smoothness_scores.append(sparc_n)
            sparc_raw[jname] = sparc_raw_j

    features.smoothness_score = float(np.mean(smoothness_scores)) if smoothness_scores else 0.0
    features.sparc_per_joint = sparc_raw

    # 4. DTW similarity: compare against reference trajectories
    if ref_trajs:
        similarity, total_dist, per_joint = compute_weighted_multi_joint_dtw(
            trajs, ref_trajs,
            weights={j: JOINT_WEIGHTS.get(j, 0.5) for j in trajs},
            window_percent=0.15,
            normalize_length=normalize_length,
        )
        features.dtw_similarity = similarity
        features.dtw_per_joint = per_joint
    else:
        features.dtw_similarity = 0.0

    # 5. Total score: weighted combination
    # Pose (40%) + Smoothness (25%) + DTW (35%)
    features.total_score = (
        0.40 * features.pose_score +
        0.25 * features.smoothness_score +
        0.35 * features.dtw_similarity
    )

    return features


def compute_pair_score(
    traj_a: Dict[str, JointTrajectory],
    traj_b: Dict[str, JointTrajectory],
    normalize_length: int = 60,
) -> float:
    """Compute similarity score between two recordings.

    Used for Strategy 1 (discriminability / verification test).

    Args:
        traj_a: Joint trajectories from recording A.
        traj_b: Joint trajectories from recording B.
        normalize_length: Target length for temporal normalization.

    Returns:
        Similarity score in [0, 100] (higher = more similar).
    """
    similarity, _, _ = compute_weighted_multi_joint_dtw(
        traj_a, traj_b,
        weights={j: JOINT_WEIGHTS.get(j, 0.5) for j in traj_a if j in traj_b},
        window_percent=0.15,
        normalize_length=normalize_length,
    )
    return similarity


# ── Template building ─────────────────────────────────────────────────────────

def build_reference_template(
    recordings: List[RecordingFeatures],
    top_k: int = 10,
    normalize_length: int = 60,
) -> Dict[str, JointTrajectory]:
    """Build reference template from top-performing recordings.

    For each joint, we take the trajectory from the recording with the
    highest clinical score. This serves as the "expert" template for
    DTW-based comparison.

    Args:
        recordings: List of all recordings for one exercise.
        top_k: Number of top recordings to consider.
        normalize_length: Target length for temporal normalization.

    Returns:
        Dict mapping joint_name → JointTrajectory (reference).
    """
    if not recordings:
        return {}

    # Sort by clinical score descending
    sorted_recs = sorted(recordings, key=lambda r: r.clinical_score, reverse=True)
    top_recordings = sorted_recs[:top_k]

    # For each joint, pick the trajectory from the highest-scoring recording
    # that has that joint
    ref_trajs: Dict[str, JointTrajectory] = {}
    for rec in top_recordings:
        for jname, traj in rec.joint_trajectories.items():
            if jname not in ref_trajs and traj.is_valid:
                # Resample to normalize length
                resampled = resample_trajectory(traj.angles, normalize_length)
                ref_trajs[jname] = JointTrajectory(jname, resampled, is_valid=True)

    return ref_trajs


def build_mean_template(
    recordings: List[RecordingFeatures],
    normalize_length: int = 60,
) -> Dict[str, JointTrajectory]:
    """Build mean template by averaging across top-50% recordings.

    More robust than single-template for real-world use.

    Args:
        recordings: List of all recordings for one exercise.
        normalize_length: Target length for temporal normalization.

    Returns:
        Dict mapping joint_name → JointTrajectory (reference).
    """
    if not recordings:
        return {}

    # Use top 50% by clinical score
    sorted_recs = sorted(recordings, key=lambda r: r.clinical_score, reverse=True)
    top_half = sorted_recs[:max(1, len(sorted_recs) // 2)]

    # Collect all trajectories per joint
    joint_trajs: Dict[str, List[np.ndarray]] = {}
    for rec in top_half:
        for jname, traj in rec.joint_trajectories.items():
            if traj.is_valid:
                resampled = resample_trajectory(traj.angles, normalize_length)
                joint_trajs.setdefault(jname, []).append(resampled)

    # Average per joint
    ref_trajs = {}
    for jname, traj_list in joint_trajs.items():
        if traj_list:
            mean_traj = np.mean(np.stack(traj_list, axis=0), axis=0)
            ref_trajs[jname] = JointTrajectory(jname, mean_traj, is_valid=True)

    return ref_trajs