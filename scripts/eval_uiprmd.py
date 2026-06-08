"""
UI-PRMD Dataset Evaluation for ADAPT-Rehab.

UI-PRMD: University of Idaho - Physical Rehabilitation Movement Data.

Dataset structure:
- 1423 samples
- 25 Kinect joints × 4 values (x, y, z, confidence)
- Confidence: 1 (occluded) or 2 (visible)

Evaluation:
1. Joint angle computation from 3D keypoints
2. Movement smoothness (SPARC)
3. Self-consistency metrics
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ============================================================================
# UI-PRMD DATA LOADING
# ============================================================================

def load_uiprmd_dataset(data_path: str) -> np.ndarray:
    """Load UI-PRMD dataset from CSV.

    Returns:
        Array of shape (num_samples, 25, 3) with 3D joint positions.
    """
    df = pd.read_csv(data_path, header=None)

    # Extract x, y, z (skip confidence values)
    # Format: joint0_x, joint0_y, joint0_z, joint0_conf, joint1_x, ...
    n_samples = df.shape[0]
    n_joints = 25

    keypoints = np.zeros((n_samples, n_joints, 3))

    for i in range(n_samples):
        row = df.iloc[i].values
        for j in range(n_joints):
            idx = j * 4  # 4 values per joint
            keypoints[i, j, 0] = row[idx]      # x
            keypoints[i, j, 1] = row[idx + 1]  # y
            keypoints[i, j, 2] = row[idx + 2]  # z

    return keypoints


def get_confidence_mask(data_path: str) -> np.ndarray:
    """Load confidence mask from UI-PRMD dataset.

    Returns:
        Boolean array of shape (num_samples, 25) where True = visible.
    """
    df = pd.read_csv(data_path, header=None)

    n_samples = df.shape[0]
    n_joints = 25

    mask = np.zeros((n_samples, n_joints), dtype=bool)

    for i in range(n_samples):
        row = df.iloc[i].values
        for j in range(n_joints):
            idx = j * 4 + 3  # confidence is 4th value
            mask[i, j] = (row[idx] == 2)  # 2 = visible

    return mask


# ============================================================================
# JOINT ANGLE COMPUTATION
# ============================================================================

# Kinect v2 joint indices
KINECT_JOINTS = {
    "spinebase": 0, "spinemid": 1, "neck": 2, "head": 3,
    "shoulderleft": 4, "elbowleft": 5, "wristleft": 6, "handleft": 7,
    "shoulderright": 8, "elbowright": 9, "wristright": 10, "handright": 11,
    "hipleft": 12, "kneeleft": 13, "ankleleft": 14, "footleft": 15,
    "hipright": 16, "kneeright": 17, "ankleright": 18, "footright": 19,
    "spineshoulder": 20, "handtipleft": 21, "thumbleft": 22,
    "handtipright": 23, "thumbright": 24,
}

ANGLE_DEFS = {
    "left_shoulder": ("spinebase", "shoulderleft", "elbowleft"),
    "right_shoulder": ("spinebase", "shoulderright", "elbowright"),
    "left_elbow": ("shoulderleft", "elbowleft", "wristleft"),
    "right_elbow": ("shoulderright", "elbowright", "wristright"),
    "left_hip": ("spinebase", "hipleft", "kneeleft"),
    "right_hip": ("spinebase", "hipright", "kneeright"),
    "left_knee": ("hipleft", "kneeleft", "ankleleft"),
    "right_knee": ("hipright", "kneeright", "ankleright"),
}


def compute_joint_angle(kps: np.ndarray, prox_name: str, vert_name: str, dist_name: str) -> float:
    """Compute joint angle from 3D keypoints."""
    try:
        p_idx = KINECT_JOINTS[prox_name]
        v_idx = KINECT_JOINTS[vert_name]
        d_idx = KINECT_JOINTS[dist_name]

        if max(p_idx, v_idx, d_idx) >= len(kps):
            return 0.0

        a, b, c = kps[p_idx], kps[v_idx], kps[d_idx]
        ba, bc = a - b, c - b
        norm_ba, norm_bc = np.linalg.norm(ba), np.linalg.norm(bc)

        if norm_ba < 1e-10 or norm_bc < 1e-10:
            return 0.0

        cos_angle = np.dot(ba, bc) / (norm_ba * norm_bc)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        return float(np.degrees(np.arccos(cos_angle)))
    except (IndexError, ValueError):
        return 0.0


def compute_all_angles(kps: np.ndarray) -> Dict[str, float]:
    """Compute all joint angles from 3D keypoints."""
    angles = {}
    for angle_name, (prox, vert, dist) in ANGLE_DEFS.items():
        angle = compute_joint_angle(kps, prox, vert, dist)
        if angle > 0:
            angles[angle_name] = angle
    return angles


# ============================================================================
# SMOOTHNESS METRICS
# ============================================================================

def compute_sparc(velocity: np.ndarray, fs: float = 30.0) -> float:
    """Compute SPARC (Spectral Arc Length)."""
    from numpy.fft import fft, fftfreq

    N = len(velocity)
    N_padded = N * 16

    Mf = np.abs(fft(velocity, n=N_padded))[:N_padded // 2 + 1]
    freq = fftfreq(N_padded, d=1.0 / fs)[:N_padded // 2 + 1]

    if Mf[0] > 0:
        Mf_norm = Mf / Mf[0]
    else:
        Mf_norm = Mf

    freq_mask = freq <= 10.0
    f_sel = freq[freq_mask]
    Mf_sel = Mf_norm[freq_mask]

    above_th = Mf_sel >= 0.05
    if np.sum(above_th) < 2:
        return 0.0

    f_sel = f_sel[above_th]
    Mf_sel = Mf_sel[above_th]

    df = np.diff(f_sel)
    dM = np.diff(Mf_sel)
    arc_length = -np.sum(np.sqrt(df**2 + dM**2))

    return float(arc_length)


# ============================================================================
# EVALUATION METRICS
# ============================================================================

def compute_self_consistency(keypoints: np.ndarray) -> Dict:
    """Compute self-consistency metrics.

    Args:
        keypoints: Shape (num_samples, num_joints, 3)

    Returns:
        Dict with self-consistency metrics
    """
    # Compute mean pose
    mean_kps = np.mean(keypoints, axis=0)

    # Compute MPJPE against mean
    mpjpe_values = []
    for i in range(len(keypoints)):
        error = np.linalg.norm(keypoints[i] - mean_kps, axis=-1)
        mpjpe_values.append(np.mean(error))

    # Compute per-joint error
    per_joint_error = np.mean(np.linalg.norm(
        keypoints - mean_kps[np.newaxis], axis=-1
    ), axis=0)

    return {
        "mpjpe_mean": float(np.mean(mpjpe_values)),
        "mpjpe_std": float(np.std(mpjpe_values)),
        "per_joint_error": per_joint_error.tolist(),
    }


def compute_angle_statistics(keypoints: np.ndarray) -> Dict:
    """Compute joint angle statistics across all samples.

    Args:
        keypoints: Shape (num_samples, num_joints, 3)

    Returns:
        Dict with angle statistics
    """
    all_angles = {name: [] for name in ANGLE_DEFS.keys()}

    for i in range(len(keypoints)):
        angles = compute_all_angles(keypoints[i])
        for name, value in angles.items():
            all_angles[name].append(value)

    stats = {}
    for name, values in all_angles.items():
        if values:
            stats[name] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
            }

    return stats


def compute_smoothness_metrics(keypoints: np.ndarray) -> Dict:
    """Compute smoothness metrics for angle trajectories.

    Args:
        keypoints: Shape (num_samples, num_joints, 3)

    Returns:
        Dict with smoothness metrics
    """
    # Extract angle trajectories
    trajectories = {}
    for angle_name in ANGLE_DEFS.keys():
        trajectory = []
        for i in range(len(keypoints)):
            angles = compute_all_angles(keypoints[i])
            if angle_name in angles:
                trajectory.append(angles[angle_name])
        if trajectory:
            trajectories[angle_name] = np.array(trajectory)

    # Compute SPARC for each trajectory
    sparc_values = {}
    for name, trajectory in trajectories.items():
        if len(trajectory) > 10:
            velocity = np.diff(trajectory) * 30.0  # Assume 30 FPS
            sparc = compute_sparc(velocity, fs=30.0)
            sparc_values[name] = sparc

    return {
        "sparc": sparc_values,
        "mean_sparc": float(np.mean(list(sparc_values.values()))) if sparc_values else 0,
    }


# ============================================================================
# MAIN EVALUATION
# ============================================================================

def run_uiprmd_evaluation():
    """Run evaluation on UI-PRMD dataset."""
    print("=" * 70)
    print("UI-PRMD Dataset Evaluation")
    print("=" * 70)

    # Load dataset
    data_path = "data/UI-PRMD/input.csv"
    if not os.path.exists(data_path):
        print(f"[Error] Dataset not found: {data_path}")
        return None

    print(f"\nLoading UI-PRMD dataset...")
    keypoints = load_uiprmd_dataset(data_path)
    confidence = get_confidence_mask(data_path)

    print(f"Shape: {keypoints.shape}")
    print(f"Confidence: {np.mean(confidence)*100:.1f}% visible")

    # 1. Self-Consistency
    print(f"\n{'='*50}")
    print("1. SELF-CONSISTENCY (Temporal Stability)")
    print(f"{'='*50}")
    consistency = compute_self_consistency(keypoints)
    print(f"  MPJPE (mean): {consistency['mpjpe_mean']:.3f}")
    print(f"  MPJPE (std): {consistency['mpjpe_std']:.3f}")

    # 2. Joint Angle Statistics
    print(f"\n{'='*50}")
    print("2. JOINT ANGLE STATISTICS")
    print(f"{'='*50}")
    angle_stats = compute_angle_statistics(keypoints)
    for name, stats in angle_stats.items():
        print(f"  {name:20s}: mean={stats['mean']:.1f}°, std={stats['std']:.1f}°, range=[{stats['min']:.1f}, {stats['max']:.1f}]")

    # 3. Smoothness
    print(f"\n{'='*50}")
    print("3. SMOOTHNESS (SPARC)")
    print(f"{'='*50}")
    smoothness = compute_smoothness_metrics(keypoints)
    print(f"  Mean SPARC: {smoothness['mean_sparc']:.3f}")
    for name, sparc in smoothness['sparc'].items():
        print(f"  {name:20s}: {sparc:.3f}")

    # 4. Per-Joint Error
    print(f"\n{'='*50}")
    print("4. PER-JOINT ERROR")
    print(f"{'='*50}")
    joint_names = list(KINECT_JOINTS.keys())
    per_joint = consistency['per_joint_error']
    for i, (name, error) in enumerate(zip(joint_names, per_joint)):
        print(f"  {name:20s}: {error:.3f}")

    # Save results
    results = {
        "dataset": "UI-PRMD",
        "num_samples": len(keypoints),
        "num_joints": keypoints.shape[1],
        "self_consistency": consistency,
        "angle_statistics": angle_stats,
        "smoothness": smoothness,
    }

    output_path = "evaluation/results/uiprmd_evaluation.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n[Results] Saved to {output_path}")

    return results


if __name__ == "__main__":
    run_uiprmd_evaluation()
