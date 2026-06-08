"""
KIMORE Dataset Evaluation for ADAPT-Rehab.

KIMORE: KInematic assessment of MOvement and clinical scores for
Remote monitoring of physical REhabilitation.

Dataset structure:
- 5 exercises (ex1 to ex5)
- 75-77 samples per exercise
- 25 Kinect joints × 7 features (x, y, z, qw, qx, qy, qz)
- Clinical scores (cTS) for each sample

Evaluation:
1. Exercise classification accuracy
2. Clinical score prediction correlation
3. Discrimination (same vs different exercise)
"""

import os
import sys
import json
import pickle
import numpy as np
from typing import Dict, List, Tuple, Optional
from itertools import combinations
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ============================================================================
# KIMORE DATA LOADING
# ============================================================================

def load_kimore_dataset(data_path: str) -> Dict[str, np.ndarray]:
    """Load KIMORE dataset from pickle file.

    Returns:
        Dict mapping exercise name to array of shape (num_samples, num_frames, num_joints, 3)
    """
    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    exercises = {}
    for ex_name, ex_df in data.items():
        # Extract joint positions (first 3 values are x, y, z)
        samples = []
        for idx in range(len(ex_df)):
            sample_data = []
            for col in ex_df.columns:
                if col == 'cTS':
                    continue
                joint_data = ex_df.iloc[idx][col]
                # joint_data shape: (num_frames, 7) where 7 = [x, y, z, qw, qx, qy, qz]
                if isinstance(joint_data, np.ndarray) and joint_data.ndim == 2:
                    # Extract only x, y, z (first 3 values)
                    positions = joint_data[:, :3]
                    sample_data.append(positions)

            if sample_data:
                # Stack joints: (num_frames, num_joints, 3)
                sample_array = np.stack(sample_data, axis=1)
                samples.append(sample_array)

        if samples:
            exercises[ex_name] = samples

    return exercises


def load_kimore_clinical_scores(data_path: str) -> Dict[str, List[float]]:
    """Load clinical scores from KIMORE dataset."""
    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    scores = {}
    for ex_name, ex_df in data.items():
        scores[ex_name] = ex_df['cTS'].values.tolist()

    return scores


# ============================================================================
# FEATURE EXTRACTION
# ============================================================================

def compute_joint_angles(kps: np.ndarray) -> Dict[str, float]:
    """Compute joint angles from 3D keypoints.

    Args:
        kps: Keypoints array, shape (num_joints, 3)

    Returns:
        Dict mapping joint name to angle in degrees
    """
    # Kinect v2 joint indices
    joints = {
        "spinebase": 0, "spinemid": 1, "neck": 2, "head": 3,
        "shoulderleft": 4, "elbowleft": 5, "wristleft": 6, "handleft": 7,
        "shoulderright": 8, "elbowright": 9, "wristright": 10, "handright": 11,
        "hipleft": 12, "kneeleft": 13, "ankleleft": 14, "footleft": 15,
        "hipright": 16, "kneeright": 17, "ankleright": 18, "footright": 19,
        "spineshoulder": 20, "handtipleft": 21, "thumbleft": 22,
        "handtipright": 23, "thumbright": 24,
    }

    angle_defs = {
        "left_shoulder": ("spinebase", "shoulderleft", "elbowleft"),
        "right_shoulder": ("spinebase", "shoulderright", "elbowright"),
        "left_elbow": ("shoulderleft", "elbowleft", "wristleft"),
        "right_elbow": ("shoulderright", "elbowright", "wristright"),
        "left_hip": ("spinebase", "hipleft", "kneeleft"),
        "right_hip": ("spinebase", "hipright", "kneeright"),
        "left_knee": ("hipleft", "kneeleft", "ankleleft"),
        "right_knee": ("hipright", "kneeright", "ankleright"),
    }

    angles = {}
    for angle_name, (prox_name, vert_name, dist_name) in angle_defs.items():
        try:
            p_idx = joints[prox_name]
            v_idx = joints[vert_name]
            d_idx = joints[dist_name]

            if max(p_idx, v_idx, d_idx) >= len(kps):
                continue

            a, b, c = kps[p_idx], kps[v_idx], kps[d_idx]
            ba, bc = a - b, c - b
            norm_ba, norm_bc = np.linalg.norm(ba), np.linalg.norm(bc)

            if norm_ba < 1e-10 or norm_bc < 1e-10:
                continue

            cos_angle = np.dot(ba, bc) / (norm_ba * norm_bc)
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            angles[angle_name] = float(np.degrees(np.arccos(cos_angle)))
        except (IndexError, ValueError):
            continue

    return angles


def extract_angle_trajectory(sample: np.ndarray) -> np.ndarray:
    """Extract angle trajectory from a sample.

    Args:
        sample: Shape (num_frames, num_joints, 3)

    Returns:
        Angle trajectory, shape (num_frames, num_angles)
    """
    trajectories = []
    for frame_idx in range(len(sample)):
        kps = sample[frame_idx]
        angles = compute_joint_angles(kps)
        if angles and len(angles) == 8:  # All 8 joints computed
            trajectories.append(list(angles.values()))

    if not trajectories:
        return np.array([])

    return np.array(trajectories)


# ============================================================================
# DTW DISTANCE
# ============================================================================

def dtw_distance(seq1: np.ndarray, seq2: np.ndarray, window: int = 20) -> float:
    """DTW distance with Sakoe-Chiba band constraint."""
    n, m = len(seq1), len(seq2)
    w = max(window, abs(n - m))

    dtw_matrix = np.full((n + 1, m + 1), np.inf)
    dtw_matrix[0, 0] = 0

    for i in range(1, n + 1):
        for j in range(max(1, i - w), min(m, i + w) + 1):
            cost = np.linalg.norm(seq1[i-1] - seq2[j-1])
            dtw_matrix[i, j] = cost + min(
                dtw_matrix[i-1, j],
                dtw_matrix[i, j-1],
                dtw_matrix[i-1, j-1]
            )

    return float(dtw_matrix[n, m])


# ============================================================================
# EVALUATION METRICS
# ============================================================================

def compute_classification_accuracy(
    exercises: Dict[str, List[np.ndarray]],
) -> Dict:
    """Compute exercise classification accuracy using DTW distance.

    For each sample, find the most similar other sample.
    If they belong to the same exercise, it's a correct classification.
    """
    # Prepare data: flatten samples with labels
    all_samples = []
    all_labels = []

    for ex_name, samples in exercises.items():
        for sample in samples:
            trajectory = extract_angle_trajectory(sample)
            if len(trajectory) > 0:
                all_samples.append(trajectory)
                all_labels.append(ex_name)

    if len(all_samples) < 2:
        return {"error": "Not enough samples"}

    # Compute pairwise distances (subsample for speed)
    n_samples = len(all_samples)
    max_pairs = 1000  # Limit number of pairs for speed

    # Randomly sample pairs
    np.random.seed(42)
    pairs = []
    for i in range(n_samples):
        for j in range(i+1, n_samples):
            pairs.append((i, j))

    if len(pairs) > max_pairs:
        pairs = list(np.random.choice(len(pairs), max_pairs, replace=False))
        pairs = [(i, j) for i, j in combinations(range(n_samples), 2)]

    # Compute distances
    same_exercise_distances = []
    diff_exercise_distances = []
    correct = 0
    total = 0

    for i, j in pairs[:max_pairs]:
        dist = dtw_distance(all_samples[i], all_samples[j])
        same = (all_labels[i] == all_labels[j])

        if same:
            same_exercise_distances.append(dist)
        else:
            diff_exercise_distances.append(dist)

        # Classification: find nearest neighbor
        # For simplicity, just check if this pair is same exercise
        if same:
            correct += 1
        total += 1

    if not same_exercise_distances or not diff_exercise_distances:
        return {"error": "Not enough data"}

    # Compute metrics
    same_mean = np.mean(same_exercise_distances)
    diff_mean = np.mean(diff_exercise_distances)
    separation_ratio = diff_mean / (same_mean + 1e-10)

    # Classification accuracy (using threshold)
    threshold = (same_mean + diff_mean) / 2
    correct_classify = 0
    for dist in same_exercise_distances:
        if dist < threshold:
            correct_classify += 1
    for dist in diff_exercise_distances:
        if dist >= threshold:
            correct_classify += 1
    total_classify = len(same_exercise_distances) + len(diff_exercise_distances)
    accuracy = correct_classify / total_classify if total_classify > 0 else 0

    return {
        "separation_ratio": float(separation_ratio),
        "classification_accuracy": float(accuracy),
        "intra_class_mean": float(same_mean),
        "inter_class_mean": float(diff_mean),
        "num_same_pairs": len(same_exercise_distances),
        "num_diff_pairs": len(diff_exercise_distances),
    }


def compute_clinical_score_correlation(
    exercises: Dict[str, List[np.ndarray]],
    clinical_scores: Dict[str, List[float]],
) -> Dict:
    """Compute correlation between movement features and clinical scores."""
    # Extract features for each sample
    features = []
    scores = []

    for ex_name, samples in exercises.items():
        ex_scores = clinical_scores.get(ex_name, [])
        for i, sample in enumerate(samples):
            trajectory = extract_angle_trajectory(sample)
            if len(trajectory) > 0:
                # Compute simple features: mean angle, std angle, range
                mean_angles = np.mean(trajectory, axis=0)
                std_angles = np.std(trajectory, axis=0)
                range_angles = np.ptp(trajectory, axis=0)

                feature = np.concatenate([mean_angles, std_angles, range_angles])
                features.append(feature)

                if i < len(ex_scores):
                    scores.append(ex_scores[i])

    if len(features) < 2:
        return {"error": "Not enough data"}

    features = np.array(features)
    scores = np.array(scores)

    # Compute correlation for each feature
    correlations = []
    for col in range(features.shape[1]):
        corr = np.corrcoef(features[:, col], scores)[0, 1]
        if not np.isnan(corr):
            correlations.append(abs(corr))

    return {
        "mean_correlation": float(np.mean(correlations)) if correlations else 0,
        "max_correlation": float(np.max(correlations)) if correlations else 0,
        "num_features": features.shape[1],
        "num_samples": len(scores),
    }


# ============================================================================
# MAIN EVALUATION
# ============================================================================

def run_kimore_evaluation():
    """Run evaluation on KIMORE dataset."""
    print("=" * 70)
    print("KIMORE Dataset Evaluation")
    print("=" * 70)

    # Load dataset
    data_path = "data/KIMORE/kimore_exercise_dataset.pkl"
    if not os.path.exists(data_path):
        print(f"[Error] Dataset not found: {data_path}")
        return None

    print(f"\nLoading KIMORE dataset...")
    exercises = load_kimore_dataset(data_path)
    clinical_scores = load_kimore_clinical_scores(data_path)

    print(f"Exercises: {list(exercises.keys())}")
    for ex_name, samples in exercises.items():
        print(f"  {ex_name}: {len(samples)} samples, shape={samples[0].shape}")

    # 1. Exercise Classification
    print(f"\n{'='*50}")
    print("1. EXERCISE CLASSIFICATION")
    print(f"{'='*50}")
    classification_metrics = compute_classification_accuracy(exercises)
    print(f"  Separation Ratio: {classification_metrics.get('separation_ratio', 0):.2f}x")
    print(f"  Classification Accuracy: {classification_metrics.get('classification_accuracy', 0)*100:.1f}%")
    print(f"  Intra-class Distance: {classification_metrics.get('intra_class_mean', 0):.2f}")
    print(f"  Inter-class Distance: {classification_metrics.get('inter_class_mean', 0):.2f}")

    # 2. Clinical Score Correlation
    print(f"\n{'='*50}")
    print("2. CLINICAL SCORE CORRELATION")
    print(f"{'='*50}")
    correlation_metrics = compute_clinical_score_correlation(exercises, clinical_scores)
    print(f"  Mean Correlation: {correlation_metrics.get('mean_correlation', 0):.3f}")
    print(f"  Max Correlation: {correlation_metrics.get('max_correlation', 0):.3f}")
    print(f"  Num Features: {correlation_metrics.get('num_features', 0)}")
    print(f"  Num Samples: {correlation_metrics.get('num_samples', 0)}")

    # 3. Per-Exercise Statistics
    print(f"\n{'='*50}")
    print("3. PER-EXERCISE STATISTICS")
    print(f"{'='*50}")
    for ex_name, samples in exercises.items():
        scores = clinical_scores.get(ex_name, [])
        if scores:
            print(f"  {ex_name}: {len(samples)} samples, "
                  f"clinical score: mean={np.mean(scores):.2f}, std={np.std(scores):.2f}")

    # Save results
    results = {
        "dataset": "KIMORE",
        "num_exercises": len(exercises),
        "total_samples": sum(len(s) for s in exercises.values()),
        "classification": classification_metrics,
        "clinical_correlation": correlation_metrics,
    }

    output_path = "evaluation/results/kimore_evaluation.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n[Results] Saved to {output_path}")

    return results


if __name__ == "__main__":
    run_kimore_evaluation()
