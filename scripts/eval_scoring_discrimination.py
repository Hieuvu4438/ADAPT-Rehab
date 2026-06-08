"""
Scoring Discrimination Evaluation for ADAPT-Rehab.

Measures whether the system can correctly distinguish:
- Same exercise pairs (should have HIGH similarity / LOW distance)
- Different exercise pairs (should have LOW similarity / HIGH distance)

Metrics:
1. Intra-class vs Inter-class distance (should be separable)
2. Classification accuracy (can we correctly identify exercise type?)
3. ROC-AUC (how well can we distinguish same vs different?)
4. Precision@K / Recall@K (retrieval quality)
"""

import os
import sys
import json
import time
import numpy as np
from typing import Dict, List, Tuple, Optional
from itertools import combinations
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.datasets import load_yoga_dataset


# ============================================================================
# DTW DISTANCE COMPUTATION
# ============================================================================

def dtw_distance(seq1: np.ndarray, seq2: np.ndarray) -> float:
    """
    Dynamic Time Warping distance between two sequences.

    Args:
        seq1: First sequence, shape (T1, D)
        seq2: Second sequence, shape (T2, D)

    Returns:
        DTW distance (lower = more similar)
    """
    n, m = len(seq1), len(seq2)

    # Cost matrix
    dtw_matrix = np.full((n + 1, m + 1), np.inf)
    dtw_matrix[0, 0] = 0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = np.linalg.norm(seq1[i-1] - seq2[j-1])
            dtw_matrix[i, j] = cost + min(
                dtw_matrix[i-1, j],      # insertion
                dtw_matrix[i, j-1],      # deletion
                dtw_matrix[i-1, j-1]     # match
            )

    return float(dtw_matrix[n, m])


def dtw_distance_fast(seq1: np.ndarray, seq2: np.ndarray, window: int = 10) -> float:
    """
    Fast DTW with Sakoe-Chiba band constraint.

    Args:
        seq1: First sequence, shape (T1, D)
        seq2: Second sequence, shape (T2, D)
        window: Window size for Sakoe-Chiba band

    Returns:
        DTW distance
    """
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
# FEATURE EXTRACTION
# ============================================================================

def extract_angle_trajectory(keypoints_list: List[np.ndarray]) -> np.ndarray:
    """
    Extract joint angle trajectory from keypoints sequence.

    Args:
        keypoints_list: List of keypoints arrays, each (J, 3)

    Returns:
        Angle trajectory array, shape (T, num_joints)
    """
    trajectories = []

    for kps in keypoints_list:
        if kps is None or len(kps) < 10:
            continue

        # Compute angles for key joints
        angles = []
        joint_defs = [
            ("left_shoulder", 11, 13, 15),   # MediaPipe indices
            ("right_shoulder", 12, 14, 16),
            ("left_elbow", 11, 13, 15),
            ("right_elbow", 12, 14, 16),
            ("left_hip", 23, 25, 27),
            ("right_hip", 24, 26, 28),
            ("left_knee", 23, 25, 27),
            ("right_knee", 24, 26, 28),
        ]

        for name, p_idx, v_idx, d_idx in joint_defs:
            if max(p_idx, v_idx, d_idx) < len(kps):
                a, b, c = kps[p_idx], kps[v_idx], kps[d_idx]
                ba, bc = a - b, c - b
                norm_ba = np.linalg.norm(ba)
                norm_bc = np.linalg.norm(bc)
                if norm_ba > 1e-10 and norm_bc > 1e-10:
                    cos_angle = np.dot(ba, bc) / (norm_ba * norm_bc)
                    angle = np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))
                    angles.append(angle)
                else:
                    angles.append(0.0)
            else:
                angles.append(0.0)

        if angles:
            trajectories.append(angles)

    if not trajectories:
        return np.array([])

    return np.array(trajectories)


def extract_keypoint_trajectory(keypoints_list: List[np.ndarray]) -> np.ndarray:
    """
    Extract flattened keypoint trajectory.

    Args:
        keypoints_list: List of keypoints arrays, each (J, 3)

    Returns:
        Trajectory array, shape (T, J*3)
    """
    trajectories = []
    for kps in keypoints_list:
        if kps is not None:
            # Use first 17 joints (body only) and flatten
            kps_flat = kps[:17].flatten() if len(kps) >= 17 else kps.flatten()
            trajectories.append(kps_flat)

    if not trajectories:
        return np.array([])

    return np.array(trajectories)


# ============================================================================
# EVALUATION METRICS
# ============================================================================

def compute_discrimination_metrics(
    video_features: Dict[str, np.ndarray],
    video_labels: Dict[str, str],
) -> Dict:
    """
    Compute discrimination metrics.

    Args:
        video_features: Dict mapping video name to feature trajectory
        video_labels: Dict mapping video name to exercise label

    Returns:
        Dict with discrimination metrics
    """
    video_names = list(video_features.keys())

    # Compute pairwise distances
    same_exercise_distances = []
    diff_exercise_distances = []
    pairwise_results = []

    for (name1, feat1), (name2, feat2) in combinations(
        zip(video_names, [video_features[n] for n in video_names]), 2
    ):
        if len(feat1) == 0 or len(feat2) == 0:
            continue

        # Compute DTW distance
        dist = dtw_distance_fast(feat1, feat2, window=15)

        label1 = video_labels[name1]
        label2 = video_labels[name2]
        same_exercise = (label1 == label2)

        if same_exercise:
            same_exercise_distances.append(dist)
        else:
            diff_exercise_distances.append(dist)

        pairwise_results.append({
            "video1": name1,
            "video2": name2,
            "label1": label1,
            "label2": label2,
            "distance": dist,
            "same_exercise": same_exercise,
        })

    if not same_exercise_distances or not diff_exercise_distances:
        return {"error": "Not enough data for comparison"}

    # 1. Intra-class vs Inter-class statistics
    same_mean = np.mean(same_exercise_distances)
    same_std = np.std(same_exercise_distances)
    diff_mean = np.mean(diff_exercise_distances)
    diff_std = np.std(diff_exercise_distances)

    # 2. Separation ratio (higher = better)
    separation_ratio = diff_mean / (same_mean + 1e-10)

    # 3. Classification accuracy (using threshold = midpoint)
    threshold = (same_mean + diff_mean) / 2
    correct = 0
    total = 0
    for pr in pairwise_results:
        predicted_same = pr["distance"] < threshold
        if predicted_same == pr["same_exercise"]:
            correct += 1
        total += 1
    classification_accuracy = correct / total if total > 0 else 0

    # 4. ROC-AUC
    labels = np.array([1 if pr["same_exercise"] else 0 for pr in pairwise_results])
    distances = np.array([pr["distance"] for pr in pairwise_results])

    # For ROC-AUC, we need to invert distances (lower distance = higher score)
    scores = -distances
    auc = compute_auc(labels, scores)

    # 5. Precision@K and Recall@K
    precision_at_k, recall_at_k = compute_precision_recall_at_k(
        pairwise_results, k_values=[1, 3, 5, 10]
    )

    return {
        "intra_class_distance": {
            "mean": float(same_mean),
            "std": float(same_std),
            "min": float(np.min(same_exercise_distances)),
            "max": float(np.max(same_exercise_distances)),
        },
        "inter_class_distance": {
            "mean": float(diff_mean),
            "std": float(diff_std),
            "min": float(np.min(diff_exercise_distances)),
            "max": float(np.max(diff_exercise_distances)),
        },
        "separation_ratio": float(separation_ratio),
        "classification_accuracy": float(classification_accuracy),
        "roc_auc": float(auc),
        "precision_at_k": precision_at_k,
        "recall_at_k": recall_at_k,
        "num_same_pairs": len(same_exercise_distances),
        "num_diff_pairs": len(diff_exercise_distances),
        "threshold": float(threshold),
    }


def compute_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Compute AUC (Area Under ROC Curve)."""
    # Sort by score (descending)
    sorted_indices = np.argsort(-scores)
    sorted_labels = labels[sorted_indices]

    # Compute TPR and FPR at each threshold
    n_pos = np.sum(labels == 1)
    n_neg = np.sum(labels == 0)

    if n_pos == 0 or n_neg == 0:
        return 0.5

    tpr_list = [0.0]
    fpr_list = [0.0]
    tp = 0
    fp = 0

    for label in sorted_labels:
        if label == 1:
            tp += 1
        else:
            fp += 1
        tpr_list.append(tp / n_pos)
        fpr_list.append(fp / n_neg)

    # Compute AUC using trapezoidal rule
    auc = 0.0
    for i in range(1, len(tpr_list)):
        auc += (fpr_list[i] - fpr_list[i-1]) * (tpr_list[i] + tpr_list[i-1]) / 2

    return float(auc)


def compute_precision_recall_at_k(
    pairwise_results: List[Dict],
    k_values: List[int] = [1, 3, 5, 10]
) -> Tuple[Dict[int, float], Dict[int, float]]:
    """Compute Precision@K and Recall@K for each exercise."""
    # Group by query exercise
    exercises = set()
    for pr in pairwise_results:
        exercises.add(pr["label1"])
        exercises.add(pr["label2"])

    precision_at_k = {k: [] for k in k_values}
    recall_at_k = {k: [] for k in k_values}

    for exercise in exercises:
        # Get all pairs involving this exercise
        relevant_pairs = [
            pr for pr in pairwise_results
            if pr["label1"] == exercise or pr["label2"] == exercise
        ]

        # Sort by distance (ascending = most similar first)
        relevant_pairs.sort(key=lambda x: x["distance"])

        # Count total relevant (same exercise)
        total_relevant = sum(1 for pr in relevant_pairs if pr["same_exercise"])

        if total_relevant == 0:
            continue

        for k in k_values:
            # Top-K retrieved
            top_k = relevant_pairs[:k]
            relevant_in_top_k = sum(1 for pr in top_k if pr["same_exercise"])

            precision_at_k[k].append(relevant_in_top_k / k)
            recall_at_k[k].append(relevant_in_top_k / total_relevant)

    return (
        {k: float(np.mean(v)) if v else 0.0 for k, v in precision_at_k.items()},
        {k: float(np.mean(v)) if v else 0.0 for k, v in recall_at_k.items()},
    )


# ============================================================================
# MAIN EVALUATION
# ============================================================================

def run_scoring_discrimination_evaluation():
    """Run scoring discrimination evaluation."""
    print("=" * 70)
    print("Scoring Discrimination Evaluation")
    print("Measures: Can the system distinguish same vs different exercises?")
    print("=" * 70)

    # Load dataset
    samples = load_yoga_dataset("data")
    if not samples:
        print("[Error] No samples found!")
        return None

    print(f"\nDataset: Yoga-Collect ({len(samples)} videos)")

    # Create estimator
    from core.pose3d import create_estimator
    estimator = create_estimator("rtmw3d")
    if not estimator.initialize():
        print("[Error] Failed to initialize RTMW3D!")
        return None

    print(f"Estimator: {estimator.model_name}")

    # Process videos and extract features
    video_features = {}
    video_labels = {}

    for i, sample in enumerate(samples):
        print(f"\r[{i+1}/{len(samples)}] {sample.person_name}_{sample.exercise_type}", end="", flush=True)

        features = extract_features_from_video(sample, estimator, max_frames=30)
        if features is not None and len(features) > 0:
            video_name = f"{sample.person_name}_{sample.exercise_type}"
            video_features[video_name] = features
            video_labels[video_name] = sample.exercise_type

    print(f"\n\nProcessed {len(video_features)} videos")

    # Run discrimination evaluation
    print("\nComputing pairwise distances...")
    metrics = compute_discrimination_metrics(video_features, video_labels)

    # Save results
    save_discrimination_results(metrics)

    # Print summary
    print_discrimination_summary(metrics)

    estimator.close()
    return metrics


def extract_features_from_video(sample, estimator, max_frames=30) -> Optional[np.ndarray]:
    """Extract features from a video."""
    import cv2

    cap = cv2.VideoCapture(sample.video_path)
    if not cap.isOpened():
        return None

    keypoints_list = []
    frame_idx = 0

    while cap.isOpened() and frame_idx < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        pose_result = estimator.estimate(frame, frame_idx * 33)
        if pose_result.is_valid and pose_result.keypoints_3d is not None:
            keypoints_list.append(pose_result.keypoints_3d.copy())

        frame_idx += 1

    cap.release()

    if len(keypoints_list) < 5:
        return None

    # Extract angle trajectory
    return extract_angle_trajectory(keypoints_list)


def save_discrimination_results(metrics: Dict):
    """Save discrimination results."""
    output_path = "evaluation/results/scoring_discrimination.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"[Results] Saved to {output_path}")


def print_discrimination_summary(metrics: Dict):
    """Print discrimination evaluation summary."""
    print("\n" + "=" * 70)
    print("SCORING DISCRIMINATION RESULTS")
    print("=" * 70)

    if "error" in metrics:
        print(f"Error: {metrics['error']}")
        return

    # Intra-class vs Inter-class
    intra = metrics["intra_class_distance"]
    inter = metrics["inter_class_distance"]

    print(f"\n{'='*50}")
    print("DISTANCE DISTRIBUTION")
    print(f"{'='*50}")
    print(f"  Same Exercise (intra-class):")
    print(f"    Mean: {intra['mean']:.2f} (±{intra['std']:.2f})")
    print(f"    Range: [{intra['min']:.2f}, {intra['max']:.2f}]")
    print(f"    Pairs: {metrics['num_same_pairs']}")
    print(f"  Different Exercise (inter-class):")
    print(f"    Mean: {inter['mean']:.2f} (±{inter['std']:.2f})")
    print(f"    Range: [{inter['min']:.2f}, {inter['max']:.2f}]")
    print(f"    Pairs: {metrics['num_diff_pairs']}")

    # Separation
    print(f"\n{'='*50}")
    print("SEPARATION METRICS")
    print(f"{'='*50}")
    print(f"  Separation Ratio: {metrics['separation_ratio']:.2f}x")
    print(f"    (>1.0 means same-exercise pairs are closer)")
    print(f"  Classification Accuracy: {metrics['classification_accuracy']*100:.1f}%")
    print(f"  ROC-AUC: {metrics['roc_auc']:.3f}")
    print(f"    (>0.5 means system can distinguish same vs different)")

    # Precision@K
    print(f"\n{'='*50}")
    print("RETRIEVAL METRICS")
    print(f"{'='*50}")
    print(f"{'K':<5} {'Precision@K':<15} {'Recall@K':<15}")
    print("-" * 35)
    for k in sorted(metrics['precision_at_k'].keys()):
        p = metrics['precision_at_k'][k]
        r = metrics['recall_at_k'][k]
        print(f"{k:<5} {p:<15.3f} {r:<15.3f}")

    # Interpretation
    print(f"\n{'='*50}")
    print("INTERPRETATION")
    print(f"{'='*50}")

    if metrics['separation_ratio'] > 2.0:
        print("  ✓ Good separation: same-exercise pairs are significantly closer")
    elif metrics['separation_ratio'] > 1.5:
        print("  ~ Moderate separation: some overlap between same/different")
    else:
        print("  ✗ Poor separation: system cannot distinguish exercises well")

    if metrics['classification_accuracy'] > 0.8:
        print("  ✓ High classification accuracy")
    elif metrics['classification_accuracy'] > 0.6:
        print("  ~ Moderate classification accuracy")
    else:
        print("  ✗ Low classification accuracy")

    if metrics['roc_auc'] > 0.8:
        print("  ✓ Strong discriminative ability (AUC > 0.8)")
    elif metrics['roc_auc'] > 0.6:
        print("  ~ Moderate discriminative ability")
    else:
        print("  ✗ Weak discriminative ability")


if __name__ == "__main__":
    run_scoring_discrimination_evaluation()
