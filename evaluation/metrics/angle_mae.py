"""
Joint angle evaluation metrics.

- Angle MAE: Mean Absolute Error between predicted and ground truth joint angles
- ICC: Intraclass Correlation Coefficient for agreement measurement
- Per-joint angle error breakdown

Joint angles are computed from 3D keypoints using the dot product method.
For evaluation, both predicted and GT angles should be in degrees.

Reference:
- ICC: Koo & Li, "A Guideline of Selecting and Reporting Intraclass
  Correlation Coefficients for Reliability Research," J Chiropr Med, 2016.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from evaluation.skeleton_mapping import (
    compute_angle_from_keypoints,
    COMMON14_FROM_SMPL24,
    COMMON14_FROM_MEDIAPIPE,
    COMMON14_FROM_KINECT_V2,
    SkeletonType,
)


# Standard joint angle definitions for evaluation
# Each entry: (joint_name, proximal, vertex, distal)
JOINT_ANGLE_DEFS = {
    "left_shoulder": ("pelvis", "left_shoulder", "left_elbow"),
    "right_shoulder": ("pelvis", "right_shoulder", "right_elbow"),
    "left_elbow": ("left_shoulder", "left_elbow", "left_wrist"),
    "right_elbow": ("right_shoulder", "right_elbow", "right_wrist"),
    "left_hip": ("spine", "left_hip", "left_knee"),
    "right_hip": ("spine", "right_hip", "right_knee"),
    "left_knee": ("left_hip", "left_knee", "left_ankle"),
    "right_knee": ("right_hip", "right_knee", "right_ankle"),
}

# Mapping from joint name to common14 indices
COMMON14_INDICES = {name: i for i, name in enumerate([
    "pelvis", "spine", "neck", "head",
    "left_shoulder", "left_elbow", "left_wrist",
    "right_shoulder", "right_elbow", "right_wrist",
    "left_hip", "left_knee", "left_ankle",
    "right_hip",
])}


def compute_angle_mae(
    predicted: Dict[str, float],
    ground_truth: Dict[str, float],
) -> float:
    """Mean Absolute Error between predicted and ground truth joint angles.

    Args:
        predicted: Dict mapping joint name to angle in degrees.
        ground_truth: Dict mapping joint name to angle in degrees.

    Returns:
        Mean absolute error in degrees. Returns 0.0 if no common joints.

    Example:
        >>> pred = {"left_elbow": 45.0, "right_elbow": 50.0}
        >>> gt = {"left_elbow": 42.0, "right_elbow": 48.0}
        >>> compute_angle_mae(pred, gt)  # mean of |45-42| and |50-48| = 2.5
    """
    common = set(predicted.keys()) & set(ground_truth.keys())
    if not common:
        return 0.0

    errors = [abs(predicted[j] - ground_truth[j]) for j in common]
    return float(np.mean(errors))


def compute_per_joint_angle_mae(
    predicted: Dict[str, float],
    ground_truth: Dict[str, float],
) -> Dict[str, float]:
    """Per-joint angle MAE breakdown.

    Args:
        predicted: Dict mapping joint name to angle in degrees.
        ground_truth: Dict mapping joint name to angle in degrees.

    Returns:
        Dict mapping joint name to absolute error in degrees.
    """
    common = set(predicted.keys()) & set(ground_truth.keys())
    return {j: float(abs(predicted[j] - ground_truth[j])) for j in common}


def compute_angles_from_keypoints(
    keypoints: np.ndarray,
    skeleton: SkeletonType,
) -> Dict[str, float]:
    """Compute all joint angles from 3D keypoints.

    Uses common14 skeleton as intermediate representation.

    Args:
        keypoints: 3D keypoints, shape (J, 3) in any skeleton format.
        skeleton: Type of skeleton.

    Returns:
        Dict mapping joint name to angle in degrees.
    """
    from evaluation.skeleton_mapping import remap_to_common14

    # Remap to common14 for consistent angle computation
    kps14 = remap_to_common14(keypoints, skeleton)

    angles = {}
    for angle_name, (prox, vert, dist) in JOINT_ANGLE_DEFS.items():
        if prox in COMMON14_INDICES and vert in COMMON14_INDICES and dist in COMMON14_INDICES:
            p_idx = COMMON14_INDICES[prox]
            v_idx = COMMON14_INDICES[vert]
            d_idx = COMMON14_INDICES[dist]

            if max(p_idx, v_idx, d_idx) < len(kps14):
                angle = compute_angle_from_keypoints(kps14, p_idx, v_idx, d_idx)
                angles[angle_name] = angle

    return angles


def compute_icc(
    predicted: np.ndarray,
    ground_truth: np.ndarray,
    model: int = 3,
) -> float:
    """Intraclass Correlation Coefficient (ICC).

    Computes ICC(3,1) by default — two-way mixed, single measures, absolute agreement.
    This is the standard ICC model for inter-rater reliability in rehabilitation research.

    Args:
        predicted: Predicted values, shape (N,).
        ground_truth: Ground truth values, shape (N,).
        model: ICC model (1, 2, or 3). Default is 3 (two-way mixed).

    Returns:
        ICC value in [0, 1]. Values > 0.75 indicate good agreement.

    Reference:
        Koo & Li (2016). A Guideline of Selecting and Reporting Intraclass
        Correlation Coefficients for Reliability Research. J Chiropr Med.
    """
    predicted = np.asarray(predicted).flatten()
    ground_truth = np.asarray(ground_truth).flatten()

    n = len(predicted)
    if n < 2:
        return 0.0

    # Build 2-column matrix: each row is one measurement, columns are raters
    # Shape: (N, 2) where col 0 = predicted, col 1 = ground truth
    data = np.column_stack([predicted, ground_truth])
    k = 2  # number of raters

    # Grand mean
    grand_mean = np.mean(data)

    # Between-subject mean
    row_means = np.mean(data, axis=1)

    # Sum of squares
    ss_total = np.sum((data - grand_mean) ** 2)
    ss_between = k * np.sum((row_means - grand_mean) ** 2)
    ss_within = np.sum((data - row_means[:, np.newaxis]) ** 2)
    ss_error = ss_within  # for model 3, error = within

    # Mean squares
    ms_between = ss_between / (n - 1) if n > 1 else 0
    ms_within = ss_within / (n * (k - 1)) if n * (k - 1) > 0 else 0

    if model == 3:
        # ICC(3,1): two-way mixed, single measures, absolute agreement
        if ms_within == 0 and ms_between == 0:
            return 0.0
        icc = (ms_between - ms_within) / (ms_between + (k - 1) * ms_within)
    elif model == 2:
        # ICC(2,1): two-way random, single measures, absolute agreement
        ms_rows = ms_between
        icc = (ms_rows - ms_within) / (ms_rows + (k - 1) * ms_within + k * (ms_between - ms_within) / n)
    elif model == 1:
        # ICC(1,1): one-way random, single measures
        ms_rows = ms_between
        icc = (ms_rows - ms_within) / (ms_rows + (k - 1) * ms_within)
    else:
        raise ValueError(f"Unsupported ICC model: {model}")

    return float(max(0.0, min(1.0, icc)))


def compute_angular_correlation(
    predicted: np.ndarray,
    ground_truth: np.ndarray,
) -> float:
    """Compute Pearson correlation between predicted and GT angle trajectories.

    Useful for evaluating temporal alignment of movement patterns.

    Args:
        predicted: Angle trajectory, shape (T,) or (T, J).
        ground_truth: Angle trajectory, same shape.

    Returns:
        Pearson correlation coefficient in [-1, 1].
    """
    predicted = np.asarray(predicted).flatten()
    ground_truth = np.asarray(ground_truth).flatten()

    if len(predicted) < 2:
        return 0.0

    # Pearson correlation
    pred_centered = predicted - np.mean(predicted)
    gt_centered = ground_truth - np.mean(ground_truth)

    numerator = np.sum(pred_centered * gt_centered)
    denominator = np.sqrt(np.sum(pred_centered ** 2) * np.sum(gt_centered ** 2))

    if denominator < 1e-10:
        return 0.0

    return float(np.clip(numerator / denominator, -1.0, 1.0))
