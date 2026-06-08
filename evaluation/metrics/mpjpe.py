"""
MPJPE (Mean Per Joint Position Error) metrics.

Standard protocols from the 3D pose estimation literature:
- MPJPE (Protocol 1): Root-aligned, mean L2 distance per joint
- P-MPJPE (Protocol 2): Procrustes-aligned (translation + rotation + scale)
- N-MPJPE: Scale-only alignment (normalized MPJPE)
- Per-joint MPJPE: Error breakdown by joint

Reference: Pavlakos et al., "Ordinal Depth Supervision for 3D Human Pose
Estimation," CVPR 2018. Implementation follows VideoPose3D (Facebook Research).

All inputs should be in the same unit (typically millimeters).
Expected shape: (num_frames, num_joints, 3) or (num_joints, 3) for single frame.
"""

import numpy as np
from typing import Optional, Dict, Tuple


def compute_mpjpe(predicted: np.ndarray, ground_truth: np.ndarray) -> float:
    """Mean Per-Joint Position Error (root-aligned).

    Protocol 1: Both skeletons are root-aligned (subtract pelvis/hip joint),
    then compute mean Euclidean distance across all joints and frames.

    Args:
        predicted: Predicted 3D keypoints, shape (J, 3) or (F, J, 3).
        ground_truth: Ground truth 3D keypoints, same shape as predicted.

    Returns:
        MPJPE in the same unit as input (typically mm).

    Example:
        >>> pred = np.random.rand(17, 3) * 500  # mm
        >>> gt = np.random.rand(17, 3) * 500
        >>> error = compute_mpjpe(pred, gt)
    """
    predicted, ground_truth = _ensure_3d(predicted, ground_truth)

    # Root-align: subtract pelvis (joint 0) from all joints
    pred_root = predicted[:, :1, :]  # (F, 1, 3)
    gt_root = ground_truth[:, :1, :]
    pred_centered = predicted - pred_root
    gt_centered = ground_truth - gt_root

    # Mean L2 distance per joint, averaged across all joints and frames
    per_joint = np.linalg.norm(pred_centered - gt_centered, axis=-1)  # (F, J)
    return float(np.mean(per_joint))


def compute_p_mpjpe(predicted: np.ndarray, ground_truth: np.ndarray) -> float:
    """Procrustes-aligned MPJPE (PA-MPJPE).

    Protocol 2: Applies optimal rigid alignment (translation + rotation + scale)
    using SVD, then computes mean L2 distance. This measures shape error only,
    removing all rigid-body differences.

    Implementation follows VideoPose3D (Facebook Research) exactly.

    Args:
        predicted: Predicted 3D keypoints, shape (J, 3) or (F, J, 3).
        ground_truth: Ground truth 3D keypoints, same shape as predicted.

    Returns:
        P-MPJPE in the same unit as input (typically mm).
    """
    predicted, ground_truth = _ensure_3d(predicted, ground_truth)
    assert predicted.shape == ground_truth.shape, \
        f"Shape mismatch: {predicted.shape} vs {ground_truth.shape}"

    # Step 1: Center both point sets
    muX = np.mean(ground_truth, axis=1, keepdims=True)  # (F, 1, 3)
    muY = np.mean(predicted, axis=1, keepdims=True)
    X0 = ground_truth - muX
    Y0 = predicted - muY

    # Step 2: Normalize by Frobenius norm
    normX = np.sqrt(np.sum(X0 ** 2, axis=(1, 2), keepdims=True))  # (F, 1, 1)
    normY = np.sqrt(np.sum(Y0 ** 2, axis=(1, 2), keepdims=True))
    # Avoid division by zero
    normX = np.maximum(normX, 1e-10)
    normY = np.maximum(normY, 1e-10)
    X0 = X0 / normX
    Y0 = Y0 / normY

    # Step 3: SVD of cross-covariance matrix to find optimal rotation
    H = np.matmul(X0.transpose(0, 2, 1), Y0)  # (F, 3, 3)
    U, s, Vt = np.linalg.svd(H)  # U, Vt: (F, 3, 3), s: (F, 3)
    V = Vt.transpose(0, 2, 1)

    # Step 4: Avoid improper rotations (reflections)
    sign_detR = np.sign(np.expand_dims(np.linalg.det(
        np.matmul(V, U.transpose(0, 2, 1))), axis=1))  # (F, 1)
    V[:, :, -1] *= sign_detR
    s[:, -1] *= sign_detR.flatten()

    # Step 5: Recover optimal rotation, scale, and translation
    R = np.matmul(V, U.transpose(0, 2, 1))  # (F, 3, 3)
    tr = np.expand_dims(np.sum(s, axis=1, keepdims=True), axis=2)  # (F, 1, 1)
    a = tr * normX / normY  # optimal scale (F, 1, 1)
    t = muX - a * np.matmul(muY, R)  # optimal translation (F, 1, 3)

    # Step 6: Apply alignment and compute error
    predicted_aligned = a * np.matmul(predicted, R) + t  # (F, J, 3)
    per_joint = np.linalg.norm(predicted_aligned - ground_truth, axis=-1)  # (F, J)
    return float(np.mean(per_joint))


def compute_n_mpjpe(predicted: np.ndarray, ground_truth: np.ndarray) -> float:
    """Normalized MPJPE (scale-only alignment).

    Aligns by translation (root) and scale (Frobenius norm), but NOT rotation.
    Useful for evaluating global orientation accuracy.

    Args:
        predicted: Predicted 3D keypoints, shape (J, 3) or (F, J, 3).
        ground_truth: Ground truth 3D keypoints, same shape.

    Returns:
        N-MPJPE in the same unit as input.
    """
    predicted, ground_truth = _ensure_3d(predicted, ground_truth)

    # Root-align
    pred_root = predicted[:, :1, :]
    gt_root = ground_truth[:, :1, :]
    pred_centered = predicted - pred_root
    gt_centered = ground_truth - gt_root

    # Scale-align: normalize by Frobenius norm
    pred_norm = np.sqrt(np.sum(pred_centered ** 2, axis=(1, 2), keepdims=True))
    gt_norm = np.sqrt(np.sum(gt_centered ** 2, axis=(1, 2), keepdims=True))
    pred_norm = np.maximum(pred_norm, 1e-10)
    gt_norm = np.maximum(gt_norm, 1e-10)

    pred_scaled = pred_centered / pred_norm
    gt_scaled = gt_centered / gt_norm

    per_joint = np.linalg.norm(pred_scaled - gt_scaled, axis=-1)
    return float(np.mean(per_joint))


def compute_per_joint_mpjpe(
    predicted: np.ndarray,
    ground_truth: np.ndarray,
    joint_names: Optional[list] = None,
) -> Dict[str, float]:
    """Per-joint MPJPE breakdown.

    Args:
        predicted: Predicted 3D keypoints, shape (J, 3) or (F, J, 3).
        ground_truth: Ground truth 3D keypoints, same shape.
        joint_names: Optional list of joint names. If None, uses indices.

    Returns:
        Dict mapping joint name/index to MPJPE in mm.
    """
    predicted, ground_truth = _ensure_3d(predicted, ground_truth)

    # Root-align
    pred_root = predicted[:, :1, :]
    gt_root = ground_truth[:, :1, :]
    pred_centered = predicted - pred_root
    gt_centered = ground_truth - gt_root

    # Per-joint L2 distance, averaged across frames
    per_joint = np.linalg.norm(pred_centered - gt_centered, axis=-1)  # (F, J)
    mean_per_joint = np.mean(per_joint, axis=0)  # (J,)

    result = {}
    for j in range(mean_per_joint.shape[0]):
        name = joint_names[j] if joint_names and j < len(joint_names) else f"joint_{j}"
        result[name] = float(mean_per_joint[j])

    return result


def compute_mpjpe_with_scale(
    predicted: np.ndarray,
    ground_truth: np.ndarray,
    pred_unit: str = "m",
    gt_unit: str = "m",
) -> Tuple[float, float, float]:
    """Compute MPJPE and P-MPJPE with automatic unit conversion to mm.

    Args:
        predicted: Predicted 3D keypoints.
        ground_truth: Ground truth 3D keypoints.
        pred_unit: Unit of predicted keypoints ("m", "cm", "mm", "normalized").
        gt_unit: Unit of ground truth keypoints.

    Returns:
        Tuple of (mpjpe_mm, p_mpjpe_mm, scale_factor_to_mm).
    """
    # Convert both to mm
    pred_mm = _convert_to_mm(predicted, pred_unit)
    gt_mm = _convert_to_mm(ground_truth, gt_unit)

    mpjpe_val = compute_mpjpe(pred_mm, gt_mm)
    p_mpjpe_val = compute_p_mpjpe(pred_mm, gt_mm)

    return mpjpe_val, p_mpjpe_val, 1.0  # already in mm


def _convert_to_mm(keypoints: np.ndarray, unit: str) -> np.ndarray:
    """Convert keypoints to millimeters."""
    factors = {"m": 1000.0, "cm": 10.0, "mm": 1.0}
    if unit == "normalized":
        raise ValueError(
            "Cannot convert normalized coordinates to mm without scale reference. "
            "Use compute_mpjpe_with_body_scale() instead."
        )
    factor = factors.get(unit, 1.0)
    return keypoints * factor


def _ensure_3d(
    predicted: np.ndarray, ground_truth: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Ensure arrays are 3D: (F, J, 3)."""
    if predicted.ndim == 2:
        predicted = predicted[np.newaxis]
    if ground_truth.ndim == 2:
        ground_truth = ground_truth[np.newaxis]
    assert predicted.shape[-1] == 3, f"Last dim must be 3, got {predicted.shape[-1]}"
    assert ground_truth.shape[-1] == 3, f"Last dim must be 3, got {ground_truth.shape[-1]}"
    return predicted, ground_truth
