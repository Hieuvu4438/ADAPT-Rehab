"""MPJPE (Mean Per Joint Position Error) metrics."""
import numpy as np


def compute_mpjpe(predicted: np.ndarray, ground_truth: np.ndarray) -> float:
    if predicted.ndim == 2:
        predicted, ground_truth = predicted[np.newaxis], ground_truth[np.newaxis]
    return float(np.mean(np.linalg.norm(predicted - ground_truth, axis=2)))


def compute_p_mpjpe(predicted: np.ndarray, ground_truth: np.ndarray) -> float:
    from scipy.spatial import procrustes
    if predicted.ndim == 3:
        return float(np.mean([compute_p_mpjpe(p, g) for p, g in zip(predicted, ground_truth)]))
    _, aligned, _ = procrustes(ground_truth, predicted)
    return float(np.mean(np.linalg.norm(aligned - ground_truth, axis=1)))
