"""Joint angle MAE and ICC metrics."""
import numpy as np
from typing import Dict


def compute_angle_mae(predicted: Dict[str, float], ground_truth: Dict[str, float]) -> float:
    errors = [abs(predicted[j] - gt) for j, gt in ground_truth.items() if j in predicted]
    return float(np.mean(errors)) if errors else 0.0


def compute_icc(predicted: np.ndarray, ground_truth: np.ndarray) -> float:
    n = len(predicted)
    if n < 2: return 0.0
    mp, mg = np.mean(predicted), np.mean(ground_truth)
    gm = (mp + mg) / 2
    ss_b = n * ((mp - gm)**2 + (mg - gm)**2)
    ss_w = np.sum((predicted - ground_truth)**2) / 2
    ms_b = ss_b / (n - 1) if n > 1 else ss_b
    ms_w = ss_w / n
    return float(max(0, min(1, (ms_b - ms_w) / (ms_b + ms_w + 1e-10))))
