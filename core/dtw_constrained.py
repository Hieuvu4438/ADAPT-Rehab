"""
Constrained Dynamic Time Warping.

DTW with Sakoe-Chiba band constraint to prevent
pathological alignments where many frames map to one frame.

Standard DTW: O(n*m)
Constrained DTW: O(n*w) where w = window width

Usage:
    from core.dtw_constrained import constrained_dtw

    distance, path = constrained_dtw(seq1, seq2, window_percent=0.1)
"""

import numpy as np
from typing import Tuple, List, Optional


def constrained_dtw(
    seq1: np.ndarray,
    seq2: np.ndarray,
    window_percent: float = 0.1,
    min_window: int = 5,
    dist_fn=None,
) -> Tuple[float, List[Tuple[int, int]]]:
    """
    DTW with Sakoe-Chiba band constraint.

    The constraint limits how far the warping path can deviate
    from the diagonal, preventing pathological alignments.

    Args:
        seq1: First sequence (1D array).
        seq2: Second sequence (1D array).
        window_percent: Window width as percentage of max sequence length.
        min_window: Minimum window width (prevents too tight constraints).
        dist_fn: Distance function. Default: absolute difference.

    Returns:
        Tuple of (distance, path) where path is list of (i, j) pairs.

    Example:
        >>> s1 = np.sin(np.linspace(0, 2*np.pi, 100))
        >>> s2 = np.sin(np.linspace(0, 2*np.pi, 120))
        >>> dist, path = constrained_dtw(s1, s2, window_percent=0.15)
        >>> print(f"Distance: {dist:.3f}")
    """
    n, m = len(seq1), len(seq2)

    if n == 0 or m == 0:
        return 0.0, []

    # Compute window width
    w = max(int(max(n, m) * window_percent), abs(n - m), min_window)

    # Distance function
    if dist_fn is None:
        dist_fn = lambda a, b: abs(a - b)

    # Initialize cost matrix with infinity
    # Only compute within the Sakoe-Chiba band
    dtw = np.full((n + 1, m + 1), np.inf)
    dtw[0, 0] = 0

    # Fill cost matrix within band
    for i in range(1, n + 1):
        # Band boundaries for row i
        j_start = max(1, i - w)
        j_end = min(m, i + w)

        for j in range(j_start, j_end + 1):
            cost = dist_fn(seq1[i - 1], seq2[j - 1])
            dtw[i, j] = cost + min(
                dtw[i - 1, j],      # Insertion
                dtw[i, j - 1],      # Deletion
                dtw[i - 1, j - 1],  # Match
            )

    # Backtrack to find optimal path
    path = []
    i, j = n, m

    while i > 0 or j > 0:
        path.append((i - 1, j - 1))

        if i == 0:
            j -= 1
        elif j == 0:
            i -= 1
        else:
            candidates = [
                (dtw[i - 1, j - 1], i - 1, j - 1),
                (dtw[i - 1, j], i - 1, j),
                (dtw[i, j - 1], i, j - 1),
            ]
            _, i, j = min(candidates, key=lambda x: x[0])

    path.reverse()
    return float(dtw[n, m]), path


def weighted_constrained_dtw(
    user_seqs: dict,
    ref_seqs: dict,
    weights: dict,
    window_percent: float = 0.1,
) -> Tuple[float, dict]:
    """
    Weighted DTW with constraints for multiple joints.

    Args:
        user_seqs: Dict mapping joint_name -> user angle sequence.
        ref_seqs: Dict mapping joint_name -> reference angle sequence.
        weights: Dict mapping joint_name -> importance weight.
        window_percent: Sakoe-Chiba window width.

    Returns:
        Tuple of (total_distance, per_joint_details).

    Example:
        >>> user = {"shoulder": [10, 20, 30], "elbow": [5, 10, 15]}
        >>> ref = {"shoulder": [12, 22, 32], "elbow": [6, 11, 16]}
        >>> weights = {"shoulder": 1.0, "elbow": 0.5}
        >>> dist, details = weighted_constrained_dtw(user, ref, weights)
    """
    total_weighted_dist = 0.0
    total_weight = 0.0
    details = {}

    for joint, user_seq in user_seqs.items():
        if joint not in ref_seqs:
            continue

        weight = weights.get(joint, 0.5)
        if weight < 1e-6:
            continue

        # Normalize sequences
        user_norm = _normalize(user_seq)
        ref_norm = _normalize(ref_seq)

        # Compute constrained DTW
        dist, path = constrained_dtw(user_norm, ref_norm, window_percent)

        # Normalize by sequence length
        seq_len = max(len(user_seq), len(ref_seq))
        normalized_dist = dist / seq_len if seq_len > 0 else 0

        total_weighted_dist += weight * normalized_dist
        total_weight += weight

        details[joint] = {
            "distance": dist,
            "normalized_distance": normalized_dist,
            "weight": weight,
            "path_length": len(path),
        }

    final_dist = total_weighted_dist / total_weight if total_weight > 0 else 0.0

    # Convert to similarity score (0-100)
    similarity = 100.0 * np.exp(-final_dist * 3)
    similarity = float(np.clip(similarity, 0, 100))

    return similarity, details


def _normalize(seq) -> np.ndarray:
    """Normalize sequence to [0, 1]."""
    arr = np.array(seq, dtype=np.float64)
    min_val, max_val = arr.min(), arr.max()
    if max_val - min_val > 1e-6:
        arr = (arr - min_val) / (max_val - min_val)
    return arr
