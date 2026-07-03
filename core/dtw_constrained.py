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
    # Default: L2 norm of the elementwise difference, which works for both
    # 1D scalars (e.g. np.float64 vs np.float64) and 2D row vectors
    # (e.g. a (3,) joint position vs another (3,) joint position).
    if dist_fn is None:
        def dist_fn(a, b):
            diff = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
            if diff.ndim == 0:
                return abs(float(diff))
            return float(np.linalg.norm(diff))

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
    weights: "Optional[dict]" = None,
    window_percent: float = 0.1,
) -> Tuple[float, float, dict]:
    """
    Weighted DTW with constraints for multiple joints.

    Args:
        user_seqs: Dict mapping joint_name -> user angle sequence.
        ref_seqs: Dict mapping joint_name -> reference angle sequence.
        weights: Dict mapping joint_name -> importance weight. If None, equal
            weight (1.0) is assigned to every joint present in ``user_seqs``.
        window_percent: Sakoe-Chiba window width.

    Returns:
        Tuple of ``(similarity_score, total_distance, per_joint_details)``
        where ``similarity_score`` is in ``[0, 100]`` (100 = identical),
        ``total_distance`` is the path-length-normalized weighted distance,
        and ``per_joint_details`` maps each joint to a dict with
        ``distance``, ``normalized_distance``, ``weight``, and ``path_length``.

    Example:
        >>> user = {"shoulder": [10, 20, 30], "elbow": [5, 10, 15]}
        >>> ref = {"shoulder": [12, 22, 32], "elbow": [6, 11, 16]}
        >>> sim, total, details = weighted_constrained_dtw(user, ref)
    """
    if weights is None:
        weights = {joint: 1.0 for joint in user_seqs}

    total_weighted_dist = 0.0
    total_weight = 0.0
    details = {}

    for joint, user_seq in user_seqs.items():
        if joint not in ref_seqs:
            continue

        weight = weights.get(joint, 0.5)
        if weight < 1e-6:
            continue

        # Mean-center both trajectories and scale by reference amplitude (ROM).
        # This removes absolute position offset while preserving amplitude info.
        # If user has same ROM as ref → both span [-0.5, 0.5] → low distance.
        # If user has less ROM → user spans smaller range → higher distance.
        user_arr = np.array(user_seq, dtype=np.float64)
        ref_arr = np.array(ref_seqs[joint], dtype=np.float64)
        ref_centered = ref_arr - np.mean(ref_arr)
        ref_amp = max(np.max(ref_centered) - np.min(ref_centered), 1e-6)
        user_norm = (user_arr - np.mean(user_arr)) / ref_amp
        ref_norm = ref_centered / ref_amp

        # Compute constrained DTW
        dist, path = constrained_dtw(user_norm, ref_norm, window_percent)

        # Normalize by path length for per-frame average distance
        path_len = max(len(path), 1)
        normalized_dist = dist / path_len

        total_weighted_dist += weight * normalized_dist
        total_weight += weight

        details[joint] = {
            "distance": dist,
            "normalized_distance": normalized_dist,
            "weight": weight,
            "path_length": len(path),
        }

    final_dist = total_weighted_dist / total_weight if total_weight > 0 else 0.0

    # Convert per-frame distance to similarity.
    # In amplitude-normalized units, identical=0, similar=0.05-0.15,
    # moderately different=0.2-0.4, very different=0.5+.
    # Use exp(-d*5): d=0→100, d=0.1→61, d=0.2→37, d=0.5→8.
    similarity = 100.0 * float(np.exp(-final_dist * 5.0))
    similarity = float(np.clip(similarity, 0, 100))

    return similarity, final_dist, details
