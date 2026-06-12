"""
DTW Analysis Module for ADAPT-Rehab.

Weighted Dynamic Time Warping for comparing movement rhythms between
user and reference video.

References:
    - Tormene, P., et al. (2009). "How to normalize sequences for dynamic
      time warping." Information Sciences, 179(13).
    - Sakoe, H., & Chiba, S. (1978). "Dynamic programming algorithm
      optimization for spoken word recognition." IEEE Trans. ASSP, 26(1).
    - Winter, D.A. (2009). Biomechanics and Motor Control of Human Movement.

Author: ADAPT-Rehab Team
Version: 2.0.0
"""

from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Union
import numpy as np

try:
    from fastdtw import fastdtw
    FASTDTW_AVAILABLE = True
except ImportError:
    FASTDTW_AVAILABLE = False

from scipy.spatial.distance import euclidean
from scipy.signal import butter, filtfilt

from .kinematics import JointType


@dataclass
class DTWResult:
    """DTW analysis result.

    Attributes:
        distance: Raw DTW distance (lower = more similar).
        normalized_distance: Distance normalized by warping path length.
        path: Optimal warping path [(i1, j1), (i2, j2), ...].
        similarity_score: Similarity score (0-100%).
        rhythm_quality: Quality rating ("excellent", "good", "fair", "poor").
        details: Detailed per-joint analysis.
    """
    distance: float
    normalized_distance: float
    path: List[Tuple[int, int]]
    similarity_score: float
    rhythm_quality: str
    details: Dict = None

    def __post_init__(self):
        if self.details is None:
            self.details = {}


def preprocess_sequence(
    sequence: Union[List[float], np.ndarray],
    smooth_window: int = 5,
    normalize: bool = True,
    fs: float = 30.0,
    filter_cutoff: float = 10.0
) -> np.ndarray:
    """Preprocess sequence before DTW computation.

    Uses Butterworth low-pass filter (biomechanics standard) instead of
    moving average for better frequency response.

    Args:
        sequence: Raw angle sequence.
        smooth_window: Minimum window size for fallback smoothing.
        normalize: Whether to normalize to [0, 1].
        fs: Sampling frequency in Hz (default: 30.0).
        filter_cutoff: Butterworth filter cutoff frequency in Hz (default: 10.0).

    Returns:
        Preprocessed sequence.
    """
    arr = np.array(sequence, dtype=np.float64)

    if len(arr) == 0:
        return arr

    # Butterworth low-pass filter (Winter, 2009) - biomechanics standard
    if len(arr) >= 32:  # Minimum for filtfilt stability
        nyquist = fs / 2.0
        normalized_cutoff = filter_cutoff / nyquist
        if normalized_cutoff < 1.0:
            b, a = butter(4, normalized_cutoff, btype='low')
            arr = filtfilt(b, a, arr)
    elif smooth_window > 1 and len(arr) >= smooth_window:
        # Fallback: simple moving average for very short sequences
        from scipy.ndimage import uniform_filter1d
        arr = uniform_filter1d(arr, size=smooth_window, mode='nearest')

    # Normalize to [0, 1] (optional - loses absolute angle information)
    if normalize:
        min_val, max_val = arr.min(), arr.max()
        if max_val - min_val > 1e-6:
            arr = (arr - min_val) / (max_val - min_val)

    return arr


def compute_dtw_distance(
    seq1: Union[List[float], np.ndarray],
    seq2: Union[List[float], np.ndarray],
    radius: int = 1
) -> Tuple[float, List[Tuple[int, int]]]:
    """Compute DTW distance between two 1D sequences.

    Uses FastDTW if available, falls back to simple O(n*m) implementation.

    Args:
        seq1: First sequence.
        seq2: Second sequence.
        radius: Search radius for FastDTW.

    Returns:
        Tuple of (distance, warping_path).
    """
    arr1 = np.array(seq1).reshape(-1, 1)
    arr2 = np.array(seq2).reshape(-1, 1)

    if len(arr1) == 0 or len(arr2) == 0:
        return 0.0, []

    if FASTDTW_AVAILABLE:
        distance, path = fastdtw(arr1, arr2, radius=radius, dist=euclidean)
    else:
        distance, path = _simple_dtw(arr1.flatten(), arr2.flatten())

    return float(distance), list(path)


def _simple_dtw(
    seq1: np.ndarray,
    seq2: np.ndarray
) -> Tuple[float, List[Tuple[int, int]]]:
    """Simple DTW implementation when fastdtw is not available.

    O(n*m) time and space complexity.
    """
    n, m = len(seq1), len(seq2)

    # Accumulated cost matrix
    dtw_matrix = np.full((n + 1, m + 1), np.inf)
    dtw_matrix[0, 0] = 0

    # Fill matrix
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = abs(seq1[i-1] - seq2[j-1])
            dtw_matrix[i, j] = cost + min(
                dtw_matrix[i-1, j],     # Insertion
                dtw_matrix[i, j-1],     # Deletion
                dtw_matrix[i-1, j-1]    # Match
            )

    # Backtrack to find optimal path
    path = []
    i, j = n, m
    while i > 0 or j > 0:
        path.append((i-1, j-1))
        if i == 0:
            j -= 1
        elif j == 0:
            i -= 1
        else:
            candidates = [
                (dtw_matrix[i-1, j-1], i-1, j-1),
                (dtw_matrix[i-1, j], i-1, j),
                (dtw_matrix[i, j-1], i, j-1),
            ]
            _, i, j = min(candidates, key=lambda x: x[0])

    path.reverse()
    return dtw_matrix[n, m], path


def compute_weighted_dtw(
    user_sequences: Dict[JointType, List[float]],
    ref_sequences: Dict[JointType, List[float]],
    weights: Optional[Dict[JointType, float]] = None,
    preprocess: bool = True
) -> DTWResult:
    """Compute Weighted DTW across multiple joints.

    Formula:
        Total Distance = Σ (weight_i × dtw_distance_i) / Σ weight_i

    Normalization uses path-length normalization per Tormene et al. (2009):
        normalized_distance = distance / len(warping_path)

    Args:
        user_sequences: Dict mapping JointType → user angle sequence.
        ref_sequences: Dict mapping JointType → reference angle sequence.
        weights: Dict mapping JointType → weight (0-1). If None, equal weight
            (1.0) is assigned to every joint present in ``user_sequences``.
        preprocess: Whether to preprocess sequences.

    Returns:
        DTWResult with analysis results.
    """
    if weights is None:
        weights = {joint: 1.0 for joint in user_sequences}
    if not user_sequences or not ref_sequences:
        return DTWResult(
            distance=0.0,
            normalized_distance=0.0,
            path=[],
            similarity_score=0.0,
            rhythm_quality="no_data"
        )

    total_weighted_distance = 0.0
    total_weight = 0.0
    joint_details = {}
    combined_path = []

    for joint_type, user_seq in user_sequences.items():
        if joint_type not in ref_sequences:
            continue

        ref_seq = ref_sequences[joint_type]
        weight = weights.get(joint_type, 0.5)
        if weight < 1e-6:
            continue

        # Joint key as string (accepts both JointType enum and str)
        joint_key = joint_type.value if hasattr(joint_type, "value") else str(joint_type)

        # Preprocess with Butterworth filter
        if preprocess:
            user_processed = preprocess_sequence(user_seq)
            ref_processed = preprocess_sequence(ref_seq)
        else:
            user_processed = np.array(user_seq)
            ref_processed = np.array(ref_seq)

        # Compute DTW for this joint
        distance, path = compute_dtw_distance(user_processed, ref_processed)

        # Path-length normalization (Tormene et al., 2009)
        # This is the correct normalization for clinical time series
        path_len = len(path)
        normalized = distance / path_len if path_len > 0 else 0

        # Accumulate
        total_weighted_distance += weight * normalized
        total_weight += weight

        joint_details[joint_key] = {
            "distance": distance,
            "normalized_distance": normalized,
            "weight": weight,
            "weighted_contribution": weight * normalized,
            "path_length": path_len,
        }

        if not combined_path:
            combined_path = path

    # Compute final weighted distance
    if total_weight > 0:
        final_distance = total_weighted_distance / total_weight
    else:
        final_distance = 0.0

    # Convert distance to similarity score using Gaussian kernel
    # sigma = 0.5 provides good discrimination for normalized distances
    sigma = 0.5
    similarity_score = 100.0 * np.exp(-final_distance**2 / (2 * sigma**2))
    similarity_score = float(np.clip(similarity_score, 0, 100))

    rhythm_quality = _evaluate_rhythm_quality(similarity_score)

    return DTWResult(
        distance=total_weighted_distance,
        normalized_distance=final_distance,
        path=combined_path,
        similarity_score=similarity_score,
        rhythm_quality=rhythm_quality,
        details={"joints": joint_details, "total_weight": total_weight}
    )


def compute_single_joint_dtw(
    user_sequence: List[float],
    ref_sequence: List[float],
    preprocess: bool = True
) -> DTWResult:
    """Compute DTW for a single joint.

    Args:
        user_sequence: User angle sequence.
        ref_sequence: Reference angle sequence.
        preprocess: Whether to preprocess sequences.

    Returns:
        DTWResult with analysis results.
    """
    if not user_sequence or not ref_sequence:
        return DTWResult(
            distance=0.0,
            normalized_distance=0.0,
            path=[],
            similarity_score=0.0,
            rhythm_quality="no_data"
        )

    if preprocess:
        user_processed = preprocess_sequence(user_sequence)
        ref_processed = preprocess_sequence(ref_sequence)
    else:
        user_processed = np.array(user_sequence)
        ref_processed = np.array(ref_sequence)

    distance, path = compute_dtw_distance(user_processed, ref_processed)

    # Path-length normalization (Tormene et al., 2009)
    path_len = len(path)
    normalized = distance / path_len if path_len > 0 else 0

    # Gaussian kernel similarity
    sigma = 0.5
    similarity_score = 100.0 * np.exp(-normalized**2 / (2 * sigma**2))
    similarity_score = float(np.clip(similarity_score, 0, 100))

    rhythm_quality = _evaluate_rhythm_quality(similarity_score)

    return DTWResult(
        distance=distance,
        normalized_distance=normalized,
        path=path,
        similarity_score=similarity_score,
        rhythm_quality=rhythm_quality
    )


def _evaluate_rhythm_quality(similarity_score: float) -> str:
    """Evaluate rhythm quality from similarity score.

    Args:
        similarity_score: Similarity score (0-100).

    Returns:
        Quality string: "excellent", "good", "fair", or "poor".
    """
    if similarity_score >= 85:
        return "excellent"
    elif similarity_score >= 70:
        return "good"
    elif similarity_score >= 50:
        return "fair"
    else:
        return "poor"


def get_rhythm_feedback(dtw_result: DTWResult) -> str:
    """Generate rhythm feedback for the user.

    Args:
        dtw_result: DTW analysis result.

    Returns:
        Feedback message string.
    """
    quality = dtw_result.rhythm_quality
    score = dtw_result.similarity_score

    feedback_map = {
        "excellent": f"Tuyệt vời! Nhịp điệu rất mượt mà ({score:.0f}%)!",
        "good": f"Tốt lắm! Nhịp điệu khá ổn ({score:.0f}%).",
        "fair": f"Được rồi! Cố gắng đều tay hơn nhé ({score:.0f}%).",
        "poor": f"Không sao! Từ từ luyện tập sẽ tốt hơn ({score:.0f}%).",
        "unknown": "Đang phân tích..."
    }

    return feedback_map.get(quality, feedback_map["unknown"])


def analyze_speed_variation(
    timestamps: List[float],
    angles: List[float]
) -> Dict[str, float]:
    """Analyze speed variation in movement.

    Args:
        timestamps: Timestamp sequence (seconds).
        angles: Corresponding angle sequence.

    Returns:
        Dict with speed metrics.
    """
    if len(timestamps) < 2 or len(angles) < 2:
        return {"mean_velocity": 0, "velocity_std": 0, "smoothness": 1.0}

    ts = np.array(timestamps)
    ang = np.array(angles)

    dt = np.diff(ts)
    da = np.diff(ang)

    dt = np.where(dt < 1e-6, 1e-6, dt)
    velocity = da / dt

    mean_vel = float(np.mean(np.abs(velocity)))
    std_vel = float(np.std(velocity))

    if mean_vel > 1e-6:
        smoothness = 1.0 / (1.0 + std_vel / mean_vel)
    else:
        smoothness = 1.0

    return {
        "mean_velocity": mean_vel,
        "velocity_std": std_vel,
        "smoothness": float(smoothness),
        "max_velocity": float(np.max(np.abs(velocity))),
        "min_velocity": float(np.min(np.abs(velocity))),
    }


def create_exercise_weights(
    exercise_type: str
) -> Dict[JointType, float]:
    """Create weight table for exercise type.

    Args:
        exercise_type: "arm_raise", "squat", "bicep_curl", etc.

    Returns:
        Dict mapping JointType to weight.
    """
    default_weights = {jt: 0.5 for jt in JointType}

    exercise_weights = {
        "arm_raise": {
            JointType.LEFT_SHOULDER: 1.0,
            JointType.RIGHT_SHOULDER: 1.0,
            JointType.LEFT_ELBOW: 0.6,
            JointType.RIGHT_ELBOW: 0.6,
            JointType.LEFT_KNEE: 0.1,
            JointType.RIGHT_KNEE: 0.1,
            JointType.LEFT_HIP: 0.2,
            JointType.RIGHT_HIP: 0.2,
        },
        "squat": {
            JointType.LEFT_KNEE: 1.0,
            JointType.RIGHT_KNEE: 1.0,
            JointType.LEFT_HIP: 0.8,
            JointType.RIGHT_HIP: 0.8,
            JointType.LEFT_SHOULDER: 0.2,
            JointType.RIGHT_SHOULDER: 0.2,
            JointType.LEFT_ELBOW: 0.1,
            JointType.RIGHT_ELBOW: 0.1,
        },
        "bicep_curl": {
            JointType.LEFT_ELBOW: 1.0,
            JointType.RIGHT_ELBOW: 1.0,
            JointType.LEFT_SHOULDER: 0.5,
            JointType.RIGHT_SHOULDER: 0.5,
            JointType.LEFT_KNEE: 0.1,
            JointType.RIGHT_KNEE: 0.1,
            JointType.LEFT_HIP: 0.2,
            JointType.RIGHT_HIP: 0.2,
        },
    }

    weights = exercise_weights.get(exercise_type.lower(), {})

    result = default_weights.copy()
    result.update(weights)

    return result
