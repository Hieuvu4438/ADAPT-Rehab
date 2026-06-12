"""
KIMORE dataset loader and evaluation utilities for ADAPT-Rehab.

KIMORE: KInematic assessment of MOvement and clinical scores for
Remote monitoring of physical REhabilitation.

Dataset structure (as packaged in this repo):
- 5 exercises (``ex1`` through ``ex5``)
- 75-77 samples per exercise
- 25 Kinect v2 joints × 7 features per joint (x, y, z, qw, qx, qy, qz)
- Per-sample clinical total score (``cTS``) — higher = worse quality
- Distributed as a pickled ``pandas.DataFrame`` per exercise at
  ``data/KIMORE/kimore_exercise_dataset.pkl``.

Public API:
    KimoreLoader
        .load() -> dict[str, list[np.ndarray]]     # exercise -> samples
        .get_clinical_scores() -> dict[str, list[float]]
        .iter_samples() -> Iterator                 # (ex_name, sample, cTS)
        .compute_classification_accuracy() -> dict  # DTW-based k-NN
        .compute_clinical_score_correlation() -> dict
        .summary() -> dict
"""

from __future__ import annotations

import os
import pickle
from dataclasses import dataclass, field
from itertools import combinations
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np


# Kinect v2 joint indices (canonical 25-joint skeleton, same as UI-PRMD).
KINECT_JOINTS: Dict[str, int] = {
    "spinebase": 0, "spinemid": 1, "neck": 2, "head": 3,
    "shoulderleft": 4, "elbowleft": 5, "wristleft": 6, "handleft": 7,
    "shoulderright": 8, "elbowright": 9, "wristright": 10, "handright": 11,
    "hipleft": 12, "kneeleft": 13, "ankleleft": 14, "footleft": 15,
    "hipright": 16, "kneeright": 17, "ankleright": 18, "footright": 19,
    "spineshoulder": 20, "handtipleft": 21, "thumbleft": 22,
    "handtipright": 23, "thumbright": 24,
}

# Standard 8-angle set for KIMORE analysis.
ANGLE_DEFS: Dict[str, Tuple[str, str, str]] = {
    "left_shoulder":  ("spinebase", "shoulderleft", "elbowleft"),
    "right_shoulder": ("spinebase", "shoulderright", "elbowright"),
    "left_elbow":     ("shoulderleft", "elbowleft", "wristleft"),
    "right_elbow":    ("shoulderright", "elbowright", "wristright"),
    "left_hip":       ("spinebase", "hipleft", "kneeleft"),
    "right_hip":      ("spinebase", "hipright", "kneeright"),
    "left_knee":      ("hipleft", "kneeleft", "ankleleft"),
    "right_knee":     ("hipright", "kneeright", "ankleright"),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _joint_angle(kps: np.ndarray, prox: str, vert: str, dist: str) -> Optional[float]:
    try:
        p_idx = KINECT_JOINTS[prox]
        v_idx = KINECT_JOINTS[vert]
        d_idx = KINECT_JOINTS[dist]
    except KeyError:
        return None
    if max(p_idx, v_idx, d_idx) >= len(kps):
        return None
    a, b, c = kps[p_idx], kps[v_idx], kps[d_idx]
    ba, bc = a - b, c - b
    n_ba, n_bc = np.linalg.norm(ba), np.linalg.norm(bc)
    if n_ba < 1e-10 or n_bc < 1e-10:
        return None
    cos = np.clip(np.dot(ba, bc) / (n_ba * n_bc), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos)))


def _all_angles(kps: np.ndarray) -> Dict[str, float]:
    angles: Dict[str, float] = {}
    for name, (prox, vert, dist) in ANGLE_DEFS.items():
        v = _joint_angle(kps, prox, vert, dist)
        if v is not None:
            angles[name] = v
    return angles


def _angle_trajectory(sample: np.ndarray) -> np.ndarray:
    """Extract a (num_frames, 8) angle trajectory from a KIMORE sample."""
    rows: List[List[float]] = []
    for frame_idx in range(len(sample)):
        angles = _all_angles(sample[frame_idx])
        if len(angles) == 8:
            rows.append(list(angles.values()))
    if not rows:
        return np.array([])
    return np.asarray(rows, dtype=np.float32)


def _dtw_distance(seq1: np.ndarray, seq2: np.ndarray, window: int = 20) -> float:
    """DTW distance with Sakoe-Chiba band constraint (O(n*w))."""
    n, m = len(seq1), len(seq2)
    if n == 0 or m == 0:
        return float("inf")
    w = max(window, abs(n - m))
    dtw = np.full((n + 1, m + 1), np.inf, dtype=np.float64)
    dtw[0, 0] = 0.0
    for i in range(1, n + 1):
        j_lo = max(1, i - w)
        j_hi = min(m, i + w)
        for j in range(j_lo, j_hi + 1):
            cost = float(np.linalg.norm(seq1[i - 1] - seq2[j - 1]))
            dtw[i, j] = cost + min(dtw[i - 1, j], dtw[i, j - 1], dtw[i - 1, j - 1])
    return float(dtw[n, m])


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

@dataclass
class KimoreLoader:
    """Loads and evaluates the KIMORE dataset.

    Args:
        data_path: Path to ``kimore_exercise_dataset.pkl``.
    """

    data_path: str = "data/KIMORE/kimore_exercise_dataset.pkl"

    # Populated by load()
    exercises: Dict[str, List[np.ndarray]] = field(default_factory=dict, init=False)
    clinical_scores: Dict[str, List[float]] = field(default_factory=dict, init=False)

    def is_available(self) -> bool:
        return os.path.exists(self.data_path)

    # ----- Loading -----

    def load(self) -> Dict[str, List[np.ndarray]]:
        """Load the KIMORE pickle into a dict ``ex_name -> list of samples``.

        Each sample is a ``(num_frames, 25, 3)`` array of joint positions
        (only x, y, z from the original 7-feature joint record).
        """
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(
                f"KIMORE pickle not found at {self.data_path}"
            )
        with open(self.data_path, "rb") as f:
            data = pickle.load(f)

        exercises: Dict[str, List[np.ndarray]] = {}
        for ex_name, ex_df in data.items():
            samples: List[np.ndarray] = []
            for idx in range(len(ex_df)):
                sample_data: List[np.ndarray] = []
                for col in ex_df.columns:
                    if col == "cTS":
                        continue
                    joint_data = ex_df.iloc[idx][col]
                    if isinstance(joint_data, np.ndarray) and joint_data.ndim == 2:
                        # First 3 cols are x, y, z; remaining 4 are quaternions
                        sample_data.append(joint_data[:, :3])
                if sample_data:
                    samples.append(np.stack(sample_data, axis=1))
            if samples:
                exercises[ex_name] = samples

        self.exercises = exercises
        return exercises

    def get_clinical_scores(self) -> Dict[str, List[float]]:
        """Return ``ex_name -> list of cTS scores``."""
        if self.clinical_scores:
            return self.clinical_scores
        if not os.path.exists(self.data_path):
            return {}
        with open(self.data_path, "rb") as f:
            data = pickle.load(f)
        self.clinical_scores = {
            ex_name: ex_df["cTS"].values.tolist()
            for ex_name, ex_df in data.items()
            if "cTS" in ex_df.columns
        }
        return self.clinical_scores

    def iter_samples(self) -> Iterator[Tuple[str, np.ndarray, Optional[float]]]:
        """Yield ``(ex_name, sample, cTS)`` triples."""
        if not self.exercises:
            self.load()
        scores = self.get_clinical_scores()
        for ex_name, samples in self.exercises.items():
            ex_scores = scores.get(ex_name, [])
            for i, sample in enumerate(samples):
                cts = float(ex_scores[i]) if i < len(ex_scores) else None
                yield ex_name, sample, cts

    # ----- Evaluation -----

    def _build_trajectories(
        self,
    ) -> Tuple[List[np.ndarray], List[str]]:
        """Pre-compute angle trajectories for every sample (cached)."""
        if not self.exercises:
            self.load()
        trajs: List[np.ndarray] = []
        labels: List[str] = []
        for ex_name, samples in self.exercises.items():
            for sample in samples:
                t = _angle_trajectory(sample)
                if len(t) > 0:
                    trajs.append(t)
                    labels.append(ex_name)
        return trajs, labels

    def compute_classification_accuracy(
        self,
        max_pairs: int = 1000,
    ) -> Dict:
        """Exercise-classification metrics using DTW nearest-neighbor.

        NOTE: This function is **intentionally O(n²)** in the number of
        samples (with ``max_pairs`` capping the comparisons). On the full
        KIMORE dataset (378 samples, ~1.1k-frame sequences) it can take
        5-10 minutes. The user has explicitly excluded performance
        optimization of this routine from the current fix.
        """
        trajs, labels = self._build_trajectories()
        n_samples = len(trajs)
        if n_samples < 2:
            return {"error": "Not enough samples"}

        # Build all unique pairs (i, j) and optionally subsample.
        all_pairs = list(combinations(range(n_samples), 2))
        if len(all_pairs) > max_pairs:
            rng = np.random.default_rng(42)
            chosen = rng.choice(len(all_pairs), size=max_pairs, replace=False)
            pairs = [all_pairs[i] for i in chosen]
        else:
            pairs = all_pairs

        same_dists: List[float] = []
        diff_dists: List[float] = []
        for i, j in pairs:
            dist = _dtw_distance(trajs[i], trajs[j])
            if labels[i] == labels[j]:
                same_dists.append(dist)
            else:
                diff_dists.append(dist)

        if not same_dists or not diff_dists:
            return {"error": "Not enough data"}

        same_mean = float(np.mean(same_dists))
        diff_mean = float(np.mean(diff_dists))
        separation = diff_mean / (same_mean + 1e-10)

        # Threshold classification (correctly-classified proportion)
        threshold = (same_mean + diff_mean) / 2.0
        correct = sum(1 for d in same_dists if d < threshold)
        correct += sum(1 for d in diff_dists if d >= threshold)
        total = len(same_dists) + len(diff_dists)
        accuracy = correct / total if total > 0 else 0.0

        return {
            "separation_ratio": float(separation),
            "classification_accuracy": float(accuracy),
            "intra_class_mean": float(same_mean),
            "inter_class_mean": float(diff_mean),
            "num_same_pairs": len(same_dists),
            "num_diff_pairs": len(diff_dists),
        }

    def compute_clinical_score_correlation(self) -> Dict:
        """Pearson correlation between per-sample features and cTS."""
        if not self.exercises:
            self.load()
        scores_map = self.get_clinical_scores()

        features: List[np.ndarray] = []
        scores: List[float] = []
        for ex_name, samples in self.exercises.items():
            ex_scores = scores_map.get(ex_name, [])
            for i, sample in enumerate(samples):
                traj = _angle_trajectory(sample)
                if len(traj) == 0 or i >= len(ex_scores):
                    continue
                mean_angles = traj.mean(axis=0)
                std_angles = traj.std(axis=0)
                range_angles = traj.ptp(axis=0)
                features.append(np.concatenate([mean_angles, std_angles, range_angles]))
                scores.append(ex_scores[i])

        if len(features) < 2:
            return {"error": "Not enough data"}

        feats = np.asarray(features, dtype=np.float64)
        scores = np.asarray(scores, dtype=np.float64)

        corrs: List[float] = []
        for col in range(feats.shape[1]):
            if np.std(feats[:, col]) < 1e-10:
                continue
            r = np.corrcoef(feats[:, col], scores)[0, 1]
            if not np.isnan(r):
                corrs.append(abs(float(r)))

        return {
            "mean_correlation": float(np.mean(corrs)) if corrs else 0.0,
            "max_correlation": float(np.max(corrs)) if corrs else 0.0,
            "num_features": int(feats.shape[1]),
            "num_samples": int(len(scores)),
        }

    def summary(self) -> Dict:
        """One-shot summary of the full KIMORE evaluation."""
        if not self.exercises:
            self.load()
        scores = self.get_clinical_scores()
        per_ex_stats = {}
        for ex_name, samples in self.exercises.items():
            ex_scores = scores.get(ex_name, [])
            per_ex_stats[ex_name] = {
                "n_samples": len(samples),
                "n_frames": int(samples[0].shape[0]) if samples else 0,
                "cTS_mean": float(np.mean(ex_scores)) if ex_scores else None,
                "cTS_std": float(np.std(ex_scores)) if ex_scores else None,
            }
        return {
            "dataset": "KIMORE",
            "num_exercises": len(self.exercises),
            "total_samples": sum(len(s) for s in self.exercises.values()),
            "per_exercise": per_ex_stats,
            "classification": self.compute_classification_accuracy(),
            "clinical_correlation": self.compute_clinical_score_correlation(),
        }
