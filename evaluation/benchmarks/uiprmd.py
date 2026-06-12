"""
UI-PRMD dataset loader and evaluation utilities for ADAPT-Rehab.

UI-PRMD: University of Idaho - Physical Rehabilitation Movement Data.

Dataset structure:
- 1423 samples (correct + incorrect repetitions of 10 exercises)
- 25 Kinect v2 joints × 4 values (x, y, z, confidence)
- Confidence: 1 (occluded) or 2 (visible)
- Filename pattern: ``input.csv`` for the data, ``label.csv`` for the
  per-sample label (1 = correct, 0 = incorrect).

Public API:
    UI_PRMDLoader
        .load() -> np.ndarray            # (N, 25, 3) 3D keypoints
        .get_confidence_mask() -> ndarray  # (N, 25) boolean
        .get_labels() -> np.ndarray        # (N,) 0/1 labels
        .compute_joint_angles() -> dict    # per-joint angle statistics
        .compute_smoothness() -> dict      # SPARC per joint
        .compute_self_consistency() -> dict
        .iter_samples() -> Iterator        # (keypoints, label) per sample
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd


# Kinect v2 joint indices used in UI-PRMD (canonical 25-joint skeleton).
KINECT_JOINTS: Dict[str, int] = {
    "spinebase": 0, "spinemid": 1, "neck": 2, "head": 3,
    "shoulderleft": 4, "elbowleft": 5, "wristleft": 6, "handleft": 7,
    "shoulderright": 8, "elbowright": 9, "wristright": 10, "handright": 11,
    "hipleft": 12, "kneeleft": 13, "ankleleft": 14, "footleft": 15,
    "hipright": 16, "kneeright": 17, "ankleright": 18, "footright": 19,
    "spineshoulder": 20, "handtipleft": 21, "thumbleft": 22,
    "handtipright": 23, "thumbright": 24,
}

# Joint angle definitions: (proximal_name, vertex_name, distal_name).
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
# Low-level helpers
# ---------------------------------------------------------------------------

def _joint_angle(kps: np.ndarray, prox_name: str, vert_name: str, dist_name: str) -> float:
    """Compute a single joint angle (degrees) from 3D keypoints."""
    try:
        p_idx = KINECT_JOINTS[prox_name]
        v_idx = KINECT_JOINTS[vert_name]
        d_idx = KINECT_JOINTS[dist_name]
    except KeyError:
        return 0.0
    if max(p_idx, v_idx, d_idx) >= len(kps):
        return 0.0

    a, b, c = kps[p_idx], kps[v_idx], kps[d_idx]
    ba, bc = a - b, c - b
    n_ba, n_bc = np.linalg.norm(ba), np.linalg.norm(bc)
    if n_ba < 1e-10 or n_bc < 1e-10:
        return 0.0
    cos = np.clip(np.dot(ba, bc) / (n_ba * n_bc), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos)))


def _all_angles(kps: np.ndarray) -> Dict[str, float]:
    """Compute all joint angles for one pose. Skips joints that return 0."""
    return {
        name: angle
        for name, (prox, vert, dist) in ANGLE_DEFS.items()
        for angle in [_joint_angle(kps, prox, vert, dist)]
        if angle > 0
    }


def _sparc(velocity: np.ndarray, fs: float = 30.0) -> float:
    """Spectral Arc Length smoothness metric (Balasubramanian 2012)."""
    from numpy.fft import fft, fftfreq
    N = len(velocity)
    if N < 4:
        return 0.0
    N_padded = N * 16
    Mf = np.abs(fft(velocity, n=N_padded))[:N_padded // 2 + 1]
    freq = fftfreq(N_padded, d=1.0 / fs)[:N_padded // 2 + 1]
    if Mf[0] > 0:
        Mf = Mf / Mf[0]
    mask = (freq <= 10.0) & (Mf >= 0.05)
    f_sel = freq[mask]
    Mf_sel = Mf[mask]
    if len(f_sel) < 2:
        return 0.0
    df = np.diff(f_sel)
    dM = np.diff(Mf_sel)
    return float(-np.sum(np.sqrt(df**2 + dM**2)))


# ---------------------------------------------------------------------------
# Loader class
# ---------------------------------------------------------------------------

@dataclass
class UI_PRMDLoader:
    """Loads and evaluates the UI-PRMD dataset.

    Args:
        data_dir: Directory containing ``input.csv`` (and optionally
            ``label.csv``). Defaults to ``data/UI-PRMD`` relative to the
            project root.
    """

    data_dir: str = "data/UI-PRMD"
    n_joints: int = 25

    # Populated by load()
    keypoints: Optional[np.ndarray] = field(default=None, init=False)
    confidence: Optional[np.ndarray] = field(default=None, init=False)
    labels: Optional[np.ndarray] = field(default=None, init=False)

    @property
    def input_csv(self) -> str:
        return os.path.join(self.data_dir, "input.csv")

    @property
    def label_csv(self) -> str:
        return os.path.join(self.data_dir, "label.csv")

    def is_available(self) -> bool:
        return os.path.exists(self.input_csv)

    def load(self) -> np.ndarray:
        """Load 3D keypoints from ``input.csv`` into a (N, 25, 3) array."""
        if not os.path.exists(self.input_csv):
            raise FileNotFoundError(
                f"UI-PRMD input.csv not found at {self.input_csv}"
            )
        df = pd.read_csv(self.input_csv, header=None)
        n_samples = df.shape[0]
        kps = np.zeros((n_samples, self.n_joints, 3), dtype=np.float32)
        for i in range(n_samples):
            row = df.iloc[i].values
            for j in range(self.n_joints):
                idx = j * 4  # 4 values per joint: x, y, z, conf
                kps[i, j, 0] = row[idx]
                kps[i, j, 1] = row[idx + 1]
                kps[i, j, 2] = row[idx + 2]
        self.keypoints = kps
        return kps

    def get_confidence_mask(self) -> np.ndarray:
        """Return a (N, 25) boolean mask — True where the joint is visible.

        Kinect v2 marks visible joints with confidence == 2; occluded
        joints have confidence == 1.
        """
        if self.confidence is not None:
            return self.confidence
        if not os.path.exists(self.input_csv):
            raise FileNotFoundError(self.input_csv)
        df = pd.read_csv(self.input_csv, header=None)
        n_samples = df.shape[0]
        mask = np.zeros((n_samples, self.n_joints), dtype=bool)
        for i in range(n_samples):
            row = df.iloc[i].values
            for j in range(self.n_joints):
                mask[i, j] = (row[j * 4 + 3] == 2)
        self.confidence = mask
        return mask

    def get_labels(self) -> np.ndarray:
        """Return the (N,) label array (1 = correct, 0 = incorrect)."""
        if self.labels is not None:
            return self.labels
        if not os.path.exists(self.label_csv):
            return np.array([], dtype=np.int64)
        df = pd.read_csv(self.label_csv, header=None)
        self.labels = df.values.flatten().astype(np.int64)
        return self.labels

    def iter_samples(self) -> Iterator[Tuple[np.ndarray, int]]:
        """Yield ``(keypoints, label)`` for every sample in the dataset."""
        if self.keypoints is None:
            self.load()
        labels = self.get_labels()
        n = len(self.keypoints)
        # If label.csv is missing or shorter than the dataset, default
        # to label=1 (correct) for every sample.
        if len(labels) < n:
            labels = np.ones(n, dtype=np.int64)
        for i, kps in enumerate(self.keypoints):
            yield kps, int(labels[i])

    # ----- Evaluation helpers -----

    def compute_self_consistency(self) -> Dict:
        """Self-consistency MPJPE against the mean pose (per-sample set)."""
        if self.keypoints is None:
            self.load()
        kps = self.keypoints
        mean_kps = kps.mean(axis=0)
        errors = np.linalg.norm(kps - mean_kps, axis=-1)  # (N, J)
        per_joint = errors.mean(axis=0)
        return {
            "mpjpe_mean": float(errors.mean()),
            "mpjpe_std": float(errors.std()),
            "per_joint_error": per_joint.tolist(),
        }

    def compute_joint_angles(self) -> Dict:
        """Mean / std / min / max of each joint angle across all samples."""
        if self.keypoints is None:
            self.load()
        all_angles: Dict[str, List[float]] = {k: [] for k in ANGLE_DEFS}
        for kps in self.keypoints:
            angles = _all_angles(kps)
            for k, v in angles.items():
                all_angles[k].append(v)
        return {
            name: {
                "mean": float(np.mean(vs)),
                "std":  float(np.std(vs)),
                "min":  float(np.min(vs)),
                "max":  float(np.max(vs)),
            }
            for name, vs in all_angles.items() if vs
        }

    def compute_smoothness(self, fs: float = 30.0) -> Dict:
        """SPARC smoothness for each joint-angle trajectory."""
        if self.keypoints is None:
            self.load()
        sparc_per_joint: Dict[str, float] = {}
        for name in ANGLE_DEFS:
            traj = []
            for kps in self.keypoints:
                angles = _all_angles(kps)
                if name in angles:
                    traj.append(angles[name])
            if len(traj) > 10:
                vel = np.diff(np.asarray(traj)) * fs
                sparc_per_joint[name] = _sparc(vel, fs=fs)
        return {
            "sparc": sparc_per_joint,
            "mean_sparc": float(np.mean(list(sparc_per_joint.values())))
            if sparc_per_joint else 0.0,
        }

    def summary(self) -> Dict:
        """One-shot summary of the full UI-PRMD evaluation."""
        if self.keypoints is None:
            self.load()
        return {
            "n_samples": int(len(self.keypoints)),
            "n_joints": int(self.keypoints.shape[1]),
            "shape": list(self.keypoints.shape),
            "self_consistency": self.compute_self_consistency(),
            "angle_statistics": self.compute_joint_angles(),
            "smoothness": self.compute_smoothness(),
        }
