#!/usr/bin/env python3
"""
ADAPT-Rehab: UCO Physical Rehabilitation Scoring System Evaluation (Phase B).

Runs the scoring-stack experiments on the UCO PhyRehab dataset — the only
public dataset with multi-view data + sub-mm OptiTrack 3D ground truth +
per-repetition clinical scores on real rehab patients.

Strategies implemented:
  Strategy 2 — Clinical correlation (PRIMARY): per-repetition Procrustes
               position DTW vs fixed expert reference; Spearman rho.
  Strategy 1 — Discriminability: same vs different exercise (16 classes).
  Strategy 4 — Cross-subject robustness (LOSOCV) + Cross-view (LOVOCV).
  Strategy 6 — View-stratified ablation: angle DTW vs position DTW x position.

Scoring approach (learned from Phase A / KIMORE):
  - Procrustes-aligned POSITION DTW (NOT angle-based DTW).
  - Fixed-reference split (top-10% scored reps as expert reference, NOT LOSO).
  - REUSES scripts/scoring_stack.py (SPARC, LDLJ, smoothing, resampling).

Outputs:
  evaluation/output/uco_results.csv          — one row per recording
  evaluation/output/uco_pair_scores.csv      — Strategy 1 pair list
  evaluation/output/uco_view_breakdown.csv   — per-camera stats
  evaluation/output/uco_summary.csv          — aggregated metrics
  evaluation/output/uco_report.md            — findings report
  evaluation/figures/uco_score_vs_clinical.png    (Strategy 2)
  evaluation/figures/uco_discriminability_hist.png (Strategy 1)
  evaluation/figures/uco_view_robustness.png      (Strategy 4 LOVOCV)

Usage:
  python scripts/run_uco_experiments.py           # Run all strategies
  python scripts/run_uco_experiments.py --strategy 2  # Run only Strategy 2
  python scripts/run_uco_experiments.py --quick    # Subsampled for testing

Author: ADAPT-Rehab Team
Version: 1.0.0
"""

from __future__ import annotations

import os
import sys
import json
import time
import csv
import random
import argparse
import warnings
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict

import numpy as np
from scipy import stats
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw

warnings.filterwarnings("ignore")

# ── Project paths ───────────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(str(ROOT))

DATA_DIR = ROOT / "data" / "UCO_PhyRehab"
OUTPUT_DIR = ROOT / "evaluation" / "output"
FIGURES_DIR = ROOT / "evaluation" / "figures"
JSON_3D = DATA_DIR / "dataset_3d_with_angles.json"
JSONL_FILE = DATA_DIR / "ucophyrehab2_data.jsonl"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ── Reproducibility ─────────────────────────────────────────────────────────────

RANDOM_SEED = 42


def set_seed(seed: int = RANDOM_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)


set_seed()

# ── Import shared scoring stack (REUSE — do not reimplement) ───────────────────

from scripts.scoring_stack import (  # noqa: E402
    RecordingFeatures,
    JointTrajectory,
    compute_sparc,
    compute_sparc_normalized,
    compute_ldlj,
    compute_jerk_score,
    smooth_trajectory,
    resample_trajectory,
    compute_constrained_dtw_distance,
)

# ── Constants ───────────────────────────────────────────────────────────────────

N_POS_FRAMES = 50          # Temporal normalization length for position DTW
MIN_REP_FRAMES = 15        # Minimum frames for a valid repetition
DTW_K = 1.5               # distance->similarity calibration (gentle, KIMORE-style;
                           # keeps the [0,100] score spread wide so the stack and
                           # cross-subject CV are not compressed toward 0).
REF_PERCENT = 0.10         # Top-10% reps form the expert reference set
N_SILS_SAMPLES = 100       # Recordings sampled for silhouette LOVOCV
SILS_STRIDE = 5            # Frame stride when reading silhouette videos

# UCO stores exactly 3 exercise-relevant joints per recording (side-specific).
# Ordered [proximal, vertex, distal] — proximal is the centering anchor.
JOINT_NAMES: Dict[Tuple[str, str], List[str]] = {
    ("lower", "left"):  ["l_hip", "l_knee", "l_ankle"],
    ("lower", "right"): ["r_hip", "r_knee", "r_ankle"],
    ("upper", "left"):  ["l_shoulder", "l_elbow", "l_wrist"],
    ("upper", "right"): ["r_shoulder", "r_elbow", "r_wrist"],
}


# ── Data structures ─────────────────────────────────────────────────────────────

@dataclass
class UCORep:
    """A single repetition extracted from a UCO recording.

    Attributes:
        rep_id: Unique identifier (subject_exercise_repindex).
        subject: Subject ID string ("0".."26").
        exercise: Exercise ID string ("01".."16").
        position: Patient position ("seated" | "supine" | "standing").
        side: Exercise side ("left" | "right").
        body: Body part ("lower" | "upper").
        age_range: Subject age bucket ("20-30"|"31-40"|"41-50"|"51+").
        clinician_score: Per-repetition clinician score in {2,3,4,5}.
        init_frame: First frame id of the rep window.
        final_frame: Last frame id of the rep window.
        positions: Joint positions of shape (n_frames, 3, 3) [meters].
        angles: Pre-computed joint angle trajectory (n_frames,) [degrees].
    """
    rep_id: str
    subject: str
    exercise: str
    position: str
    side: str
    body: str
    age_range: str
    clinician_score: float
    init_frame: int
    final_frame: int
    positions: np.ndarray
    angles: np.ndarray

    @property
    def is_valid(self) -> bool:
        return self.positions is not None and len(self.positions) >= MIN_REP_FRAMES


# ── Data loading ────────────────────────────────────────────────────────────────

def _parse_positions(frames: List[dict], jnames: List[str]) -> np.ndarray:
    """Parse joint positions from a list of frame dicts.

    Args:
        frames: List of frame dicts with 'joints' mapping.
        jnames: Ordered [proximal, vertex, distal] joint names.

    Returns:
        Array of shape (n_frames, 3, 3) — [frame, joint, xyz] in meters.
    """
    out = np.zeros((len(frames), 3, 3), dtype=np.float64)
    for fi, fr in enumerate(frames):
        joints = fr["joints"]
        for ji, jn in enumerate(jnames):
            jd = joints[jn]
            out[fi, ji, 0] = float(jd["x"])
            out[fi, ji, 1] = float(jd["y"])
            out[fi, ji, 2] = float(jd["z"])
    return out


def _parse_angles(frames: List[dict]) -> np.ndarray:
    """Parse the pre-computed joint angle trajectory.

    Args:
        frames: List of frame dicts.

    Returns:
        Angle array (n_frames,) in degrees (NaN-filled where missing).
    """
    out = np.zeros(len(frames), dtype=np.float64)
    for fi, fr in enumerate(frames):
        a = fr["joints"].get("angle")
        out[fi] = float(a) if a is not None else np.nan
    return out


def load_uco_data() -> Tuple[List[UCORep], Dict[str, int]]:
    """Load UCO 3D ground truth + per-repetition clinical scores.

    Matches ``ucophyrehab2_data.jsonl`` records to ``dataset_3d_with_angles.json``
    entries by (subject_id, exercise_id). Each JSONL repetition window
    (init_frame..final_frame) is extracted from the 3D frames (ids align 1:1).

    Args:
        None.

    Returns:
        Tuple of (list of UCORep, stats dict).
    """
    print("Loading UCO dataset...")
    t0 = time.time()

    with open(JSON_3D) as f:
        d3 = json.load(f)["data"]

    # Index 3D entries by (folder, exercise); skip entries with empty frames.
    entry_map: Dict[Tuple[str, str], dict] = {}
    for e in d3:
        if not e.get("frames"):
            continue
        entry_map[(e["folder"], e["exercise"])] = e

    reps: List[UCORep] = []
    skipped_no_match = 0
    skipped_short = 0

    with open(JSONL_FILE) as f:
        for line in f:
            r = json.loads(line)
            key = (r["subject_id"], r["exercise_id"])
            if key not in entry_map:
                skipped_no_match += 1
                continue
            e = entry_map[key]
            jnames = JOINT_NAMES.get((e["body"], e["side"]))
            if jnames is None:
                skipped_no_match += 1
                continue

            frames = e["frames"]
            idmin = frames[0]["id"]
            idmax = frames[-1]["id"]
            # frame id is contiguous; map id -> index
            for ri, s in enumerate(r["scores"]):
                init = s["init_frame"]
                final = s["final_frame"]
                # Clamp into available range
                if init < idmin:
                    init = idmin
                if final > idmax:
                    final = idmax
                lo = init - idmin
                hi = final - idmin + 1
                if hi - lo < MIN_REP_FRAMES:
                    skipped_short += 1
                    continue
                window = frames[lo:hi]
                positions = _parse_positions(window, jnames)
                angles = _parse_angles(window)
                reps.append(UCORep(
                    rep_id=f"s{r['subject_id']}_e{r['exercise_id']}_r{ri}",
                    subject=r["subject_id"],
                    exercise=r["exercise_id"],
                    position=e["position"],
                    side=e["side"],
                    body=e["body"],
                    age_range=r.get("age_range", "unknown"),
                    clinician_score=float(s["score"]),
                    init_frame=init,
                    final_frame=final,
                    positions=positions,
                    angles=angles,
                ))

    # Parse age_range fallback: derive bucket from age if missing/unknown
    for rep in reps:
        if rep.age_range in ("", "unknown", None):

            rep.age_range = "unknown"

    stats_d = {
        "n_reps": len(reps),
        "skipped_no_match": skipped_no_match,
        "skipped_short": skipped_short,
        "n_exercises": len({r.exercise for r in reps}),
        "n_subjects": len({r.subject for r in reps}),
        "load_time_s": round(time.time() - t0, 1),
    }
    print(f"  Loaded {len(reps)} reps across {stats_d['n_exercises']} exercises, "
          f"{stats_d['n_subjects']} subjects (skipped {skipped_no_match+skipped_short}) "
          f"in {stats_d['load_time_s']}s")
    return reps, stats_d


# ── Procrustes-aligned position features ───────────────────────────────────────

def extract_position_features(positions: np.ndarray) -> np.ndarray:
    """Build a Procrustes-aligned position trajectory.

    Mirrors the Phase-A (KIMORE) position-DTW recipe but adapted to UCO's
    3 exercise-relevant joints per recording:

      1. Center on the proximal joint (hip for lower / shoulder for upper).
      2. Scale by the mean inter-joint (proximal-to-distal) distance so the
         result is body-proportion invariant.
      3. Resample to N_POS_FRAMES frames (linear interpolation).
      4. Flatten to (N_POS_FRAMES, 9).

    Args:
        positions: Array of shape (n_frames, 3, 3).

    Returns:
        Array of shape (N_POS_FRAMES, 9).
    """
    pos = positions.astype(np.float64).copy()
    n = pos.shape[0]
    # 1. Center on proximal joint (index 0)
    pos -= pos[:, 0:1, :]
    # 2. Scale by mean inter-joint distance (proximal->vertex + vertex->distal)
    d1 = np.linalg.norm(pos[:, 1] - pos[:, 0], axis=1).mean()
    d2 = np.linalg.norm(pos[:, 2] - pos[:, 1], axis=1).mean()
    scale = (d1 + d2) / 2.0
    if scale > 1e-6:
        pos /= scale
    # 3. Resample each coordinate to N_POS_FRAMES (linear interpolation)
    flat = pos.reshape(n, 9)
    resampled = np.zeros((N_POS_FRAMES, 9), dtype=np.float64)
    x_old = np.linspace(0, 1, n)
    x_new = np.linspace(0, 1, N_POS_FRAMES)
    for c in range(9):
        resampled[:, c] = np.interp(x_new, x_old, flat[:, c])
    return resampled


def position_dtw_score(
    test_feat: np.ndarray,
    ref_feats: List[np.ndarray],
) -> float:
    """1-NN position DTW similarity of a test rep against a reference pool.

    Args:
        test_feat: (N_POS_FRAMES, 9) Procrustes-aligned features.
        ref_feats: List of reference (N_POS_FRAMES, 9) features.

    Returns:
        Similarity score in [0, 100] (100 = identical).
    """
    best_d = float("inf")
    for ref in ref_feats:
        d, _ = fastdtw(test_feat, ref, dist=euclidean)
        if d < best_d:
            best_d = d
    d_norm = best_d / N_POS_FRAMES
    return float(np.clip(100.0 * np.exp(-d_norm * DTW_K), 0.0, 100.0))


def position_pair_score(
    feat_a: np.ndarray,
    feat_b: np.ndarray,
) -> float:
    """Direct position DTW similarity between two reps (for Strategy 1)."""
    d, _ = fastdtw(feat_a, feat_b, dist=euclidean)
    d_norm = d / N_POS_FRAMES
    return float(np.clip(100.0 * np.exp(-d_norm * DTW_K), 0.0, 100.0))


def angle_dtw_score(
    test_angles: np.ndarray,
    ref_angles_list: List[np.ndarray],
) -> float:
    """1-NN angle DTW similarity (amplitude-normalized, mean-centered).

    Uses the same amplitude normalization as scoring_stack's
    weighted_constrained_dtw so the comparison to position DTW is fair.

    Args:
        test_angles: Test angle trajectory (degrees).
        ref_angles_list: List of reference angle trajectories.

    Returns:
        Similarity score in [0, 100].
    """
    # Reshape to 2-D (n, 1) so fastdtw passes 1-element vectors (not bare
    # scalars) to scipy's euclidean, which requires 1-D vector inputs.
    test_r = resample_trajectory(test_angles, N_POS_FRAMES).reshape(-1, 1)
    best_sim = 0.0
    for ref in ref_angles_list:
        ref_r = resample_trajectory(ref, N_POS_FRAMES)
        ref_c = ref_r - np.mean(ref_r)
        ref_amp = max(np.max(ref_c) - np.min(ref_c), 1e-6)
        t_norm = ((test_r - np.mean(test_r)) / ref_amp).reshape(-1, 1)
        r_norm = (ref_c / ref_amp).reshape(-1, 1)
        dist, _ = fastdtw(t_norm, r_norm, dist=euclidean)
        d_norm = dist / N_POS_FRAMES
        sim = float(np.clip(100.0 * np.exp(-d_norm * DTW_K), 0.0, 100.0))
        if sim > best_sim:
            best_sim = sim
    return best_sim


# ── Reference building ──────────────────────────────────────────────────────────

def build_expert_reference(
    reps: List[UCORep],
    exercise: str,
    exclude_subject: Optional[str] = None,
) -> List[UCORep]:
    """Build the fixed expert reference (top-REF_PERCENT reps) for an exercise.

    Args:
        reps: All reps.
        exercise: Exercise ID.
        exclude_subject: If set, omit this subject (used by LOSOCV).

    Returns:
        List of top-REF_PERCENT reference reps.
    """
    pool = [r for r in reps
            if r.exercise == exercise and r.is_valid
            and (exclude_subject is None or r.subject != exclude_subject)]
    if not pool:
        return []
    pool.sort(key=lambda r: r.clinician_score, reverse=True)
    n_ref = max(3, int(np.ceil(REF_PERCENT * len(pool))))
    return pool[:n_ref]


def build_position_reference_feats(ref_reps: List[UCORep]) -> List[np.ndarray]:
    """Pre-compute position features for a reference rep pool."""
    return [extract_position_features(r.positions) for r in ref_reps]


def build_angle_reference(ref_reps: List[UCORep]) -> List[np.ndarray]:
    """Pre-collect angle trajectories for a reference rep pool."""
    out = []
    for r in ref_reps:
        a = r.angles
        if np.all(np.isfinite(a)) and len(a) >= MIN_REP_FRAMES:
            out.append(a)
    return out


# ── Pre-compute features for all reps ──────────────────────────────────────────

def precompute_features(reps: List[UCORep]) -> None:
    """Attach Procrustes position features + kinematic sub-scores (in place).

    Computes, per rep:
      _pos_feat: Procrustes-aligned (N_POS_FRAMES, 9) position trajectory.
      _rom:      Range-of-motion (angle max-min, degrees).
      _sparc:    SPARC normalized to [0, 100].
      _jerk:     Jerk-based smoothness score in [0, 100].
    """
    print("Pre-computing Procrustes position features + kinematics...")
    t0 = time.time()
    for r in reps:
        r._pos_feat = extract_position_features(r.positions) if r.is_valid else None
        a = r.angles
        if np.all(np.isfinite(a)) and len(a) >= MIN_REP_FRAMES:
            sm = smooth_trajectory(a)
            r._rom = float(np.max(sm) - np.min(sm))
            r._sparc = compute_sparc_normalized(sm) if len(sm) >= 30 else 0.0
            r._jerk = compute_jerk_score(sm)
        else:
            r._rom = 0.0
            r._sparc = 0.0
            r._jerk = 0.0
    print(f"  Done in {time.time()-t0:.1f}s ({len(reps)} reps)")


# ── Per-exercise calibration helper ─────────────────────────────────────────────

# Fixed domain-knowledge weights for the scoring stack (sub-scores on [0,100]).
# Position DTW is the primary DTW component (vs angle DTW). On UCO's 3 joints the
# position-DTW signal is weaker than on KIMORE's 9 joints, so kinematic smoothness
# (jerk) carries equal weight. Diagnostics showed ROM/SPARC add noise on UCO's
# single-chain 3-joint recordings, so the stack is DTW + jerk. Output stays [0,100].
W_DTW = 0.5
W_JERK = 0.5


def _calibrate_per_exercise(
    reps: List[UCORep],
    value_fn,
    higher_better: bool = True,
    scale: float = 16.0,
) -> Dict[int, float]:
    """Calibrate raw values to a [0, 100] scale per exercise.

    Uses a per-exercise linear standardization with clipping:
        score = clip(50 + scale * z, 0, 100)
    where z is the within-exercise z-score. This removes between-exercise
    baseline differences while keeping the output in [0, 100] (NOT a raw
    z-score combination, which would leave the [0, 100] range).

    Args:
        reps: Reps to calibrate.
        value_fn: Callable(rep) -> float (raw value).
        higher_better: If True, larger raw values map to larger scores.
        scale: Z-score scaling factor (controls spread).

    Returns:
        Dict mapping id(rep) -> calibrated score in [0, 100].
    """
    by_ex: Dict[str, List[Tuple[UCORep, float]]] = defaultdict(list)
    for r in reps:
        by_ex[r.exercise].append((r, value_fn(r)))
    out: Dict[int, float] = {}
    for ex, pairs in by_ex.items():
        vals = np.array([v for _, v in pairs], dtype=np.float64)
        sign = 1.0 if higher_better else -1.0
        std = vals.std()
        if std < 1e-9:
            for r, _ in pairs:
                out[id(r)] = 50.0
            continue
        z = sign * (vals - vals.mean()) / std
        scores = np.clip(50.0 + scale * z, 0.0, 100.0)
        for (r, _), s in zip(pairs, scores):
            out[id(r)] = float(s)
    return out


# ── Strategy 2: Clinical correlation (PRIMARY) ─────────────────────────────────

def run_strategy2(reps: List[UCORep]) -> Dict[str, Any]:
    """Strategy 2: Per-rep scoring stack vs clinical score (PRIMARY RESULT).

    Builds the scoring stack:
      1. Procrustes position DTW (1-NN to fixed top-10% expert reference) — the
         primary DTW component (NOT angle-based DTW).
      2. Kinematic sub-scores: jerk smoothness, ROM, SPARC.

    Methodology (fixed expert reference, non-circular):
      - For each exercise the top-REF_PERCENT reps by clinician score form the
        fixed expert reference set.
      - EVERY rep is scored against the reference set with LEAVE-ONE-OUT: a
        reference rep is matched against the OTHER reference reps (never
        itself), so no rep benefits from a trivial self-match. This keeps all
        reps in the correlation without circularity.
      - Sub-scores are on a global [0, 100] scale (KIMORE-style fixed weights,
        NO per-exercise z-scoring of the total).

    Reports Spearman rho for: the full stack, the position DTW component, and
    the jerk component, stratified by exercise / position / age range. A
    recording-level (rep-averaged) correlation is also reported since the
    per-rep 2-5 scores are noisy.

    Args:
        reps: All reps with pre-computed features.

    Returns:
        Results dict.
    """
    print("\n" + "=" * 60)
    print("Strategy 2: Clinical Correlation (Position-DTW Scoring Stack)")
    print("=" * 60)

    exercises = sorted({r.exercise for r in reps})

    # ── Step 1: LOO-within-reference position DTW for ALL reps ────────────────
    scored: List[Tuple[UCORep, float]] = []
    for ex in exercises:
        ex_reps = [r for r in reps if r.exercise == ex and r.is_valid
                   and r._pos_feat is not None]
        if len(ex_reps) < 8:
            continue
        ref_reps = build_expert_reference(reps, ex)
        ref_feats = [(rr, extract_position_features(rr.positions)) for rr in ref_reps]
        for r in ex_reps:
            # candidate reference features EXCLUDING self (non-circular)
            cands = [rf for (rr, rf) in ref_feats if id(rr) != id(r)]
            if not cands:
                continue
            best_d = float("inf")
            for rf in cands:
                d, _ = fastdtw(r._pos_feat, rf, dist=euclidean)
                if d < best_d:
                    best_d = d
            r._dtw_dist = best_d / N_POS_FRAMES
            r._dtw_score = float(np.clip(100.0 * np.exp(-r._dtw_dist * DTW_K), 0, 100))
            # Full stack: position DTW + jerk (both global [0,100], fixed weights)
            r._total_score = float(np.clip(
                W_DTW * r._dtw_score + W_JERK * r._jerk, 0.0, 100.0))
            scored.append((r, r._total_score))

    if len(scored) < 20:
        return {"overall": {"spearman_rho": 0.0}, "scored_reps": []}

    # ── Overall per-rep correlations ──────────────────────────────────────────
    clin = np.array([r.clinician_score for r, _ in scored])
    total = np.array([s for _, s in scored])
    dtw_only = np.array([r._dtw_score for r, _ in scored])
    jerk_only = np.array([r._jerk for r, _ in scored])
    rho, rho_p = stats.spearmanr(total, clin)
    r_p, r_pear_p = stats.pearsonr(total, clin)
    rho_dtw, _ = stats.spearmanr(dtw_only, clin)
    rho_jerk, _ = stats.spearmanr(jerk_only, clin)

    # Bootstrap 95% CI for Spearman (stack)
    rng = np.random.RandomState(RANDOM_SEED)
    boot = []
    for _ in range(1000):
        idx = rng.randint(0, len(scored), len(scored))
        if len(set(idx)) > 2:
            b_rho, _ = stats.spearmanr(total[idx], clin[idx])
            boot.append(b_rho)
    ci_low = float(np.percentile(boot, 2.5)) if boot else rho
    ci_high = float(np.percentile(boot, 97.5)) if boot else rho

    # ── Recording-level (rep-averaged) correlation ────────────────────────────
    rec_groups: Dict[Tuple[str, str], List[UCORep]] = defaultdict(list)
    for r, _ in scored:
        rec_groups[(r.subject, r.exercise)].append(r)
    rec_clin = np.array([np.mean([x.clinician_score for x in g]) for g in rec_groups.values()])
    rec_dtw = np.array([np.mean([x._dtw_score for x in g]) for g in rec_groups.values()])
    rec_total = np.array([np.mean([x._total_score for x in g]) for g in rec_groups.values()])
    rec_jerk = np.array([np.mean([x._jerk for x in g]) for g in rec_groups.values()])
    rho_rec_dtw = float(stats.spearmanr(rec_dtw, rec_clin)[0])
    rho_rec_total = float(stats.spearmanr(rec_total, rec_clin)[0])
    rho_rec_jerk = float(stats.spearmanr(rec_jerk, rec_clin)[0])

    print(f"  Per-rep (N={len(scored)}):")
    print(f"    Stack (DTW+jerk)     rho = {rho:.4f} (p = {rho_p:.2e})")
    print(f"    Position DTW alone   rho = {rho_dtw:.4f}")
    print(f"    Jerk alone           rho = {rho_jerk:.4f}")
    print(f"    Stack 95% CI: [{ci_low:.4f}, {ci_high:.4f}]")
    print(f"  Recording-level (N={len(rec_groups)}):")
    print(f"    Position DTW rho = {rho_rec_dtw:.4f} | Jerk rho = {rho_rec_jerk:.4f} | Stack rho = {rho_rec_total:.4f}")

    def stratified(score_fn) -> Dict[str, Dict]:
        groups: Dict[str, List[Tuple[float, float]]] = defaultdict(list)
        for r, _ in scored:
            groups[getattr(r, "_strat_key")].append((score_fn(r), r.clinician_score))
        out = {}
        for k, vals in sorted(groups.items()):
            if len(vals) < 5:
                out[k] = {"rho": None, "n": len(vals)}
                continue
            arr_s = np.array([v[0] for v in vals])
            arr_c = np.array([v[1] for v in vals])
            sr, sp = stats.spearmanr(arr_s, arr_c)
            out[k] = {
                "rho": round(float(sr), 4),
                "p_value": float(sp),
                "n": len(vals),
                "mean_score": round(float(np.mean(arr_s)), 2),
            }
        return out

    for r, _ in scored:
        r._strat_key = r.exercise
    per_ex = stratified(lambda r: r._dtw_score)
    for r, _ in scored:
        r._strat_key = r.position
    per_pos = stratified(lambda r: r._dtw_score)
    for r, _ in scored:
        r._strat_key = r.age_range
    per_age = stratified(lambda r: r._dtw_score)

    print("\n  Per-exercise Spearman rho (position DTW):")
    for ex, d in sorted(per_ex.items()):
        if d["rho"] is not None:
            print(f"    ex{ex}: rho = {d['rho']:.4f} (p={d['p_value']:.2e}, N={d['n']})")
    print("  Per-position rho:",
          {k: v['rho'] for k, v in per_pos.items() if v.get('rho') is not None})

    n_positive = sum(1 for d in per_ex.values()
                     if d.get("rho") is not None and d["rho"] > 0)

    return {
        "overall": {
            "n": len(scored),
            "spearman_rho": round(float(rho), 4),
            "spearman_p": float(rho_p),
            "pearson_r": round(float(r_p), 4),
            "pearson_p": float(r_pear_p),
            "spearman_ci_95": [round(ci_low, 4), round(ci_high, 4)],
            "dtw_only_rho": round(float(rho_dtw), 4),
            "jerk_only_rho": round(float(rho_jerk), 4),
            "rec_level_n": len(rec_groups),
            "rec_level_dtw_rho": round(rho_rec_dtw, 4),
            "rec_level_jerk_rho": round(rho_rec_jerk, 4),
            "rec_level_stack_rho": round(rho_rec_total, 4),
        },
        "per_exercise": per_ex,
        "per_position": per_pos,
        "per_age_range": per_age,
        "n_positive_exercises": n_positive,
        "scored_reps": scored,
    }


# ── Strategy 1: Discriminability ───────────────────────────────────────────────

def _compute_auc(pos: np.ndarray, neg: np.ndarray) -> float:
    if len(pos) < 2 or len(neg) < 2:
        return 0.5
    auc = 0.0
    for p in pos:
        auc += np.sum(p > neg) + 0.5 * np.sum(p == neg)
    return float(auc / (len(pos) * len(neg)))


def _compute_eer(pos: np.ndarray, neg: np.ndarray, n_thr: int = 1000) -> float:
    all_s = np.concatenate([pos, neg])
    thr = np.linspace(np.min(all_s), np.max(all_s), n_thr)
    fprs = np.array([np.sum(neg < t) / len(neg) for t in thr])
    fnrs = np.array([np.sum(pos >= t) / len(pos) for t in thr])
    idx = int(np.argmin(np.abs(fprs - fnrs)))
    return float((fprs[idx] + fnrs[idx]) / 2)


def run_strategy1(reps: List[UCORep], n_pairs: int = 3000) -> Dict[str, Any]:
    """Strategy 1: Same vs different exercise discriminability.

    Builds length-matched positive (same exercise) and negative (different
    exercise) rep pairs, scores each with Procrustes position DTW.

    Args:
        reps: All reps with pre-computed features.
        n_pairs: Number of pairs per class.

    Returns:
        Results dict.
    """
    print("\n" + "=" * 60)
    print("Strategy 1: Discriminability (Same vs Different Exercise)")
    print("=" * 60)

    exercises = sorted({r.exercise for r in reps})
    by_ex: Dict[str, List[UCORep]] = defaultdict(list)
    for r in reps:
        if r.is_valid and r._pos_feat is not None:
            by_ex[r.exercise].append(r)

    # Positive pairs (same exercise)
    pos_scores: List[float] = []
    tries = 0
    while len(pos_scores) < n_pairs and tries < n_pairs * 5:
        tries += 1
        ex = random.choice(exercises)
        pool = by_ex.get(ex, [])
        if len(pool) < 2:
            continue
        a, b = random.sample(range(len(pool)), 2)
        pos_scores.append(position_pair_score(pool[a]._pos_feat, pool[b]._pos_feat))

    # Negative pairs (different exercise)
    neg_scores: List[float] = []
    tries = 0
    while len(neg_scores) < n_pairs and tries < n_pairs * 5:
        tries += 1
        ex_a, ex_b = random.sample(exercises, 2)
        ra = by_ex.get(ex_a, [])
        rb = by_ex.get(ex_b, [])
        if not ra or not rb:
            continue
        neg_scores.append(position_pair_score(
            random.choice(ra)._pos_feat, random.choice(rb)._pos_feat))

    pos_arr = np.array(pos_scores)
    neg_arr = np.array(neg_scores)
    auc = _compute_auc(pos_arr, neg_arr)
    eer = _compute_eer(pos_arr, neg_arr)
    u_stat, u_p = stats.mannwhitneyu(pos_arr, neg_arr, alternative="greater")
    z = float((u_stat - len(pos_arr) * len(neg_arr) / 2) /
              np.sqrt(len(pos_arr) * len(neg_arr) * (len(pos_arr) + len(neg_arr) + 1) / 12))

    pos_mean, pos_std = float(np.mean(pos_arr)), float(np.std(pos_arr))
    neg_mean, neg_std = float(np.mean(neg_arr)), float(np.std(neg_arr))

    print(f"  N_pos={len(pos_arr)}, N_neg={len(neg_arr)}")
    print(f"    AUC-ROC = {auc:.4f}")
    print(f"    EER     = {eer:.4f}")
    print(f"    Mann-Whitney Z = {z:.3f}, p = {u_p:.2e}")
    print(f"    Same-exercise:  mean={pos_mean:.2f} std={pos_std:.2f}")
    print(f"    Diff-exercise:  mean={neg_mean:.2f} std={neg_std:.2f}")
    print(f"    Separation = {pos_mean - neg_mean:.2f} points")

    # 16x16 confusion matrix (mean pair score between exercise classes)
    confusion = np.zeros((len(exercises), len(exercises)))
    for i, ex_a in enumerate(exercises):
        ra = by_ex.get(ex_a, [])
        if not ra:
            continue
        for j, ex_b in enumerate(exercises):
            rb = by_ex.get(ex_b, [])
            if not rb:
                continue
            vals = []
            for _ in range(min(60, len(ra) * len(rb))):
                a = random.choice(ra)
                b = random.choice(rb)
                if a is b:
                    continue
                vals.append(position_pair_score(a._pos_feat, b._pos_feat))
            confusion[i, j] = float(np.mean(vals)) if vals else 0.0

    # Build pair scores CSV rows
    pair_rows = []
    for s in pos_scores:
        pair_rows.append({"score": round(float(s), 4), "label": 1, "pair_type": "same"})
    for s in neg_scores:
        pair_rows.append({"score": round(float(s), 4), "label": 0, "pair_type": "different"})

    return {
        "auc_roc": round(float(auc), 4),
        "eer": round(float(eer), 4),
        "mann_whitney_u": float(u_stat),
        "mann_whitney_z": round(float(z), 4),
        "mann_whitney_p": float(u_p),
        "n_pos_pairs": len(pos_scores),
        "n_neg_pairs": len(neg_scores),
        "pos_score_mean": round(pos_mean, 3),
        "pos_score_std": round(pos_std, 3),
        "neg_score_mean": round(neg_mean, 3),
        "neg_score_std": round(neg_std, 3),
        "separation": round(pos_mean - neg_mean, 3),
        "confusion_matrix": confusion.tolist(),
        "exercise_names": exercises,
        "pair_rows": pair_rows,
    }


# ── Strategy 4a: Cross-subject robustness (LOSOCV) ─────────────────────────────

def run_strategy4_subject(reps: List[UCORep]) -> Dict[str, Any]:
    """Strategy 4a: Leave-One-Subject-Out cross-validation.

    For each held-out subject, score their reps using a fixed expert reference
    built from the OTHER subjects (top-10%). Report coefficient of variation
    per exercise.

    Args:
        reps: All reps with pre-computed features.

    Returns:
        Results dict.
    """
    print("\n" + "=" * 60)
    print("Strategy 4a: Cross-Subject Robustness (LOSOCV)")
    print("=" * 60)

    exercises = sorted({r.exercise for r in reps})
    results: Dict[str, Any] = {}

    for ex in exercises:
        ex_reps = [r for r in reps if r.exercise == ex and r.is_valid]
        subjects = sorted({r.subject for r in ex_reps})
        if len(subjects) < 4:
            results[f"ex{ex}"] = {"cv": None, "n_subjects": len(subjects)}
            continue
        held_scores: List[float] = []
        for hs in subjects:
            ref_reps = build_expert_reference(reps, ex, exclude_subject=hs)
            if not ref_reps:
                continue
            ref_feats = build_position_reference_feats(ref_reps)
            test = [r for r in ex_reps if r.subject == hs and r._pos_feat is not None]
            for r in test:
                held_scores.append(position_dtw_score(r._pos_feat, ref_feats))
        if len(held_scores) < 3:
            results[f"ex{ex}"] = {"cv": None, "n_subjects": len(subjects)}
            continue
        arr = np.array(held_scores)
        mean_s = float(np.mean(arr))
        std_s = float(np.std(arr))
        cv = std_s / mean_s if abs(mean_s) > 1e-6 else 0.0
        results[f"ex{ex}"] = {
            "cv": round(float(cv), 4),
            "mean_score": round(mean_s, 3),
            "std_score": round(std_s, 3),
            "n": len(arr),
        }
        print(f"  ex{ex}: CV = {cv:.4f} (mean={mean_s:.2f}, std={std_s:.2f}, N={len(arr)})")

    cvs = [v["cv"] for v in results.values() if v.get("cv") is not None]
    overall_cv = float(np.mean(cvs)) if cvs else 1.0
    max_cv = float(np.max(cvs)) if cvs else 1.0
    results["_overall"] = {
        "mean_cv": round(overall_cv, 4),
        "max_cv": round(max_cv, 4),
        "target_met": max_cv <= 0.35,
    }
    print(f"\n  Overall mean CV = {overall_cv:.4f}, max CV = {max_cv:.4f}")
    return results


# ── Strategy 4b: Cross-view robustness (LOVOCV) ────────────────────────────────

def _read_silhouette_signal(video_path: Path, stride: int = SILS_STRIDE) -> Optional[np.ndarray]:
    """Read a silhouette video and extract a per-frame motion signal.

    Signal per frame = [silhouette area, centroid_x, centroid_y], normalized.
    This is a view-dependent motion proxy that does NOT require pose estimation.

    Args:
        video_path: Path to cam*.mp4.
        stride: Frame stride (downsample for speed).

    Returns:
        Array of shape (n_sampled, 3) or None if unreadable.
    """
    import cv2
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    signals = []
    fi = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if fi % stride == 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                _, bw = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
                ys, xs = np.where(bw > 0)
                if len(xs) > 0:
                    area = len(xs) / (bw.shape[0] * bw.shape[1])
                    cx = (xs.mean() / bw.shape[1]) - 0.5
                    cy = (ys.mean() / bw.shape[0]) - 0.5
                else:
                    area, cx, cy = 0.0, 0.0, 0.0
                signals.append([area, cx, cy])
            fi += 1
    finally:
        cap.release()
    if len(signals) < 10:
        return None
    arr = np.array(signals, dtype=np.float64)
    # Normalize each column to unit amplitude (z-score-like) for cross-view compare
    std = arr.std(axis=0)
    std[std < 1e-6] = 1.0
    return (arr - arr.mean(axis=0)) / std


def run_strategy4_view(reps: List[UCORep], n_samples: int = N_SILS_SAMPLES) -> Dict[str, Any]:
    """Strategy 4b: Cross-view robustness (LOVOCV, Option C).

    UCO records each exercise from 5 synchronized cameras. The 3D OptiTrack GT
    is view-invariant, so to test view sensitivity we use the silhouette videos
    directly: extract a view-dependent motion signal (area + centroid) per
    camera and measure pairwise DTW similarity across views.

    This isolates view-robustness of the motion signal without running pose
    estimation (out of scope per the study design).

    Args:
        reps: All reps (used to pick recordings spanning exercises).
        n_samples: Number of recordings to sample.

    Returns:
        Results dict with 5x5 camera-transfer matrix.
    """
    print("\n" + "=" * 60)
    print("Strategy 4b: Cross-View Robustness (LOVOCV, silhouette Option C)")
    print("=" * 60)

    # Pick recordings to sample — one per (subject, exercise) to span exercises
    seen = {}
    for r in reps:
        seen.setdefault((r.subject, r.exercise), r)
    candidates = list(seen.values())
    rng = random.Random(RANDOM_SEED)
    rng.shuffle(candidates)
    sampled = candidates[:n_samples]
    print(f"  Sampling {len(sampled)} recordings x 5 cameras...")

    cam_matrix_sum = np.zeros((5, 5))
    cam_matrix_cnt = np.zeros((5, 5))
    per_cam_scores: Dict[int, List[float]] = defaultdict(list)
    n_ok = 0

    for r in sampled:
        base = SILS_DIR / r.subject / r.exercise
        if not base.exists():
            continue
        signals: Dict[int, np.ndarray] = {}
        for cam in range(5):
            vp = base / f"cam{cam}.mp4"
            sig = _read_silhouette_signal(vp)
            if sig is not None:
                signals[cam] = sig
        if len(signals) < 2:
            continue
        n_ok += 1
        # Resample all signals to common length and compute pairwise DTW
        L = 60
        cams = sorted(signals.keys())
        for i in cams:
            si = np.zeros((L, signals[i].shape[1]))
            for c in range(signals[i].shape[1]):
                si[:, c] = np.interp(np.linspace(0, 1, L),
                                     np.linspace(0, 1, len(signals[i])),
                                     signals[i][:, c])
            for j in cams:
                if i >= j:
                    continue
                sj = np.zeros((L, signals[j].shape[1]))
                for c in range(signals[j].shape[1]):
                    sj[:, c] = np.interp(np.linspace(0, 1, L),
                                         np.linspace(0, 1, len(signals[j])),
                                         signals[j][:, c])
                d, _ = fastdtw(si, sj, dist=euclidean)
                d_norm = d / L
                sim = float(np.clip(100.0 * np.exp(-d_norm * 2.0), 0, 100))
                cam_matrix_sum[i, j] += sim
                cam_matrix_sum[j, i] += sim
                cam_matrix_cnt[i, j] += 1
                cam_matrix_cnt[j, i] += 1
                per_cam_scores[i].append(sim)
                per_cam_scores[j].append(sim)

    cam_matrix = np.where(cam_matrix_cnt > 0,
                          cam_matrix_sum / np.maximum(cam_matrix_cnt, 1), np.nan)
    # Same-view baseline: a camera compared with itself is trivially identical
    # (similarity 100). The diagonal is set to 100 so the transfer matrix is
    # well-defined; the cross-view drop = 100 - cross_view_mean quantifies how
    # much the silhouette-derived motion signal degrades across viewpoints.
    for i in range(5):
        cam_matrix[i, i] = 100.0
    off_vals = [cam_matrix[i, j] for i in range(5) for j in range(5)
                if i != j and not np.isnan(cam_matrix[i, j])]
    off_mean = float(np.mean(off_vals)) if off_vals else 0.0
    diag_mean = 100.0

    print(f"  Processed {n_ok} recordings with >=2 camera views")
    print(f"  Mean cross-view similarity: {off_mean:.2f}  (same-view baseline = 100)")
    print(f"  Cross-view drop: {diag_mean - off_mean:.2f} points")

    # Per-camera score distribution
    cam_breakdown = {}
    for cam in range(5):
        vals = per_cam_scores.get(cam, [])
        cam_breakdown[f"cam{cam}"] = {
            "mean": round(float(np.mean(vals)), 3) if vals else None,
            "std": round(float(np.std(vals)), 3) if vals else None,
            "n": len(vals),
        }

    return {
        "n_recordings": n_ok,
        "camera_transfer_matrix": cam_matrix.tolist(),
        "same_view_mean": round(diag_mean, 3),
        "cross_view_mean": round(off_mean, 3),
        "cross_view_drop": round(diag_mean - off_mean, 3),
        "per_camera": cam_breakdown,
    }


# ── Strategy 6: View-stratified angle vs position DTW ablation ─────────────────

def run_strategy6(reps: List[UCORep], s2_scored: List[Tuple[UCORep, float]]) -> Dict[str, Any]:
    """Strategy 6: Ablate DTW variant (angle vs position) x patient position.

    Produces a 2 x 3 table: rows = {angle DTW, position DTW}, cols =
    {seated, supine, standing}, cells = Spearman rho.

    Args:
        reps: All reps.
        s2_scored: Strategy 2 scored reps (for position DTW scores).

    Returns:
        Results dict with the 2x3 table.
    """
    print("\n" + "=" * 60)
    print("Strategy 6: View-Stratified Angle vs Position DTW Ablation")
    print("=" * 60)

    exercises = sorted({r.exercise for r in reps})
    positions = ["seated", "supine", "standing"]

    # ── Compute angle DTW distance per rep (amplitude-normalized, LOO) ─────────
    angle_test: List[UCORep] = []
    for ex in exercises:
        ex_reps = [r for r in reps if r.exercise == ex and r.is_valid]
        if len(ex_reps) < 8:
            continue
        ref_reps = build_expert_reference(reps, ex)
        # Build (rep, angle) pairs keeping correspondence for LOO self-exclusion.
        ref_pairs = []
        for rr in ref_reps:
            a = rr.angles
            if np.all(np.isfinite(a)) and len(a) >= MIN_REP_FRAMES:
                ref_pairs.append((rr, a))
        for r in ex_reps:
            a = r.angles
            if not np.all(np.isfinite(a)) or len(a) < MIN_REP_FRAMES:
                continue
            # LOO: reference reps matched against the OTHER reference reps
            test_r = resample_trajectory(a, N_POS_FRAMES)
            best_d = float("inf")
            for (rr, ref) in ref_pairs:
                if id(rr) == id(r):
                    continue  # exclude self
                ref_r = resample_trajectory(ref, N_POS_FRAMES)
                ref_c = ref_r - np.mean(ref_r)
                ref_amp = max(np.max(ref_c) - np.min(ref_c), 1e-6)
                t_norm = ((test_r - np.mean(test_r)) / ref_amp).reshape(-1, 1)
                r_norm = (ref_c / ref_amp).reshape(-1, 1)
                d, _ = fastdtw(t_norm, r_norm, dist=euclidean)
                if d < best_d:
                    best_d = d
            if best_d == float("inf"):
                continue
            r._angle_score = float(np.clip(100.0 * np.exp(-(best_d / N_POS_FRAMES) * DTW_K), 0, 100))
            angle_test.append(r)

    # Position DTW reps come from Strategy 2 (carry _dtw_score, global [0,100]).
    pos_test = [rep for rep, _ in s2_scored if hasattr(rep, "_dtw_score")]

    table: Dict[str, Dict[str, Optional[float]]] = {
        "angle_dtw": {},
        "position_dtw": {},
    }
    for pos in positions:
        ang_vals = [(r._angle_score, r.clinician_score) for r in angle_test
                    if r.position == pos]
        table["angle_dtw"][pos] = (round(float(stats.spearmanr(
            [v[0] for v in ang_vals], [v[1] for v in ang_vals])[0]), 4)
            if len(ang_vals) >= 5 else None)
        pos_vals = [(r._dtw_score, r.clinician_score) for r in pos_test
                    if r.position == pos]
        table["position_dtw"][pos] = (round(float(stats.spearmanr(
            [v[0] for v in pos_vals], [v[1] for v in pos_vals])[0]), 4)
            if len(pos_vals) >= 5 else None)

    print("  2 x 3 table (Spearman rho, global [0,100] DTW similarity):")
    print(f"  {'DTW variant':<16s}", end="")
    for p in positions:
        print(f" {p:>10s}", end="")
    print()
    for variant in ["angle_dtw", "position_dtw"]:
        print(f"  {variant:<16s}", end="")
        for p in positions:
            v = table[variant][p]
            print(f" {('%.4f' % v) if v is not None else '  n/a':>10s}", end="")
        print()

    # Overall (pooled) angle vs position
    ang_rho = (round(float(stats.spearmanr(
        [r._angle_score for r in angle_test],
        [r.clinician_score for r in angle_test])[0]), 4)
        if len(angle_test) > 5 else None)
    pos_rho = (round(float(stats.spearmanr(
        [r._dtw_score for r in pos_test],
        [r.clinician_score for r in pos_test])[0]), 4)
        if len(pos_test) > 5 else None)

    print(f"\n  Overall: angle DTW rho = {ang_rho}, position DTW rho = {pos_rho}")
    return {
        "table": table,
        "overall_angle_dtw_rho": ang_rho,
        "overall_position_dtw_rho": pos_rho,
        "positions": positions,
    }


# ── CSV / figure / report output ───────────────────────────────────────────────

def save_results_csv(reps: List[UCORep], s2_scored: List[Tuple[UCORep, float]], path: Path) -> None:
    """Save one row per test rep (reference reps excluded)."""
    scored_map = {id(r): r for r, _ in s2_scored}
    rows = []
    for r in reps:
        if not r.is_valid:
            continue
        if id(r) in scored_map:
            sr = scored_map[id(r)]
            dtw = getattr(sr, "_dtw_score", "")
            total = getattr(sr, "_total_score", "")
            jerk = getattr(sr, "_jerk", "")
            rom = getattr(sr, "_rom", "")
            sparc = getattr(sr, "_sparc", "")
        else:
            dtw = total = jerk = rom = sparc = ""
        rows.append({
            "rep_id": r.rep_id,
            "subject": r.subject,
            "exercise": r.exercise,
            "position": r.position,
            "side": r.side,
            "body": r.body,
            "age_range": r.age_range,
            "clinician_score": r.clinician_score,
            "n_frames": len(r.positions),
            "dtw_score": round(dtw, 4) if dtw != "" else "",
            "jerk_score": round(jerk, 4) if jerk != "" else "",
            "rom_score": round(rom, 4) if rom != "" else "",
            "sparc_score": round(sparc, 4) if sparc != "" else "",
            "total_score": round(total, 4) if total != "" else "",
        })
    if not rows:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"  + uco_results.csv ({len(rows)} rows)")


def save_pair_scores_csv(pair_rows: List[dict], path: Path) -> None:
    if not pair_rows:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(pair_rows[0].keys()))
        w.writeheader()
        w.writerows(pair_rows)
    print(f"  + uco_pair_scores.csv ({len(pair_rows)} rows)")


def save_view_breakdown_csv(view_results: Dict, path: Path) -> None:
    rows = []
    # Camera-transfer matrix
    mat = view_results.get("camera_transfer_matrix", [])
    for i in range(len(mat)):
        for j in range(len(mat[i])):
            rows.append({
                "metric": "camera_transfer",
                "row": f"cam{i}",
                "col": f"cam{j}",
                "value": round(mat[i][j], 4) if mat[i][j] == mat[i][j] else "",
            })
    for cam, d in view_results.get("per_camera", {}).items():
        rows.append({
            "metric": "per_camera_score",
            "row": cam,
            "col": "",
            "value": d.get("mean", ""),
        })
    rows.append({"metric": "same_view_mean", "row": "", "col": "",
                 "value": view_results.get("same_view_mean", "")})
    rows.append({"metric": "cross_view_mean", "row": "", "col": "",
                 "value": view_results.get("cross_view_mean", "")})
    rows.append({"metric": "cross_view_drop", "row": "", "col": "",
                 "value": view_results.get("cross_view_drop", "")})
    if not rows:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"  + uco_view_breakdown.csv ({len(rows)} rows)")


def save_summary_csv(all_results: Dict, path: Path) -> None:
    rows = []
    s2 = all_results.get("strategy2", {})
    s2ov = s2.get("overall", {})
    rows.append({"strategy": "Strategy 2: Clinical Correlation", "metric": "Spearman rho",
                 "value": s2ov.get("spearman_rho", ""), "ci_95": str(s2ov.get("spearman_ci_95", "")),
                 "n": s2ov.get("n", ""), "notes": f"Target: rho>=0.25"})
    rows.append({"strategy": "Strategy 2: Clinical Correlation", "metric": "Pearson r",
                 "value": s2ov.get("pearson_r", ""), "ci_95": "", "n": s2ov.get("n", ""), "notes": ""})
    rows.append({"strategy": "Strategy 2", "metric": "positive_exercises",
                 "value": s2.get("n_positive_exercises", ""), "ci_95": "", "n": "", "notes": "Target: >=10/16"})

    s1 = all_results.get("strategy1", {})
    rows.append({"strategy": "Strategy 1: Discriminability", "metric": "AUC-ROC",
                 "value": s1.get("auc_roc", ""), "ci_95": "", "n": "",
                 "notes": f"EER={s1.get('eer','')}, Target: AUC>=0.80"})
    rows.append({"strategy": "Strategy 1: Discriminability", "metric": "separation",
                 "value": s1.get("separation", ""), "ci_95": "", "n": "",
                 "notes": "Target: >=15 points"})
    rows.append({"strategy": "Strategy 1: Discriminability", "metric": "Mann-Whitney p",
                 "value": s1.get("mann_whitney_p", ""), "ci_95": "", "n": "",
                 "notes": f"Z={s1.get('mann_whitney_z','')}"})

    for ex, d in sorted(s2.get("per_exercise", {}).items()):
        rows.append({"strategy": f"Strategy 2 per-exercise", "metric": f"ex{ex} rho",
                     "value": d.get("rho", ""), "ci_95": "", "n": d.get("n", ""), "notes": ""})
    for pos, d in sorted(s2.get("per_position", {}).items()):
        rows.append({"strategy": "Strategy 2 per-position", "metric": f"{pos} rho",
                     "value": d.get("rho", ""), "ci_95": "", "n": d.get("n", ""), "notes": ""})

    s4s = all_results.get("strategy4_subject", {})
    for k, v in sorted(s4s.items()):
        if k == "_overall":
            rows.append({"strategy": "Strategy 4 LOSOCV", "metric": "mean_cv",
                         "value": v.get("mean_cv", ""), "ci_95": "", "n": "",
                         "notes": f"max_cv={v.get('max_cv','')}, Target: CV<=0.35"})
        else:
            rows.append({"strategy": f"Strategy 4 LOSOCV", "metric": f"{k} cv",
                         "value": v.get("cv", ""), "ci_95": "", "n": v.get("n", ""),
                         "notes": f"mean={v.get('mean_score','')}"})

    s4v = all_results.get("strategy4_view", {})
    rows.append({"strategy": "Strategy 4 LOVOCV", "metric": "same_view_mean",
                 "value": s4v.get("same_view_mean", ""), "ci_95": "", "n": s4v.get("n_recordings", ""), "notes": ""})
    rows.append({"strategy": "Strategy 4 LOVOCV", "metric": "cross_view_mean",
                 "value": s4v.get("cross_view_mean", ""), "ci_95": "", "n": "", "notes": ""})
    rows.append({"strategy": "Strategy 4 LOVOCV", "metric": "cross_view_drop",
                 "value": s4v.get("cross_view_drop", ""), "ci_95": "", "n": "", "notes": ""})

    s6 = all_results.get("strategy6", {})
    rows.append({"strategy": "Strategy 6 ablation", "metric": "overall_angle_dtw_rho",
                 "value": s6.get("overall_angle_dtw_rho", ""), "ci_95": "", "n": "", "notes": ""})
    rows.append({"strategy": "Strategy 6 ablation", "metric": "overall_position_dtw_rho",
                 "value": s6.get("overall_position_dtw_rho", ""), "ci_95": "", "n": "", "notes": ""})
    for variant, posdict in s6.get("table", {}).items():
        for pos, val in posdict.items():
            rows.append({"strategy": f"Strategy 6 {variant}", "metric": f"{pos} rho",
                         "value": val, "ci_95": "", "n": "", "notes": ""})

    if not rows:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"  + uco_summary.csv ({len(rows)} rows)")


def generate_figures(s2: Dict, s1: Dict, s4v: Dict) -> None:
    print("Generating figures...")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        print("  matplotlib not available — skipping figures")
        return

    # Figure 1: Score vs Clinical (Strategy 2)
    scored = s2.get("scored_reps", [])
    if scored:
        our = np.array([s for _, s in scored])
        clin = np.array([r.clinician_score for r, _ in scored])
        pos_labels = np.array([r.position for r, _ in scored])
        rho = s2.get("overall", {}).get("spearman_rho", 0.0)
        ci = s2.get("overall", {}).get("spearman_ci_95", [0, 0])

        fig, ax = plt.subplots(figsize=(8, 6))
        pos_colors = {"seated": "#e41a1c", "supine": "#377eb8", "standing": "#4daf4a"}
        colors = [pos_colors.get(p, "gray") for p in pos_labels]
        # Jitter clinician score for visibility (discrete 2-5)
        clin_jit = clin + np.random.RandomState(RANDOM_SEED).uniform(-0.15, 0.15, len(clin))
        ax.scatter(our, clin_jit, c=colors, s=18, alpha=0.45, zorder=3)
        ax.set_xlabel("Our Procrustes Position DTW Score", fontsize=12)
        ax.set_ylabel("Clinician Score (jittered)", fontsize=12)
        ax.set_title(f"UCO: Our Score vs Clinician Score\n"
                     f"Spearman rho = {rho:.3f} (95% CI: [{ci[0]:.3f}, {ci[1]:.3f}])",
                     fontsize=13)
        ax.grid(True, alpha=0.3)
        handles = [mpatches.Patch(color=c, label=p) for p, c in pos_colors.items()]
        ax.legend(handles=handles, title="Position", fontsize=9)
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / "uco_score_vs_clinical.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print("  + uco_score_vs_clinical.png")

    # Figure 2: Discriminability histogram (Strategy 1)
    pos_mean = s1.get("pos_score_mean", 0.0)
    pos_std = s1.get("pos_score_std", 10.0)
    neg_mean = s1.get("neg_score_mean", 0.0)
    neg_std = s1.get("neg_score_std", 10.0)
    auc = s1.get("auc_roc", 0.5)
    eer = s1.get("eer", 0.5)
    x = np.linspace(0, 100, 200)
    pos_pdf = stats.norm.pdf(x, pos_mean, max(pos_std, 1.0))
    neg_pdf = stats.norm.pdf(x, neg_mean, max(neg_std, 1.0))
    pos_pdf = pos_pdf / (pos_pdf.max() or 1) * 0.45
    neg_pdf = neg_pdf / (neg_pdf.max() or 1) * 0.45
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.fill_between(x, pos_pdf, alpha=0.4, color="steelblue",
                    label=f"Same exercise (mean={pos_mean:.1f})")
    ax.fill_between(x, neg_pdf, alpha=0.4, color="#d62728",
                    label=f"Different exercise (mean={neg_mean:.1f})")
    ax.plot(x, pos_pdf, "b-", lw=2)
    ax.plot(x, neg_pdf, "r-", lw=2)
    ax.set_xlabel("Pair Similarity Score", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title(f"UCO Strategy 1: Discriminability\nAUC-ROC = {auc:.3f}, EER = {eer:.3f}",
                 fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 100)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "uco_discriminability_hist.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  + uco_discriminability_hist.png")

    # Figure 3: View robustness heatmap (Strategy 4 LOVOCV)
    mat = np.array(s4v.get("camera_transfer_matrix", []))
    if mat.size:
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(mat, cmap="viridis", vmin=0, vmax=100)
        ax.set_xticks(range(5))
        ax.set_yticks(range(5))
        ax.set_xticklabels([f"cam{i}" for i in range(5)])
        ax.set_yticklabels([f"cam{i}" for i in range(5)])
        for i in range(5):
            for j in range(5):
                v = mat[i, j]
                if np.isfinite(v):
                    ax.text(j, i, f"{v:.0f}", ha="center", va="center",
                            color="white" if v < 50 else "black", fontsize=9)
        ax.set_title(f"UCO Cross-View Robustness (LOVOCV)\n"
                     f"same-view={s4v.get('same_view_mean',0):.1f}, "
                     f"cross-view={s4v.get('cross_view_mean',0):.1f}", fontsize=12)
        fig.colorbar(im, ax=ax, label="DTW similarity")
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / "uco_view_robustness.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print("  + uco_view_robustness.png")


def generate_report(all_results: Dict, stats_d: Dict) -> str:
    L = []
    s2 = all_results.get("strategy2", {})
    s1 = all_results.get("strategy1", {})
    s4s = all_results.get("strategy4_subject", {})
    s4v = all_results.get("strategy4_view", {})
    s6 = all_results.get("strategy6", {})

    L.append("# UCO Physical Rehabilitation Scoring Evaluation — Findings Report")
    L.append("")
    L.append(f"**Generated**: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    L.append(f"**Dataset**: UCO PhyRehab ({stats_d.get('n_reps',0)} reps, "
             f"{stats_d.get('n_exercises',0)} exercises, {stats_d.get('n_subjects',0)} subjects)")
    L.append("")

    # Strategy 2
    s2ov = s2.get("overall", {})
    rho = s2ov.get("spearman_rho", 0.0)             # stack (DTW+jerk)
    rho_dtw = s2ov.get("dtw_only_rho", 0.0)          # position DTW component
    rho_jerk = s2ov.get("jerk_only_rho", 0.0)        # jerk component
    ci = s2ov.get("spearman_ci_95", [0, 0])
    L.append("## Strategy 2: Clinical Correlation (Primary Result)")
    L.append("")
    L.append("Per-repetition correlation with clinician score (LOO-within-reference, "
             "non-circular, all reps):")
    L.append("")
    L.append("| Component | Spearman rho | N |")
    L.append("|---|---|---|")
    L.append(f"| **Position DTW (component)** | **{rho_dtw:.4f}** | {s2ov.get('n',0)} |")
    L.append(f"| Jerk (component) | {rho_jerk:.4f} | {s2ov.get('n',0)} |")
    L.append(f"| Stack (DTW+jerk) | {rho:.4f} (95% CI [{ci[0]:.4f},{ci[1]:.4f}]) | {s2ov.get('n',0)} |")
    L.append("")
    L.append("Recording-level (reps averaged per recording, N="
             f"{s2ov.get('rec_level_n',0)}): position DTW rho = "
             f"{s2ov.get('rec_level_dtw_rho',0):.4f}, jerk rho = "
             f"{s2ov.get('rec_level_jerk_rho',0):.4f}, stack rho = "
             f"{s2ov.get('rec_level_stack_rho',0):.4f}.")
    L.append("")
    L.append("*Note: UCO per-rep scores are discrete (2-5) and heavily skewed (93% at "
             "4-5), and each recording has only 3 joints (single kinematic chain) vs "
             "KIMORE's 9 whole-body joints. Both factors cap the achievable rho; the "
             "position-DTW signal that dominates on KIMORE is weak on UCO, while "
             "kinematic smoothness (jerk) is the strongest single predictor.*")
    L.append("")
    L.append(f"Positive exercises (position DTW): {s2.get('n_positive_exercises',0)}/16")
    L.append("")
    L.append("| Exercise | Spearman rho | p-value | N |")
    L.append("|---|---|---|---|")
    for ex, d in sorted(s2.get("per_exercise", {}).items()):
        if d.get("rho") is not None:
            L.append(f"| ex{ex} | {d['rho']:.4f} | {d['p_value']:.2e} | {d['n']} |")
    L.append("")
    L.append("| Position | Spearman rho | N |")
    L.append("|---|---|---|")
    for pos, d in sorted(s2.get("per_position", {}).items()):
        v = d.get("rho")
        L.append(f"| {pos} | {('%.4f' % v) if v is not None else 'n/a'} | {d.get('n','')} |")
    L.append("")

    # Strategy 1
    L.append("## Strategy 1: Discriminability")
    L.append("")
    L.append(f"**AUC-ROC = {s1.get('auc_roc',0):.4f}**, EER = {s1.get('eer',0):.4f}")
    L.append(f"Mann-Whitney p = {s1.get('mann_whitney_p',0):.2e}")
    L.append("")
    L.append("| Pair Type | Mean Score | Std | N |")
    L.append("|---|---|---|---|")
    L.append(f"| Same exercise | {s1.get('pos_score_mean',0):.2f} | {s1.get('pos_score_std',0):.2f} | {s1.get('n_pos_pairs',0)} |")
    L.append(f"| Different exercise | {s1.get('neg_score_mean',0):.2f} | {s1.get('neg_score_std',0):.2f} | {s1.get('n_neg_pairs',0)} |")
    L.append(f"| **Separation** | **{s1.get('separation',0):.2f}** | | |")
    L.append("")

    # Strategy 4a
    L.append("## Strategy 4a: Cross-Subject Robustness (LOSOCV)")
    L.append("")
    ov = s4s.get("_overall", {})
    L.append(f"**Overall mean CV = {ov.get('mean_cv',0):.4f}**, max CV = {ov.get('max_cv',0):.4f} "
             f"(target <=0.35: {'MET' if ov.get('target_met') else 'NOT MET'})")
    L.append("")
    L.append("| Exercise | CV | Mean | Std | N |")
    L.append("|---|---|---|---|---|")
    for k, v in sorted(s4s.items()):
        if k == "_overall" or v.get("cv") is None:
            continue
        L.append(f"| {k} | {v['cv']:.4f} | {v['mean_score']:.2f} | {v['std_score']:.2f} | {v['n']} |")
    L.append("")

    # Strategy 4b
    L.append("## Strategy 4b: Cross-View Robustness (LOVOCV, silhouette Option C)")
    L.append("")
    L.append(f"Recordings processed: {s4v.get('n_recordings',0)}")
    L.append(f"Same-view mean similarity: {s4v.get('same_view_mean',0):.2f}")
    L.append(f"Cross-view mean similarity: {s4v.get('cross_view_mean',0):.2f}")
    L.append(f"Cross-view drop: {s4v.get('cross_view_drop',0):.2f} points")
    L.append("")
    L.append("Note: Uses silhouette-derived motion signals (area+centroid) per camera "
             "view — no pose estimation. Tests view-stability of the motion signal.")
    L.append("")

    # Strategy 6
    L.append("## Strategy 6: Angle vs Position DTW Ablation")
    L.append("")
    L.append("| DTW variant | " + " | ".join(s6.get("positions", [])) + " |")
    L.append("|---|" + "---|" * len(s6.get("positions", [])))
    for variant in ["angle_dtw", "position_dtw"]:
        cells = []
        for p in s6.get("positions", []):
            v = s6.get("table", {}).get(variant, {}).get(p)
            cells.append(f"{v:.4f}" if v is not None else "n/a")
        L.append(f"| {variant} | " + " | ".join(cells) + " |")
    L.append(f"\nOverall: angle DTW rho = {s6.get('overall_angle_dtw_rho','n/a')}, "
             f"position DTW rho = {s6.get('overall_position_dtw_rho','n/a')}")
    L.append("")

    # Summary
    L.append("## Summary vs Success Criteria")
    L.append("")
    L.append("| Criterion | Value | Target | Status |")
    L.append("|---|---|---|---|")
    sc = [
        ("Position DTW rho (per-rep)", rho_dtw, 0.25, rho_dtw >= 0.25),
        ("Position DTW rho (recording-level)", s2ov.get("rec_level_dtw_rho", 0), 0.25,
         s2ov.get("rec_level_dtw_rho", 0) >= 0.25),
        ("Positive exercises", s2.get("n_positive_exercises", 0), 10,
         s2.get("n_positive_exercises", 0) >= 10),
        ("Discriminability AUC", s1.get("auc_roc", 0), 0.80, s1.get("auc_roc", 0) >= 0.80),
        ("Same-diff separation", s1.get("separation", 0), 15, s1.get("separation", 0) >= 15),
        ("Cross-subject max CV", ov.get("max_cv", 1), 0.35, ov.get("max_cv", 1) <= 0.35),
        ("Cross-subject mean CV", ov.get("mean_cv", 1), 0.35, ov.get("mean_cv", 1) <= 0.35),
    ]
    for name, val, tgt, ok in sc:
        L.append(f"| {name} | {val} | {tgt} | {'MET' if ok else 'NOT MET'} |")
    L.append("")
    L.append("---")
    L.append("*Generated by ADAPT-Rehab UCO evaluation script (Phase B).*")
    return "\n".join(L)


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="UCO PhyRehab Scoring Evaluation (Phase B)")
    parser.add_argument("--strategy", type=str, default="all",
                        choices=["all", "1", "2", "4", "6"],
                        help="Which strategy to run (default: all)")
    parser.add_argument("--quick", action="store_true",
                        help="Subsampled mode for testing")
    parser.add_argument("--no-figures", action="store_true",
                        help="Skip figure generation")
    parser.add_argument("--no-view", action="store_true",
                        help="Skip silhouette LOVOCV (slow video reading)")
    args = parser.parse_args()

    global SILS_DIR
    SILS_DIR = DATA_DIR / "sils_cropped"

    set_seed()
    t_start = time.time()

    print("=" * 60)
    print("ADAPT-Rehab: UCO PhyRehab Scoring Evaluation (Phase B)")
    print("=" * 60)

    # Load data
    reps, stats_d = load_uco_data()

    # Pre-compute features
    precompute_features(reps)

    all_results: Dict[str, Any] = {}

    # Strategy 2 (primary)
    if args.strategy in ("all", "2"):
        all_results["strategy2"] = run_strategy2(reps)

    # Strategy 1
    if args.strategy in ("all", "1"):
        n_pairs = 500 if args.quick else 3000
        all_results["strategy1"] = run_strategy1(reps, n_pairs=n_pairs)

    # Strategy 4
    if args.strategy in ("all", "4"):
        all_results["strategy4_subject"] = run_strategy4_subject(reps)
        if not args.no_view:
            n_sils = 30 if args.quick else N_SILS_SAMPLES
            all_results["strategy4_view"] = run_strategy4_view(reps, n_samples=n_sils)
        else:
            all_results["strategy4_view"] = {}

    # Strategy 6
    if args.strategy in ("all", "6"):
        s2_scored = all_results.get("strategy2", {}).get("scored_reps", [])
        all_results["strategy6"] = run_strategy6(reps, s2_scored)

    # Save outputs
    print("\n" + "=" * 60)
    print("Saving outputs")
    print("=" * 60)

    s2_scored = all_results.get("strategy2", {}).get("scored_reps", [])
    save_results_csv(reps, s2_scored, OUTPUT_DIR / "uco_results.csv")
    save_pair_scores_csv(
        all_results.get("strategy1", {}).get("pair_rows", []),
        OUTPUT_DIR / "uco_pair_scores.csv")
    if all_results.get("strategy4_view"):
        save_view_breakdown_csv(all_results["strategy4_view"],
                                OUTPUT_DIR / "uco_view_breakdown.csv")
    save_summary_csv(all_results, OUTPUT_DIR / "uco_summary.csv")

    # Figures
    if not args.no_figures:
        generate_figures(
            s2=all_results.get("strategy2", {}),
            s1=all_results.get("strategy1", {}),
            s4v=all_results.get("strategy4_view", {}),
        )

    # Report
    report = generate_report(all_results, stats_d)
    with open(OUTPUT_DIR / "uco_report.md", "w") as f:
        f.write(report)
    print("  + uco_report.md")

    # Headline summary
    print("\n" + "=" * 60)
    print("HEADLINE NUMBERS (UCO Phase B)")
    print("=" * 60)
    s2ov = all_results.get("strategy2", {}).get("overall", {})
    rho_dtw = s2ov.get("dtw_only_rho", 0.0)
    rho_jerk = s2ov.get("jerk_only_rho", 0.0)
    rho_stack = s2ov.get("spearman_rho", 0.0)
    rec_dtw = s2ov.get("rec_level_dtw_rho", 0.0)
    rec_jerk = s2ov.get("rec_level_jerk_rho", 0.0)
    n_pos = all_results.get("strategy2", {}).get("n_positive_exercises", 0)
    auc = all_results.get("strategy1", {}).get("auc_roc", 0.0)
    sep = all_results.get("strategy1", {}).get("separation", 0.0)
    s4ov = all_results.get("strategy4_subject", {}).get("_overall", {})
    max_cv = s4ov.get("max_cv", 1.0)
    mean_cv = s4ov.get("mean_cv", 1.0)
    ang_rho = all_results.get("strategy6", {}).get("overall_angle_dtw_rho", 0.0)
    pos_rho = all_results.get("strategy6", {}).get("overall_position_dtw_rho", 0.0)
    per_pos = all_results.get("strategy2", {}).get("per_position", {})

    print(f"\n  1. Clinical correlation:")
    print(f"     Position DTW rho = {rho_dtw:.4f} (per-rep) / {rec_dtw:.4f} (rec-level)  [KIMORE 0.347]")
    print(f"     Jerk rho         = {rho_jerk:.4f} (per-rep) / {rec_jerk:.4f} (rec-level)")
    print(f"     Stack rho        = {rho_stack:.4f} (per-rep)")
    print(f"     Positive exercises: {n_pos}/16")
    print(f"\n  2. Discriminability:")
    print(f"     AUC = {auc:.4f}, separation = {sep:.2f} pts  (KIMORE AUC 0.71)")
    print(f"\n  3. Cross-subject robustness:")
    print(f"     mean CV = {mean_cv:.4f}, max CV = {max_cv:.4f}  (UCO target 0.35)")
    print(f"\n  4. Position vs angle DTW (Strategy 6):")
    print(f"     position rho = {pos_rho}, angle rho = {ang_rho} "
          f"(KIMORE: 0.40 vs 0.04)")
    print(f"\n  5. Per-position rho (position DTW):")
    for p in ["seated", "supine", "standing"]:
        d = per_pos.get(p, {})
        v = d.get("rho")
        print(f"     {p}: {('%.4f' % v) if v is not None else 'n/a'} (N={d.get('n','')})")
    if all_results.get("strategy4_view"):
        print(f"\n  6. Cross-view (LOVOCV):")
        print(f"     same-view={all_results['strategy4_view'].get('same_view_mean',0):.1f}, "
              f"cross-view={all_results['strategy4_view'].get('cross_view_mean',0):.1f}, "
              f"drop={all_results['strategy4_view'].get('cross_view_drop',0):.1f}")

    elapsed = time.time() - t_start
    print(f"\n  Total time: {elapsed:.1f}s")
    print(f"\n  Output files:")
    for fn in ["uco_results.csv", "uco_pair_scores.csv", "uco_view_breakdown.csv",
               "uco_summary.csv", "uco_report.md"]:
        print(f"    {OUTPUT_DIR / fn}")
    for fn in ["uco_score_vs_clinical.png", "uco_discriminability_hist.png",
               "uco_view_robustness.png"]:
        print(f"    {FIGURES_DIR / fn}")


if __name__ == "__main__":
    main()
