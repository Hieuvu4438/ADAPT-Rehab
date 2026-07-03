"""Probe v2: Procrustes-aligned DTW + correct exercise-joint mapping.

Key fixes vs v1:
1. Procrustes alignment removes subject body-proportion differences
2. Correct KIMORE exercise descriptions (from Capecci 2020):
   ex1: shoulder abduction (arms out to sides)
   ex2: shoulder flexion (arms forward/up)
   ex3: lateral leg raises (hip abduction)
   ex4: squat (knee + hip flexion)
   ex5: trunk lateral tilt
3. Multi-joint DTW on all 9 angle series (fixed)
4. Multiple DTW variants: position-based, angle-based, combined
"""
import sys
import pickle
import numpy as np
from pathlib import Path
from scipy import stats
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
KIMORE_PATH = ROOT / "data" / "KIMORE" / "kimore_exercise_dataset.pkl"

with open(KIMORE_PATH, "rb") as f:
    raw = pickle.load(f)

KINECT_IDX = {
    "spinebase": 0, "spinemid": 1, "neck": 2, "head": 3,
    "shoulderleft": 4, "elbowleft": 5, "wristleft": 6, "handleft": 7,
    "shoulderright": 8, "elbowright": 9, "wristright": 10, "handright": 11,
    "hipleft": 12, "kneeleft": 13, "ankleleft": 14, "footleft": 15,
    "hipright": 16, "kneeright": 17, "ankleright": 18, "footright": 19,
    "spineshoulder": 20,
}

# KIMORE exercise → primary joints (based on Capecci 2020 paper)
EXERCISE_PRIMARY = {
    "ex1": [(0, 8, 9), (0, 4, 5)],     # shoulder abduction: R+L shoulder
    "ex2": [(0, 8, 9), (0, 4, 5)],     # shoulder flexion: R+L shoulder
    "ex3": [(0, 12, 13), (0, 16, 17)], # hip abduction: R+L hip
    "ex4": [(12, 13, 14), (16, 17, 18), (0, 12, 13), (0, 16, 17)],  # squat: knees + hips
    "ex5": [(0, 1, 2), (0, 20, 2)],    # trunk lateral tilt: spine angles
}

JOINT_SPECS_9 = {
    "l_shoulder":  (0, 4, 5),
    "r_shoulder":  (0, 8, 9),
    "l_knee":      (12, 13, 14),
    "r_knee":      (16, 17, 18),
    "l_hip":       (0, 12, 13),
    "r_hip":       (0, 16, 17),
    "l_elbow":     (4, 5, 6),
    "r_elbow":     (8, 9, 10),
    "spine":       (0, 1, 2),
}


def load_exercise(df):
    out = []
    for idx in range(len(df)):
        row = df.iloc[idx]
        cts = float(row["cTS"])
        frames = []
        for col in df.columns:
            if col == "cTS":
                continue
            jdata = row[col]
            if isinstance(jdata, np.ndarray) and jdata.ndim == 2 and jdata.shape[1] >= 7:
                frames.append(jdata[:, 4:7])
        if not frames:
            continue
        kp = np.stack(frames, axis=1)
        out.append((kp, cts))
    return out


def procrustes_align(kp, ref):
    """Z-score normalize per joint per axis, then center.
    Removes subject-specific scale/offset while preserving motion."""
    # Center each frame on spinebase
    centered = kp - kp[:, 0:1, :]
    # Scale normalize by shoulder-width
    sw = np.linalg.norm(centered[:, 8] - centered[:, 4], axis=1).mean()
    if sw > 1e-6:
        centered = centered / sw
    return centered


def angle_series(kp, spec):
    p, v, d = spec
    v1 = kp[:, p] - kp[:, v]
    v2 = kp[:, d] - kp[:, v]
    n1 = np.linalg.norm(v1, axis=1, keepdims=True)
    n2 = np.linalg.norm(v2, axis=1, keepdims=True)
    mask = (n1[:, 0] > 1e-10) & (n2[:, 0] > 1e-10)
    cos = np.zeros(kp.shape[0])
    cos[mask] = np.sum(v1[mask] / n1[mask] * v2[mask] / n2[mask], axis=1)
    return np.degrees(np.arccos(np.clip(cos, -1, 1)))


def resample(arr, n=100):
    if len(arr) == n:
        return arr
    if len(arr) < 2:
        return np.full(n, arr[0] if len(arr) == 1 else 0.0)
    return np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(arr)), arr)


def run_dtw_eval(ex_name, feature_fn, label, n_template=5, n_resample=100):
    """Run DTW-to-template evaluation for one feature type."""
    df = raw[ex_name]
    data = load_exercise(df)
    cts_arr = np.array([d[1] for d in data])

    # Compute features
    feats = []
    for kp, _ in data:
        f = feature_fn(kp, ex_name)
        feats.append(f)

    # Build template from top-N cTS
    top_idx = np.argsort(-cts_arr)[:n_template]
    templates = [feats[i] for i in top_idx]

    # DTW to nearest template
    distances = []
    for i, f in enumerate(feats):
        best = float('inf')
        for t in templates:
            if f.ndim == 1:
                d, _ = fastdtw(f.reshape(-1, 1), t.reshape(-1, 1), dist=euclidean)
            else:
                d, _ = fastdtw(f, t, dist=euclidean)
            best = min(best, d)
        distances.append(best)
    distances = np.array(distances)

    rho, p = stats.spearmanr(-distances, cts_arr)
    return rho, p, len(cts_arr)


# Feature functions
def feat_primary_angle(kp, ex):
    specs = EXERCISE_PRIMARY.get(ex, [(0, 8, 9)])
    angles = [resample(angle_series(kp, s), 100) for s in specs]
    return np.mean(angles, axis=0)


def feat_all_9_angles(kp, ex):
    angles = [resample(angle_series(kp, s), 50) for s in JOINT_SPECS_9.values()]
    return np.stack(angles, axis=1)  # (50, 9)


def feat_procrustes_pos(kp, ex):
    """Procrustes-aligned position trajectory of all 9 main joints."""
    main_joints = [0, 1, 2, 8, 4, 12, 16, 13, 17]  # spine, head, shoulders, hips, knees
    kp9 = kp[:, main_joints, :]
    # Procrustes: center on spinebase, scale by shoulder width
    centered = kp9 - kp9[:, 0:1, :]
    sw = np.linalg.norm(kp[:, 8] - kp[:, 4], axis=1).mean()
    if sw > 1e-6:
        centered = centered / sw
    # Subsample
    n_frames = centered.shape[0]
    if n_frames > 50:
        idx = np.linspace(0, n_frames - 1, 50).astype(int)
        centered = centered[idx]
    return centered.reshape(-1, 9 * 3)  # (50, 27)


print("=" * 80)
print("KIMORE DTW Evaluation: 3 feature types × 5 exercises")
print("=" * 80)
print()

results = {}
for feat_name, feat_fn, label in [
    ("primary_angle", feat_primary_angle, "Primary joint angle DTW"),
    ("all_9_angles", feat_all_9_angles, "9-joint angle DTW"),
    ("procrustes_pos", feat_procrustes_pos, "Procrustes 9-joint position DTW"),
]:
    print(f"--- {label} ---")
    rhos = []
    for ex in ["ex1", "ex2", "ex3", "ex4", "ex5"]:
        rho, p, n = run_dtw_eval(ex, feat_fn, label)
        rhos.append(rho)
        sig = "*" if p < 0.05 else ""
        print(f"  {ex}: ρ = {rho:+.4f} (p={p:.4f}){sig}  N={n}")
    mean_rho = np.mean(rhos)
    print(f"  MEAN ρ = {mean_rho:+.4f}")
    results[feat_name] = (mean_rho, rhos)
    print()

# Summary
print("=" * 80)
print("SUMMARY: Best feature per exercise")
print("=" * 80)
for i, ex in enumerate(["ex1", "ex2", "ex3", "ex4", "ex5"]):
    best_feat = max(results.keys(), key=lambda k: results[k][1][i])
    best_rho = results[best_feat][1][i]
    print(f"  {ex}: best = {best_feat} (ρ={best_rho:+.4f})")
print()
print("Overall best feature:", max(results.keys(), key=lambda k: results[k][0]))
print(f"  Mean ρ = {results[max(results.keys(), key=lambda k: results[k][0])][0]:+.4f}")
