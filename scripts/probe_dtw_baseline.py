"""Probe DTW-to-template approach (the actual Capecci 2020 method).

Tests whether DTW distance from a reference template (built from top-cTS subjects)
correlates with clinical scores. This is what the literature actually does.
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

def load_exercise_positions(df):
    """Load all subjects' position arrays for one exercise DataFrame.
    Returns list of (keypoints, cTS) tuples."""
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
        kp = np.stack(frames, axis=1)  # (n_frames, 25, 3)
        out.append((kp, cts))
    return out


def compute_joint_angle_series(kp, joint_spec):
    """Compute angle time-series for a joint specified as (prox, vert, dist)."""
    p, v, d = joint_spec
    v1 = kp[:, p] - kp[:, v]
    v2 = kp[:, d] - kp[:, v]
    n1 = np.linalg.norm(v1, axis=1, keepdims=True)
    n2 = np.linalg.norm(v2, axis=1, keepdims=True)
    mask = (n1[:, 0] > 1e-10) & (n2[:, 0] > 1e-10)
    cos = np.zeros(kp.shape[0])
    cos[mask] = np.sum(v1[mask] / n1[mask] * v2[mask] / n2[mask], axis=1)
    cos = np.clip(cos, -1, 1)
    return np.degrees(np.arccos(cos))


def resample(arr, target_len=100):
    """Resample 1D array to target length."""
    if len(arr) == target_len:
        return arr
    if len(arr) < 2:
        return np.full(target_len, arr[0] if len(arr) == 1 else 0.0)
    x_old = np.linspace(0, 1, len(arr))
    x_new = np.linspace(0, 1, target_len)
    return np.interp(x_new, x_old, arr)


# Per-exercise primary joint angle (prox, vert, dist) — what each exercise targets
EXERCISE_JOINT = {
    "ex1": (0, 8, 9),    # spinebase-shoulderright-elbowright (shoulder flexion)
    "ex2": (0, 8, 9),    # same
    "ex3": (16, 18, 17) if False else (12, 14, 13),  # hip-knee-ankle left (hip work)
    "ex4": (0, 8, 9),    # shoulder
    "ex5": (16, 18, 17),  # hipright-ankleright-kneeright (knee bend right)
}

print("=" * 70)
print("DTW-TO-TEMPLATE: per-exercise Spearman ρ with cTS")
print("=" * 70)
print("Method: top-5 cTS subjects = template; DTW(test, template); -distance = score")
print()

all_rho = []
all_n = []

for ex_name in ["ex1", "ex2", "ex3", "ex4", "ex5"]:
    df = raw[ex_name]
    data = load_exercise_positions(df)
    n = len(data)
    cts_arr = np.array([d[1] for d in data])

    # Pick primary joint for this exercise
    if ex_name in EXERCISE_JOINT:
        joint_spec = EXERCISE_JOINT[ex_name]
    else:
        joint_spec = (16, 18, 17)  # default right knee

    # Compute angle series for all subjects
    angle_series = []
    for kp, _ in data:
        ang = compute_joint_angle_series(kp, joint_spec)
        ang = resample(ang, 100)  # uniform length
        angle_series.append(ang)

    # Find top-5 cTS subjects (template)
    top5_idx = np.argsort(-cts_arr)[:5]
    templates = [angle_series[i] for i in top5_idx]

    # For each subject, compute DTW distance to nearest template
    distances = []
    for i, ang in enumerate(angle_series):
        best_d = float('inf')
        for tpl in templates:
            d, _ = fastdtw(ang.reshape(-1, 1), tpl.reshape(-1, 1), dist=euclidean)
            if d < best_d:
                best_d = d
        distances.append(best_d)

    distances = np.array(distances)
    # Score = -distance (lower distance = higher score, should correlate + with cTS)
    rho, p = stats.spearmanr(-distances, cts_arr)
    print(f"{ex_name}: ρ = {rho:+.4f} (p={p:.4f}), N={n}, "
          f"dist range [{distances.min():.1f}, {distances.max():.1f}]")
    all_rho.append(rho)
    all_n.append(n)

    # Per-exercise detail for ex5
    if ex_name == "ex5":
        print(f"  Distance percentiles:")
        for q in [10, 25, 50, 75, 90]:
            d_q = np.percentile(distances, q)
            cts_q = np.percentile(cts_arr, q)
            print(f"    P{q}: distance={d_q:.1f}, cTS={cts_q:.3f}")

print()
print(f"Mean ρ across exercises: {np.mean(all_rho):+.4f}")
print(f"Weighted by N: {np.average(all_rho, weights=all_n):+.4f}")

# Try multi-joint DTW: combine primary joints for each exercise
print()
print("=" * 70)
print("MULTI-JOINT DTW: combine all 9 standard joints")
print("=" * 70)

JOINT_SPECS_9 = {
    "left_shoulder":  (0, 4, 5),
    "right_shoulder": (0, 8, 9),
    "left_knee":      (12, 13, 14),
    "right_knee":     (16, 17, 18),
    "left_hip":       (0, 12, 13),
    "right_hip":      (0, 16, 17),
    "left_elbow":     (4, 5, 6),
    "right_elbow":    (8, 9, 10),
    "spine":          (0, 1, 2),
}

all_rho_mj = []
for ex_name in ["ex1", "ex2", "ex3", "ex4", "ex5"]:
    df = raw[ex_name]
    data = load_exercise_positions(df)
    n = len(data)
    cts_arr = np.array([d[1] for d in data])

    # Compute all 9 angle series
    multi_series = []
    for kp, _ in data:
        angles = []
        for jname, jspec in JOINT_SPECS_9.items():
            ang = compute_joint_angle_series(kp, jspec)
            angles.append(resample(ang, 50))  # shorter for multi-dim DTW
        # Stack to (50, 9) feature matrix
        feat = np.stack(angles, axis=1)
        multi_series.append(feat)

    # Top-5 template
    top5_idx = np.argsort(-cts_arr)[:5]
    templates = [multi_series[i] for i in top5_idx]

    distances = []
    for i, feat in enumerate(multi_series):
        best_d = float('inf')
        for tpl in templates:
            d, _ = fastdtw(feat, tpl, dist=euclidean)
            if d < best_d:
                best_d = d
        distances.append(best_d)

    rho, p = stats.spearmanr(-distances, cts_arr)
    print(f"{ex_name}: multi-joint DTW ρ = {rho:+.4f} (p={p:.4f}), N={n}")
    all_rho_mj.append(rho)

print(f"\nMean ρ (multi-joint): {np.mean(all_rho_mj):+.4f}")
