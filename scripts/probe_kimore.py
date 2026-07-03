"""Hands-on KIMORE diagnostic.

Verifies:
1. Positions are actually positions (not quaternions)
2. Per-joint angle sanity (knee should oscillate for knee-bend exercises)
3. Why baseline correlation is still 0.054
"""
import sys
import pickle
import numpy as np
from pathlib import Path
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

KIMORE_PATH = ROOT / "data" / "KIMORE" / "kimore_exercise_dataset.pkl"

print("=" * 70)
print("LOADING KIMORE")
print("=" * 70)
with open(KIMORE_PATH, "rb") as f:
    raw = pickle.load(f)

# Pick ex5 (knee bends) — knee angle should oscillate strongly
df = raw["ex5"]
print(f"ex5: {len(df)} subjects, {len(df.columns)} columns")
print(f"Columns: {list(df.columns)}")

# Inspect one row's data structure
row0 = df.iloc[0]
col0 = "kneeright"
jd = row0[col0]
print(f"\n--- {col0} raw data ---")
print(f"Type: {type(jd).__name__}")
if isinstance(jd, np.ndarray):
    print(f"Shape: {jd.shape}")
    print(f"dtype: {jd.dtype}")
    print(f"First frame: {jd[0]}")
    print(f"Last frame:  {jd[-1]}")
    print(f"Frame 100:   {jd[100] if len(jd) > 100 else 'N/A'}")

# Now check what [:, :3] vs [:, 4:7] gives us
print(f"\n--- Slice analysis on {col0} ---")
if jd.ndim == 2 and jd.shape[1] >= 7:
    qxyz = jd[:, :3]
    pos = jd[:, 4:7]
    print(f"jd[:, :3]  (quaternion xyz)  range: x={qxyz[:,0].min():.3f}..{qxyz[:,0].max():.3f}, "
          f"y={qxyz[:,1].min():.3f}..{qxyz[:,1].max():.3f}, z={qxyz[:,2].min():.3f}..{qxyz[:,2].max():.3f}")
    print(f"                            norm: {np.linalg.norm(qxyz, axis=1).mean():.4f} (≈1.0 = quaternion)")
    print(f"jd[:, 4:7] (position xyz)   range: x={pos[:,0].min():.3f}..{pos[:,0].max():.3f}, "
          f"y={pos[:,1].min():.3f}..{pos[:,1].max():.3f}, z={pos[:,2].min():.3f}..{pos[:,2].max():.3f}")
    print(f"                            norm: {np.linalg.norm(pos, axis=1).mean():.4f} (meters from origin)")
elif jd.ndim == 2 and jd.shape[1] == 3:
    print(f"Already 3D positions. Shape: {jd.shape}")
    pos = jd
else:
    print(f"Unexpected shape: {jd.shape}")

# Build keypoints for first 5 subjects using positions
print("\n" + "=" * 70)
print("ANGLE SANITY CHECK — ex5 (knee bends)")
print("=" * 70)
print("Expected: right_knee angle oscillates between ~30° (flexion) and ~170° (extension)")
print()

# KIMORE Kinect v2 joint indices (matching KINECT_JOINT_INDEX)
KINECT_IDX = {
    "spinebase": 0, "spinemid": 1, "neck": 2, "head": 3,
    "shoulderleft": 4, "elbowleft": 5, "wristleft": 6, "handleft": 7,
    "shoulderright": 8, "elbowright": 9, "wristright": 10, "handright": 11,
    "hipleft": 12, "kneeleft": 13, "ankleleft": 14, "footleft": 15,
    "hipright": 16, "kneeright": 17, "ankleright": 18, "footright": 19,
    "spineshoulder": 20,
}

for subj_idx in range(5):
    row = df.iloc[subj_idx]
    cts = float(row["cTS"])

    # Stack positions: (n_frames, 25, 3)
    frames = []
    for col in df.columns:
        if col == "cTS":
            continue
        jdata = row[col]
        if isinstance(jdata, np.ndarray) and jdata.ndim == 2:
            if jdata.shape[1] >= 7:
                frames.append(jdata[:, 4:7])  # positions
            else:
                frames.append(jdata[:, :3])
    if not frames:
        continue
    kp = np.stack(frames, axis=1)  # (n_frames, 25, 3)

    # Right knee angle: hipright(16) - kneeright(17) - ankleright(18)
    v1 = kp[:, KINECT_IDX["hipright"]] - kp[:, KINECT_IDX["kneeright"]]
    v2 = kp[:, KINECT_IDX["ankleright"]] - kp[:, KINECT_IDX["kneeright"]]
    n1 = np.linalg.norm(v1, axis=1, keepdims=True)
    n2 = np.linalg.norm(v2, axis=1, keepdims=True)
    mask = (n1[:, 0] > 1e-10) & (n2[:, 0] > 1e-10)
    cos = np.zeros(kp.shape[0])
    cos[mask] = np.sum(v1[mask] / n1[mask] * v2[mask] / n2[mask], axis=1)
    cos = np.clip(cos, -1, 1)
    angles = np.degrees(np.arccos(cos))

    # Also: vertical distance head-spinebase (subject should be upright)
    head_y = kp[:, KINECT_IDX["head"], 1].mean()
    spine_y = kp[:, KINECT_IDX["spinebase"], 1].mean()
    upright = head_y > spine_y

    print(f"subj {subj_idx} cTS={cts:.3f}: upright={upright}, "
          f"knee angle: min={angles.min():.1f}° max={angles.max():.1f}° "
          f"range={angles.max()-angles.min():.1f}° mean={angles.mean():.1f}°")

# Now check: does knee angle RANGE correlate with cTS across all subjects?
print("\n" + "=" * 70)
print("CORRELATION: knee angle range vs cTS — ex5 (ALL subjects)")
print("=" * 70)
all_ranges = []
all_cts = []
for idx in range(len(df)):
    row = df.iloc[idx]
    cts = float(row["cTS"])
    frames = []
    for col in df.columns:
        if col == "cTS":
            continue
        jdata = row[col]
        if isinstance(jdata, np.ndarray) and jdata.ndim == 2:
            if jdata.shape[1] >= 7:
                frames.append(jdata[:, 4:7])
            else:
                frames.append(jdata[:, :3])
    if not frames:
        continue
    kp = np.stack(frames, axis=1)

    # Right knee
    v1 = kp[:, 16] - kp[:, 17]
    v2 = kp[:, 18] - kp[:, 17]
    n1 = np.linalg.norm(v1, axis=1, keepdims=True)
    n2 = np.linalg.norm(v2, axis=1, keepdims=True)
    mask = (n1[:, 0] > 1e-10) & (n2[:, 0] > 1e-10)
    if not np.any(mask):
        continue
    cos = np.zeros(kp.shape[0])
    cos[mask] = np.sum(v1[mask] / n1[mask] * v2[mask] / n2[mask], axis=1)
    cos = np.clip(cos, -1, 1)
    angles = np.degrees(np.arccos(cos))
    all_ranges.append(angles.max() - angles.min())
    all_cts.append(cts)

all_ranges = np.array(all_ranges)
all_cts = np.array(all_cts)
rho, p = stats.spearmanr(all_ranges, all_cts)
print(f"N={len(all_ranges)}, Spearman ρ = {rho:.4f}, p = {p:.4f}")
print(f"knee range: min={all_ranges.min():.1f}°, max={all_ranges.max():.1f}°, "
      f"mean={all_ranges.mean():.1f}°, std={all_ranges.std():.1f}°")
print(f"cTS:        min={all_cts.min():.3f}, max={all_cts.max():.3f}, "
      f"mean={all_cts.mean():.3f}, std={all_cts.std():.3f}")

# The KEY question: is the sign correct? Maybe higher cTS = WORSE?
print(f"\nQuartiles of cTS vs knee range:")
for q_lo, q_hi in [(0.0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.01)]:
    lo = np.quantile(all_cts, q_lo)
    hi = np.quantile(all_cts, q_hi)
    mask = (all_cts >= lo) & (all_cts < hi)
    if mask.sum() > 0:
        print(f"  cTS [{lo:.3f}, {hi:.3f}): n={mask.sum()}, mean knee range = {all_ranges[mask].mean():.1f}°")
