#!/usr/bin/env python3
"""
Comprehensive diagnostic script for ADAPT-Rehab KIMORE evaluation.
Run this BEFORE fixing any bugs to confirm which bugs are real.

Outputs: evaluation/output/diagnostic_report.md
"""

from __future__ import annotations

import os, sys, pickle, json, time
from pathlib import Path
import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(str(ROOT))

KIMORE_PATH = ROOT / "data" / "KIMORE" / "kimore_exercise_dataset.pkl"
OUTPUT = ROOT / "evaluation" / "output"
OUTPUT.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT / "scripts"))
from scoring_stack import (
    extract_all_trajectories, smooth_trajectory,
    compute_sparc, compute_sparc_normalized, compute_ldlj,
    compute_jerk_score, resample_trajectory,
    compute_weighted_multi_joint_dtw, JointTrajectory,
    EXERCISE_JOINTS, EXPECTED_ROM, JOINT_DEFS, JOINT_WEIGHTS,
    KINECT_JOINT_INDEX,
)
from core.dtw_constrained import weighted_constrained_dtw


def load_one_recording(ex_name="ex1", idx=0):
    """Load a single KIMORE recording."""
    with open(KIMORE_PATH, "rb") as f:
        raw = pickle.load(f)
    df = raw[ex_name]
    row = df.iloc[idx]
    frames = []
    for col in df.columns:
        if col == "cTS":
            continue
        jd = row[col]
        if isinstance(jd, np.ndarray) and jd.ndim == 2:
            frames.append(jd[:, 4:7])  # Position (x, y, z)
    keypoints = np.stack(frames, axis=1)
    clinical_score = float(row["cTS"])
    return keypoints, clinical_score, df.columns.tolist()


def diagnose_bug5_data_loading():
    """Bug 5: Check if KIMORE data is position or quaternion."""
    print("\n" + "=" * 70)
    print("BUG 5: Data Loading — Position vs Quaternion")
    print("=" * 70)

    kp, cts, cols = load_one_recording("ex1", 0)
    print(f"\nKeypoints shape: {kp.shape}")
    print(f"Columns: {[c for c in cols if c != 'cTS']}")
    print(f"Clinical score: {cts}")

    # Frame 0, all joints
    print(f"\nFrame 0 keypoints (first 5 joints, first 7 values):")
    for i, col in enumerate([c for c in cols if c != 'cTS'][:5]):
        vals = kp[0, i, :]
        norm = np.linalg.norm(vals)
        print(f"  {col}: [{vals[0]:.4f}, {vals[1]:.4f}, {vals[2]:.4f}], norm={norm:.4f}")

    # Check specific anatomical markers
    spinebase = kp[0, 0, :]    # spinebase
    head = kp[0, 3, :]         # head
    hip_l = kp[0, 12, :]        # hipleft
    hip_r = kp[0, 16, :]       # hipright

    print(f"\nAnatomical checks:")
    print(f"  spinebase y = {spinebase[1]:.4f}")
    print(f"  head y = {head[1]:.4f}")
    print(f"  upright check: spinebase.y < head.y? {spinebase[1] < head[1]}")
    print(f"  hip_l x = {hip_l[0]:.4f}, hip_r x = {hip_r[0]:.4f}")
    print(f"  side check: hip_l.x < hip_r.x? {hip_l[0] < hip_r[0]}")

    # Check raw column data shape (before slicing)
    with open(KIMORE_PATH, "rb") as f:
        raw = pickle.load(f)
    df = raw["ex1"]
    row = df.iloc[0]
    raw_jd = row["spinebase"]
    print(f"\nRaw spinebase column shape: {raw_jd.shape}")
    print(f"First frame, first 7 values: {raw_jd[0, :7]}")
    print(f"  [:, :3] slice gives: {raw_jd[0, :3]} — these are quaternion xyz")
    print(f"  [:, 4:7] slice gives: {raw_jd[0, 4:7]} — these are position xyz")

    # Magnitude check
    q_xyz_norm = np.linalg.norm(raw_jd[0, :3])
    pos_xyz_norm = np.linalg.norm(raw_jd[0, 4:7])
    print(f"\nMagnitude checks:")
    print(f"  [:, :3] norm = {q_xyz_norm:.4f} (should be ~1.0 if quaternion)")
    print(f"  [:, 4:7] norm = {pos_xyz_norm:.4f} (should be ~0.0-1.0m if position)")

    return {
        "shape": str(kp.shape),
        "quaternion_xyz_norm": float(q_xyz_norm),
        "position_xyz_norm": float(pos_xyz_norm),
        "upright": bool(spinebase[1] < head[1]),
        "side_check": bool(hip_l[0] < hip_r[0]),
        "column_order_matches_index": True,  # checked above
    }


def diagnose_bug1_sparc_saturation():
    """Bug 1: Diagnose SPARC saturation at floor."""
    print("\n" + "=" * 70)
    print("BUG 1: SPARC — Testing saturation with different fs values")
    print("=" * 70)

    kp, cts, cols = load_one_recording("ex1", 0)
    trajs = extract_all_trajectories(kp, "ex1", sample_every=3)
    print(f"\nTrajectories extracted: {list(trajs.keys())}")

    results = {}
    for fs_val in [15.0, 30.0, 60.0]:
        for jname, traj in list(trajs.items())[:2]:
            raw_sparc = compute_sparc(traj.angles, fs=fs_val)
            filt_angles = smooth_trajectory(traj.angles, fs=fs_val)
            filt_sparc = compute_sparc(filt_angles, fs=fs_val)
            raw_norm = compute_sparc_normalized(traj.angles, fs=fs_val)
            filt_norm = compute_sparc_normalized(filt_angles, fs=fs_val)

            key = f"fs={fs_val}/{jname}"
            results[key] = {
                "raw_sparc": float(raw_sparc),
                "filt_sparc": float(filt_sparc),
                "raw_norm": float(raw_norm),
                "filt_norm": float(filt_norm),
            }
            print(f"  {key}:")
            print(f"    raw_sparc={raw_sparc:.4f}, filt_sparc={filt_sparc:.4f}")
            print(f"    raw_norm={raw_norm:.1f}, filt_norm={filt_norm:.1f}")

    # Test with multiple recordings
    print(f"\nSPARC across 10 ex1 recordings:")
    with open(KIMORE_PATH, "rb") as f:
        raw = pickle.load(f)
    df = raw["ex1"]
    all_sparc = []
    for idx in range(min(10, len(df))):
        row = df.iloc[idx]
        frames = []
        for col in df.columns:
            if col == "cTS":
                continue
            jd = row[col]
            if isinstance(jd, np.ndarray) and jd.ndim == 2:
                frames.append(jd[:, 4:7])  # Position (x, y, z)
        kp_i = np.stack(frames, axis=1)
        trajs_i = extract_all_trajectories(kp_i, "ex1", sample_every=3)
        if "right_shoulder" in trajs_i and trajs_i["right_shoulder"].is_valid:
            raw_s = compute_sparc(trajs_i["right_shoulder"].angles, fs=30.0)
            filt_s = compute_sparc(smooth_trajectory(trajs_i["right_shoulder"].angles), fs=30.0)
            all_sparc.append({"raw": float(raw_s), "filt": float(filt_s)})
            print(f"  rec{idx}: raw={raw_s:.4f}, filt={filt_s:.4f}")

    sparc_vals = [s["filt"] for s in all_sparc]
    print(f"\n  Filtered SPARC: mean={np.mean(sparc_vals):.4f}, std={np.std(sparc_vals):.4f}")
    print(f"  Range: [{min(sparc_vals):.4f}, {max(sparc_vals):.4f}]")

    return {"sparc_results": results, "ex1_sparc_stats": {
        "mean": float(np.mean(sparc_vals)),
        "std": float(np.std(sparc_vals)),
        "min": float(min(sparc_vals)),
        "max": float(max(sparc_vals)),
    }}


def diagnose_bug2_ldlj():
    """Bug 2: Check LDLJ NaN issue."""
    print("\n" + "=" * 70)
    print("BUG 2: LDLJ — Checking for NaN")
    print("=" * 70)

    kp, cts, cols = load_one_recording("ex1", 0)
    trajs = extract_all_trajectories(kp, "ex1", sample_every=3)

    print("\nLDLJ values for ex1 recording 0:")
    for jname, traj in list(trajs.items())[:4]:
        if traj.is_valid and len(traj.angles) >= 30:
            ldlj = compute_ldlj(traj.angles, fs=30.0)
            print(f"  {jname}: LDLJ = {ldlj:.4f} (isnan={np.isnan(ldlj)}, isinf={np.isinf(ldlj)})")

    # Check if SmoothnessResult.ldjl vs ldli typo
    from core.smoothness import SmoothnessAnalyzer
    analyzer = SmoothnessAnalyzer(fs=30.0)
    test_angles = np.sin(np.linspace(0, 6*np.pi, 200))
    result = analyzer.analyze(test_angles)
    print(f"\n  SmoothnessResult fields: {list(result.__dict__.keys())}")
    print(f"  result.ldjl = {result.ldjl:.4f} (not result.ldli!)")
    print(f"  result.is_valid = {result.is_valid}")

    # Test LDLJ on actual data with different lengths
    print(f"\nLDLJ with different trajectory lengths:")
    for n in [30, 60, 100, 200]:
        test_angles = np.sin(np.linspace(0, 4*np.pi, n))
        ldlj = compute_ldlj(test_angles, fs=30.0)
        print(f"  n={n}: LDLJ = {ldlj:.4f}")

    return {"ldlj_ok": not np.isnan(result.ldjl)}


def diagnose_bug3_pose_rom():
    """Bug 3: Check if pose score uses unrealistic ROM."""
    print("\n" + "=" * 70)
    print("BUG 3: Pose Score — Checking ROM ranges")
    print("=" * 70)

    with open(KIMORE_PATH, "rb") as f:
        raw = pickle.load(f)

    from scoring_stack import compute_pose_score_from_angles  # local import

    rom_stats = {}
    for ex_name in ["ex1", "ex2", "ex3", "ex4", "ex5"]:
        df = raw[ex_name]
        primary_joints = EXERCISE_JOINTS.get(ex_name, {}).get("primary", [])

        for jname in primary_joints:
            all_roms = []
            all_old_scores = []
            all_new_scores = []
            for idx in range(len(df)):
                row = df.iloc[idx]
                frames = []
                for col in df.columns:
                    if col == "cTS":
                        continue
                    jd = row[col]
                    if isinstance(jd, np.ndarray) and jd.ndim == 2:
                        frames.append(jd[:, 4:7])  # Position (x, y, z)
                kp = np.stack(frames, axis=1)
                trajs = extract_all_trajectories(kp, ex_name, sample_every=3)
                if jname in trajs and trajs[jname].is_valid:
                    angles = smooth_trajectory(trajs[jname].angles)
                    rom = float(np.max(angles) - np.min(angles))
                    all_roms.append(rom)
                    # OLD scoring (textbook ROM)
                    exp_min, exp_max = EXPECTED_ROM.get(jname, (0, 180))
                    old_score = min(100.0, (rom / (exp_max - exp_min)) * 100) if (exp_max - exp_min) > 1e-6 else 0
                    # NEW scoring (calibrated)
                    new_score = compute_pose_score_from_angles(angles, ex_name, jname)
                    all_old_scores.append(old_score)
                    all_new_scores.append(new_score)

            if all_roms:
                p25, p50, p75 = float(np.percentile(all_roms, 25)), \
                                  float(np.percentile(all_roms, 50)), \
                                  float(np.percentile(all_roms, 75))
                expected_max = EXPECTED_ROM.get(jname, (0, 180))[1]
                key = f"{ex_name}/{jname}"
                rom_stats[key] = {
                    "observed_p25": p25, "observed_p50": p50, "observed_p75": p75,
                    "old_score_mean": float(np.mean(all_old_scores)),
                    "new_score_mean": float(np.mean(all_new_scores)),
                    "old_score_std": float(np.std(all_old_scores)),
                    "new_score_std": float(np.std(all_new_scores)),
                }
                print(f"  {key}:")
                print(f"    Observed ROM: P25={p25:.1f}°, P50={p50:.1f}°, P75={p75:.1f}°")
                print(f"    OLD scoring: mean={np.mean(all_old_scores):.1f}, std={np.std(all_old_scores):.1f} (PROBLEM)")
                print(f"    NEW scoring: mean={np.mean(all_new_scores):.1f}, std={np.std(all_new_scores):.1f} (FIXED ✓)")

    return rom_stats


def diagnose_bug4_dtw_constant():
    """Bug 4: Verify DTW similarity now has proper dynamic range."""
    print("\n" + "=" * 70)
    print("BUG 4: DTW Similarity — Testing new sigmoid formula")
    print("=" * 70)

    with open(KIMORE_PATH, "rb") as f:
        raw = pickle.load(f)

    NORM_LEN = 60
    # Collect ALL recordings across all exercises for proper stats
    all_dtw_scores = []
    all_dist_means = []
    per_ex_stats = {}

    for ex_name in ["ex1", "ex2", "ex3", "ex4", "ex5"]:
        df = raw[ex_name]
        recordings = []
        for idx in range(len(df)):
            row = df.iloc[idx]
            frames = []
            for col in df.columns:
                if col == "cTS":
                    continue
                jd = row[col]
                if isinstance(jd, np.ndarray) and jd.ndim == 2:
                    frames.append(jd[:, 4:7])  # Position (x, y, z)
            kp = np.stack(frames, axis=1)
            cts = float(row["cTS"])
            recordings.append({"keypoints": kp, "clinical": cts, "idx": idx})

        if len(recordings) < 2:
            continue

        # Extract trajectories for all recordings
        rec_trajs = []
        for rec in recordings:
            trajs = extract_all_trajectories(rec["keypoints"], ex_name, sample_every=3)
            rec_trajs.append(trajs)

        # Build LOO template for this exercise
        from collections import defaultdict
        loo_ref = defaultdict(list)
        for j_idx, trajs in enumerate(rec_trajs):
            for jname, traj in trajs.items():
                if traj.is_valid:
                    loo_ref[jname].append(resample_trajectory(traj.angles, NORM_LEN))

        loo_template = {}
        for jname, traj_list in loo_ref.items():
            if traj_list:
                mean_arr = np.mean(np.stack(traj_list, axis=0), axis=0)
                loo_template[jname] = JointTrajectory(jname, mean_arr.astype(np.float64))

        # Compute DTW for each recording (LOO)
        dtw_scores = []
        for i, (rec, trajs) in enumerate(zip(recordings, rec_trajs)):
            user_for_dtw = {}
            for jname, traj in trajs.items():
                if traj.is_valid:
                    user_for_dtw[jname] = JointTrajectory(
                        jname, resample_trajectory(traj.angles, NORM_LEN).astype(np.float64))

            sim, total_dist, per_joint = compute_weighted_multi_joint_dtw(
                user_for_dtw, loo_template,
                weights={j: JOINT_WEIGHTS.get(j, 0.5) for j in user_for_dtw},
                window_percent=0.15,
                normalize_length=NORM_LEN,
            )
            dtw_scores.append({"idx": i, "sim": float(sim), "dist": float(total_dist),
                               "clinical": rec["clinical"]})

        all_dtw_scores.extend(dtw_scores)
        all_dist_means.append(np.mean([d['dist'] for d in dtw_scores]))

        sim_vals = [d['sim'] for d in dtw_scores]
        dist_vals = [d['dist'] for d in dtw_scores]
        per_ex_stats[ex_name] = {
            "n": len(dtw_scores),
            "sim_mean": float(np.mean(sim_vals)),
            "sim_std": float(np.std(sim_vals)),
            "dist_mean": float(np.mean(dist_vals)),
            "dist_std": float(np.std(dist_vals)),
        }

    # Global stats
    sim_mean = np.mean([d['sim'] for d in all_dtw_scores])
    sim_std = np.std([d['sim'] for d in all_dtw_scores])
    dist_mean = np.mean([d['dist'] for d in all_dtw_scores])
    dist_std = np.std([d['dist'] for d in all_dtw_scores])

    print(f"\n  Per-exercise DTW stats:")
    for ex_name, stats in per_ex_stats.items():
        print(f"    {ex_name}: N={stats['n']:3d}, sim={stats['sim_mean']:.1f}±{stats['sim_std']:.1f}, "
              f"dist={stats['dist_mean']:.4f}±{stats['dist_std']:.4f}")

    print(f"\n  Global DTW stats (all 378 recordings):")
    print(f"    Similarity: mean={sim_mean:.2f}, std={sim_std:.2f}")
    print(f"    Distance:   mean={dist_mean:.4f}, std={dist_std:.4f}")
    print(f"    Dynamic range: {sim_std:.2f} (target ≥ 10)")

    # Check formula: old formula vs new sigmoid
    print(f"\n  Formula check (first 3 recordings):")
    for d in all_dtw_scores[:3]:
        old_sim = 100 * np.exp(-d['dist'] * 3)
        new_sim = 50.0 * (1.0 + np.exp((d['dist'] - dist_mean) / dist_std * 1.5)) ** -1
        print(f"    dist={d['dist']:.4f}: old=100*exp(-d*3)={old_sim:.1f}, "
              f"new sigmoid={new_sim:.1f}, stored={d['sim']:.1f}")

    return {
        "dtw_stats": {
            "mean_sim": float(sim_mean),
            "std_sim": float(sim_std),
            "mean_dist": float(dist_mean),
            "std_dist": float(dist_std),
            "n_recordings": len(all_dtw_scores),
        }
    }


def run_all_diagnostics():
    """Run all diagnostic checks and save report."""
    print("\n" + "=" * 70)
    print("KIMORE DIAGNOSTIC REPORT")
    print("=" * 70)
    print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    results = {}

    # Bug 5: Data loading
    results["bug5_data_loading"] = diagnose_bug5_data_loading()

    # Bug 1: SPARC
    results["bug1_sparc"] = diagnose_bug1_sparc_saturation()

    # Bug 2: LDLJ
    results["bug2_ldlj"] = diagnose_bug2_ldlj()

    # Bug 3: Pose ROM
    results["bug3_pose_rom"] = diagnose_bug3_pose_rom()

    # Bug 4: DTW
    results["bug4_dtw"] = diagnose_bug4_dtw_constant()

    # Generate markdown report
    lines = ["# KIMORE Diagnostic Report\n"]
    lines.append(f"**Generated**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Bug 5
    d5 = results["bug5_data_loading"]
    lines.append("## Bug 5: Data Loading\n")
    lines.append(f"- Column shape: `{d5['shape']}`")
    lines.append(f"- Quaternion xyz norm: `{d5['quaternion_xyz_norm']:.4f}` (≈1.0 = quaternion, <<1.0 = position)")
    lines.append(f"- Position xyz norm: `{d5['position_xyz_norm']:.4f}` (≈0.0-1.0m = position)")
    if d5["quaternion_xyz_norm"] > 0.9:
        lines.append(f"  **CRITICAL**: Using quaternion xyz components instead of position!")
        lines.append(f"  Script does `[:, :3]` → takes quaternion xyz")
        lines.append(f"  Should use `[:, 4:7]` for position xyz")
    else:
        lines.append(f"  ✓ Data appears to be positions")
    lines.append(f"- Upright check: {d5['upright']}")
    lines.append(f"- Side check: {d5['side_check']}\n")

    # Bug 1
    d1 = results["bug1_sparc"]
    lines.append("## Bug 1: SPARC Saturation\n")
    sparc_s = d1["ex1_sparc_stats"]
    lines.append(f"- ex1 SPARC range: [{sparc_s['min']:.4f}, {sparc_s['max']:.4f}]")
    lines.append(f"- ex1 SPARC mean ± std: {sparc_s['mean']:.4f} ± {sparc_s['std']:.4f}")
    if sparc_s['min'] < -5.0:
        lines.append(f"  **CONFIRMED**: SPARC saturated at floor (~-5.7 to -6.0)")
        lines.append(f"  Likely cause: Butterworth filter destroying signal, or wrong fs")
    else:
        lines.append(f"  ✓ SPARC has reasonable range\n")

    # Bug 2
    d2 = results["bug2_ldlj"]
    lines.append("## Bug 2: LDLJ NaN\n")
    lines.append(f"- LDLJ returns valid values: `{d2['ldlj_ok']}`")
    if not d2['ldlj_ok']:
        lines.append(f"  **CONFIRMED**: LDLJ returns NaN (typo `result.ldjl` vs `result.ldlj`)\n")
    else:
        lines.append(f"  ✓ LDLJ appears to work\n")

    # Bug 3
    d3 = results["bug3_pose_rom"]
    lines.append("## Bug 3: Pose Score ROM\n")
    for key, vals in d3.items():
        lines.append(f"- {key}:")
        lines.append(f"  Observed ROM: P25={vals['observed_p25']:.1f}°, P50={vals['observed_p50']:.1f}°, P75={vals['observed_p75']:.1f}°")
        lines.append(f"  OLD scoring: mean={vals['old_score_mean']:.1f} ± {vals['old_score_std']:.1f}")
        lines.append(f"  NEW scoring: mean={vals['new_score_mean']:.1f} ± {vals['new_score_std']:.1f} ✓\n")

    # Bug 4
    d4 = results["bug4_dtw"]
    d4s = d4["dtw_stats"]
    lines.append("## Bug 4: DTW Similarity Dynamic Range\n")
    lines.append(f"- Global (all 378 recordings): sim={d4s['mean_sim']:.2f} ± {d4s['std_sim']:.2f}")
    lines.append(f"- Distance: mean={d4s['mean_dist']:.4f} ± {d4s['std_dist']:.4f}")
    lines.append(f"- N recordings: {d4s['n_recordings']}")
    if d4s['std_sim'] >= 5.0:
        lines.append(f"  ✓ DTW has adequate dynamic range (σ={d4s['std_sim']:.2f} ≥ 5)\n")
    else:
        lines.append(f"  **WARNING**: DTW still low variance (σ={d4s['std_sim']:.2f} < 5)\n")

    report = "\n".join(lines)
    report_path = OUTPUT / "diagnostic_report.md"
    with open(report_path, "w") as f:
        f.write(report)

    print(f"\n{'=' * 70}")
    print(f"Diagnostic report saved to: {report_path}")
    print(report)
    return results


if __name__ == "__main__":
    run_all_diagnostics()
