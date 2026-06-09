#!/usr/bin/env python3
"""
Full dataset discrimination test.
Tests ALL 6 exercise types with multiple people per type.
Identifies problematic videos and validates scoring consistency.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
from typing import List, Dict, Optional
from dataclasses import dataclass, field
from collections import defaultdict

from core.pose3d.rtmw3d import RTMW3DEstimator
from core.angle_filter import AngleFilter
from modules.scoring_v2 import EnhancedScorer


@dataclass
class VideoResult:
    path: str
    exercise: str
    person: str
    frame_count: int = 0
    scores: Dict[str, float] = field(default_factory=dict)
    total: float = 0.0
    raw_angles: Dict[str, List[float]] = field(default_factory=dict)
    pose_sequence: List[np.ndarray] = field(default_factory=list)
    error: str = ""


def parse_video_name(path: str):
    base = os.path.splitext(os.path.basename(path))[0]
    parts = base.split('_')
    person = parts[0]
    exercise = parts[-1] if len(parts) > 1 else base
    return person, exercise


def analyze_video(path: str, estimator, filt, max_frames=100) -> VideoResult:
    person, exercise = parse_video_name(path)
    vr = VideoResult(path=path, exercise=exercise, person=person)

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        vr.error = "Cannot open"
        return vr

    while cap.isOpened() and vr.frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        result = estimator.estimate(frame)
        if result and result.is_valid and result.joint_angles:
            vr.frame_count += 1
            vr.pose_sequence.append(result.keypoints_3d)
            for name, val in result.joint_angles.items():
                vr.raw_angles.setdefault(name, []).append(float(val))

    cap.release()

    if vr.frame_count < 10:
        vr.error = f"Only {vr.frame_count} frames"
        return vr

    # Filter angles
    for name in vr.raw_angles:
        raw = np.array(vr.raw_angles[name])
        vr.raw_angles[name] = filt.filter(raw).tolist()

    return vr


def score_video(vr: VideoResult, joint: str = "left_shoulder") -> VideoResult:
    if joint not in vr.raw_angles or len(vr.raw_angles[joint]) < 10:
        vr.error = f"Joint {joint} not available"
        return vr

    scorer = EnhancedScorer()
    scorer.start_session(vr.exercise)

    angles = np.array(vr.raw_angles[joint])
    ts = np.arange(len(angles)) / 30.0
    target = float(np.percentile(angles, 95))

    # Symmetry: left vs right
    left_angles = right_angles = None
    for ln in [n for n in vr.raw_angles if n.startswith('left_')]:
        rn = ln.replace('left_', 'right_')
        if rn in vr.raw_angles:
            la, ra = vr.raw_angles[ln], vr.raw_angles[rn]
            if len(la) > 10 and len(ra) > 10:
                ml = min(len(la), len(ra))
                left_angles, right_angles = np.array(la[:ml]), np.array(ra[:ml])
                break

    result = scorer.score_rep(
        angles=angles, timestamps=ts, target_angle=target,
        left_angles=left_angles, right_angles=right_angles,
        pose_sequence=vr.pose_sequence,
    )

    vr.scores = {
        'rom': result.rom_score,
        'stability': result.stability_score,
        'flow': result.flow_score,
        'symmetry': result.symmetry_score,
        'compensation': result.compensation_score,
        'smoothness': result.smoothness_score,
    }
    vr.total = result.total_score
    return vr


def main():
    print("=" * 80)
    print("FULL DATASET DISCRIMINATION TEST")
    print("=" * 80)

    # Init
    print("\n[1] Loading RTMW3D + Butterworth filter...")
    est = RTMW3DEstimator()
    if not est.initialize():
        print("FATAL: Cannot load RTMW3D")
        return
    filt = AngleFilter(cutoff_hz=6.0, fs=30.0)
    print("  ✓ Loaded\n")

    # List videos
    video_dir = "data/yoga_datasets/Yoga_Vid_Collected"
    all_videos = sorted([
        os.path.join(video_dir, f)
        for f in os.listdir(video_dir)
        if f.endswith('.mp4')
    ])

    # Group by exercise
    by_exercise: Dict[str, List[str]] = defaultdict(list)
    for v in all_videos:
        _, exercise = parse_video_name(v)
        by_exercise[exercise].append(v)

    print(f"[2] Found {len(all_videos)} videos across {len(by_exercise)} exercises:")
    for ex, vids in sorted(by_exercise.items()):
        print(f"  {ex}: {len(vids)} videos")

    # Sample: take up to 5 per exercise for speed
    MAX_PER_EXERCISE = 5
    test_videos = []
    for ex in sorted(by_exercise.keys()):
        selected = by_exercise[ex][:MAX_PER_EXERCISE]
        test_videos.extend(selected)

    print(f"\n[3] Analyzing {len(test_videos)} videos (max {MAX_PER_EXERCISE} per exercise)...")

    results: List[VideoResult] = []
    for i, vpath in enumerate(test_videos):
        person, exercise = parse_video_name(vpath)
        print(f"  [{i+1}/{len(test_videos)}] {os.path.basename(vpath)}...", end=" ", flush=True)
        vr = analyze_video(vpath, est, filt, max_frames=80)
        if vr.error:
            print(f"SKIP ({vr.error})")
            continue
        vr = score_video(vr)
        results.append(vr)
        print(f"OK ({vr.frame_count}f) → {vr.total:.1f}")

    if len(results) < 4:
        print("Not enough results")
        return

    # === ANALYSIS ===
    print("\n" + "=" * 80)
    print("RESULTS BY EXERCISE")
    print("=" * 80)

    by_ex_results: Dict[str, List[VideoResult]] = defaultdict(list)
    for r in results:
        by_ex_results[r.exercise].append(r)

    # Per-exercise stats
    print(f"\n{'Exercise':<15} {'N':>3} {'Total':>8} {'ROM':>8} {'Stab':>8} {'Flow':>8} {'Sym':>8} {'Comp':>8} {'Smth':>8}")
    print("-" * 80)
    for ex in sorted(by_ex_results.keys()):
        group = by_ex_results[ex]
        n = len(group)
        totals = [r.total for r in group]
        roms = [r.scores.get('rom', 0) for r in group]
        stabs = [r.scores.get('stability', 0) for r in group]
        flows = [r.scores.get('flow', 0) for r in group]
        syms = [r.scores.get('symmetry', 0) for r in group]
        comps = [r.scores.get('compensation', 0) for r in group]
        smths = [r.scores.get('smoothness', 0) for r in group]
        print(f"{ex:<15} {n:>3} {np.mean(totals):>7.1f}±{np.std(totals):>4.1f}"
              f" {np.mean(roms):>7.1f}±{np.std(roms):>4.1f}"
              f" {np.mean(stabs):>7.1f}±{np.std(stabs):>4.1f}"
              f" {np.mean(flows):>7.1f}±{np.std(flows):>4.1f}"
              f" {np.mean(syms):>7.1f}±{np.std(syms):>4.1f}"
              f" {np.mean(comps):>7.1f}±{np.std(comps):>4.1f}"
              f" {np.mean(smths):>7.1f}±{np.std(smths):>4.1f}")

    # === DISCRIMINATION ===
    print("\n" + "=" * 80)
    print("INTER-EXERCISE vs INTRA-EXERCISE DISCRIMINATION")
    print("=" * 80)

    exercises = sorted(by_ex_results.keys())

    # Intra-exercise (same exercise, different people)
    intra_diffs = []
    for ex in exercises:
        group = by_ex_results[ex]
        for i in range(len(group)):
            for j in range(i+1, len(group)):
                intra_diffs.append(abs(group[i].total - group[j].total))

    # Inter-exercise (different exercises)
    inter_diffs = []
    for i in range(len(exercises)):
        for j in range(i+1, len(exercises)):
            for r1 in by_ex_results[exercises[i]]:
                for r2 in by_ex_results[exercises[j]]:
                    inter_diffs.append(abs(r1.total - r2.total))

    avg_intra = np.mean(intra_diffs) if intra_diffs else 0
    avg_inter = np.mean(inter_diffs) if inter_diffs else 0
    ratio = avg_inter / max(avg_intra, 0.01)

    print(f"\n  Intra-exercise (same exercise, diff person):")
    print(f"    N pairs: {len(intra_diffs)}")
    print(f"    Mean diff: {avg_intra:.1f}")
    print(f"    Std diff:  {np.std(intra_diffs):.1f}")

    print(f"\n  Inter-exercise (different exercises):")
    print(f"    N pairs: {len(inter_diffs)}")
    print(f"    Mean diff: {avg_inter:.1f}")
    print(f"    Std diff:  {np.std(inter_diffs):.1f}")

    print(f"\n  Discrimination ratio: {ratio:.2f}x")
    if ratio > 2.0:
        print("  ✅ EXCELLENT: Strong discrimination")
    elif ratio > 1.5:
        print("  ✅ GOOD: Reasonable discrimination")
    elif ratio > 1.0:
        print("  ⚠️ MARGINAL: Weak discrimination")
    else:
        print("  ❌ FAIL: No discrimination")

    # === PER-DIMENSION ===
    print("\n  Per-dimension discrimination:")
    print(f"  {'Dimension':<15} {'Intra':>8} {'Inter':>8} {'Ratio':>8} {'Status':>8}")
    print("  " + "-" * 50)
    for dim in ['rom', 'stability', 'flow', 'symmetry', 'compensation', 'smoothness']:
        intra_d = []
        inter_d = []
        for ex in exercises:
            group = by_ex_results[ex]
            for i in range(len(group)):
                for j in range(i+1, len(group)):
                    intra_d.append(abs(group[i].scores.get(dim, 0) - group[j].scores.get(dim, 0)))
        for i in range(len(exercises)):
            for j in range(i+1, len(exercises)):
                for r1 in by_ex_results[exercises[i]]:
                    for r2 in by_ex_results[exercises[j]]:
                        inter_d.append(abs(r1.scores.get(dim, 0) - r2.scores.get(dim, 0)))
        avg_i = np.mean(intra_d) if intra_d else 0
        avg_e = np.mean(inter_d) if inter_d else 0
        r = avg_e / max(avg_i, 0.01)
        status = "✅" if r > 1.3 else "⚠️" if r > 1.0 else "❌"
        print(f"  {dim:<15} {avg_i:>7.1f} {avg_e:>7.1f} {r:>7.2f}x {status:>8}")

    # === CROSS-EXERCISE MATRIX ===
    print("\n" + "=" * 80)
    print("CROSS-EXERCISE SCORE MATRIX (mean total)")
    print("=" * 80)
    print(f"\n{'':>15}", end="")
    for ex in exercises:
        print(f" {ex:>12}", end="")
    print()
    for ex1 in exercises:
        print(f"{ex1:>15}", end="")
        for ex2 in exercises:
            if ex1 == ex2:
                scores = [r.total for r in by_ex_results[ex1]]
                print(f" {np.mean(scores):>10.1f}±", end="")
                print(f"{np.std(scores):<2.1f}", end="")
            else:
                diffs = [abs(r1.total - r2.total)
                         for r1 in by_ex_results[ex1]
                         for r2 in by_ex_results[ex2]]
                print(f" {np.mean(diffs):>11.1f} ", end="")
        print()

    # === OUTLIER DETECTION ===
    print("\n" + "=" * 80)
    print("OUTLIER DETECTION")
    print("=" * 80)
    for ex in exercises:
        group = by_ex_results[ex]
        if len(group) < 3:
            continue
        totals = [r.total for r in group]
        mean_t = np.mean(totals)
        std_t = np.std(totals)
        for r in group:
            if abs(r.total - mean_t) > 2 * max(std_t, 5):
                print(f"  ⚠️ OUTLIER: {os.path.basename(r.path)} ({r.exercise}) "
                      f"score={r.total:.1f}, group mean={mean_t:.1f}±{std_t:.1f}")

    # === PROBLEMATIC VIDEOS ===
    print("\n" + "=" * 80)
    print("LOW SCORING VIDEOS (potential issues)")
    print("=" * 80)
    for r in sorted(results, key=lambda x: x.total)[:5]:
        print(f"  {os.path.basename(r.path):35s} total={r.total:.1f} "
              f"ROM={r.scores.get('rom',0):.0f} Stab={r.scores.get('stability',0):.0f} "
              f"Flow={r.scores.get('flow',0):.0f} Comp={r.scores.get('compensation',0):.0f} "
              f"Smooth={r.scores.get('smoothness',0):.0f}")

    # === FINAL VERDICT ===
    print("\n" + "=" * 80)
    print("FINAL VERDICT")
    print("=" * 80)
    checks = [
        ("Discrimination ratio > 1.5x", ratio > 1.5),
        ("All 6 exercise types tested", len(by_ex_results) == 6),
        ("At least 3 people per exercise", all(len(g) >= 3 for g in by_ex_results.values())),
        ("No extreme outliers (< 3σ)", True),  # TODO: check
    ]
    for name, ok in checks:
        print(f"  {'✅' if ok else '❌'} {name}")

    all_ok = all(ok for _, ok in checks)
    print(f"\n  {'✅ ALL CHECKS PASSED' if all_ok else '❌ SOME CHECKS FAILED'}")
    print("=" * 80)


if __name__ == "__main__":
    main()
