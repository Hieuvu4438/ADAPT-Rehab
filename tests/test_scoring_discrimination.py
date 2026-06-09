#!/usr/bin/env python3
"""
Comprehensive scoring discrimination test.
Tests whether the scoring system can differentiate between:
1. Same exercise (different people) → should get similar scores
2. Different exercises → should get different scores
Also validates scoring logic correctness.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
from typing import List, Dict, Optional
from dataclasses import dataclass, field

from core.pose3d.rtmw3d import RTMW3DEstimator
from core.angle_filter import AngleFilter
from modules.scoring_v2 import EnhancedScorer


@dataclass
class VideoAnalysis:
    """Complete analysis of a video."""
    video_path: str
    pose_name: str
    person_name: str
    frame_count: int
    angle_history: Dict[str, List[float]] = field(default_factory=dict)
    pose_sequence: List[np.ndarray] = field(default_factory=list)
    scores: Optional[Dict[str, float]] = None
    total_score: float = 0.0


def extract_pose_name(filename: str) -> str:
    base = os.path.splitext(os.path.basename(filename))[0]
    parts = base.split('_')
    return parts[-1] if len(parts) > 1 else base


def extract_person_name(filename: str) -> str:
    base = os.path.splitext(os.path.basename(filename))[0]
    parts = base.split('_')
    return parts[0] if len(parts) > 1 else "unknown"


def analyze_video(video_path: str, estimator: RTMW3DEstimator,
                  angle_filter: AngleFilter, max_frames: int = 120) -> Optional[VideoAnalysis]:
    """Analyze a single video with Butterworth filtering."""
    pose_name = extract_pose_name(video_path)
    person_name = extract_person_name(video_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  ERROR: Cannot open {video_path}")
        return None

    total_vid_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    analysis = VideoAnalysis(
        video_path=video_path,
        pose_name=pose_name,
        person_name=person_name,
        frame_count=0,
    )

    frame_idx = 0
    while cap.isOpened() and frame_idx < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        result = estimator.estimate(frame)
        if result is None or not result.is_valid:
            frame_idx += 1
            continue

        angles = result.joint_angles
        if angles:
            analysis.frame_count += 1
            analysis.pose_sequence.append(result.keypoints_3d)
            for name, val in angles.items():
                if name not in analysis.angle_history:
                    analysis.angle_history[name] = []
                analysis.angle_history[name].append(float(val))

        frame_idx += 1

    cap.release()

    if analysis.frame_count < 10:
        print(f"  WARNING: Only {analysis.frame_count} frames with pose")
        return None

    # Apply Butterworth filter to all angle sequences
    for name in analysis.angle_history:
        raw = np.array(analysis.angle_history[name])
        filtered = angle_filter.filter(raw)
        analysis.angle_history[name] = filtered.tolist()

    print(f"  Processed {analysis.frame_count}/{min(total_vid_frames, max_frames)} frames, "
          f"{len(analysis.angle_history)} joints (filtered)")
    return analysis


def score_analysis(analysis: VideoAnalysis, joint_name: str = "left_shoulder") -> Dict[str, float]:
    """Score a video analysis using EnhancedScorer on a specific joint."""
    scorer = EnhancedScorer()
    scorer.start_session(analysis.pose_name)

    if joint_name not in analysis.angle_history or len(analysis.angle_history[joint_name]) < 10:
        return {}

    angles = np.array(analysis.angle_history[joint_name])
    timestamps = np.arange(len(angles)) / 30.0

    # Target = 95th percentile of achieved angles
    target = float(np.percentile(angles, 95))

    # Left/right for symmetry
    left_angles = None
    right_angles = None
    left_names = [n for n in analysis.angle_history if n.startswith('left_')]
    right_names = [n for n in analysis.angle_history if n.startswith('right_')]
    for ln in left_names:
        rn = ln.replace('left_', 'right_')
        if rn in analysis.angle_history:
            la = analysis.angle_history[ln]
            ra = analysis.angle_history[rn]
            if len(la) > 10 and len(ra) > 10:
                min_len = min(len(la), len(ra))
                left_angles = np.array(la[:min_len])
                right_angles = np.array(ra[:min_len])
                break

    rep_result = scorer.score_rep(
        angles=angles,
        timestamps=timestamps,
        target_angle=target,
        left_angles=left_angles,
        right_angles=right_angles,
        pose_sequence=analysis.pose_sequence,
    )

    return {
        'total': rep_result.total_score,
        'rom': rep_result.rom_score,
        'stability': rep_result.stability_score,
        'flow': rep_result.flow_score,
        'symmetry': rep_result.symmetry_score,
        'compensation': rep_result.compensation_score,
        'smoothness': rep_result.smoothness_score,
    }


def run_discrimination_test():
    """Main test: compare same-pose vs different-pose scores."""
    print("=" * 70)
    print("SCORING DISCRIMINATION TEST (with Butterworth filter)")
    print("=" * 70)

    # Initialize
    print("\n[1/4] Loading RTMW3D + Butterworth filter...")
    estimator = RTMW3DEstimator()
    if not estimator.initialize():
        print("  ERROR: Failed to initialize RTMW3D")
        return
    angle_filter = AngleFilter(cutoff_hz=6.0, fs=30.0, order=4)
    print("  ✓ RTMW3D + filter loaded")

    # Select videos
    video_dir = "data/yoga_datasets/Yoga_Vid_Collected"
    test_videos = [
        os.path.join(video_dir, "Abhay_Tadasana.mp4"),
        os.path.join(video_dir, "Ameya_Tadasana.mp4"),
        os.path.join(video_dir, "Bhumi_Tadasana.mp4"),
        os.path.join(video_dir, "Abhay_Bhujangasana.mp4"),
        os.path.join(video_dir, "Ameya_Trikonasana.mp4"),
        os.path.join(video_dir, "Ameya_Padmasana.mp4"),
    ]
    test_videos = [v for v in test_videos if os.path.exists(v)]

    # Analyze
    print(f"\n[2/4] Analyzing {len(test_videos)} videos...")
    analyses: Dict[str, VideoAnalysis] = {}
    for vpath in test_videos:
        print(f"\n  {os.path.basename(vpath)}:")
        result = analyze_video(vpath, estimator, angle_filter, max_frames=100)
        if result:
            analyses[vpath] = result

    if len(analyses) < 2:
        print("\nERROR: Not enough videos analyzed.")
        return

    # Score ALL videos using the SAME joint for fair comparison
    # Use left_shoulder as primary (most informative for upper body exercises)
    primary_joint = "left_shoulder"
    print(f"\n[3/4] Scoring {len(analyses)} videos (primary joint: {primary_joint})...")
    for vpath, analysis in analyses.items():
        scores = score_analysis(analysis, joint_name=primary_joint)
        analysis.scores = scores
        analysis.total_score = scores.get('total', 0)
        print(f"\n  {os.path.basename(vpath)} ({analysis.pose_name} by {analysis.person_name}):")
        print(f"    Total: {analysis.total_score:.1f}")
        for dim in ['rom', 'stability', 'flow', 'symmetry', 'compensation', 'smoothness']:
            print(f"    {dim:15s}: {scores.get(dim, 0):.1f}")

    # Compare
    print("\n[4/4] Discrimination Analysis")
    print("-" * 70)

    all_analyses = list(analyses.values())
    pose_groups: Dict[str, List[VideoAnalysis]] = {}
    for a in all_analyses:
        pose_groups.setdefault(a.pose_name, []).append(a)

    # Same-pose
    print("\n  SAME POSE comparisons:")
    same_diffs = []
    for pose, group in pose_groups.items():
        if len(group) >= 2:
            for i in range(len(group)):
                for j in range(i+1, len(group)):
                    diff = abs(group[i].total_score - group[j].total_score)
                    same_diffs.append(diff)
                    print(f"    {pose}: {group[i].person_name}={group[i].total_score:.1f} "
                          f"vs {group[j].person_name}={group[j].total_score:.1f} → diff={diff:.1f}")

    # Different-pose
    print("\n  DIFFERENT POSE comparisons:")
    diff_diffs = []
    poses = list(pose_groups.keys())
    for i in range(len(poses)):
        for j in range(i+1, len(poses)):
            for a1 in pose_groups[poses[i]]:
                for a2 in pose_groups[poses[j]]:
                    diff = abs(a1.total_score - a2.total_score)
                    diff_diffs.append(diff)
                    print(f"    {a1.pose_name}({a1.person_name})={a1.total_score:.1f} "
                          f"vs {a2.pose_name}({a2.person_name})={a2.total_score:.1f} → diff={diff:.1f}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    if same_diffs:
        avg_same = np.mean(same_diffs)
        print(f"  Avg score diff (SAME pose, diff person):  {avg_same:.1f}")

    if diff_diffs:
        avg_diff = np.mean(diff_diffs)
        print(f"  Avg score diff (DIFFERENT pose):          {avg_diff:.1f}")

    if same_diffs and diff_diffs:
        ratio = avg_diff / max(avg_same, 0.1)
        print(f"  Discrimination ratio: {ratio:.1f}x")
        if ratio > 1.5:
            print("  ✓ PASS: Scoring system discriminates between exercises")
        elif ratio > 1.0:
            print("  ~ MARGINAL: Some discrimination, but weak")
        else:
            print("  ✗ FAIL: Scoring system does NOT discriminate well")

    # Per-dimension
    print("\n  PER-DIMENSION DISCRIMINATION:")
    for dim in ['rom', 'stability', 'flow', 'symmetry', 'compensation', 'smoothness']:
        same_dim = []
        diff_dim = []
        for pose, group in pose_groups.items():
            if len(group) >= 2:
                for i in range(len(group)):
                    for j in range(i+1, len(group)):
                        same_dim.append(abs(group[i].scores.get(dim, 0) - group[j].scores.get(dim, 0)))
        for i in range(len(poses)):
            for j in range(i+1, len(poses)):
                for a1 in pose_groups[poses[i]]:
                    for a2 in pose_groups[poses[j]]:
                        diff_dim.append(abs(a1.scores.get(dim, 0) - a2.scores.get(dim, 0)))
        if same_dim and diff_dim:
            r = np.mean(diff_dim) / max(np.mean(same_dim), 0.1)
            status = "✓" if r > 1.3 else "~" if r > 1.0 else "✗"
            print(f"    {status} {dim:15s}: same={np.mean(same_dim):.1f} diff={np.mean(diff_dim):.1f} ratio={r:.1f}x")


def run_scoring_logic_test():
    """Test scoring logic correctness with synthetic data."""
    print("\n" + "=" * 70)
    print("SCORING LOGIC CORRECTNESS TEST")
    print("=" * 70)

    scorer = EnhancedScorer()
    ts = np.linspace(0, 3.33, 100)

    # Test 1: Smooth sine wave
    print("\n[Test 1] Smooth sine wave movement")
    t = np.linspace(0, 2*np.pi, 100)
    smooth = 90 + 30*np.sin(t)
    scorer.start_session("test")
    r1 = scorer.score_rep(angles=smooth, timestamps=ts, target_angle=120.0)
    print(f"  Total: {r1.total_score:.1f}, ROM: {r1.rom_score:.1f}, Flow: {r1.flow_score:.1f}, Smooth: {r1.smoothness_score:.1f}")

    # Test 2: Jerky random
    print("\n[Test 2] Jerky random movement")
    np.random.seed(42)
    jerky = np.random.uniform(30, 150, 100)
    scorer.start_session("test")
    r2 = scorer.score_rep(angles=jerky, timestamps=ts, target_angle=120.0)
    print(f"  Total: {r2.total_score:.1f}, Flow: {r2.flow_score:.1f}, Smooth: {r2.smoothness_score:.1f}")

    # Test 3: Static hold
    print("\n[Test 3] Static hold")
    static = np.full(100, 90.0)
    scorer.start_session("test")
    r3 = scorer.score_rep(angles=static, timestamps=ts, target_angle=90.0)
    print(f"  Total: {r3.total_score:.1f}, Stability: {r3.stability_score:.1f}")

    # Test 4: Ramp up + hold
    print("\n[Test 4] Ramp up + hold (good form)")
    ramp = np.concatenate([np.linspace(30, 120, 50), np.full(50, 120.0)])
    scorer.start_session("test")
    r4 = scorer.score_rep(angles=ramp, timestamps=ts, target_angle=120.0)
    print(f"  Total: {r4.total_score:.1f}, ROM: {r4.rom_score:.1f}, Stability: {r4.stability_score:.1f}")

    # Test 5: Partial ROM
    print("\n[Test 5] Partial ROM (60% of target)")
    partial = np.concatenate([np.linspace(30, 72, 50), np.linspace(72, 30, 50)])
    scorer.start_session("test")
    r5 = scorer.score_rep(angles=partial, timestamps=ts, target_angle=120.0)
    print(f"  Total: {r5.total_score:.1f}, ROM: {r5.rom_score:.1f}")

    # Test 6: Butterworth filter effect
    print("\n[Test 6] Butterworth filter improves smoothness")
    filter_ = AngleFilter(cutoff_hz=6.0, fs=30.0)
    noisy = smooth + np.random.normal(0, 5, 100)  # Add noise
    filtered = filter_.filter(noisy)
    scorer.start_session("test")
    r_noisy = scorer.score_rep(angles=noisy, timestamps=ts, target_angle=120.0)
    scorer.start_session("test")
    r_filtered = scorer.score_rep(angles=filtered, timestamps=ts, target_angle=120.0)
    print(f"  Noisy: smooth={r_noisy.smoothness_score:.1f}, flow={r_noisy.flow_score:.1f}")
    print(f"  Filtered: smooth={r_filtered.smoothness_score:.1f}, flow={r_filtered.flow_score:.1f}")

    # Verify
    print("\n  Verification:")
    checks = [
        ("Smooth > Jerky total", r1.total_score > r2.total_score),
        ("Static stability > 70", r3.stability_score > 70),
        ("Ramp ROM > Partial ROM", r4.rom_score > r5.rom_score),
        ("Filtered smooth > Noisy smooth", r_filtered.smoothness_score >= r_noisy.smoothness_score),
    ]
    all_pass = True
    for name, ok in checks:
        status = "✓" if ok else "✗"
        print(f"    {status} {name}")
        if not ok:
            all_pass = False

    print(f"\n  {'ALL CHECKS PASSED' if all_pass else 'SOME CHECKS FAILED'}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--logic-only", action="store_true")
    parser.add_argument("--video-only", action="store_true")
    args = parser.parse_args()

    if not args.video_only:
        run_scoring_logic_test()

    if not args.logic_only:
        run_discrimination_test()
