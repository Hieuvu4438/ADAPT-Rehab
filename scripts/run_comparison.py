#!/usr/bin/env python3
"""
ADAPT-Rehab: Reference Video Comparison Pipeline

Compares user exercise video against a reference video using:
- 3D pose estimation (RTMW3D / MediaPipe fallback)
- Quaternion-based joint angle extraction
- Constrained DTW for rhythm matching
- 6-dimension scoring with reference-based targets

Usage:
    python scripts/run_comparison.py \
        --reference data/yoga_datasets/Yoga_Vid_Collected/Abhay_Tadasana.mp4 \
        --user data/yoga_datasets/Yoga_Vid_Collected/Ameya_Tadasana.mp4

Author: ADAPT-Rehab Team
Version: 3.1.0
"""

import argparse
import sys
import os
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.pose3d.base import create_estimator, PoseEstimator3D, Pose3DResult
from core.angle_filter import AngleFilter
from core.smoothness import SmoothnessAnalyzer
from core.dtw_constrained import constrained_dtw, weighted_constrained_dtw
from core.procrustes import align_skeleton_to_reference, extract_core_landmarks
from modules.scoring_v2 import EnhancedScorer, RepScoreV2
from modules.compensation import CompensationDetector
from modules.calibration import UserProfile, SafeMaxCalibrator
from modules.target_generator import compute_scale_factor, rescale_reference_motion


def extract_video_angles(
    estimator: PoseEstimator3D,
    video_path: str,
    angle_filter: AngleFilter,
    max_frames: int = 0,
    timestamp_offset_ms: int = 0,
) -> Dict[str, np.ndarray]:
    """Extract joint angle sequences from a video file.

    Args:
        estimator: Initialized pose estimator.
        video_path: Path to video file.
        angle_filter: Butterworth filter for smoothing.
        max_frames: Max frames to process (0 = all).
        timestamp_offset_ms: Offset for timestamps (to avoid monotonic issues).

    Returns:
        Dict mapping joint name -> filtered angle sequence (degrees).
        Also includes metadata: 'timestamps', 'raw_angles_per_frame', 'keypoints_sequence'.
    """
    import cv2

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if max_frames > 0:
        total = min(total, max_frames)

    print(f"  Processing: {Path(video_path).name} ({total} frames @ {fps:.1f} FPS)")

    # Collect per-frame data
    joint_names = [
        "left_shoulder", "right_shoulder",
        "left_elbow", "right_elbow",
        "left_hip", "right_hip",
        "left_knee", "right_knee",
    ]
    angle_sequences: Dict[str, List[float]] = {j: [] for j in joint_names}
    timestamps: List[float] = []
    raw_angles_per_frame: List[Dict[str, float]] = []
    keypoints_sequence: List[np.ndarray] = []

    frame_idx = 0
    while frame_idx < total:
        ret, frame = cap.read()
        if not ret:
            break

        ts_ms = int(frame_idx * (1000 / fps)) + timestamp_offset_ms
        ts_s = frame_idx / fps

        result = estimator.estimate(frame, ts_ms)

        if result.is_valid and result.joint_angles:
            for jname in joint_names:
                angle_sequences[jname].append(result.joint_angles.get(jname, 0.0))
            timestamps.append(ts_s)
            raw_angles_per_frame.append(result.joint_angles)
            if result.keypoints_3d is not None:
                keypoints_sequence.append(result.keypoints_3d)
        elif frame_idx < 3:
            print(f"    ⚠ Frame {frame_idx}: valid={result.is_valid}, "
                  f"angles={bool(result.joint_angles)}, "
                  f"error={result.error_message}")

        frame_idx += 1
        if frame_idx % 30 == 0:
            print(f"    Frame {frame_idx}/{total} (valid so far: {len(timestamps)})")

    cap.release()

    # Apply Butterworth filter to each joint
    filtered: Dict[str, np.ndarray] = {}
    for jname in joint_names:
        raw = np.array(angle_sequences[jname], dtype=np.float64)
        if len(raw) >= 10:
            filtered[jname] = angle_filter.filter(raw)
        else:
            filtered[jname] = raw

    filtered["timestamps"] = np.array(timestamps)
    filtered["_raw_angles_per_frame"] = raw_angles_per_frame
    filtered["_keypoints_sequence"] = keypoints_sequence

    print(f"  ✓ Extracted {len(timestamps)} frames, {len(joint_names)} joints")
    return filtered


def compute_reference_target(
    ref_angles: np.ndarray,
    user_max_angle: Optional[float] = None,
    challenge_factor: float = 0.05,
) -> Tuple[float, float]:
    """Compute target angle from reference video using target_generator formula.

    Personalization formula (from target_generator.py):
        θ_target = θ_ref × (θ_user_max / max(θ_ref)) × (1 + α)

    If no calibration data, uses reference max directly.

    Args:
        ref_angles: Reference angle sequence.
        user_max_angle: User's calibrated max ROM (optional, from Safe-Max calibration).
        challenge_factor: Extra challenge factor α (default 5%).

    Returns:
        Tuple of (target_angle, scale_factor).
    """
    ref_max = float(np.max(ref_angles))

    if user_max_angle is not None and user_max_angle > 0 and ref_max > 0:
        # Use target_generator's compute_scale_factor
        scale = compute_scale_factor(user_max_angle, ref_max, challenge_factor)
        target = ref_max * scale
        # Cap at reference max (never exceed)
        target = min(target, ref_max)
        return target, scale
    else:
        # No calibration — use reference max as target
        return ref_max, 1.0


def compute_procrustes_similarity(
    user_keypoints: List[np.ndarray],
    ref_keypoints: List[np.ndarray],
) -> float:
    """Compute Procrustes similarity between user and reference skeletons.

    Uses mean skeleton from each video, extracts core landmarks,
    and computes Procrustes alignment similarity.

    Args:
        user_keypoints: List of user 3D keypoints per frame.
        ref_keypoints: List of reference 3D keypoints per frame.

    Returns:
        Similarity score (0-100). Higher = more similar shape.
    """
    if not user_keypoints or not ref_keypoints:
        return 0.0

    # Compute mean skeleton for each video
    user_mean = np.mean(user_keypoints, axis=0)
    ref_mean = np.mean(ref_keypoints, axis=0)

    # Extract core landmarks (12 joints: shoulders, elbows, wrists, hips, knees, ankles)
    user_core = extract_core_landmarks(user_mean)
    ref_core = extract_core_landmarks(ref_mean)

    if user_core.shape[0] < 6 or ref_core.shape[0] < 6:
        return 0.0

    # Procrustes alignment
    result = align_skeleton_to_reference(user_core, ref_core)

    # Convert disparity to similarity (0-100)
    similarity = float(np.exp(-result.disparity * 10) * 100)
    return float(np.clip(similarity, 0, 100))


def compute_dtw_comparison(
    user_data: Dict[str, np.ndarray],
    ref_data: Dict[str, np.ndarray],
) -> Dict[str, float]:
    """Compute DTW similarity between user and reference for all joints.

    Args:
        user_data: User's angle sequences per joint.
        ref_data: Reference angle sequences per joint.

    Returns:
        Dict with 'overall_similarity', 'overall_distance', 'per_joint' details.
    """
    joint_names = [
        "left_shoulder", "right_shoulder",
        "left_elbow", "right_elbow",
        "left_hip", "right_hip",
        "left_knee", "right_knee",
    ]

    # Exercise-specific weights (default: equal)
    weights = {j: 1.0 for j in joint_names}

    user_seqs = {}
    ref_seqs = {}
    for jname in joint_names:
        if jname in user_data and jname in ref_data:
            u = user_data[jname]
            r = ref_data[jname]
            if len(u) >= 5 and len(r) >= 5:
                user_seqs[jname] = u
                ref_seqs[jname] = r

    if not user_seqs:
        return {"overall_similarity": 0.0, "overall_distance": float("inf"), "per_joint": {}}

    # Weighted constrained DTW
    similarity, total_dist, details = weighted_constrained_dtw(
        user_seqs, ref_seqs, weights=weights, window_percent=0.15
    )

    return {
        "overall_similarity": similarity,
        "overall_distance": total_dist,
        "per_joint": details,
    }


def run_full_comparison(
    reference_path: str,
    user_path: str,
    output_path: Optional[str] = None,
    max_frames: int = 0,
    pose_backend: str = "mediapipe_fallback",
    calibration_path: Optional[str] = None,
    primary_joint: str = "left_shoulder",
) -> dict:
    """Run full comparison pipeline between reference and user videos.

    Integrates all ADAPT-Rehab modules:
    - RTMW3D/MediaPipe 3D Pose Estimation
    - Quaternion-based Joint Angles
    - Safe-Max Calibration (if profile provided)
    - Target Generator (personalized rescaling)
    - Procrustes Skeleton Alignment
    - Constrained DTW (rhythm matching)
    - SPARC Smoothness Metric
    - 6-Dimension Scoring

    Args:
        reference_path: Path to reference exercise video.
        user_path: Path to user exercise video.
        output_path: Optional path to save JSON results.
        max_frames: Max frames per video (0 = all).
        pose_backend: Pose estimation backend.
        calibration_path: Optional path to user calibration JSON.
        primary_joint: Primary joint for scoring (default: left_shoulder).

    Returns:
        Complete results dict.
    """
    print("=" * 60)
    print("ADAPT-Rehab: Reference Video Comparison Pipeline")
    print("=" * 60)

    # 1. Initialize pose estimator
    print("\n[1/6] Initializing Pose Estimator...")
    estimator = None
    for candidate in [pose_backend, "mediapipe_fallback"]:
        try:
            est = create_estimator(candidate)
            if est.initialize():
                estimator = est
                print(f"  ✓ Using: {estimator.model_name}")
                break
        except Exception as e:
            print(f"  ⚠ {candidate} failed: {e}")

    if estimator is None:
        print("  ✗ No pose estimator available!")
        return {}

    angle_filter = AngleFilter(cutoff_hz=6.0, fs=30.0, order=4)
    smoothness_analyzer = SmoothnessAnalyzer()
    scorer = EnhancedScorer()

    # Load calibration if provided
    user_profile = None
    user_max_angle = None
    if calibration_path and os.path.exists(calibration_path):
        print(f"\n  Loading calibration: {calibration_path}")
        try:
            with open(calibration_path, "r") as f:
                profile_data = json.load(f)
            user_profile = UserProfile.from_dict(profile_data)
            # Get max angle for the primary joint
            from core.kinematics import JointType
            for jt in JointType:
                if jt.value == primary_joint or primary_joint in jt.value:
                    user_max_angle = user_profile.get_max_angle(jt)
                    break
            if user_max_angle:
                print(f"  ✓ Calibration: {primary_joint} max = {user_max_angle:.1f}°")
            else:
                print(f"  ⚠ No calibration data for {primary_joint}")
        except Exception as e:
            print(f"  ⚠ Calibration load failed: {e}")

    # 2. Extract reference video angles
    print("\n[2/6] Extracting Reference Video Angles...")
    ref_data = extract_video_angles(estimator, reference_path, angle_filter, max_frames)

    # 3. Extract user video angles — re-create estimator to reset internal timestamp state
    print("\n[3/6] Extracting User Video Angles...")
    try:
        estimator.close()
    except Exception:
        pass
    estimator2 = create_estimator(pose_backend)
    if not estimator2.initialize():
        # Fallback to the same backend that worked
        estimator2 = create_estimator("mediapipe_fallback")
        estimator2.initialize()
    user_data = extract_video_angles(estimator2, user_path, angle_filter, max_frames,
                                      timestamp_offset_ms=0)

    # 4. DTW Comparison + Procrustes Alignment
    print("\n[4/6] Computing DTW Comparison...")
    dtw_results = compute_dtw_comparison(user_data, ref_data)
    print(f"  Overall DTW Similarity: {dtw_results['overall_similarity']:.1f}%")
    for jname, detail in dtw_results.get("per_joint", {}).items():
        print(f"    {jname}: distance={detail['normalized_distance']:.4f}, "
              f"path_len={detail['path_length']}")

    # 4b. Procrustes skeleton alignment
    print("\n[5/6] Computing Procrustes Skeleton Alignment...")
    user_kps = user_data.get("_keypoints_sequence", [])
    ref_kps = ref_data.get("_keypoints_sequence", [])
    procrustes_sim = compute_procrustes_similarity(user_kps, ref_kps)
    print(f"  Procrustes Similarity: {procrustes_sim:.1f}%")

    # 5. Scoring with reference comparison
    print("\n[6/6] Scoring with Reference Comparison...")
    scorer.start_session("comparison")

    # Use the primary joint for single-joint scoring
    user_angles = user_data.get(primary_joint, np.array([]))
    ref_angles = ref_data.get(primary_joint, np.array([]))
    timestamps = user_data.get("timestamps", np.array([]))

    # Compute personalized target using target_generator formula
    target_angle, scale_factor = compute_reference_target(
        ref_angles, user_max_angle=user_max_angle
    )
    if user_max_angle:
        print(f"  Personalized target: {target_angle:.1f}° "
              f"(scale={scale_factor:.3f}, user_max={user_max_angle:.1f}°, "
              f"ref_max={float(np.max(ref_angles)):.1f}°)")
    else:
        print(f"  Target (no calibration): {target_angle:.1f}°")

    # Build multi-joint dicts for weighted DTW
    joint_names = [
        "left_shoulder", "right_shoulder",
        "left_elbow", "right_elbow",
        "left_hip", "right_hip",
        "left_knee", "right_knee",
    ]
    multi_user = {j: user_data[j] for j in joint_names if j in user_data and len(user_data[j]) >= 5}
    multi_ref = {j: ref_data[j] for j in joint_names if j in ref_data and len(ref_data[j]) >= 5}

    # Score
    score = scorer.score_rep(
        angles=user_angles,
        timestamps=timestamps,
        target_angle=target_angle,
        left_angles=user_data.get("left_shoulder"),
        right_angles=user_data.get("right_shoulder"),
        pose_sequence=user_data.get("_keypoints_sequence", [])[-30:],
        ref_angles=ref_angles,
        ref_left_angles=ref_data.get("left_shoulder"),
        ref_right_angles=ref_data.get("right_shoulder"),
        multi_joint_user=multi_user,
        multi_joint_ref=multi_ref,
    )

    # Build results
    results = {
        "pipeline": "ADAPT-Rehab v3.1 Reference Comparison (Full Integration)",
        "reference_video": str(reference_path),
        "user_video": str(user_path),
        "pose_model": estimator.model_name,
        "modules_used": [
            "RTMW3D/MediaPipe 3D Pose",
            "Quaternion Kinematics",
            "Butterworth Filter (4th order, 6Hz)",
            "Safe-Max Calibration" if user_profile else "Calibration (not provided)",
            "Target Generator (personalized rescaling)" if user_max_angle else "Target (reference direct)",
            "Procrustes Skeleton Alignment",
            "Constrained DTW (Sakoe-Chiba)",
            "SPARC Smoothness",
            "Compensation Detection",
            "Fatigue Analysis (4 indicators)",
            "6-Dimension Scoring",
        ],
        "calibration": {
            "provided": user_profile is not None,
            "user_max_angle": user_max_angle,
            "scale_factor": scale_factor,
            "challenge_factor": 0.05,
            "formula": "θ_target = θ_ref × (θ_user_max / max(θ_ref)) × (1 + α)",
        },
        "reference": {
            "total_frames": len(ref_data.get("timestamps", [])),
            "max_angle": float(np.max(ref_angles)) if len(ref_angles) > 0 else 0,
            "mean_angle": float(np.mean(ref_angles)) if len(ref_angles) > 0 else 0,
            "duration_s": float(ref_data["timestamps"][-1]) if len(ref_data.get("timestamps", [])) > 0 else 0,
        },
        "user": {
            "total_frames": len(user_data.get("timestamps", [])),
            "max_angle": float(np.max(user_angles)) if len(user_angles) > 0 else 0,
            "mean_angle": float(np.mean(user_angles)) if len(user_angles) > 0 else 0,
            "duration_s": float(user_data["timestamps"][-1]) if len(user_data.get("timestamps", [])) > 0 else 0,
        },
        "target_angle": target_angle,
        "procrustes_similarity": procrustes_sim,
        "dtw_comparison": {
            "overall_similarity": dtw_results["overall_similarity"],
            "overall_distance": dtw_results["overall_distance"],
            "per_joint": {
                j: {
                    "distance": d["distance"],
                    "normalized_distance": d["normalized_distance"],
                    "path_length": d["path_length"],
                }
                for j, d in dtw_results.get("per_joint", {}).items()
            },
        },
        "scoring": score.to_dict(),
        "score_breakdown": {
            "ROM (25%)": f"{score.rom_score:.1f} — max_angle vs reference target ({target_angle:.1f}°)",
            "Stability (15%)": f"{score.stability_score:.1f} — angle variability in hold phase",
            "Flow (20%)": f"{score.flow_score:.1f} — DTW similarity with reference" if score.ref_comparison_used else f"{score.flow_score:.1f} — velocity smoothness (no reference)",
            "Symmetry (15%)": f"{score.symmetry_score:.1f} — left-right balance vs reference",
            "Compensation (15%)": f"{score.compensation_score:.1f} — compensatory movements",
            "Smoothness (10%)": f"{score.smoothness_score:.1f} — SPARC metric",
        },
    }

    # Print summary
    print("\n" + "=" * 60)
    print("COMPARISON RESULTS")
    print("=" * 60)
    print(f"Reference: {Path(reference_path).name}")
    print(f"User:      {Path(user_path).name}")
    print(f"Model:     {estimator.model_name}")
    print(f"\nReference: {results['reference']['total_frames']} frames, "
          f"max={results['reference']['max_angle']:.1f}°, "
          f"duration={results['reference']['duration_s']:.1f}s")
    print(f"User:      {results['user']['total_frames']} frames, "
          f"max={results['user']['max_angle']:.1f}°, "
          f"duration={results['user']['duration_s']:.1f}s")
    print(f"Target:    {target_angle:.1f}°")
    print(f"\nDTW Similarity: {dtw_results['overall_similarity']:.1f}%")
    print(f"Procrustes Similarity: {procrustes_sim:.1f}%")
    if user_max_angle:
        print(f"Calibration: user_max={user_max_angle:.1f}°, scale={scale_factor:.3f}")
    print(f"\n--- SCORE BREAKDOWN ---")
    for dim, detail in results["score_breakdown"].items():
        print(f"  {dim}")
    print(f"\n  TOTAL: {score.total_score:.1f}/100")
    print(f"  Reference comparison: {'YES' if score.ref_comparison_used else 'NO'}")
    print("=" * 60)

    # Save results
    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to: {output_path}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="ADAPT-Rehab: Compare user exercise vs reference video"
    )
    parser.add_argument("--reference", "-r", required=True,
                        help="Path to reference exercise video")
    parser.add_argument("--user", "-u", required=True,
                        help="Path to user exercise video")
    parser.add_argument("--output", "-o", default="evaluation/output/comparison_results.json",
                        help="Output JSON path")
    parser.add_argument("--max-frames", type=int, default=0,
                        help="Max frames per video (0 = all)")
    parser.add_argument("--pose-backend", type=str, default="mediapipe_fallback",
                        choices=["rtmw3d", "mediapipe_fallback"],
                        help="Pose estimation backend")
    parser.add_argument("--calibration", type=str, default=None,
                        help="Path to user calibration JSON (from Safe-Max calibration)")
    parser.add_argument("--joint", type=str, default="left_shoulder",
                        help="Primary joint for scoring (default: left_shoulder)")
    args = parser.parse_args()

    if not os.path.exists(args.reference):
        print(f"Error: Reference video not found: {args.reference}")
        sys.exit(1)
    if not os.path.exists(args.user):
        print(f"Error: User video not found: {args.user}")
        sys.exit(1)

    results = run_full_comparison(
        reference_path=args.reference,
        user_path=args.user,
        output_path=args.output,
        max_frames=args.max_frames,
        pose_backend=args.pose_backend,
        calibration_path=args.calibration,
        primary_joint=args.joint,
    )

    if not results:
        sys.exit(1)


if __name__ == "__main__":
    main()
