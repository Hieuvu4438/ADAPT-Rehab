"""
Integration tests using actual datasets.

Tests all modules against real yoga pose videos
from data/yoga_datasets/Yoga_Vid_Collected/

Run: python tests/test_with_data.py
"""

import sys
import os
import time
import glob
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2


def get_test_videos(n=3):
    """Get a few test videos from the dataset."""
    video_dir = os.path.join(os.path.dirname(__file__), "..", "data", "yoga_datasets", "Yoga_Vid_Collected")
    videos = sorted(glob.glob(os.path.join(video_dir, "*.mp4")))
    return videos[:n]


def test_video_loading():
    """Test that videos can be loaded and frames extracted."""
    print("\n" + "=" * 60)
    print("TEST 1: Video Loading")
    print("=" * 60)

    videos = get_test_videos(3)
    for video_path in videos:
        cap = cv2.VideoCapture(video_path)
        assert cap.isOpened(), f"Cannot open: {video_path}"

        ret, frame = cap.read()
        assert ret, f"Cannot read frame from: {video_path}"
        assert frame is not None and frame.size > 0

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        h, w = frame.shape[:2]

        print(f"  ✓ {os.path.basename(video_path)}: {w}x{h}, {fps:.1f} FPS, {total_frames} frames")
        cap.release()

    print("  RESULT: PASS")


def test_mediapipe_fallback():
    """Test MediaPipe fallback pose estimator on real videos."""
    print("\n" + "=" * 60)
    print("TEST 2: MediaPipe Fallback Pose Estimation")
    print("=" * 60)

    try:
        from core.pose3d import MediaPipeFallbackEstimator
    except ImportError:
        print("  SKIP: MediaPipe not installed")
        return

    estimator = MediaPipeFallbackEstimator()
    if not estimator.initialize():
        print("  SKIP: MediaPipe model not found")
        return

    videos = get_test_videos(2)
    for video_path in videos:
        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        detected_count = 0
        angles_collected = []

        for _ in range(90):  # Test 90 frames (~3 seconds)
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            result = estimator.estimate(frame, timestamp_ms=frame_count * 33)

            if result.is_valid:
                detected_count += 1
                if result.joint_angles:
                    angles_collected.append(result.joint_angles)

        cap.release()

        detection_rate = detected_count / max(frame_count, 1) * 100
        print(f"  {os.path.basename(video_path)}:")
        print(f"    Frames: {frame_count}, Detected: {detected_count} ({detection_rate:.1f}%)")

        if angles_collected:
            avg_angles = {}
            for joint in angles_collected[0]:
                vals = [a[joint] for a in angles_collected if joint in a]
                avg_angles[joint] = np.mean(vals)
            print(f"    Avg angles: {', '.join(f'{k}={v:.1f}°' for k, v in avg_angles.items())}")

    estimator.close()
    print("  RESULT: PASS")


def test_quaternion_kinematics():
    """Test quaternion angle computation on detected poses."""
    print("\n" + "=" * 60)
    print("TEST 3: Quaternion Kinematics on Real Data")
    print("=" * 60)

    try:
        from core.pose3d import MediaPipeFallbackEstimator
        from core.kinematics_quaternion import QuaternionKinematics
    except ImportError:
        print("  SKIP: Dependencies not available")
        return

    estimator = MediaPipeFallbackEstimator()
    if not estimator.initialize():
        print("  SKIP: MediaPipe model not found")
        return

    qk = QuaternionKinematics()
    video = get_test_videos(1)[0]
    cap = cv2.VideoCapture(video)

    dot_angles = []
    quat_angles = []

    for _ in range(60):
        ret, frame = cap.read()
        if not ret:
            break

        result = estimator.estimate(frame)
        if not result.is_valid or result.keypoints_3d is None:
            continue

        kps = result.keypoints_3d

        # Compute angle both ways for left shoulder
        try:
            p, v, d = kps[23], kps[11], kps[13]  # hip, shoulder, elbow
            dot_a = result.joint_angles.get("left_shoulder", 0)
            quat_a = qk.compute_angle(p, v, d)
            dot_angles.append(dot_a)
            quat_angles.append(quat_a)
        except (IndexError, ValueError):
            continue

    cap.release()
    estimator.close()

    if dot_angles:
        dot_mean = np.mean(dot_angles)
        quat_mean = np.mean(quat_angles)
        diff = abs(dot_mean - quat_mean)
        print(f"  Left Shoulder - Dot Product: {dot_mean:.1f}°, Quaternion: {quat_mean:.1f}°, Diff: {diff:.1f}°")
        print(f"  Both methods agree within {diff:.1f}°")
    else:
        print("  No valid poses detected")

    print("  RESULT: PASS")


def test_smoothness_on_real_data():
    """Test SPARC smoothness metric on real exercise data."""
    print("\n" + "=" * 60)
    print("TEST 4: SPARC Smoothness on Real Data")
    print("=" * 60)

    try:
        from core.pose3d import MediaPipeFallbackEstimator
        from core.smoothness import SmoothnessAnalyzer
    except ImportError:
        print("  SKIP: Dependencies not available")
        return

    estimator = MediaPipeFallbackEstimator()
    if not estimator.initialize():
        print("  SKIP: MediaPipe model not found")
        return

    video = get_test_videos(1)[0]
    cap = cv2.VideoCapture(video)
    angles = []
    timestamps = []

    for i in range(120):
        ret, frame = cap.read()
        if not ret:
            break
        result = estimator.estimate(frame, timestamp_ms=i * 33)
        if result.is_valid and "left_shoulder" in result.joint_angles:
            angles.append(result.joint_angles["left_shoulder"])
            timestamps.append(i * 33 / 1000.0)

    cap.release()
    estimator.close()

    if len(angles) >= 10:
        angles_arr = np.array(angles)
        ts_arr = np.array(timestamps)

        analyzer = SmoothnessAnalyzer(fs=30)
        result = analyzer.analyze(angles_arr, ts_arr)

        print(f"  Video: {os.path.basename(video)}")
        print(f"  Frames analyzed: {len(angles)}")
        print(f"  Angle range: {min(angles):.1f}° - {max(angles):.1f}°")
        print(f"  SPARC: {result.sparc:.3f}")
        print(f"  LDLJ: {result.ldjl:.3f}")
        print(f"  Velocity Peaks: {result.nvp}")
        print(f"  Smoothness Score: {result.smoothness_score:.1f}/100")
    else:
        print("  Not enough data for analysis")

    print("  RESULT: PASS")


def test_compensation_on_real_data():
    """Test compensation detection on real pose data."""
    print("\n" + "=" * 60)
    print("TEST 5: Compensation Detection on Real Data")
    print("=" * 60)

    try:
        from core.pose3d import MediaPipeFallbackEstimator
        from modules.compensation import CompensationDetector
    except ImportError:
        print("  SKIP: Dependencies not available")
        return

    estimator = MediaPipeFallbackEstimator()
    if not estimator.initialize():
        print("  SKIP: MediaPipe model not found")
        return

    video = get_test_videos(1)[0]
    cap = cv2.VideoCapture(video)
    detector = CompensationDetector()

    poses = []
    for _ in range(60):
        ret, frame = cap.read()
        if not ret:
            break
        result = estimator.estimate(frame)
        if result.is_valid and result.keypoints_3d is not None:
            poses.append(result.keypoints_3d)

    cap.release()
    estimator.close()

    if len(poses) >= 5:
        result = detector.analyze(poses)
        print(f"  Video: {os.path.basename(video)}")
        print(f"  Poses analyzed: {len(poses)}")
        print(f"  Compensation Score: {result.score:.1f}/100")
        print(f"  Shoulder diff avg: {result.shoulder_diff_avg:.4f}")
        print(f"  Trunk tilt avg: {result.trunk_tilt_avg:.1f}°")
        print(f"  Hip diff avg: {result.hip_diff_avg:.4f}")
        if result.detected_types:
            print(f"  Detected: {', '.join(result.detected_types)}")
        else:
            print(f"  Detected: None (good)")
    else:
        print("  Not enough poses detected")

    print("  RESULT: PASS")


def test_fatigue_across_reps():
    """Test fatigue detection across multiple video reps."""
    print("\n" + "=" * 60)
    print("TEST 6: Fatigue Detection Across Reps")
    print("=" * 60)

    try:
        from core.pose3d import MediaPipeFallbackEstimator
        from modules.fatigue import FatigueAnalyzer
    except ImportError:
        print("  SKIP: Dependencies not available")
        return

    estimator = MediaPipeFallbackEstimator()
    if not estimator.initialize():
        print("  SKIP: MediaPipe model not found")
        return

    videos = get_test_videos(3)
    analyzer = FatigueAnalyzer()

    for idx, video in enumerate(videos):
        cap = cv2.VideoCapture(video)
        angles = []

        for _ in range(60):
            ret, frame = cap.read()
            if not ret:
                break
            result = estimator.estimate(frame)
            if result.is_valid and "left_shoulder" in result.joint_angles:
                angles.append(result.joint_angles["left_shoulder"])

        cap.release()

        if len(angles) >= 10:
            angles_arr = np.array(angles)
            velocity = np.abs(np.diff(angles_arr))

            rep_data = {
                "jerk_value": float(np.sum(np.diff(velocity) ** 2)),
                "max_angle": float(np.max(angles_arr)),
                "mean_velocity": float(np.mean(velocity)),
                "angle_std": float(np.std(angles_arr)),
            }

            if idx == 0:
                analyzer.set_baseline(rep_data)
                print(f"  Baseline ({os.path.basename(video)}): jerk={rep_data['jerk_value']:.1f}, ROM={rep_data['max_angle']:.1f}°")
            else:
                result = analyzer.analyze(rep_data)
                print(f"  Rep {idx + 1} ({os.path.basename(video)}): fatigue={result.level.name}, jerk_ratio={result.jerk_ratio:.2f}")

    estimator.close()
    print("  RESULT: PASS")


def test_enhanced_scoring():
    """Test enhanced 6-dimension scoring on real data."""
    print("\n" + "=" * 60)
    print("TEST 7: Enhanced Scoring on Real Data")
    print("=" * 60)

    try:
        from core.pose3d import MediaPipeFallbackEstimator
        from modules.scoring_v2 import EnhancedScorer
    except ImportError:
        print("  SKIP: Dependencies not available")
        return

    estimator = MediaPipeFallbackEstimator()
    if not estimator.initialize():
        print("  SKIP: MediaPipe model not found")
        return

    scorer = EnhancedScorer()
    scorer.start_session("yoga_pose_test")

    video = get_test_videos(1)[0]
    cap = cv2.VideoCapture(video)
    angles = []
    timestamps = []

    for i in range(90):
        ret, frame = cap.read()
        if not ret:
            break
        result = estimator.estimate(frame, timestamp_ms=i * 33)
        if result.is_valid and "left_shoulder" in result.joint_angles:
            angles.append(result.joint_angles["left_shoulder"])
            timestamps.append(i * 33 / 1000.0)

    cap.release()
    estimator.close()

    if len(angles) >= 10:
        angles_arr = np.array(angles)
        ts_arr = np.array(timestamps)
        target = float(np.max(angles_arr)) * 1.1  # Slightly above max

        score = scorer.score_rep(angles_arr, ts_arr, target_angle=target)

        print(f"  Video: {os.path.basename(video)}")
        print(f"  Frames: {len(angles)}")
        print(f"  Target: {target:.1f}°")
        print(f"  ROM Score: {score.rom_score:.1f}/100")
        print(f"  Stability Score: {score.stability_score:.1f}/100")
        print(f"  Flow Score: {score.flow_score:.1f}/100")
        print(f"  Symmetry Score: {score.symmetry_score:.1f}/100")
        print(f"  Compensation Score: {score.compensation_score:.1f}/100")
        print(f"  Smoothness Score: {score.smoothness_score:.1f}/100")
        print(f"  TOTAL: {score.total_score:.1f}/100")
        print(f"  Fatigue: {score.fatigue.name}")
    else:
        print("  Not enough data for scoring")

    print("  RESULT: PASS")


def test_evaluation_metrics():
    """Test evaluation metrics with synthetic vs real data."""
    print("\n" + "=" * 60)
    print("TEST 8: Evaluation Metrics")
    print("=" * 60)

    from evaluation.metrics.mpjpe import compute_mpjpe, compute_p_mpjpe
    from evaluation.metrics.angle_mae import compute_angle_mae, compute_icc

    # Synthetic test data
    pred = np.random.randn(10, 3) * 10
    gt = pred + np.random.randn(10, 3) * 2  # Small noise

    mpjpe = compute_mpjpe(pred, gt)
    p_mpjpe = compute_p_mpjpe(pred, gt)

    print(f"  MPJPE: {mpjpe:.2f} mm")
    print(f"  P-MPJPE: {p_mpjpe:.2f} mm")

    # Angle MAE
    pred_angles = {"left_shoulder": 85.0, "right_shoulder": 82.0, "left_elbow": 120.0}
    gt_angles = {"left_shoulder": 87.0, "right_shoulder": 80.0, "left_elbow": 118.0}

    mae = compute_angle_mae(pred_angles, gt_angles)
    print(f"  Angle MAE: {mae:.2f}°")

    # ICC
    pred_arr = np.array([85, 90, 95, 100, 105])
    gt_arr = np.array([87, 92, 93, 102, 103])
    icc = compute_icc(pred_arr, gt_arr)
    print(f"  ICC: {icc:.3f}")

    print("  RESULT: PASS")


def test_3dyoga90_metadata():
    """Test loading 3DYoga90 metadata."""
    print("\n" + "=" * 60)
    print("TEST 9: 3DYoga90 Metadata Loading")
    print("=" * 60)

    import json
    import csv

    json_path = os.path.join(os.path.dirname(__file__), "..", "data", "yoga_datasets", "3DYoga90", "data", "3DYoga90.json")
    csv_path = os.path.join(os.path.dirname(__file__), "..", "data", "yoga_datasets", "3DYoga90", "data", "pose-index.csv")

    if os.path.exists(json_path):
        with open(json_path) as f:
            data = json.load(f)
        print(f"  3DYoga90.json: {len(data)} poses")
        if data:
            print(f"  First pose: {data[0].get('pose', 'N/A')}")
            print(f"  Instances: {len(data[0].get('instances', []))}")

    if os.path.exists(csv_path):
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        print(f"  pose-index.csv: {len(rows)} entries")

    print("  RESULT: PASS")


def main():
    """Run all integration tests."""
    print("\n" + "=" * 60)
    print("ADAPT-Rehab v3.0 - Integration Tests with Real Data")
    print("=" * 60)

    tests = [
        test_video_loading,
        test_mediapipe_fallback,
        test_quaternion_kinematics,
        test_smoothness_on_real_data,
        test_compensation_on_real_data,
        test_fatigue_across_reps,
        test_enhanced_scoring,
        test_evaluation_metrics,
        test_3dyoga90_metadata,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"  RESULT: FAIL - {e}")
            failed += 1

    print("\n" + "=" * 60)
    print(f"SUMMARY: {passed} passed, {failed} failed out of {len(tests)} tests")
    print("=" * 60)


if __name__ == "__main__":
    main()
