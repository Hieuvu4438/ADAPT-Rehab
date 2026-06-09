"""
Phase 3: Face Analysis Tests (OpenFace 3.0 + AU-based State Detection).

Tests:
1. MediaPipe Face Mesh detection
2. Facial State Detector (AU-based formulas)
3. OpenFace 3.0 analyzer integration
4. PSPI pain detection (Prkachin & Solomon, 2008)
5. PERCLOS fatigue detection (Wierwille et al., 1994)
6. Engagement/boredom detection (Whitehill et al., 2014)

Run: python tests/test_phase3_face.py
"""

import sys
import os
import glob
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def get_test_videos(n=3):
    """Get test videos."""
    video_dir = os.path.join(os.path.dirname(__file__), "..", "data", "yoga_datasets", "Yoga_Vid_Collected")
    videos = sorted(glob.glob(os.path.join(video_dir, "*.mp4")))
    return videos[:n]


def test_face_detector():
    """Test MediaPipe Face Mesh detection on real videos."""
    print("\n" + "=" * 60)
    print("TEST 1: Face Detection (MediaPipe Face Mesh)")
    print("=" * 60)

    from modules.perception import FaceDetector

    detector = FaceDetector()
    if not detector.initialize():
        print("  SKIP: Face model not found (need face_landmarker.task)")
        print("  Download: https://developers.google.com/mediapipe/solutions/vision/face_landmarker")
        return False

    videos = get_test_videos(3)
    total_frames = 0
    total_detected = 0

    for video_path in videos:
        cap = cv2.VideoCapture(video_path)
        detected = 0
        frames = 0

        for _ in range(60):  # Test 60 frames per video
            ret, frame = cap.read()
            if not ret:
                break
            frames += 1
            result = detector.detect(frame)
            if result.is_valid:
                detected += 1

        cap.release()

        rate = detected / max(frames, 1) * 100
        print(f"  {os.path.basename(video_path)}: {detected}/{frames} ({rate:.1f}%)")
        total_frames += frames
        total_detected += detected

    overall_rate = total_detected / max(total_frames, 1) * 100
    print(f"  Overall: {total_detected}/{total_frames} ({overall_rate:.1f}%)")

    detector.close()
    print("  RESULT: PASS")
    return True


def test_facial_state_detector():
    """Test AU-based facial state detection with synthetic AU data."""
    print("\n" + "=" * 60)
    print("TEST 2: Facial State Detection (AU-based Formulas)")
    print("=" * 60)

    from modules.perception.facial_state_detector import (
        FacialStateDetector, FacialState, AUData,
        PSPICalculator, PERCLOSCalculator, EngagementCalculator,
        BlinkDetector, YawnDetector, EARCalculator,
    )

    # Test PSPI calculator
    print("\n  [PSPI Pain Calculator - Prkachin & Solomon, 2008]")
    pspi = PSPICalculator()

    # No pain case
    au_no_pain = AUData(au4=0.0, au6=0.0, au9=0.0, au43_approx=0.0)
    score_no_pain = pspi.compute(au_no_pain)
    print(f"    No pain: PSPI = {score_no_pain:.1f} (expected 0.0)")
    assert score_no_pain == 0.0

    # Moderate pain case
    # Corrected PSPI formula: AU4 + 2*AU6 + AU9 + 2*AU43
    # = 2.0 + 2*1.5 + 1.0 + 2*0.0 = 6.0
    au_moderate = AUData(au4=2.0, au6=1.5, au9=1.0, au43_approx=0.0)
    score_moderate = pspi.compute(au_moderate)
    print(f"    Moderate pain: PSPI = {score_moderate:.1f} (expected 6.0)")
    assert abs(score_moderate - 6.0) < 0.1

    # Severe pain case
    # Corrected PSPI formula: AU4 + 2*AU6 + AU9 + 2*AU43
    # = 5.0 + 2*4.0 + 3.0 + 2*1.0 = 18.0
    au_severe = AUData(au4=5.0, au6=4.0, au9=3.0, au43_approx=1.0)
    score_severe = pspi.compute(au_severe)
    print(f"    Severe pain: PSPI = {score_severe:.1f} (expected 18.0)")
    assert abs(score_severe - 18.0) < 0.1

    level, conf = pspi.classify(score_severe)
    print(f"    Classification: {level} (confidence: {conf:.1f})")
    assert level == "SEVERE"

    # Test PERCLOS calculator
    print("\n  [PERCLOS Fatigue Calculator - Wierwille et al., 1994]")
    perclos = PERCLOSCalculator(window_seconds=5.0, fps=30.0)

    # Simulate 150 frames (5 seconds at 30fps) with no eye closure
    for _ in range(150):
        val = perclos.update(0.0)
    print(f"    Alert state: PERCLOS = {val:.1f}% (expected ~0%)")
    assert val < 5.0

    # Simulate eye closure
    perclos2 = PERCLOSCalculator(window_seconds=5.0, fps=30.0)
    for _ in range(150):
        val2 = perclos2.update(4.0)  # High AU43 = eyes closed
    print(f"    Drowsy state: PERCLOS = {val2:.1f}% (expected ~100%)")
    assert val2 > 80.0

    # Test Engagement calculator
    print("\n  [Engagement Calculator - Whitehill et al., 2014]")
    engagement = EngagementCalculator()

    # Engaged (smiling)
    au_engaged = AUData(au12=3.0, au1=0.0)
    score_engaged = engagement.compute_engagement(au_engaged)
    print(f"    Engaged (smile): {score_engaged:.2f} (expected > 0.5)")
    assert score_engaged > 0.5

    # Disengaged (sad)
    au_disengaged = AUData(au12=0.0, au1=3.0)
    score_disengaged = engagement.compute_engagement(au_disengaged)
    print(f"    Disengaged (sad): {score_disengaged:.2f} (expected < 0.5)")
    assert score_disengaged < 0.5

    # Test full state detector pipeline
    print("\n  [Full State Detector Pipeline]")
    detector = FacialStateDetector(fps=30.0)

    # Normal state
    result_normal = detector.process_frame({"AU1": 0.5, "AU12": 1.0, "AU4": 0.0, "AU6": 0.0, "AU9": 0.0, "AU25": 0.0, "AU26": 0.0})
    print(f"    Normal: state={result_normal.state.value}, conf={result_normal.confidence:.2f}")
    assert result_normal.state == FacialState.NORMAL

    # Pain state (high AU4 + AU6 + AU9)
    result_pain = detector.process_frame({"AU1": 0.0, "AU12": 0.0, "AU4": 4.0, "AU6": 3.0, "AU9": 2.0, "AU25": 0.0, "AU26": 0.0})
    print(f"    Pain: state={result_pain.state.value}, PSPI={result_pain.pspi_raw:.1f}, conf={result_pain.confidence:.2f}")
    assert result_pain.state == FacialState.PAIN

    print("\n  RESULT: PASS")
    return True


def test_openface_analyzer():
    """Test OpenFace 3.0 analyzer integration."""
    print("\n" + "=" * 60)
    print("TEST 3: OpenFace 3.0 Analyzer Integration")
    print("=" * 60)

    from modules.perception.openface_analyzer import OpenFaceAnalyzer

    analyzer = OpenFaceAnalyzer(device="cpu")
    if not analyzer.initialize():
        print("  SKIP: OpenFace 3.0 not available")
        return False

    # Test with blank frame (should handle gracefully)
    blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    result = analyzer.analyze(blank_frame, timestamp_ms=0)

    print(f"  Blank frame: valid={result.is_valid}")
    if result.is_valid:
        print(f"    Emotion: {result.emotion_label}")
        if result.state_result:
            print(f"    State: {result.state_result.state.value}")

    analyzer.close()
    print("  RESULT: PASS")
    return True


def test_blink_detector():
    """Test blink detection from EAR pattern."""
    print("\n" + "=" * 60)
    print("TEST 4: Blink Detection (EAR-based)")
    print("=" * 60)

    from modules.perception.facial_state_detector import BlinkDetector

    detector = BlinkDetector(fps=30.0, ear_threshold=0.2)

    # Simulate normal blinks (brief EAR drops)
    blink_count = 0
    for i in range(900):  # 30 seconds
        if i % 45 == 0:  # ~40 blinks/min (elevated)
            # Blink: 3 frames of low EAR
            for j in range(3):
                result = detector.update(0.1)  # Closed
                if result:
                    blink_count += 1
        else:
            detector.update(0.3)  # Open

    blink_rate = detector.get_blink_rate(window_seconds=30.0)
    print(f"  Simulated blinks: {blink_count}")
    print(f"  Blink rate: {blink_rate:.1f} blinks/min")
    print(f"  Expected: ~40 blinks/min (elevated fatigue)")
    assert blink_rate > 20.0, f"Blink rate too low: {blink_rate}"

    print("  RESULT: PASS")
    return True


def test_yawn_detector():
    """Test yawn detection from AU25+AU26."""
    print("\n" + "=" * 60)
    print("TEST 5: Yawn Detection (AU25 + AU26)")
    print("=" * 60)

    from modules.perception.facial_state_detector import YawnDetector

    detector = YawnDetector(fps=30.0, au_threshold=1.5, min_duration_s=1.0)

    # Simulate yawns (AU25 + AU26 active for > 1 second)
    # Note: YawnDetector returns event when yawn ENDS (transition to non-yawn),
    # so we must check return value in ALL branches.
    yawn_count = 0
    for i in range(3000):  # 100 seconds
        if i % 600 == 0:  # Yawn every 20 seconds
            for j in range(90):  # 3-second yawn
                result = detector.update(au25=3.0, au26=3.0)
                if result:
                    yawn_count += 1
        else:
            result = detector.update(au25=0.0, au26=0.0)
            if result:
                yawn_count += 1

    yawn_freq = detector.get_yawn_frequency(window_seconds=100.0)
    print(f"  Simulated yawns: {yawn_count}")
    print(f"  Yawn frequency: {yawn_freq:.2f} yawns/min")
    print(f"  Expected: ~3 yawns/min")
    assert yawn_count >= 3, f"Yawn count too low: {yawn_count}"

    print("  RESULT: PASS")
    return True


def test_body_state_detector():
    """Test body state detection with synthetic keypoints."""
    print("\n" + "=" * 60)
    print("TEST 6: Body State Detection (RTMW3D Keypoints)")
    print("=" * 60)

    from modules.analysis.body_state_detector import (
        BodyStateDetector, BodyState, JointAngleCalculator, KeypointIndex,
    )

    detector = BodyStateDetector(fps=30.0)

    # Create synthetic 3D keypoints (standing pose)
    keypoints = np.zeros((20, 3))
    # Head
    keypoints[0] = [0, 1.7, 0]  # Nose
    # Shoulders
    keypoints[5] = [-0.2, 1.5, 0]  # Left shoulder
    keypoints[6] = [0.2, 1.5, 0]   # Right shoulder
    # Elbows
    keypoints[7] = [-0.3, 1.2, 0]  # Left elbow
    keypoints[8] = [0.3, 1.2, 0]   # Right elbow
    # Wrists
    keypoints[9] = [-0.3, 0.9, 0]  # Left wrist
    keypoints[10] = [0.3, 0.9, 0]  # Right wrist
    # Hips
    keypoints[11] = [-0.15, 0.9, 0]  # Left hip
    keypoints[12] = [0.15, 0.9, 0]   # Right hip
    # Knees
    keypoints[13] = [-0.15, 0.5, 0]  # Left knee
    keypoints[14] = [0.15, 0.5, 0]   # Right knee
    # Ankles
    keypoints[15] = [-0.15, 0.0, 0]  # Left ankle
    keypoints[16] = [0.15, 0.0, 0]   # Right ankle
    # Neck and pelvis
    keypoints[17] = [0, 1.6, 0]   # Neck
    keypoints[18] = [0, 1.8, 0]   # Head top
    keypoints[19] = [0, 0.9, 0]   # Pelvis

    # Process frames
    for i in range(100):
        result = detector.process_frame(keypoints, timestamp_s=i / 30.0)
        assert result.is_valid, f"Frame {i} should be valid"

    print(f"  Frames processed: {detector._frame_count}")
    print(f"  State: {result.state.value}")
    print(f"  Trunk inclination: {result.trunk_inclination_deg:.1f}°")
    print(f"  Asymmetry: {result.asymmetry_pct:.1f}%")

    # Test joint angle computation
    print("\n  [Joint Angle Computation]")
    angles = JointAngleCalculator.compute_all_angles(keypoints)
    for joint, angle in sorted(angles.items()):
        print(f"    {joint}: {angle:.1f}°")

    # Test with asymmetric pose (pain simulation)
    print("\n  [Asymmetric Pose - Pain Simulation]")
    detector2 = BodyStateDetector(fps=30.0)
    asymmetric_kps = keypoints.copy()
    asymmetric_kps[5] = [-0.2, 1.4, 0]  # Left shoulder dropped (asymmetry)

    for i in range(100):
        result2 = detector2.process_frame(asymmetric_kps, timestamp_s=i / 30.0)

    print(f"    Asymmetry: {result2.asymmetry_pct:.1f}%")
    print(f"    State: {result2.state.value}")

    print("  RESULT: PASS")
    return True


def main():
    """Run all Phase 3 tests."""
    print("\n" + "=" * 60)
    print("PHASE 3: Face Analysis Tests (OpenFace 3.0 + AU Formulas)")
    print("=" * 60)

    tests = [
        ("Face Detection", test_face_detector),
        ("Facial State Detector (AU Formulas)", test_facial_state_detector),
        ("OpenFace 3.0 Analyzer", test_openface_analyzer),
        ("Blink Detection", test_blink_detector),
        ("Yawn Detection", test_yawn_detector),
        ("Body State Detection", test_body_state_detector),
    ]

    passed = 0
    skipped = 0
    failed = 0

    for name, test_fn in tests:
        try:
            result = test_fn()
            if result:
                passed += 1
            else:
                skipped += 1
        except Exception as e:
            print(f"  RESULT: FAIL - {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 60)
    print(f"SUMMARY: {passed} passed, {skipped} skipped, {failed} failed out of {len(tests)} tests")
    print("=" * 60)


if __name__ == "__main__":
    main()
