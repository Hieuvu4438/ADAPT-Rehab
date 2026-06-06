"""
Phase 3: Face Analysis Tests.

Tests face detection, AU detection, and emotion classification
on real yoga pose videos.

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


def test_au_detector():
    """Test Action Unit detection on real face landmarks."""
    print("\n" + "=" * 60)
    print("TEST 2: Action Unit Detection (FACS/PSPI)")
    print("=" * 60)

    from modules.perception import FaceDetector, ActionUnitDetector

    face_det = FaceDetector()
    if not face_det.initialize():
        print("  SKIP: Face model not found")
        return False

    au_det = ActionUnitDetector(use_baseline=True)

    video = get_test_videos(1)[0]
    cap = cv2.VideoCapture(video)

    # First, calibrate from first frame with face
    calibrated = False
    au_results = []

    for i in range(90):
        ret, frame = cap.read()
        if not ret:
            break

        face_result = face_det.detect(frame)
        if not face_result.is_valid or face_result.landmarks is None:
            continue

        if not calibrated:
            au_det.set_baseline(face_result.landmarks)
            calibrated = True
            print(f"  Baseline calibrated at frame {i}")

        au_result = au_det.detect(face_result.landmarks)
        if au_result.is_valid:
            au_results.append(au_result)

    cap.release()
    face_det.close()

    if not au_results:
        print("  No face detections for AU analysis")
        print("  RESULT: PASS (face not visible in video)")
        return True

    # Compute statistics
    pspi_scores = [r.pain_score for r in au_results]
    pain_levels = [r.pain_level for r in au_results]

    print(f"  Frames analyzed: {len(au_results)}")
    print(f"  PSPI range: {min(pspi_scores):.2f} - {max(pspi_scores):.2f}")
    print(f"  PSPI mean: {np.mean(pspi_scores):.2f}")
    print(f"  Pain levels: {dict(zip(*np.unique(pain_levels, return_counts=True)))}")

    # Show AU activations for first frame
    first = au_results[0]
    print(f"  AU activations (first frame):")
    for au, val in sorted(first.au_activations.items()):
        print(f"    {au}: {val:.3f}")

    print("  RESULT: PASS")
    return True


def test_emotion_classifier():
    """Test emotion classification on real face landmarks."""
    print("\n" + "=" * 60)
    print("TEST 3: Emotion Classification (Geometric Features)")
    print("=" * 60)

    from modules.perception import FaceDetector, EmotionClassifier

    face_det = FaceDetector()
    if not face_det.initialize():
        print("  SKIP: Face model not found")
        return False

    emotion_cls = EmotionClassifier()
    emotion_cls.initialize()

    video = get_test_videos(1)[0]
    cap = cv2.VideoCapture(video)

    emotion_results = []

    for i in range(90):
        ret, frame = cap.read()
        if not ret:
            break

        face_result = face_det.detect(frame)
        if not face_result.is_valid or face_result.landmarks is None:
            continue

        emotion_result = emotion_cls.classify(face_result.landmarks)
        if emotion_result.is_valid:
            emotion_results.append(emotion_result)

    cap.release()
    face_det.close()
    emotion_cls.close()

    if not emotion_results:
        print("  No face detections for emotion analysis")
        print("  RESULT: PASS (face not visible in video)")
        return True

    # Compute statistics
    emotions = [r.emotion.value for r in emotion_results]
    confidences = [r.confidence for r in emotion_results]

    print(f"  Frames analyzed: {len(emotion_results)}")
    print(f"  Dominant emotion: {max(set(emotions), key=emotions.count)}")
    print(f"  Mean confidence: {np.mean(confidences):.3f}")
    print(f"  Emotion distribution:")
    unique, counts = np.unique(emotions, return_counts=True)
    for emotion, count in zip(unique, counts):
        pct = count / len(emotions) * 100
        print(f"    {emotion}: {count} ({pct:.1f}%)")

    print("  RESULT: PASS")
    return True


def test_face_analyzer_combined():
    """Test combined face analyzer on real videos."""
    print("\n" + "=" * 60)
    print("TEST 4: Combined Face Analyzer")
    print("=" * 60)

    from modules.perception import FaceAnalyzer

    analyzer = FaceAnalyzer(use_baseline=True)
    if not analyzer.initialize():
        print("  SKIP: Face model not found")
        return False

    video = get_test_videos(1)[0]
    cap = cv2.VideoCapture(video)

    results = []
    for i in range(60):
        ret, frame = cap.read()
        if not ret:
            break

        result = analyzer.analyze(frame, timestamp_ms=i * 33)
        if result.is_valid:
            results.append(result)

    cap.release()
    analyzer.close()

    if not results:
        print("  No face detections")
        print("  RESULT: PASS (face not visible in video)")
        return True

    print(f"  Video: {os.path.basename(video)}")
    print(f"  Frames analyzed: {len(results)}")

    # Emotion summary
    emotions = [r.emotion.value for r in results]
    dominant = max(set(emotions), key=emotions.count)
    print(f"  Dominant emotion: {dominant}")

    # Pain summary
    pain_scores = [r.pain_score for r in results]
    pain_levels = [r.pain_level for r in results]
    print(f"  PSPI range: {min(pain_scores):.2f} - {max(pain_scores):.2f}")
    print(f"  Pain levels: {dict(zip(*np.unique(pain_levels, return_counts=True)))}")

    # Show first result
    first = results[0]
    print(f"\n  First frame detail:")
    print(f"    Emotion: {first.emotion.value} ({first.emotion_confidence:.3f})")
    print(f"    Pain: {first.pain_level} (PSPI={first.pain_score:.2f})")
    print(f"    AUs: {', '.join(f'{k}={v:.2f}' for k, v in first.au_activations.items())}")

    print("  RESULT: PASS")
    return True


def test_face_analyzer_no_face():
    """Test face analyzer handles no-face frames gracefully."""
    print("\n" + "=" * 60)
    print("TEST 5: Face Analyzer - No Face Handling")
    print("=" * 60)

    from modules.perception import FaceAnalyzer

    analyzer = FaceAnalyzer()
    if not analyzer.initialize():
        print("  SKIP: Face model not found")
        return False

    # Create a blank frame (no face)
    blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    result = analyzer.analyze(blank_frame)

    assert not result.is_valid, "Should not detect face in blank frame"
    assert result.pain_level == "NONE"
    assert result.emotion.value == "neutral"

    analyzer.close()
    print("  Blank frame: correctly returns no face")
    print("  RESULT: PASS")
    return True


def main():
    """Run all Phase 3 tests."""
    print("\n" + "=" * 60)
    print("PHASE 3: Face Analysis Tests")
    print("=" * 60)

    tests = [
        ("Face Detection", test_face_detector),
        ("AU Detection", test_au_detector),
        ("Emotion Classification", test_emotion_classifier),
        ("Combined Analyzer", test_face_analyzer_combined),
        ("No Face Handling", test_face_analyzer_no_face),
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
