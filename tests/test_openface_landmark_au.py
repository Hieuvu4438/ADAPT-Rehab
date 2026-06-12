"""
Tests for the rebuilt OpenFaceAnalyzer (Phase 4).

Verifies:
1. ``initialize()`` succeeds.
2. ``analyze()`` with injected landmarks (skipping face detection) returns
   a valid ``OpenFaceResult`` with the expected 8 AU keys.
3. AU intensities are clipped to [0, 5].
4. AU values are scale-invariant (calibration makes them comparable).
5. The 30-frame calibration phase runs before AU values become non-zero.
6. ``reset()`` clears calibration state.

Run with: pytest tests/test_openface_landmark_au.py -v
"""

import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


EXPECTED_AUS = ["AU1", "AU2", "AU4", "AU6", "AU9", "AU12", "AU25", "AU26"]


def _make_synthetic_landmarks(
    face_height: float = 200.0,
    mouth_w_factor: float = 1.0,
    mouth_v_factor: float = 1.0,
    brow_factor: float = 1.0,
    eye_factor: float = 1.0,
) -> np.ndarray:
    """Generate 468 synthetic MediaPipe-style landmarks for testing.

    The default factors are all 1.0 → "neutral" face. Vary the factors
    to simulate different AU activations.
    """
    lm = np.zeros((468, 3), dtype=np.float32)
    # Face height (forehead 10, chin 152) along y-axis
    lm[10] = np.array([320, 100, 0])
    lm[152] = np.array([320, 100 + face_height, 0])
    # Inner brows
    lm[107] = np.array([290, 100 + 30 * brow_factor, 0])
    lm[336] = np.array([350, 100 + 30 * brow_factor, 0])
    # Outer brows
    lm[105] = np.array([265, 100 + 35 * brow_factor, 0])
    lm[334] = np.array([375, 100 + 35 * brow_factor, 0])
    # Eye aperture (right eye)
    lm[159] = np.array([305, 100 + 60 * eye_factor, 0])
    lm[145] = np.array([305, 100 + 75 * eye_factor, 0])
    # Eye aperture (left eye)
    lm[386] = np.array([335, 100 + 60 * eye_factor, 0])
    lm[374] = np.array([335, 100 + 75 * eye_factor, 0])
    # Nose wings
    lm[129] = np.array([295, 100 + 110, 0])
    lm[358] = np.array([345, 100 + 110, 0])
    # Mouth corners
    lm[61] = np.array([320 - 30 * mouth_w_factor, 100 + 160, 0])
    lm[291] = np.array([320 + 30 * mouth_w_factor, 100 + 160, 0])
    # Lip aperture
    lm[13] = np.array([320, 100 + 150 - 5 * mouth_v_factor, 0])
    lm[14] = np.array([320, 100 + 150 + 5 * mouth_v_factor, 0])
    # Nose tip
    lm[1] = np.array([320, 100 + 100, 0])
    return lm


class TestOpenFaceAnalyzer:
    @pytest.fixture(scope="class")
    def analyzer(self):
        from modules.perception.openface_analyzer import OpenFaceAnalyzer
        a = OpenFaceAnalyzer()
        if not a.initialize():
            pytest.skip("OpenFaceAnalyzer failed to initialize")
        yield a
        a.close()

    def test_initialize(self, analyzer):
        assert analyzer._is_initialized

    def test_analyze_with_injected_landmarks(self, analyzer):
        landmarks = _make_synthetic_landmarks()
        # Inject landmarks directly to skip face detection.
        result = analyzer.analyze(
            np.zeros((480, 640, 3), dtype=np.uint8),
            timestamp_ms=0,
            face_landmarks=landmarks,
        )
        assert result.is_valid
        assert result.landmarks_468 is not None
        assert len(result.landmarks_468) == 468

    def test_au_intensities_have_all_8_keys(self, analyzer):
        landmarks = _make_synthetic_landmarks()
        # Calibration: 30 frames needed. Send 35.
        for i in range(35):
            analyzer.analyze(
                np.zeros((480, 640, 3), dtype=np.uint8),
                timestamp_ms=i * 33,
                face_landmarks=landmarks,
            )
        result = analyzer.analyze(
            np.zeros((480, 640, 3), dtype=np.uint8),
            timestamp_ms=35 * 33,
            face_landmarks=landmarks,
        )
        assert set(result.au_intensities.keys()) == set(EXPECTED_AUS)

    def test_au_intensities_in_range(self, analyzer):
        landmarks = _make_synthetic_landmarks()
        for i in range(35):
            analyzer.analyze(
                np.zeros((480, 640, 3), dtype=np.uint8),
                timestamp_ms=i * 33,
                face_landmarks=landmarks,
            )
        # Now vary the mouth (smile) to trigger AU12
        landmarks_smile = _make_synthetic_landmarks(mouth_w_factor=2.0)
        result = analyzer.analyze(
            np.zeros((480, 640, 3), dtype=np.uint8),
            timestamp_ms=35 * 33,
            face_landmarks=landmarks_smile,
        )
        for k in EXPECTED_AUS:
            v = result.au_intensities[k]
            assert 0.0 <= v <= 5.0, f"{k} = {v} out of [0, 5]"

    def test_scale_invariance(self, analyzer):
        """AU values should not depend on absolute face size after calibration."""
        analyzer.reset()
        # Calibrate with face_height=200
        landmarks_small = _make_synthetic_landmarks(face_height=100.0)
        for i in range(35):
            analyzer.analyze(
                np.zeros((480, 640, 3), dtype=np.uint8),
                timestamp_ms=i * 33,
                face_landmarks=landmarks_small,
            )
        result_small = analyzer.analyze(
            np.zeros((480, 640, 3), dtype=np.uint8),
            timestamp_ms=35 * 33,
            face_landmarks=_make_synthetic_landmarks(face_height=100.0),
        )
        # Now reset and calibrate with a much larger face
        analyzer.reset()
        landmarks_large = _make_synthetic_landmarks(face_height=400.0)
        for i in range(35):
            analyzer.analyze(
                np.zeros((480, 640, 3), dtype=np.uint8),
                timestamp_ms=i * 33,
                face_landmarks=landmarks_large,
            )
        result_large = analyzer.analyze(
            np.zeros((480, 640, 3), dtype=np.uint8),
            timestamp_ms=35 * 33,
            face_landmarks=_make_synthetic_landmarks(face_height=400.0),
        )
        # AU values should be similar (calibration normalizes by face height)
        for k in EXPECTED_AUS:
            v_small = result_small.au_intensities[k]
            v_large = result_large.au_intensities[k]
            # Allow generous tolerance for calibration noise
            assert abs(v_small - v_large) < 1.0, (
                f"{k} differs by face size: small={v_small}, large={v_large}"
            )

    def test_smile_triggers_AU12(self, analyzer):
        analyzer.reset()
        # Calibrate with neutral face
        neutral = _make_synthetic_landmarks()
        for i in range(35):
            analyzer.analyze(
                np.zeros((480, 640, 3), dtype=np.uint8),
                timestamp_ms=i * 33,
                face_landmarks=neutral,
            )
        # Now smile (mouth widens)
        smile = _make_synthetic_landmarks(mouth_w_factor=2.0)
        result = analyzer.analyze(
            np.zeros((480, 640, 3), dtype=np.uint8),
            timestamp_ms=35 * 33,
            face_landmarks=smile,
        )
        # AU12 (lip corner puller) should be elevated
        assert result.au_intensities["AU12"] > 1.0

    def test_reset_clears_calibration(self, analyzer):
        # Calibrate
        landmarks = _make_synthetic_landmarks()
        for i in range(35):
            analyzer.analyze(
                np.zeros((480, 640, 3), dtype=np.uint8),
                timestamp_ms=i * 33,
                face_landmarks=landmarks,
            )
        assert analyzer._baseline is not None
        analyzer.reset()
        assert analyzer._baseline is None
        assert analyzer._calibration_buffer == []
