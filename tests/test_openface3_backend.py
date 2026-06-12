"""
Tests for OpenFace 3.0 MTL backend integration.

Validates that the OpenFace 3.0 GNN-based AU/Emotion/Gaze model
loads and produces correct output shapes and ranges.

Run: python -m pytest tests/test_openface3_backend.py -v
"""

import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------------------------------------------------------------------------
# Skip markers — skip gracefully if openface-test is not installed
# ---------------------------------------------------------------------------

of3_available = False
try:
    from openface.multitask_model import MultitaskPredictor  # type: ignore
    _mtl_path = os.path.join(
        os.path.dirname(__file__), "..", "models", "openface3", "MTL_backbone.pth",
    )
    of3_available = os.path.exists(_mtl_path)
except ImportError:
    pass

skip_no_of3 = pytest.mark.skipif(
    not of3_available,
    reason="openface-test not installed or MTL_backbone.pth missing",
)


# ---------------------------------------------------------------------------
# Constants tests
# ---------------------------------------------------------------------------

def test_required_aus():
    """The 8 required AUs match what FacialStateDetector expects."""
    from modules.perception.openface_analyzer import _REQUIRED_AUS
    expected = ["AU1", "AU2", "AU4", "AU6", "AU9", "AU12", "AU25", "AU26"]
    assert _REQUIRED_AUS == expected


def test_of3_au_index_map():
    """AU index map covers all 8 indices and maps to valid AU names."""
    from modules.perception.openface_analyzer import _OF3_AU_INDEX_MAP, _REQUIRED_AUS
    assert len(_OF3_AU_INDEX_MAP) == 8
    for idx in range(8):
        assert idx in _OF3_AU_INDEX_MAP
        assert _OF3_AU_INDEX_MAP[idx] in _REQUIRED_AUS


def test_of3_emotion_labels():
    """OF3 emotion label list has exactly 8 entries."""
    from modules.perception.openface_analyzer import _OF3_EMOTION_LABELS
    assert len(_OF3_EMOTION_LABELS) == 8
    for label in _OF3_EMOTION_LABELS:
        assert isinstance(label, str)
        assert len(label) > 0


# ---------------------------------------------------------------------------
# Geometric fallback tests (always run)
# ---------------------------------------------------------------------------

def test_openface_analyzer_init_geometric():
    """OpenFaceAnalyzer initializes even without OF3."""
    from modules.perception.openface_analyzer import OpenFaceAnalyzer
    a = OpenFaceAnalyzer(device="cpu")
    ok = a.initialize()
    assert ok is True
    a.close()


def test_analyze_random_frame_no_face():
    """Random noise frame → no face detected → valid error message."""
    from modules.perception.openface_analyzer import OpenFaceAnalyzer
    a = OpenFaceAnalyzer(device="cpu")
    a.initialize()
    frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    result = a.analyze(frame, timestamp_ms=0)
    assert result.is_valid is False
    assert "face" in result.error_message.lower() or "not" in result.error_message.lower()
    a.close()


def test_analyze_with_synthetic_landmarks():
    """Synthetic landmarks → valid result with AU dict and state."""
    from modules.perception.openface_analyzer import OpenFaceAnalyzer
    a = OpenFaceAnalyzer(device="cpu")
    a.initialize()

    # Create a simple frame and landmarks
    frame = np.ones((480, 640, 3), dtype=np.uint8) * 128
    landmarks = np.random.rand(468, 3).astype(np.float32)
    landmarks[:, 0] = landmarks[:, 0] * 240 + 200
    landmarks[:, 1] = landmarks[:, 1] * 250 + 100
    landmarks[:, 2] = 0.0
    landmarks[10] = [320, 100, 0]   # forehead
    landmarks[152] = [320, 350, 0]  # chin

    # Run multiple frames to get past calibration
    for i in range(35):
        result = a.analyze(frame, timestamp_ms=i * 33, face_landmarks=landmarks)

    assert result.is_valid is True
    assert result.au_intensities is not None
    for au in ["AU1", "AU2", "AU4", "AU6", "AU9", "AU12", "AU25", "AU26"]:
        assert au in result.au_intensities
        assert 0.0 <= result.au_intensities[au] <= 5.0
    assert result.state_result is not None
    assert result.emotion_label in [
        "neutral", "happy", "sad", "surprise", "fear", "disgust", "anger", "contempt",
    ]
    a.close()


# ---------------------------------------------------------------------------
# OpenFace 3.0 MTL tests (skip if not installed)
# ---------------------------------------------------------------------------

@skip_no_of3
def test_of3_multitask_predictor_loads():
    """MultitaskPredictor loads MTL_backbone.pth successfully."""
    predictor = MultitaskPredictor(model_path=_mtl_path, device="cpu")
    assert predictor is not None
    assert predictor.model is not None


@skip_no_of3
def test_of3_predict_output_shapes():
    """predict() returns correct tensor shapes."""
    predictor = MultitaskPredictor(model_path=_mtl_path, device="cpu")
    face = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    emotion, gaze, au = predictor.predict(face)
    assert emotion.shape == (1, 8)
    assert gaze.shape == (1, 2)
    assert au.shape == (1, 8)


@skip_no_of3
def test_of3_au_cosine_range():
    """AU cosine similarities are in roughly [-1, 1]."""
    predictor = MultitaskPredictor(model_path=_mtl_path, device="cpu")
    face = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    _, _, au = predictor.predict(face)
    au_np = au.squeeze(0).detach().cpu().numpy()
    assert au_np.shape == (8,)
    # Cosine similarities should be bounded
    assert np.all(au_np >= -1.5) and np.all(au_np <= 1.5)


@skip_no_of3
def test_analyze_with_of3_active():
    """Full analyze() path with OF3 produces AU dict with all 8 keys."""
    from modules.perception.openface_analyzer import OpenFaceAnalyzer
    a = OpenFaceAnalyzer(device="cpu")
    ok = a.initialize()
    assert ok
    assert a._of3_available is True

    frame = np.ones((480, 640, 3), dtype=np.uint8) * 128
    landmarks = np.random.rand(468, 3).astype(np.float32)
    landmarks[:, 0] = landmarks[:, 0] * 240 + 200
    landmarks[:, 1] = landmarks[:, 1] * 250 + 100
    landmarks[:, 2] = 0.0
    landmarks[10] = [320, 100, 0]
    landmarks[152] = [320, 350, 0]

    result = a.analyze(frame, timestamp_ms=0, face_landmarks=landmarks)
    assert result.is_valid is True
    assert result.au_intensities is not None
    for au in ["AU1", "AU2", "AU4", "AU6", "AU9", "AU12", "AU25", "AU26"]:
        assert au in result.au_intensities
    # Gaze should be populated (non-zero) when OF3 is active
    # (may be near zero but the model should have run)
    assert isinstance(result.gaze_yaw, float)
    assert isinstance(result.gaze_pitch, float)
    a.close()


@skip_no_of3
def test_crop_face():
    """_crop_face returns a valid crop from a known bbox."""
    from modules.perception.openface_analyzer import OpenFaceAnalyzer
    frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    bbox = np.array([100.0, 100.0, 300.0, 300.0])
    crop = OpenFaceAnalyzer._crop_face(frame, bbox, padding=0.2)
    assert crop is not None
    assert crop.shape[0] > 0 and crop.shape[1] > 0
    # Crop should be larger than bbox due to padding
    assert crop.shape[0] >= 200
    assert crop.shape[1] >= 200


@skip_no_of3
def test_of3_fallback_on_bad_crop():
    """If face crop is degenerate, falls back to geometric gracefully."""
    from modules.perception.openface_analyzer import OpenFaceAnalyzer
    a = OpenFaceAnalyzer(device="cpu")
    a.initialize()

    # Empty crop
    result_empty = a._crop_face(
        np.zeros((10, 10, 3), dtype=np.uint8),
        np.array([0.0, 0.0, 0.0, 0.0]),
    )
    assert result_empty is None

    a.close()
