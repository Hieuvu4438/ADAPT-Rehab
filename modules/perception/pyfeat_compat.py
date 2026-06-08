"""
py-feat compatibility layer.

Provides AU detection using py-feat with monkey-patches for
Python 3.13 compatibility issues.

Usage:
    from modules.perception.pyfeat_compat import PyFeatDetector
    detector = PyFeatDetector()
    aus = detector.detect_aus(face_image)
"""

import sys
import types
import numpy as np

# ---------------------------------------------------------------------------
# Monkey-patches for Python 3.13 compatibility
# ---------------------------------------------------------------------------

# 1. Fix scipy.integrate.simps -> scipy.integrate.simpson
import scipy.integrate
if not hasattr(scipy.integrate, 'simps'):
    scipy.integrate.simps = scipy.integrate.simpson

# 2. Fix scipy.stats.binom_test
import scipy.stats
if not hasattr(scipy.stats, 'binom_test'):
    from scipy.stats import binomtest
    scipy.stats.binom_test = lambda x, n=None, p=0.5, alternative='two-sided': binomtest(x, n, p, alternative).pvalue

# 3. Fix numpy.ComplexWarning
if not hasattr(np, 'ComplexWarning'):
    class ComplexWarning(UserWarning):
        pass
    np.ComplexWarning = ComplexWarning

# 4. Fix torchvision.io.read_video
import torchvision.io
if not hasattr(torchvision.io, 'read_video'):
    def read_video(*args, **kwargs):
        return None, None, None
    torchvision.io.read_video = read_video

# 5. Create nltools mock if needed
try:
    import nltools
except ImportError:
    # Create mock nltools
    nltools = types.ModuleType('nltools')
    nltools.__path__ = []
    nltools.__package__ = 'nltools'

    nltools_data = types.ModuleType('nltools.data')
    nltools_stats = types.ModuleType('nltools.stats')
    nltools_utils = types.ModuleType('nltools.utils')

    class Adjacency:
        pass
    class Brain_Data:
        pass
    class Groupby:
        pass
    class Design_Matrix:
        pass
    class Design_Matrix_Series:
        pass

    nltools_data.Adjacency = Adjacency
    nltools_data.Brain_Data = Brain_Data
    nltools_data.Groupby = Groupby
    nltools_data.Design_Matrix = Design_Matrix
    nltools_data.Design_Matrix_Series = Design_Matrix_Series

    nltools_stats.downsample = lambda data, n: data
    nltools_stats.upsample = lambda data, n: data
    nltools_stats.regress = lambda X, y: np.linalg.lstsq(X, y, rcond=None)[0]
    nltools_stats.bootstrap = lambda *a, **kw: None
    nltools_stats.effsize = lambda *a, **kw: None

    nltools_utils.set_decomposition_algorithm = lambda *a, **kw: None
    nltools_utils.get_resource_path = lambda: ""
    nltools_utils.is_iterable = lambda i: hasattr(i, '__iter__')

    nltools.data = nltools_data
    nltools.stats = nltools_stats
    nltools.utils = nltools_utils

    sys.modules['nltools'] = nltools
    sys.modules['nltools.data'] = nltools_data
    sys.modules['nltools.stats'] = nltools_stats
    sys.modules['nltools.utils'] = nltools_utils


# ---------------------------------------------------------------------------
# py-feat wrapper
# ---------------------------------------------------------------------------

class PyFeatDetector:
    """Wrapper around py-feat Detector with compatibility fixes."""

    def __init__(self, au_model: str = "svm"):
        """Initialize py-feat detector.

        Args:
            au_model: AU detection model ("svm" or "xgb")
        """
        from feat import Detector as FeatDetector
        self._detector = FeatDetector(
            au_model=au_model,
            face_model="retinaface",
            landmark_model="mobilenet",
        )

    def detect_aus(self, face_image: np.ndarray) -> dict:
        """Detect Action Units from face image.

        Args:
            face_image: BGR face image

        Returns:
            Dict mapping AU name to intensity (0-5).
        """
        import cv2

        # Convert BGR to RGB
        if len(face_image.shape) == 3 and face_image.shape[2] == 3:
            rgb_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
        else:
            rgb_image = face_image

        # Detect AUs
        result = self._detector.detect_aus(rgb_image)

        # Extract AU values
        aus = {}
        au_names = ["AU1", "AU2", "AU4", "AU5", "AU6", "AU7", "AU9",
                     "AU10", "AU12", "AU14", "AU15", "AU17", "AU20",
                     "AU23", "AU24", "AU25", "AU26", "AU28", "AU43"]

        for au in au_names:
            if hasattr(result, au.lower()):
                aus[au] = float(getattr(result, au.lower()))
            elif au in result:
                aus[au] = float(result[au])
            else:
                aus[au] = 0.0

        return aus

    def detect_emotions(self, face_image: np.ndarray) -> dict:
        """Detect emotions from face image.

        Args:
            face_image: BGR face image

        Returns:
            Dict mapping emotion to probability.
        """
        import cv2

        # Convert BGR to RGB
        if len(face_image.shape) == 3 and face_image.shape[2] == 3:
            rgb_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
        else:
            rgb_image = face_image

        # Detect emotions
        result = self._detector.detect_emotions(rgb_image)

        # Extract emotion probabilities
        emotions = {}
        emotion_names = ["anger", "disgust", "fear", "happiness",
                         "neutral", "sadness", "surprise"]

        for emo in emotion_names:
            if emo in result:
                emotions[emo] = float(result[emo])
            else:
                emotions[emo] = 0.0

        return emotions


def create_pyfeat_detector(au_model: str = "jaanet") -> PyFeatDetector:
    """Create a py-feat detector with compatibility fixes.

    Args:
        au_model: AU detection model ("jaanet" or "svm")

    Returns:
        PyFeatDetector instance.
    """
    return PyFeatDetector(au_model)
