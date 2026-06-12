"""
Tests for the evaluation benchmark loaders (Phase 5).

Verifies UI_PRMDLoader and KimoreLoader:
1. ``is_available()`` returns True when the dataset is on disk.
2. ``load()`` returns data with the expected shape.
3. ``compute_self_consistency()`` returns valid metrics.
4. ``iter_samples()`` yields the right number of items.

Run with: pytest tests/test_benchmark_loaders.py -v
"""

import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestUIPRMDLoader:
    @pytest.fixture
    def loader(self):
        from evaluation.benchmarks.uiprmd import UI_PRMDLoader
        return UI_PRMDLoader(data_dir="data/UI-PRMD")

    def test_is_available(self, loader):
        assert loader.is_available(), "data/UI-PRMD/input.csv not found"

    def test_load_shape(self, loader):
        kps = loader.load()
        assert kps.ndim == 3
        assert kps.shape[1] == 25  # 25 Kinect joints
        assert kps.shape[2] == 3   # x, y, z
        assert kps.shape[0] == 1423  # known sample count

    def test_confidence_mask_shape(self, loader):
        loader.load()
        mask = loader.get_confidence_mask()
        assert mask.shape == (1423, 25)
        assert mask.dtype == bool
        # Most joints should be visible
        assert mask.mean() > 0.5

    def test_compute_self_consistency(self, loader):
        loader.load()
        result = loader.compute_self_consistency()
        assert "mpjpe_mean" in result
        assert "mpjpe_std" in result
        assert result["mpjpe_mean"] >= 0
        assert result["mpjpe_std"] >= 0

    def test_compute_joint_angles(self, loader):
        loader.load()
        stats = loader.compute_joint_angles()
        assert len(stats) > 0
        for name, s in stats.items():
            assert "mean" in s and "std" in s
            assert 0.0 <= s["mean"] <= 180.0

    def test_compute_smoothness(self, loader):
        loader.load()
        result = loader.compute_smoothness()
        assert "sparc" in result
        assert "mean_sparc" in result
        # SPARC should be negative (lower = smoother)
        for name, v in result["sparc"].items():
            assert v < 0

    def test_iter_samples(self, loader):
        loader.load()
        count = sum(1 for _ in loader.iter_samples())
        assert count == 1423

    def test_summary(self, loader):
        summary = loader.summary()
        assert summary["n_samples"] == 1423
        assert "self_consistency" in summary
        assert "angle_statistics" in summary
        assert "smoothness" in summary


class TestKimoreLoader:
    @pytest.fixture
    def loader(self):
        from evaluation.benchmarks.kimore import KimoreLoader
        return KimoreLoader(data_path="data/KIMORE/kimore_exercise_dataset.pkl")

    def test_is_available(self, loader):
        assert loader.is_available()

    def test_load_shape(self, loader):
        exercises = loader.load()
        assert len(exercises) == 5
        for name in ["ex1", "ex2", "ex3", "ex4", "ex5"]:
            assert name in exercises
            samples = exercises[name]
            assert len(samples) > 0
            # Each sample: (num_frames, 25, 3)
            assert samples[0].ndim == 3
            assert samples[0].shape[1] == 25
            assert samples[0].shape[2] == 3

    def test_clinical_scores(self, loader):
        loader.load()
        scores = loader.get_clinical_scores()
        assert len(scores) == 5
        for ex_name, vals in scores.items():
            assert len(vals) > 0
            assert all(isinstance(v, float) for v in vals)

    def test_iter_samples(self, loader):
        loader.load()
        count = sum(1 for _ in loader.iter_samples())
        # 5 exercises × ~75 samples = ~378
        assert 350 <= count <= 400

    def test_total_samples(self, loader):
        loader.load()
        total = sum(len(s) for s in loader.exercises.values())
        assert total == 378
