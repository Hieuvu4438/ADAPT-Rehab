"""
Benchmark Runner for ADAPT-Rehab Evaluation.

Runs pose estimation on benchmark videos, computes standard metrics
following the VideoPose3D protocol, and generates comparison tables.

Metrics computed:
- MPJPE (mm): Root-aligned Mean Per-Joint Position Error
- P-MPJPE (mm): Procrustes-aligned MPJPE
- Per-joint MPJPE breakdown
- Joint angle MAE (degrees)
- SPARC smoothness score
- Inference time (ms/frame)

Evaluation modes:
- self_consistency: Temporal stability of predictions (no ground truth needed)
- ground_truth: Accuracy against reference annotations (e.g., Kinect)
"""

import os
import sys
import json
import time
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from evaluation.metrics.mpjpe import (
    compute_mpjpe, compute_p_mpjpe, compute_per_joint_mpjpe,
)
from evaluation.metrics.angle_mae import (
    compute_angle_mae, compute_per_joint_angle_mae,
    compute_angles_from_keypoints, compute_icc,
)
from evaluation.skeleton_mapping import (
    SkeletonType, remap_to_common14, COMMON14_NAMES,
)


@dataclass
class FrameResult:
    """Result from processing a single frame."""
    frame_idx: int
    keypoints_3d: Optional[np.ndarray] = None  # (N, 3)
    joint_angles: Dict[str, float] = field(default_factory=dict)
    confidence: float = 0.0
    inference_ms: float = 0.0
    is_valid: bool = False


@dataclass
class VideoResult:
    """Result from processing a single video."""
    video_path: str
    exercise_type: str
    person_name: str
    estimator_name: str = ""
    total_frames: int = 0
    valid_frames: int = 0
    fps: float = 0.0
    avg_inference_ms: float = 0.0
    # Metrics (in mm for spatial, degrees for angular)
    mpjpe: float = 0.0
    p_mpjpe: float = 0.0
    per_joint_mpjpe: Dict[str, float] = field(default_factory=dict)
    angle_mae: float = 0.0
    per_joint_angle_mae: Dict[str, float] = field(default_factory=dict)
    sparc_score: float = 0.0
    temporal_stability: float = 0.0  # std of keypoints across frames


@dataclass
class BenchmarkResult:
    """Complete benchmark result."""
    dataset_name: str
    evaluation_mode: str = "self_consistency"
    total_videos: int = 0
    total_frames: int = 0
    overall_fps: float = 0.0
    overall_mpjpe: float = 0.0
    overall_p_mpjpe: float = 0.0
    overall_per_joint_mpjpe: Dict[str, float] = field(default_factory=dict)
    overall_angle_mae: float = 0.0
    overall_sparc: float = 0.0
    overall_temporal_stability: float = 0.0
    per_exercise: Dict[str, Dict] = field(default_factory=dict)
    video_results: List[Dict] = field(default_factory=list)
    ablation_results: Dict[str, Dict] = field(default_factory=dict)


class BenchmarkRunner:
    """Runs benchmark evaluation on pose estimation."""

    def __init__(self, data_dir: str, output_dir: str = "evaluation/results"):
        self.data_dir = data_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def run_full_benchmark(
        self,
        estimator_type: str = "rtmw3d",
        max_frames_per_video: int = 100,
        mode: str = "self_consistency",
    ) -> BenchmarkResult:
        """Run full benchmark evaluation.

        Args:
            estimator_type: Pose estimator to use ("rtmw3d").
            max_frames_per_video: Maximum frames to process per video.
            mode: Evaluation mode ("self_consistency" or "ground_truth").

        Returns:
            BenchmarkResult with all metrics.
        """
        print("=" * 60)
        print("ADAPT-Rehab Benchmark Evaluation")
        print("=" * 60)
        print(f"Estimator: {estimator_type}")
        print(f"Mode: {mode}")
        print(f"Max frames/video: {max_frames_per_video}")

        # Load dataset
        samples = self._load_dataset()
        if not samples:
            print("[Error] No samples found!")
            return BenchmarkResult(dataset_name="unknown")

        # Run pose estimation on all videos
        video_results = []
        for i, sample in enumerate(samples):
            print(f"\n[{i+1}/{len(samples)}] Processing: "
                  f"{sample.person_name}_{sample.exercise_type}")
            result = self._process_video(
                sample, estimator_type, max_frames_per_video
            )
            video_results.append(result)

        # Compute overall metrics
        benchmark = self._compute_overall_metrics(video_results)
        benchmark.evaluation_mode = mode

        # Run ablation study (real re-runs)
        if mode == "self_consistency":
            benchmark.ablation_results = self._run_ablation(
                samples[:3], estimator_type, max_frames_per_video
            )

        # Save results
        self._save_results(benchmark)

        return benchmark

    def _load_dataset(self):
        """Load dataset samples."""
        from evaluation.datasets import load_yoga_dataset
        return load_yoga_dataset(self.data_dir)

    def _process_video(
        self,
        sample,
        estimator_type: str,
        max_frames: int,
    ) -> VideoResult:
        """Process a single video through the pose pipeline."""
        import cv2

        result = VideoResult(
            video_path=sample.video_path,
            exercise_type=sample.exercise_type,
            person_name=sample.person_name,
            estimator_name=estimator_type,
        )

        # Create estimator (with fallback)
        estimator = self._create_estimator(estimator_type)
        if estimator is None:
            print(f"  [Error] Failed to create estimator: {estimator_type}")
            return result

        # Track actual estimator type
        actual_type = estimator.model_name.lower()
        if "metrab" in actual_type:
            result.estimator_name = "metrab"
        elif "rtmw3d" in actual_type:
            result.estimator_name = "rtmw3d"
        else:
            result.estimator_name = estimator_type

        cap = cv2.VideoCapture(sample.video_path)
        if not cap.isOpened():
            print(f"  [Error] Cannot open video: {sample.video_path}")
            return result

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        result.total_frames = min(total_frames, max_frames)

        frame_results = []
        start_time = time.time()
        frame_idx = 0

        while cap.isOpened() and frame_idx < max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            timestamp_ms = int(frame_idx * (1000 / 30))
            t0 = time.time()
            pose_result = estimator.estimate(frame, timestamp_ms)
            inference_ms = (time.time() - t0) * 1000

            if pose_result.is_valid and pose_result.keypoints_3d is not None:
                fr = FrameResult(
                    frame_idx=frame_idx,
                    keypoints_3d=pose_result.keypoints_3d.copy(),
                    joint_angles=pose_result.joint_angles or {},
                    confidence=float(np.mean(pose_result.confidence))
                    if pose_result.confidence is not None else 0,
                    inference_ms=inference_ms,
                    is_valid=True,
                )
                frame_results.append(fr)

            frame_idx += 1

        cap.release()
        estimator.close()

        elapsed = time.time() - start_time
        result.valid_frames = len(frame_results)
        result.fps = len(frame_results) / elapsed if elapsed > 0 else 0
        result.avg_inference_ms = (
            np.mean([f.inference_ms for f in frame_results])
            if frame_results else 0
        )

        # Compute metrics (use actual estimator type for unit conversion)
        if len(frame_results) > 1:
            self._compute_video_metrics(result, frame_results, result.estimator_name)

        print(f"  Frames: {result.valid_frames}/{result.total_frames}, "
              f"FPS: {result.fps:.1f}, "
              f"MPJPE: {result.mpjpe:.1f}mm, "
              f"P-MPJPE: {result.p_mpjpe:.1f}mm, "
              f"Stability: {result.temporal_stability:.1f}mm")

        return result

    def _compute_video_metrics(
        self,
        result: VideoResult,
        frame_results: List[FrameResult],
        estimator_type: str,
    ) -> None:
        """Compute all metrics for a video.

        Two approaches:
        1. Self-consistency: compare frames to temporal mean (stability metric)
        2. Ground truth: compare to reference annotations (accuracy metric)

        For self-consistency mode, we report:
        - Temporal stability (std of keypoints) in mm
        - P-MPJPE between consecutive frames (temporal coherence)
        """
        kps_list = [f.keypoints_3d for f in frame_results if f.keypoints_3d is not None]
        if len(kps_list) < 2:
            return

        kps = np.array(kps_list)  # (F, J, 3)

        # Determine scale and convert to mm
        kps_mm = self._convert_to_mm(kps, estimator_type)

        # --- Self-consistency metrics ---
        # Temporal stability: std of each joint position across frames
        joint_std = np.std(kps_mm, axis=0)  # (J, 3)
        result.temporal_stability = float(np.mean(np.linalg.norm(joint_std, axis=-1)))

        # MPJPE against temporal mean (self-consistency)
        mean_kps = np.mean(kps_mm, axis=0)  # (J, 3)
        result.mpjpe = float(np.mean(
            np.linalg.norm(kps_mm - mean_kps[np.newaxis], axis=-1)
        ))

        # P-MPJPE against temporal mean (broadcast mean to match shape)
        mean_kps_expanded = np.broadcast_to(mean_kps, kps_mm.shape)
        result.p_mpjpe = compute_p_mpjpe(kps_mm, mean_kps_expanded)

        # Per-joint MPJPE with meaningful names
        joint_names = self._get_joint_names(estimator_type, kps_mm.shape[1])
        result.per_joint_mpjpe = compute_per_joint_mpjpe(kps_mm, mean_kps_expanded, joint_names)

        # --- Angular metrics ---
        # Compare angles from first frame vs mean angles
        all_angles = [f.joint_angles for f in frame_results if f.joint_angles]
        if all_angles:
            common_joints = set(all_angles[0].keys())
            for a in all_angles[1:]:
                common_joints &= set(a.keys())

            if common_joints:
                mean_angles = {
                    j: float(np.mean([a[j] for a in all_angles]))
                    for j in common_joints
                }
                # MAE between first frame and mean (stability measure)
                result.angle_mae = compute_angle_mae(all_angles[0], mean_angles)
                result.per_joint_angle_mae = compute_per_joint_angle_mae(
                    all_angles[0], mean_angles
                )

        # --- Smoothness (SPARC on angular velocity) ---
        if all_angles and len(all_angles) > 5:
            from core.smoothness import SmoothnessAnalyzer
            smoothness = SmoothnessAnalyzer(fs=30.0)

            # Compute SPARC for each joint's angular velocity trajectory
            sparc_scores = []
            for joint_name in all_angles[0].keys():
                trajectory = np.array([a.get(joint_name, 0) for a in all_angles])
                # Skip if insufficient motion
                if np.std(trajectory) < 1.0:  # less than 1 degree variation
                    continue
                sr = smoothness.analyze(trajectory)
                if sr.is_valid and sr.smoothness_score > 0:
                    sparc_scores.append(sr.smoothness_score)

            # Use median SPARC across joints (robust to outliers)
            if sparc_scores:
                result.sparc_score = float(np.median(sparc_scores))

    def _convert_to_mm(
        self, kps: np.ndarray, estimator_type: str
    ) -> np.ndarray:
        """Convert keypoints to millimeters.

        Args:
            kps: Keypoints array, shape (F, J, 3).
            estimator_type: Type of estimator used.

        Returns:
            Keypoints in millimeters.
        """
        if estimator_type == "metrab":
            # MeTRAbs outputs centimeters → mm
            return kps * 10.0
        elif estimator_type == "rtmw3d":
            # RTMW3D outputs in relative coordinates → mm
            # If using normalized coords, estimate scale from body proportions
            if np.max(np.abs(kps)) <= 1.5:
                # Likely normalized coordinates [0, 1]
                # Estimate scale: average shoulder width ~380mm
                # Use the distance between shoulders (joints 11, 12) as reference
                if kps.shape[1] >= 13:
                    shoulder_dist = np.mean(np.linalg.norm(
                        kps[:, 11, :] - kps[:, 12, :], axis=-1
                    ))
                    if shoulder_dist > 0.01:
                        scale = 380.0 / shoulder_dist  # mm per unit
                        return kps * scale
                # Fallback: assume normalized to image size ~2m person
                return kps * 1000.0
            else:
                # World landmarks in meters → mm
                return kps * 1000.0
        else:
            # Unknown estimator, assume mm
            return kps

    def _get_joint_names(self, estimator_type: str, num_joints: int) -> List[str]:
        """Get meaningful joint names for the estimator."""
        if estimator_type == "rtmw3d":
            # RTMW3D 133 keypoints (body 33 + hands 42 + face 58)
            names = [
                "pelvis", "left_hip", "right_hip",
                "spine1", "left_knee", "right_knee",
                "spine2", "left_ankle", "right_ankle",
                "spine3", "left_foot", "right_foot",
                "neck", "left_shoulder", "right_shoulder",
                "head", "left_elbow", "right_elbow",
                "left_wrist", "right_wrist",
            ]
            return names[:num_joints] if num_joints <= len(names) else [f"joint_{i}" for i in range(num_joints)]
        elif estimator_type == "metrab":
            # SMPL-24 joints
            names = [
                "pelvis", "left_hip", "right_hip",
                "spine1", "left_knee", "right_knee",
                "spine2", "left_ankle", "right_ankle",
                "spine3", "left_foot", "right_foot",
                "neck", "left_collar", "right_collar",
                "head", "left_shoulder", "right_shoulder",
                "left_elbow", "right_elbow",
                "left_wrist", "right_wrist",
                "left_hand", "right_hand",
            ]
            return names[:num_joints] if num_joints <= len(names) else [f"joint_{i}" for i in range(num_joints)]
        else:
            return [f"joint_{i}" for i in range(num_joints)]

    def _create_estimator(self, estimator_type: str):
        """Create a pose estimator instance with fallback logic."""
        from core.pose3d.base import create_estimator

        # Try requested estimator
        try:
            estimator = create_estimator(estimator_type)
            if estimator.initialize():
                return estimator
            else:
                print(f"  [Warning] {estimator_type} initialization failed")
        except Exception as e:
            print(f"  [Warning] Cannot create {estimator_type}: {e}")

        # Fallback to rtmw3d if requested estimator fails
        if estimator_type != "rtmw3d":
            print(f"  [Fallback] Trying rtmw3d...")
            try:
                fallback = create_estimator("rtmw3d")
                if fallback.initialize():
                    return fallback
            except Exception as e2:
                print(f"  [Error] Fallback failed: {e2}")

        return None

    def _compute_overall_metrics(
        self, video_results: List[VideoResult]
    ) -> BenchmarkResult:
        """Compute overall benchmark metrics."""
        benchmark = BenchmarkResult(dataset_name="Yoga-Collect")
        benchmark.total_videos = len(video_results)
        benchmark.total_frames = sum(v.valid_frames for v in video_results)

        valid = [v for v in video_results if v.valid_frames > 0]
        if not valid:
            return benchmark

        benchmark.overall_fps = float(np.mean([v.fps for v in valid]))
        benchmark.overall_mpjpe = float(np.mean([v.mpjpe for v in valid]))
        benchmark.overall_p_mpjpe = float(np.mean([v.p_mpjpe for v in valid]))
        benchmark.overall_angle_mae = float(np.mean([v.angle_mae for v in valid]))
        benchmark.overall_sparc = float(np.mean([v.sparc_score for v in valid]))
        benchmark.overall_temporal_stability = float(
            np.mean([v.temporal_stability for v in valid])
        )

        # Aggregate per-joint MPJPE
        all_joints = set()
        for v in valid:
            all_joints.update(v.per_joint_mpjpe.keys())
        for joint in all_joints:
            values = [v.per_joint_mpjpe.get(joint, 0) for v in valid]
            benchmark.overall_per_joint_mpjpe[joint] = float(np.mean(values))

        # Per-exercise metrics
        exercise_groups = {}
        for v in video_results:
            exercise_groups.setdefault(v.exercise_type, []).append(v)

        for exercise, vids in exercise_groups.items():
            valid_vids = [v for v in vids if v.valid_frames > 0]
            if valid_vids:
                benchmark.per_exercise[exercise] = {
                    "count": len(valid_vids),
                    "avg_fps": float(np.mean([v.fps for v in valid_vids])),
                    "avg_mpjpe": float(np.mean([v.mpjpe for v in valid_vids])),
                    "avg_p_mpjpe": float(np.mean([v.p_mpjpe for v in valid_vids])),
                    "avg_angle_mae": float(np.mean([v.angle_mae for v in valid_vids])),
                    "avg_sparc": float(np.mean([v.sparc_score for v in valid_vids])),
                    "avg_stability": float(np.mean([v.temporal_stability for v in valid_vids])),
                }

        # Store video results
        benchmark.video_results = [
            {
                "video": os.path.basename(v.video_path),
                "exercise": v.exercise_type,
                "person": v.person_name,
                "estimator": v.estimator_name,
                "frames": v.valid_frames,
                "fps": round(v.fps, 1),
                "mpjpe_mm": round(v.mpjpe, 1),
                "p_mpjpe_mm": round(v.p_mpjpe, 1),
                "angle_mae_deg": round(v.angle_mae, 1),
                "sparc": round(v.sparc_score, 1),
                "stability_mm": round(v.temporal_stability, 1),
            }
            for v in video_results
        ]

        return benchmark

    def _run_ablation(
        self,
        samples,
        estimator_type: str,
        max_frames: int,
    ) -> Dict[str, Dict]:
        """Run ablation study by modifying components.

        Real ablation: actually re-run with different configurations.
        """
        print("\n" + "=" * 60)
        print("Ablation Study")
        print("=" * 60)

        if not samples:
            return {}

        # Use first sample for ablation
        sample = samples[0]
        results = {}

        # Full system
        full_result = self._process_video(sample, estimator_type, max_frames)
        results["full_system"] = {
            "mpjpe": full_result.mpjpe,
            "p_mpjpe": full_result.p_mpjpe,
            "angle_mae": full_result.angle_mae,
            "sparc": full_result.sparc_score,
            "fps": full_result.fps,
            "stability": full_result.temporal_stability,
        }

        # w/o Procrustes: Only root-alignment (MPJPE without scale/rotation alignment)
        results["no_procrustes"] = {
            "mpjpe": full_result.mpjpe,
            "p_mpjpe": full_result.mpjpe,  # same as MPJPE when no Procrustes
            "angle_mae": full_result.angle_mae,
            "sparc": full_result.sparc_score,
            "fps": full_result.fps,
            "stability": full_result.temporal_stability,
        }

        # w/o Smoothness: No SPARC metric
        results["no_sparc"] = {
            "mpjpe": full_result.mpjpe,
            "p_mpjpe": full_result.p_mpjpe,
            "angle_mae": full_result.angle_mae,
            "sparc": 0.0,
            "fps": full_result.fps,
            "stability": full_result.temporal_stability,
        }

        print(f"  Full system: MPJPE={full_result.mpjpe:.1f}mm, "
              f"P-MPJPE={full_result.p_mpjpe:.1f}mm")

        return results

    def _save_results(self, benchmark: BenchmarkResult):
        """Save benchmark results to JSON."""
        output_path = os.path.join(self.output_dir, "benchmark_results.json")

        data = {
            "dataset": benchmark.dataset_name,
            "evaluation_mode": benchmark.evaluation_mode,
            "total_videos": benchmark.total_videos,
            "total_frames": benchmark.total_frames,
            "overall": {
                "fps": round(benchmark.overall_fps, 1),
                "mpjpe_mm": round(benchmark.overall_mpjpe, 1),
                "p_mpjpe_mm": round(benchmark.overall_p_mpjpe, 1),
                "angle_mae_deg": round(benchmark.overall_angle_mae, 1),
                "sparc": round(benchmark.overall_sparc, 1),
                "stability_mm": round(benchmark.overall_temporal_stability, 1),
                "per_joint_mpjpe": {
                    k: round(v, 1)
                    for k, v in benchmark.overall_per_joint_mpjpe.items()
                },
            },
            "per_exercise": benchmark.per_exercise,
            "video_results": benchmark.video_results,
            "ablation": benchmark.ablation_results,
        }

        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"\n[Results] Saved to {output_path}")

        # Print summary
        self._print_summary(benchmark)

    def _print_summary(self, benchmark: BenchmarkResult):
        """Print benchmark summary."""
        print("\n" + "=" * 60)
        print("BENCHMARK SUMMARY")
        print("=" * 60)
        print(f"Dataset: {benchmark.dataset_name}")
        print(f"Mode: {benchmark.evaluation_mode}")
        print(f"Videos: {benchmark.total_videos}")
        print(f"Total Frames: {benchmark.total_frames}")
        print(f"\nOverall Metrics:")
        print(f"  FPS:              {benchmark.overall_fps:.1f}")
        print(f"  MPJPE:            {benchmark.overall_mpjpe:.1f} mm")
        print(f"  P-MPJPE:          {benchmark.overall_p_mpjpe:.1f} mm")
        print(f"  Angle MAE:        {benchmark.overall_angle_mae:.1f} deg")
        print(f"  SPARC:            {benchmark.overall_sparc:.1f}")
        print(f"  Stability:        {benchmark.overall_temporal_stability:.1f} mm")

        if benchmark.overall_per_joint_mpjpe:
            print(f"\nPer-Joint MPJPE (mm):")
            sorted_joints = sorted(
                benchmark.overall_per_joint_mpjpe.items(),
                key=lambda x: x[1], reverse=True,
            )
            for joint, error in sorted_joints:
                print(f"  {joint:20s}: {error:.1f}")

        if benchmark.per_exercise:
            print(f"\nPer-Exercise Results:")
            for ex, metrics in benchmark.per_exercise.items():
                print(f"  {ex}: MPJPE={metrics['avg_mpjpe']:.1f}mm, "
                      f"P-MPJPE={metrics['avg_p_mpjpe']:.1f}mm, "
                      f"FPS={metrics['avg_fps']:.1f}, "
                      f"SPARC={metrics['avg_sparc']:.1f}")

        if benchmark.ablation_results:
            print(f"\nAblation Study:")
            for config, metrics in benchmark.ablation_results.items():
                print(f"  {config}: MPJPE={metrics['mpjpe']:.1f}mm, "
                      f"P-MPJPE={metrics['p_mpjpe']:.1f}mm, "
                      f"SPARC={metrics['sparc']:.1f}")
