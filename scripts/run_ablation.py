"""
Ablation Study for ADAPT-Rehab.

Following the pattern from MotionBERT (ICCV 2023) and MotionAGFormer (CVPR 2024):
- Full system vs. w/o components
- Report MPJPE, P-MPJPE, FPS for each configuration
"""

import os
import sys
import json
import time
import numpy as np
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.metrics.mpjpe import compute_mpjpe, compute_p_mpjpe, compute_per_joint_mpjpe
from evaluation.datasets import load_yoga_dataset


def run_ablation_study():
    """Run ablation study on Yoga-Collect dataset."""
    print("=" * 70)
    print("ABLATION STUDY")
    print("Following MotionBERT/MotionAGFormer ablation pattern")
    print("=" * 70)

    # Load dataset
    samples = load_yoga_dataset("data")
    if not samples:
        print("[Error] No samples found!")
        return None

    print(f"\nDataset: Yoga-Collect ({len(samples)} videos)")

    # Define ablation configurations
    configs = {
        "full_system": {
            "description": "Full system with all components",
            "pose": "rtmw3d",
            "use_quaternion": True,
            "use_sparc": True,
            "use_compensation": True,
            "use_llm": True,
        },
        "no_3d_pose": {
            "description": "w/o 3D Pose (use RTMW3D without quaternion)",
            "pose": "rtmw3d",
            "use_quaternion": True,
            "use_sparc": True,
            "use_compensation": True,
            "use_llm": True,
        },
        "no_quaternion": {
            "description": "w/o Quaternion (use dot-product angles)",
            "pose": "rtmw3d",
            "use_quaternion": False,
            "use_sparc": True,
            "use_compensation": True,
            "use_llm": True,
        },
        "no_sparc": {
            "description": "w/o SPARC (use jerk only)",
            "pose": "rtmw3d",
            "use_quaternion": True,
            "use_sparc": False,
            "use_compensation": True,
            "use_llm": True,
        },
        "no_compensation": {
            "description": "w/o Compensation detection",
            "pose": "rtmw3d",
            "use_quaternion": True,
            "use_sparc": True,
            "use_compensation": False,
            "use_llm": True,
        },
        "no_llm": {
            "description": "w/o LLM (rule-based feedback)",
            "pose": "rtmw3d",
            "use_quaternion": True,
            "use_sparc": True,
            "use_compensation": True,
            "use_llm": False,
        },
    }

    # Run each configuration
    results = {}
    for config_name, config in configs.items():
        print(f"\n{'='*50}")
        print(f"Running: {config_name} - {config['description']}")
        print(f"{'='*50}")

        result = run_single_config(config, samples[:10])  # Use 10 videos for speed
        results[config_name] = result

    # Compute deltas
    baseline = results["full_system"]
    for config_name, result in results.items():
        if config_name != "full_system":
            result["delta_mpjpe"] = result["mpjpe"] - baseline["mpjpe"]
            result["delta_p_mpjpe"] = result["p_mpjpe"] - baseline["p_mpjpe"]
            result["delta_fps"] = result["fps"] - baseline["fps"]

    # Save results
    save_ablation_results(results)

    # Print summary
    print_ablation_summary(results)

    return results


def run_single_config(config: Dict, samples: list) -> Dict:
    """Run a single ablation configuration."""
    from core.pose3d import create_estimator

    # Create estimator
    pose_type = config["pose"]
    try:
        estimator = create_estimator(pose_type)
        if not estimator.initialize():
            print(f"  [Warning] {pose_type} init failed")
            return {"mpjpe": 0, "p_mpjpe": 0, "fps": 0, "error": f"{pose_type} init failed"}
    except Exception as e:
        print(f"  [Error] {e}")
        return {"mpjpe": 0, "p_mpjpe": 0, "fps": 0, "error": str(e)}

    # Process videos
    all_keypoints = []
    fps_list = []
    angles_list = []

    for i, sample in enumerate(samples):
        print(f"  [{i+1}/{len(samples)}] {sample.person_name}_{sample.exercise_type}", end="")
        result = process_video_ablation(sample, estimator, config)
        if result:
            all_keypoints.append(result["keypoints"])
            fps_list.append(result["fps"])
            if result["angles"]:
                angles_list.extend(result["angles"])
        print(f" - FPS: {result['fps']:.1f}" if result else " - Failed")

    estimator.close()

    # Compute metrics
    if not all_keypoints:
        return {"mpjpe": 0, "p_mpjpe": 0, "fps": 0, "error": "No valid frames"}

    # MPJPE (self-consistency)
    mpjpe_values = []
    p_mpjpe_values = []

    for kps in all_keypoints:
        if len(kps) > 1:
            mean_kps = np.mean(kps, axis=0)
            mean_kps_expanded = np.broadcast_to(mean_kps, kps.shape)

            mpjpe_val = compute_mpjpe(kps, mean_kps_expanded)
            mpjpe_values.append(mpjpe_val)

            p_mpjpe_val = compute_p_mpjpe(kps, mean_kps_expanded)
            p_mpjpe_values.append(p_mpjpe_val)

    # SPARC
    sparc_values = []
    if config["use_sparc"] and angles_list:
        from scripts.comprehensive_eval import sparc_balasubramanian
        for angle_dict in angles_list:
            if isinstance(angle_dict, dict):
                for joint, angle in angle_dict.items():
                    if isinstance(angle, (int, float)):
                        sparc_val, _, _ = sparc_balasubramanian(np.array([angle]), fs=30.0)
                        sparc_values.append(sparc_val)

    return {
        "mpjpe": float(np.mean(mpjpe_values)) if mpjpe_values else 0,
        "p_mpjpe": float(np.mean(p_mpjpe_values)) if p_mpjpe_values else 0,
        "fps": float(np.mean(fps_list)) if fps_list else 0,
        "num_videos": len(all_keypoints),
        "total_frames": sum(len(kps) for kps in all_keypoints),
        "sparc": float(np.mean(sparc_values)) if sparc_values else 0,
    }


def process_video_ablation(sample, estimator, config: Dict) -> Optional[Dict]:
    """Process a single video for ablation study."""
    import cv2

    cap = cv2.VideoCapture(sample.video_path)
    if not cap.isOpened():
        return None

    keypoints_list = []
    angles_list = []
    inference_times = []

    frame_idx = 0
    max_frames = 30  # Use fewer frames for ablation speed

    while cap.isOpened() and frame_idx < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        t0 = time.time()
        pose_result = estimator.estimate(frame, frame_idx * 33)
        inference_ms = (time.time() - t0) * 1000

        if pose_result.is_valid and pose_result.keypoints_3d is not None:
            keypoints_list.append(pose_result.keypoints_3d.copy())
            if config["use_quaternion"]:
                angles_list.append(pose_result.joint_angles_quaternion or pose_result.joint_angles)
            else:
                angles_list.append(pose_result.joint_angles)
            inference_times.append(inference_ms)

        frame_idx += 1

    cap.release()

    if len(keypoints_list) < 2:
        return None

    return {
        "keypoints": np.array(keypoints_list),
        "angles": angles_list,
        "fps": 1000.0 / np.mean(inference_times) if inference_times else 0,
    }


def save_ablation_results(results: Dict):
    """Save ablation results to JSON."""
    output_path = "evaluation/results/ablation_results.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Convert to serializable format
    serializable = {}
    for config_name, result in results.items():
        serializable[config_name] = {
            k: float(v) if isinstance(v, (np.floating, float)) else v
            for k, v in result.items()
        }

    with open(output_path, 'w') as f:
        json.dump(serializable, f, indent=2)

    print(f"\n[Results] Saved to {output_path}")


def print_ablation_summary(results: Dict):
    """Print ablation study summary."""
    print("\n" + "=" * 70)
    print("ABLATION STUDY RESULTS")
    print("=" * 70)

    baseline = results.get("full_system", {})

    print(f"\n{'Config':<25} {'MPJPE (mm)':<12} {'Δ MPJPE':<10} {'P-MPJPE':<10} {'FPS':<8} {'Δ FPS':<8}")
    print("-" * 75)

    for config_name, result in results.items():
        mpjpe = result.get("mpjpe", 0)
        p_mpjpe = result.get("p_mpjpe", 0)
        fps = result.get("fps", 0)

        if config_name == "full_system":
            delta_mpjpe = "-"
            delta_fps = "-"
        else:
            delta_mpjpe = f"+{result.get('delta_mpjpe', 0):.2f}" if result.get('delta_mpjpe', 0) > 0 else f"{result.get('delta_mpjpe', 0):.2f}"
            delta_fps = f"{result.get('delta_fps', 0):.1f}"

        print(f"{config_name:<25} {mpjpe:<12.2f} {delta_mpjpe:<10} {p_mpjpe:<10.2f} {fps:<8.1f} {delta_fps:<8}")

    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)

    if "full_system" in results:
        baseline_mpjpe = results["full_system"].get("mpjpe", 0)
        for config_name, result in results.items():
            if config_name != "full_system":
                delta = result.get("delta_mpjpe", 0)
                if delta > 0:
                    print(f"  {config_name}: +{delta:.2f}mm MPJPE (worse)")
                elif delta < 0:
                    print(f"  {config_name}: {delta:.2f}mm MPJPE (better)")
                else:
                    print(f"  {config_name}: no change")


if __name__ == "__main__":
    run_ablation_study()
