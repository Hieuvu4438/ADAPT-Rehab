#!/usr/bin/env python3
"""
ADAPT-Rehab: Cross-Pose Experiment

Runs comparison on all same-pose and different-pose video pairs
to verify that same-pose pairs score significantly higher.

Usage:
    python scripts/run_experiment.py \
        --dataset data/yoga_datasets/Yoga_Vid_Collected \
        --max-frames 60 \
        --max-pairs 10

Author: ADAPT-Rehab Team
Version: 3.1.0
"""

import argparse
import sys
import os
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.run_comparison import run_full_comparison


def discover_videos(dataset_dir: str) -> Dict[str, List[str]]:
    """Discover videos grouped by pose type.

    Args:
        dataset_dir: Path to dataset directory.

    Returns:
        Dict mapping pose_name -> list of video paths.
    """
    videos = defaultdict(list)
    dataset_path = Path(dataset_dir)

    for f in sorted(dataset_path.glob("*.mp4")):
        # Parse: PersonName_PoseName.mp4
        name = f.stem
        parts = name.split("_", 1)
        if len(parts) == 2:
            person, pose = parts
            videos[pose].append(str(f))

    return dict(videos)


def create_pairs(
    videos_by_pose: Dict[str, List[str]],
    max_pairs_per_type: int = 5,
) -> Tuple[List[Tuple[str, str, str]], List[Tuple[str, str, str]]]:
    """Create same-pose and different-pose pairs.

    Args:
        videos_by_pose: Dict mapping pose -> list of video paths.
        max_pairs_per_type: Max pairs per pose type.

    Returns:
        Tuple of (same_pose_pairs, diff_pose_pairs).
        Each pair is (video_a, video_b, label).
    """
    same_pose_pairs = []
    diff_pose_pairs = []

    poses = list(videos_by_pose.keys())

    # Same-pose pairs: pick 2 videos from same pose (different people)
    for pose in poses:
        vids = videos_by_pose[pose]
        if len(vids) < 2:
            continue
        count = 0
        for i in range(len(vids)):
            for j in range(i + 1, len(vids)):
                if count >= max_pairs_per_type:
                    break
                same_pose_pairs.append((vids[i], vids[j], f"same_{pose}"))
                count += 1
            if count >= max_pairs_per_type:
                break

    # Different-pose pairs: pick 1 video from each of 2 different poses
    count = 0
    for i in range(len(poses)):
        for j in range(i + 1, len(poses)):
            if count >= max_pairs_per_type * 2:
                break
            v1 = videos_by_pose[poses[i]][0]
            v2 = videos_by_pose[poses[j]][0]
            diff_pose_pairs.append((v1, v2, f"diff_{poses[i]}_vs_{poses[j]}"))
            count += 1
        if count >= max_pairs_per_type * 2:
            break

    return same_pose_pairs, diff_pose_pairs


def run_experiment(
    dataset_dir: str,
    output_dir: str,
    max_frames: int = 60,
    max_pairs: int = 5,
    pose_backend: str = "mediapipe_fallback",
) -> dict:
    """Run the full cross-pose experiment.

    Args:
        dataset_dir: Path to dataset directory.
        output_dir: Path to save results.
        max_frames: Max frames per video.
        max_pairs: Max pairs per type.
        pose_backend: Pose estimation backend.

    Returns:
        Experiment results dict.
    """
    print("=" * 70)
    print("ADAPT-Rehab: Cross-Pose Experiment")
    print("=" * 70)

    # 1. Discover videos
    print("\n[1/4] Discovering videos...")
    videos_by_pose = discover_videos(dataset_dir)
    total_videos = sum(len(v) for v in videos_by_pose.values())
    print(f"  Found {total_videos} videos across {len(videos_by_pose)} poses:")
    for pose, vids in sorted(videos_by_pose.items()):
        print(f"    {pose}: {len(vids)} videos")

    # 2. Create pairs
    print("\n[2/4] Creating pairs...")
    same_pairs, diff_pairs = create_pairs(videos_by_pose, max_pairs)
    print(f"  Same-pose pairs: {len(same_pairs)}")
    print(f"  Diff-pose pairs: {len(diff_pairs)}")
    print(f"  Total comparisons: {len(same_pairs) + len(diff_pairs)}")

    # 3. Run comparisons
    print("\n[3/4] Running comparisons...")
    os.makedirs(output_dir, exist_ok=True)

    same_results = []
    diff_results = []

    all_pairs = same_pairs + diff_pairs
    for idx, (v1, v2, label) in enumerate(all_pairs):
        pair_type = "SAME" if label.startswith("same_") else "DIFF"
        print(f"\n  [{idx+1}/{len(all_pairs)}] {pair_type}: {Path(v1).stem} vs {Path(v2).stem}")

        try:
            result = run_full_comparison(
                reference_path=v1,
                user_path=v2,
                output_path=None,  # Don't save individual results
                max_frames=max_frames,
                pose_backend=pose_backend,
            )

            entry = {
                "pair_type": pair_type,
                "label": label,
                "video_a": Path(v1).stem,
                "video_b": Path(v2).stem,
                "total_score": result["scoring"]["total_score"],
                "rom_score": result["scoring"]["rom_score"],
                "stability_score": result["scoring"]["stability_score"],
                "flow_score": result["scoring"]["flow_score"],
                "symmetry_score": result["scoring"]["symmetry_score"],
                "compensation_score": result["scoring"]["compensation_score"],
                "smoothness_score": result["scoring"]["smoothness_score"],
                "dtw_similarity": result["dtw_comparison"]["overall_similarity"],
                "procrustes_similarity": result["procrustes_similarity"],
            }

            if pair_type == "SAME":
                same_results.append(entry)
            else:
                diff_results.append(entry)

            print(f"    Total: {entry['total_score']:.1f} | DTW: {entry['dtw_similarity']:.1f}% | Proc: {entry['procrustes_similarity']:.1f}%")

        except Exception as e:
            print(f"    ✗ Error: {e}")

    # 4. Analyze results
    print("\n[4/4] Analyzing results...")

    def stats(values):
        if not values:
            return {"mean": 0, "std": 0, "min": 0, "max": 0, "n": 0}
        arr = np.array(values)
        return {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "n": len(arr),
        }

    same_scores = [r["total_score"] for r in same_results]
    diff_scores = [r["total_score"] for r in diff_results]
    same_dtw = [r["dtw_similarity"] for r in same_results]
    diff_dtw = [r["dtw_similarity"] for r in diff_results]
    same_proc = [r["procrustes_similarity"] for r in same_results]
    diff_proc = [r["procrustes_similarity"] for r in diff_results]

    # Per-dimension comparison
    dimensions = ["rom_score", "stability_score", "flow_score",
                   "symmetry_score", "compensation_score", "smoothness_score"]
    dim_comparison = {}
    for dim in dimensions:
        same_vals = [r[dim] for r in same_results]
        diff_vals = [r[dim] for r in diff_results]
        dim_comparison[dim] = {
            "same": stats(same_vals),
            "diff": stats(diff_vals),
            "gap": stats(same_vals)["mean"] - stats(diff_vals)["mean"],
        }

    analysis = {
        "total_score": {
            "same_pose": stats(same_scores),
            "diff_pose": stats(diff_scores),
            "gap": stats(same_scores)["mean"] - stats(diff_scores)["mean"],
            "separation_ratio": stats(same_scores)["mean"] / max(stats(diff_scores)["mean"], 0.01),
        },
        "dtw_similarity": {
            "same_pose": stats(same_dtw),
            "diff_pose": stats(diff_dtw),
            "gap": stats(same_dtw)["mean"] - stats(diff_dtw)["mean"],
        },
        "procrustes_similarity": {
            "same_pose": stats(same_proc),
            "diff_pose": stats(diff_proc),
            "gap": stats(same_proc)["mean"] - stats(diff_proc)["mean"],
        },
        "per_dimension": dim_comparison,
    }

    # Print summary
    print("\n" + "=" * 70)
    print("EXPERIMENT RESULTS")
    print("=" * 70)

    print(f"\n{'Metric':<25} {'Same-Pose':>12} {'Diff-Pose':>12} {'Gap':>10} {'Ratio':>8}")
    print("-" * 70)

    ts = analysis["total_score"]
    print(f"{'Total Score':<25} {ts['same_pose']['mean']:>10.1f}±{ts['same_pose']['std']:.1f} "
          f"{ts['diff_pose']['mean']:>10.1f}±{ts['diff_pose']['std']:.1f} "
          f"{ts['gap']:>+8.1f} {ts['separation_ratio']:>7.2f}x")

    dtw = analysis["dtw_similarity"]
    print(f"{'DTW Similarity (%)':<25} {dtw['same_pose']['mean']:>10.1f}±{dtw['same_pose']['std']:.1f} "
          f"{dtw['diff_pose']['mean']:>10.1f}±{dtw['diff_pose']['std']:.1f} "
          f"{dtw['gap']:>+8.1f}")

    proc = analysis["procrustes_similarity"]
    print(f"{'Procrustes Similarity (%)':<25} {proc['same_pose']['mean']:>10.1f}±{proc['same_pose']['std']:.1f} "
          f"{proc['diff_pose']['mean']:>10.1f}±{proc['diff_pose']['std']:.1f} "
          f"{proc['gap']:>+8.1f}")

    print(f"\n{'Dimension':<25} {'Same-Pose':>12} {'Diff-Pose':>12} {'Gap':>10}")
    print("-" * 60)
    for dim, data in dim_comparison.items():
        dim_name = dim.replace("_score", "").replace("_", " ").title()
        print(f"{dim_name:<25} {data['same']['mean']:>10.1f}±{data['same']['std']:.1f} "
              f"{data['diff']['mean']:>10.1f}±{data['diff']['std']:.1f} "
              f"{data['gap']:>+8.1f}")

    # Verification
    print(f"\n{'='*70}")
    print("VERIFICATION")
    print(f"{'='*70}")
    score_ok = ts["gap"] > 0
    dtw_ok = dtw["gap"] > 0
    print(f"  Same-pose total > Diff-pose total: {'✅ YES' if score_ok else '❌ NO'} (gap={ts['gap']:+.1f})")
    print(f"  Same-pose DTW > Diff-pose DTW:     {'✅ YES' if dtw_ok else '❌ NO'} (gap={dtw['gap']:+.1f})")
    if score_ok and dtw_ok:
        print(f"\n  ✅ HYPOTHESIS CONFIRMED: Same-pose pairs score higher than different-pose pairs")
    else:
        print(f"\n  ⚠ HYPOTHESIS NOT CONFIRMED — need more data or tuning")

    # Save full results
    experiment_results = {
        "experiment": "Cross-Pose Comparison",
        "dataset": dataset_dir,
        "total_videos": total_videos,
        "total_poses": len(videos_by_pose),
        "same_pose_pairs": len(same_results),
        "diff_pose_pairs": len(diff_results),
        "analysis": analysis,
        "same_pose_details": same_results,
        "diff_pose_details": diff_results,
    }

    output_path = os.path.join(output_dir, "experiment_results.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(experiment_results, f, indent=2, ensure_ascii=False)
    print(f"\nFull results saved to: {output_path}")

    return experiment_results


def main():
    parser = argparse.ArgumentParser(description="ADAPT-Rehab Cross-Pose Experiment")
    parser.add_argument("--dataset", type=str,
                        default="data/yoga_datasets/Yoga_Vid_Collected",
                        help="Path to dataset directory")
    parser.add_argument("--output", type=str,
                        default="evaluation/output",
                        help="Output directory for results")
    parser.add_argument("--max-frames", type=int, default=60,
                        help="Max frames per video (0 = all)")
    parser.add_argument("--max-pairs", type=int, default=5,
                        help="Max pairs per pose type")
    parser.add_argument("--pose-backend", type=str, default="mediapipe_fallback",
                        choices=["rtmw3d", "mediapipe_fallback"])
    args = parser.parse_args()

    if not os.path.exists(args.dataset):
        print(f"Error: Dataset not found: {args.dataset}")
        sys.exit(1)

    run_experiment(
        dataset_dir=args.dataset,
        output_dir=args.output,
        max_frames=args.max_frames,
        max_pairs=args.max_pairs,
        pose_backend=args.pose_backend,
    )


if __name__ == "__main__":
    main()
