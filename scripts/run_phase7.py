#!/usr/bin/env python3
"""
Phase 7: Benchmark Evaluation Runner.

Runs the full benchmark evaluation pipeline:
1. Load dataset
2. Run pose estimation on all videos
3. Compute metrics (MPJPE, angle MAE, SPARC, FPS)
4. Run ablation study
5. Generate figures for paper

Usage:
    python scripts/run_phase7.py [--data-dir DATA_DIR] [--max-frames N]
"""
import os
import sys
import argparse

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)


def main():
    parser = argparse.ArgumentParser(description="Phase 7: Benchmark Evaluation")
    parser.add_argument("--data-dir", default="data",
                        help="Path to data directory (default: data)")
    parser.add_argument("--max-frames", type=int, default=100,
                        help="Max frames per video (default: 100)")
    parser.add_argument("--output-dir", default="evaluation/results",
                        help="Output directory for results")
    parser.add_argument("--generate-figures", action="store_true", default=True,
                        help="Generate paper figures")
    args = parser.parse_args()

    # Resolve paths
    data_dir = os.path.join(project_root, args.data_dir)
    output_dir = os.path.join(project_root, args.output_dir)

    print("=" * 60)
    print("ADAPT-Rehab: Phase 7 - Benchmark Evaluation")
    print("=" * 60)
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Max frames per video: {args.max_frames}")
    print()

    # Run benchmark
    from evaluation.benchmark_runner import BenchmarkRunner
    runner = BenchmarkRunner(data_dir, output_dir)
    results = runner.run_full_benchmark(max_frames_per_video=args.max_frames)

    # Generate figures
    if args.generate_figures:
        from evaluation.visualize import generate_all_figures
        results_path = os.path.join(output_dir, "benchmark_results.json")
        figures_dir = os.path.join(project_root, "paper", "figures")
        generate_all_figures(results_path, figures_dir)

    print("\n" + "=" * 60)
    print("Phase 7 Complete!")
    print("=" * 60)
    print(f"Results: {output_dir}/benchmark_results.json")
    print(f"Figures: paper/figures/")


if __name__ == "__main__":
    main()
