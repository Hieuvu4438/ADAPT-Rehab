"""
Visualization module for generating paper figures from benchmark results.
"""
import os
import json
import numpy as np

# Use non-interactive backend
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load_results(results_path: str) -> dict:
    """Load benchmark results from JSON."""
    with open(results_path) as f:
        return json.load(f)


def generate_mpjpe_comparison(results: dict, output_path: str):
    """Generate MPJPE comparison bar chart (Figure 2)."""
    # Compare our system vs baselines from literature
    methods = ['MediaPipe\n(BlazePose)', 'OpenPose', 'MotionBERT', 'Ours\n(ADAPT-Rehab)']

    # Our MPJPE from benchmark
    our_mpjpe = results['overall']['mpjpe']

    # Literature baselines (from RESEARCH_STRATEGY.md)
    mpjpe_values = [63.0, 55.0, 37.8, our_mpjpe]
    colors = ['#cccccc', '#cccccc', '#cccccc', '#2196F3']

    fig, ax = plt.subplots(figsize=(6, 3.5))
    bars = ax.bar(methods, mpjpe_values, color=colors, edgecolor='black', linewidth=0.5)

    # Add value labels
    for bar, val in zip(bars, mpjpe_values):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                f'{val:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_ylabel('MPJPE (mm)', fontsize=10)
    ax.set_title('Pose Estimation Accuracy Comparison', fontsize=11, fontweight='bold')
    ax.set_ylim(0, max(mpjpe_values) * 1.15)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='x', labelsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[Figure] Saved MPJPE comparison: {output_path}")


def generate_ablation_table(results: dict, output_path: str):
    """Generate ablation study table as figure (for paper)."""
    ablation = results.get('ablation', {})
    if not ablation:
        return

    configs = list(ablation.keys())
    metrics = ['mpjpe', 'sparc', 'fps']

    fig, axes = plt.subplots(1, 3, figsize=(8, 3))

    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        values = [ablation[c].get(metric, 0) for c in configs]
        colors = ['#2196F3' if c == 'full_system' else '#ff9800' for c in configs]

        short_names = [c.replace('no_', 'w/o ').replace('_', ' ').title() for c in configs]
        bars = ax.barh(short_names, values, color=colors, edgecolor='black', linewidth=0.5)

        ax.set_xlabel(metric.upper(), fontsize=9)
        ax.set_title(metric.upper(), fontsize=10, fontweight='bold')
        ax.tick_params(axis='y', labelsize=7)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[Figure] Saved ablation study: {output_path}")


def generate_per_exercise_chart(results: dict, output_path: str):
    """Generate per-exercise performance chart."""
    per_ex = results.get('per_exercise', {})
    if not per_ex:
        return

    exercises = list(per_ex.keys())
    mpjpe_vals = [per_ex[e]['avg_mpjpe'] for e in exercises]
    sparc_vals = [per_ex[e]['avg_sparc'] for e in exercises]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3.5))

    x = np.arange(len(exercises))
    ax1.bar(x, mpjpe_vals, color='#2196F3', edgecolor='black', linewidth=0.5)
    ax1.set_xticks(x)
    ax1.set_xticklabels(exercises, rotation=30, ha='right', fontsize=8)
    ax1.set_ylabel('MPJPE (mm)', fontsize=9)
    ax1.set_title('MPJPE by Exercise', fontsize=10, fontweight='bold')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    ax2.bar(x, sparc_vals, color='#4CAF50', edgecolor='black', linewidth=0.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels(exercises, rotation=30, ha='right', fontsize=8)
    ax2.set_ylabel('SPARC Score', fontsize=9)
    ax2.set_title('Smoothness by Exercise', fontsize=10, fontweight='bold')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[Figure] Saved per-exercise chart: {output_path}")


def generate_all_figures(results_path: str, output_dir: str = "paper/figures"):
    """Generate all paper figures from benchmark results."""
    os.makedirs(output_dir, exist_ok=True)
    results = load_results(results_path)

    generate_mpjpe_comparison(results, os.path.join(output_dir, "fig2_mpjpe_comparison.pdf"))
    generate_ablation_table(results, os.path.join(output_dir, "fig3_ablation_study.pdf"))
    generate_per_exercise_chart(results, os.path.join(output_dir, "fig4_per_exercise.pdf"))

    print(f"\n[Figures] All figures saved to {output_dir}/")
