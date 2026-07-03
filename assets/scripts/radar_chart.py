"""
6-D Performance Scoring Radar Chart
ADAPT-Rehab — conference figure quality (300 DPI)

Dimensions: ROM · Flow · Symmetry · Stability · Compensation · Smoothness
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import os

# ─── Data ───────────────────────────────────────────────────────────────────
DIMS = ["ROM", "Flow\n(DTW)", "Symmetry", "Stability", "Compensation", "Smoothness\n(SPARC)"]
N = len(DIMS)

# Example scores: Session A (good), Session B (struggling)
SESSION_A = np.array([88, 82, 91, 76, 85, 79])   # strong session
SESSION_B = np.array([63, 55, 70, 48, 72, 58])   # fatigued session
TARGET    = np.array([100]*N)                      # full target ring (dashed)

# ─── Setup angles ────────────────────────────────────────────────────────────
angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
angles += angles[:1]   # close polygon

def close(arr):
    return np.append(arr, arr[0])

# ─── Color palette (muted, conference-friendly) ───────────────────────────
COLOR_A   = "#3B82F6"   # blue
COLOR_B   = "#F97316"   # orange
FILL_A    = "#BFDBFE"
FILL_B    = "#FED7AA"
GRID_CLR  = "#CBD5E1"
BG        = "#FFFFFF"
LABEL_CLR = "#1E293B"

# ─── Figure ──────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(
    1, 2,
    figsize=(9, 4.2),
    subplot_kw=dict(polar=True),
    facecolor=BG,
)
fig.patch.set_facecolor(BG)

TITLES = ["Session A  (Good Performance)", "Session B  (Fatigued / Struggling)"]
SCORES  = [SESSION_A, SESSION_B]
COLORS  = [COLOR_A, COLOR_B]
FILLS   = [FILL_A, FILL_B]

for ax, title, scores, color, fill in zip(axes, TITLES, SCORES, COLORS, FILLS):
    ax.set_facecolor(BG)

    # ── Grid rings ──────────────────────────────────────────────────────────
    ax.set_ylim(0, 100)
    ax.set_yticks([25, 50, 75, 100])
    ax.set_yticklabels(["25", "50", "75", "100"],
                       color="#94A3B8", fontsize=6.5, fontfamily="DejaVu Sans")
    ax.yaxis.set_tick_params(pad=2)
    ax.set_rlabel_position(30)

    for ring in [25, 50, 75, 100]:
        ring_vals = [ring] * (N + 1)
        ax.plot(angles, ring_vals,
                color=GRID_CLR, linewidth=0.6, linestyle="--" if ring == 100 else "-",
                zorder=1)

    ax.set_thetagrids(np.degrees(angles[:-1]), DIMS,
                      fontsize=8.5, fontfamily="DejaVu Sans",
                      color=LABEL_CLR, fontweight="semibold")

    ax.spines["polar"].set_visible(False)
    ax.grid(color=GRID_CLR, linewidth=0.5, linestyle="dotted", zorder=1)
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    # ── Score polygon ────────────────────────────────────────────────────────
    vals = close(scores)
    ax.plot(angles, vals, color=color, linewidth=2.2, zorder=3)
    ax.fill(angles, vals, color=fill, alpha=0.55, zorder=2)

    # ── Score dots ───────────────────────────────────────────────────────────
    for ang, val in zip(angles[:-1], scores):
        ax.scatter(ang, val, color=color, s=32, zorder=4, linewidth=0)
        ax.text(ang, val + 7, f"{val}",
                ha="center", va="center",
                fontsize=7, color=color, fontweight="bold",
                fontfamily="DejaVu Sans")

    # ── Total score badge ────────────────────────────────────────────────────
    weights = np.array([0.25, 0.20, 0.15, 0.15, 0.15, 0.10])
    total = float(np.dot(scores, weights))
    ax.text(0, 0, f"{total:.0f}",
            ha="center", va="center",
            fontsize=16, color=color, fontweight="bold",
            fontfamily="DejaVu Sans")
    ax.text(0, -22, "/100",
            ha="center", va="center",
            fontsize=7, color="#94A3B8", fontfamily="DejaVu Sans")

    # ── Subplot title ─────────────────────────────────────────────────────────
    ax.set_title(title, pad=18, fontsize=9, fontfamily="DejaVu Sans",
                 color=LABEL_CLR, fontweight="bold")

# ─── Shared legend ───────────────────────────────────────────────────────────
legend_patches = [
    mpatches.Patch(facecolor=FILL_A, edgecolor=COLOR_A, linewidth=1.5,
                   label="Session A  (Good)"),
    mpatches.Patch(facecolor=FILL_B, edgecolor=COLOR_B, linewidth=1.5,
                   label="Session B  (Fatigued)"),
]
fig.legend(handles=legend_patches, loc="lower center", ncol=2,
           fontsize=8, frameon=False,
           bbox_to_anchor=(0.5, -0.01))

# ─── Figure annotation ────────────────────────────────────────────────────────
fig.text(0.5, 1.01,
         "Fig. X.  6-D Performance Scoring — ADAPT-Rehab",
         ha="center", va="bottom",
         fontsize=9.5, fontweight="bold",
         fontfamily="DejaVu Sans", color=LABEL_CLR)
fig.text(0.5, -0.06,
         "Weights: ROM 25% · Flow 20% · Symmetry 15% · Stability 15% · "
         "Compensation 15% · Smoothness 10%",
         ha="center", va="top",
         fontsize=7, color="#64748B", fontfamily="DejaVu Sans")

plt.tight_layout(pad=1.4)

# ─── Save ─────────────────────────────────────────────────────────────────────
out_dir = os.path.join(os.path.dirname(__file__), "..", "figures")
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "radar_6d_scoring.png")
fig.savefig(out_path, dpi=300, bbox_inches="tight",
            facecolor=BG, transparent=False)
print(f"Saved → {os.path.abspath(out_path)}")

# ── Single-chart version (embed-ready for draw.io) ───────────────────────────
fig2, ax2 = plt.subplots(figsize=(3.6, 3.6),
                         subplot_kw=dict(polar=True),
                         facecolor=BG)
fig2.patch.set_facecolor(BG)
ax2.set_facecolor(BG)
ax2.set_ylim(0, 100)
ax2.set_yticks([25, 50, 75, 100])
ax2.set_yticklabels(["25", "50", "75", "100"],
                    color="#94A3B8", fontsize=6, fontfamily="DejaVu Sans")
ax2.set_rlabel_position(30)
for ring in [25, 50, 75, 100]:
    ring_vals = [ring] * (N + 1)
    ax2.plot(angles, ring_vals,
             color=GRID_CLR, linewidth=0.6,
             linestyle="--" if ring == 100 else "-", zorder=1)
ax2.set_thetagrids(np.degrees(angles[:-1]), DIMS,
                   fontsize=8, fontfamily="DejaVu Sans",
                   color=LABEL_CLR, fontweight="semibold")
ax2.spines["polar"].set_visible(False)
ax2.grid(color=GRID_CLR, linewidth=0.5, linestyle="dotted", zorder=1)
ax2.set_theta_offset(np.pi / 2)
ax2.set_theta_direction(-1)

vals_a = close(SESSION_A)
ax2.plot(angles, vals_a, color=COLOR_A, linewidth=2.2, zorder=3)
ax2.fill(angles, vals_a, color=FILL_A, alpha=0.55, zorder=2)
for ang, val in zip(angles[:-1], SESSION_A):
    ax2.scatter(ang, val, color=COLOR_A, s=28, zorder=4)
total_a = float(np.dot(SESSION_A, weights))
ax2.text(0, 0, f"{total_a:.0f}", ha="center", va="center",
         fontsize=15, color=COLOR_A, fontweight="bold", fontfamily="DejaVu Sans")
ax2.text(0, -22, "/100", ha="center", va="center",
         fontsize=7, color="#94A3B8", fontfamily="DejaVu Sans")

plt.tight_layout(pad=0.5)
out_single = os.path.join(out_dir, "radar_6d_single.png")
fig2.savefig(out_single, dpi=300, bbox_inches="tight",
             facecolor=BG, transparent=False)
print(f"Saved → {os.path.abspath(out_single)}")
