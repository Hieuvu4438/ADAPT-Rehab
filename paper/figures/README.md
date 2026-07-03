# Figures for ADAPT-Rehab Paper

This documents the figures actually referenced by `main.tex` (via `sections/*.tex`).

## Figures used in the paper

### Figure 1 — System Architecture (`fig1_architecture_body.tex`)
- **Location**: `sections/methodology.tex` (§III.A, `figure*` environment)
- **Included via**: `\input{figures/fig1_architecture_body.tex}` inside `figure*`
- **Content**: Five-stage pipeline (Input → Perception → Kinematics & Alignment → Physiological Evaluation → Intelligent Coaching), color-coded by role; blue = perception (3D pose), red = safety evaluation (pain/fatigue), green = coaching output. Includes the red-dashed pain-adapt feedback loop.
- **Standalone build**: `fig1_architecture.tex` is the standalone (non-IEEE) version for preview; the `_body` file is what the paper `\input`s.
- **Palette**: A* two-accent — AccentBlue `#1F4E79`, AccentRed `#C0392B`, OutputGreen `#27AE60`.

### Figure 2 — KIMORE score-vs-clinical scatter (`kimore_score_vs_clinical.png`)
- **Location**: `sections/experiments.tex` (§IV.C, Fig.~\ref{fig:kimore_scatter})
- **Content**: DTW-based score vs. clinical total score on KIMORE (Spearman ρ=0.347, p<10⁻¹¹) under the fixed-reference protocol, with regression line across all five exercises.

### Figure 3 — UCO discriminability histogram (`uco_discriminability_hist.png`)
- **Location**: `sections/experiments.tex` (§IV.D, Fig.~\ref{fig:uco_discrim})
- **Content**: Same-exercise vs. different-exercise score distributions on UCO (16 classes, 6,000 pairs; 48-point separation, AUC=0.974).

## Generated figure assets used by experiments

- `kimore_ablation_bars.png` — KIMORE DTW-variant ablation (used internally; the variant table is `tab:dtw_ablation` in text form).
- `uco_view_breakdown.csv` — source data for the cross-view "87-point similarity drop" sentence (§IV.D).

## Other on-disk figures (NOT referenced by `main.tex`)

The following exist in this folder but are **not** included in the current paper build. They are kept for potential future use (e.g., ground-truth pose-accuracy work, ablation appendix):
- `fig2_mpjpe_comparison.pdf`
- `fig3_ablation_study.pdf`
- `fig4_per_exercise.pdf`
- `fig_dtw_warping.tex`, `fig_safemax_calibration.tex`, `fig_sparc_spectrum.tex`
- `adapt_rehab_workflow_diagram.md`

If you add any of these to the paper, also update this README and the `\includegraphics` / `\input` calls in `sections/*.tex`.

## Conventions

- Single-column figure: `width=\columnwidth`. Two-column-wide figure: `figure*` (as used for the architecture diagram).
- Resolution: 300 DPI minimum for raster PNGs. Vector PDF/TikZ preferred.
- TikZ figures that `\input` a body file must define the colors/macros the body expects (see `main.tex` preamble for the A* palette and `\system`).
