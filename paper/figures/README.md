# Figure Templates for ADAPT-Rehab Paper

## Required Figures

### Figure 1: System Architecture
- **Location**: `sections/methodology.tex`
- **Content**: Five-layer architecture diagram (Input → Perception → Analysis → Intelligence → Output)
- **Tool**: Draw in draw.io, Figma, or PowerPoint, export as PDF

### Figure 2: Pose Estimation Results
- **Location**: `sections/experiments.tex`
- **Content**: Bar chart comparing MPJPE across methods (MediaPipe, OpenPose, MotionBERT, Ours)
- **Data source**: Run benchmarks on UI-PRMD dataset
- **Tool**: matplotlib/seaborn in Python

### Figure 3: Pain and Emotion Detection
- **Location**: `sections/experiments.tex`
- **Content**: AU detection examples and emotion classification results
- **Data source**: Run face analysis on UNBC-McMaster dataset
- **Tool**: matplotlib/seaborn in Python

## How to Replace Placeholders

1. Generate figures using your preferred tool
2. Save as PDF (preferred) or PNG (300 DPI minimum)
3. Place in `/paper/figures/` directory
4. In the corresponding `.tex` file, replace the `\fbox{\parbox{...}}` block with:
   ```latex
   \includegraphics[width=0.9\columnwidth]{figures/your_figure.pdf}
   ```

## Recommended Sizes
- Single column figure: `width=0.9\columnwidth`
- Resolution: 300 DPI for print quality
- Format: PDF (vector) preferred over PNG (raster)
