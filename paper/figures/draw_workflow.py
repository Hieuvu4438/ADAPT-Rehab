#!/usr/bin/env python3
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

def create_workflow_diagram():
    # Setup figure size and high resolution for conference papers
    fig, ax = plt.subplots(figsize=(22, 16))
    ax.set_xlim(0, 22)
    ax.set_ylim(0, 15)
    ax.axis('off')
    
    # Draw.io Standard Color Palette (Clean Hex Codes)
    colors = {
        'blue_fill': '#dae8fc',
        'blue_border': '#6c8ebf',
        'green_fill': '#d5e8d4',
        'green_border': '#82b366',
        'yellow_fill': '#fff2cc',
        'yellow_border': '#d6b656',
        'red_fill': '#f8cecc',
        'red_border': '#b85450',
        'purple_fill': '#e1d5e7',
        'purple_border': '#9673a6',
        'grey_fill': '#f5f5f5',
        'grey_border': '#666666',
        'dark_text': '#000000',
        'sub_text': '#333333'
    }

    # Helper function to draw a styled box (rounded rectangle with text)
    def draw_box(x, y, w, h, fill, border, title, text_lines, align='left'):
        # Draw shadow for clean UI effect
        shadow = patches.FancyBboxPatch(
            (x + 0.06, y - 0.06), w, h, boxstyle="round,pad=0.08",
            facecolor='#e9e9e9', edgecolor='none', zorder=1
        )
        ax.add_patch(shadow)
        
        # Draw main box
        rect = patches.FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.08",
            facecolor=fill, edgecolor=border, linewidth=1.8, zorder=2
        )
        ax.add_patch(rect)
        
        # Draw title
        if title:
            ax.text(
                x + w/2 if align=='center' else x + 0.2, 
                y + h - 0.3, 
                title, 
                fontsize=11.5, 
                fontweight='bold', 
                color=colors['dark_text'],
                ha='center' if align=='center' else 'left', 
                va='top', 
                zorder=3
            )
            
        # Draw lines of text
        current_y = y + h - 0.65
        for line in text_lines:
            if not line:
                current_y -= 0.1
                continue
            if line.startswith("SUBTITLE:"):
                line_text = line.replace("SUBTITLE:", "")
                ax.text(x + 0.2, current_y, line_text, fontsize=9.5, fontweight='bold', color='#1a1a1a', ha='left', va='top', zorder=3)
                current_y -= 0.28
            elif line.startswith("FORMULA:"):
                line_text = line.replace("FORMULA:", "")
                ax.text(x + 0.35, current_y, line_text, fontsize=9.5, fontfamily='serif', color='#000000', ha='left', va='top', zorder=3)
                current_y -= 0.32
            else:
                ax.text(x + 0.2, current_y, line, fontsize=8.5, color=colors['sub_text'], ha='left', va='top', zorder=3)
                current_y -= 0.22

    # Helper function to draw multi-column boxes for dense modules
    def draw_box_multicol(x, y, w, h, fill, border, title, col1_text, col2_text):
        shadow = patches.FancyBboxPatch(
            (x + 0.06, y - 0.06), w, h, boxstyle="round,pad=0.08",
            facecolor='#e9e9e9', edgecolor='none', zorder=1
        )
        ax.add_patch(shadow)
        
        rect = patches.FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.08",
            facecolor=fill, edgecolor=border, linewidth=1.8, zorder=2
        )
        ax.add_patch(rect)
        
        ax.text(x + 0.2, y + h - 0.3, title, fontsize=11.5, fontweight='bold', color=colors['dark_text'], ha='left', va='top', zorder=3)
        
        # Col 1
        current_y = y + h - 0.65
        for line in col1_text:
            if not line:
                current_y -= 0.1
                continue
            if line.startswith("SUBTITLE:"):
                line_text = line.replace("SUBTITLE:", "")
                ax.text(x + 0.2, current_y, line_text, fontsize=9.5, fontweight='bold', color='#1a1a1a', ha='left', va='top', zorder=3)
                current_y -= 0.28
            elif line.startswith("FORMULA:"):
                line_text = line.replace("FORMULA:", "")
                ax.text(x + 0.35, current_y, line_text, fontsize=9.5, fontfamily='serif', color='#000000', ha='left', va='top', zorder=3)
                current_y -= 0.32
            else:
                ax.text(x + 0.2, current_y, line, fontsize=8.5, color=colors['sub_text'], ha='left', va='top', zorder=3)
                current_y -= 0.22
                
        # Col 2
        current_y = y + h - 0.65
        for line in col2_text:
            if not line:
                current_y -= 0.1
                continue
            if line.startswith("SUBTITLE:"):
                line_text = line.replace("SUBTITLE:", "")
                ax.text(x + w/2 + 0.1, current_y, line_text, fontsize=9.5, fontweight='bold', color='#1a1a1a', ha='left', va='top', zorder=3)
                current_y -= 0.28
            elif line.startswith("FORMULA:"):
                line_text = line.replace("FORMULA:", "")
                ax.text(x + w/2 + 0.25, current_y, line_text, fontsize=9.5, fontfamily='serif', color='#000000', ha='left', va='top', zorder=3)
                current_y -= 0.32
            else:
                ax.text(x + w/2 + 0.1, current_y, line, fontsize=8.5, color=colors['sub_text'], ha='left', va='top', zorder=3)
                current_y -= 0.22

    # Helper function to draw connection arrows
    def draw_arrow(x1, y1, x2, y2, text="", style='-|>', color='#4a5568', lw=1.5, dashed=False):
        ls = '--' if dashed else '-'
        arrow = patches.FancyArrowPatch(
            (x1, y1), (x2, y2), arrowstyle=style, mutation_scale=10,
            linestyle=ls, linewidth=lw, color=color, zorder=2
        )
        ax.add_patch(arrow)
        if text:
            ax.text((x1 + x2)/2, (y1 + y2)/2 + 0.12, text, fontsize=8.0, ha='center', va='bottom', color='#2d3748', zorder=3)

    # Helper function to draw dashed grouping borders
    def draw_group_border(x, y, w, h, label, border_color):
        rect = patches.Rectangle(
            (x, y), w, h, fill=False, edgecolor=border_color, 
            linestyle='--', linewidth=1.2, zorder=1
        )
        ax.add_patch(rect)
        ax.text(
            x + 0.15, y + h - 0.2, label, fontsize=10, fontweight='bold', 
            color=border_color, ha='left', va='top', zorder=2
        )

    # =========================================================================
    # MAIN WORKFLOW PIPELINE (Top Row: Y = 11.2 to 13.8)
    # =========================================================================
    
    # Title Banner
    ax.text(11, 14.5, "ADAPT-Rehab System Workflow & Core Algorithm Architecture", 
            fontsize=17, fontweight='bold', ha='center', va='center', 
            color='#1a365d', bbox=dict(facecolor='#f7fafc', edgecolor='#e2e8f0', boxstyle='round,pad=0.4'))

    # Group: INPUT DATA (Y=11.5)
    draw_group_border(0.5, 11.2, 3.2, 2.5, "INPUTS", colors['grey_border'])
    draw_box(
        0.7, 11.5, 2.8, 1.9, colors['grey_fill'], colors['grey_border'],
        "Input Streams",
        [
            "• Webcam / Video feed",
            "• Reference exercise",
            "• Calibration profile"
        ]
    )

    # Group: PERCEPTION LAYER (Y=11.5)
    draw_group_border(4.1, 11.2, 6.7, 2.5, "MULTIMODAL SENSING", colors['blue_border'])
    draw_box(
        4.3, 11.5, 3.0, 1.9, colors['blue_fill'], colors['blue_border'],
        "3D Pose Tracker",
        [
            "• RTMW3D (133 KPs)",
            "• MediaPipe Fallback",
            "• Skeletal 3D joints"
        ]
    )
    draw_box(
        7.5, 11.5, 3.0, 1.9, colors['blue_fill'], colors['blue_border'],
        "Face AU Analyzer",
        [
            "• OpenFace 3.0 GNN",
            "• 8 Action Units (AUs)",
            "• Emotion / Gaze logits"
        ]
    )

    # Group: ANALYSIS LAYER (Y=11.5)
    draw_group_border(11.2, 11.2, 6.6, 2.5, "PROCESSING & ALGORITHMS", colors['green_border'])
    draw_box(
        11.4, 11.5, 3.0, 1.9, colors['green_fill'], colors['green_border'],
        "Analytical Engines",
        [
            "• Quaternion Kinematics",
            "• Smoothness (SPARC)",
            "• Behavior State Class"
        ]
    )
    draw_box(
        14.6, 11.5, 2.9, 1.9, colors['green_fill'], colors['green_border'],
        "6D Scoring Core",
        [
            "• Weighted multi-dim",
            "• ROM, Flow (DTW),",
            "  Stability, Symmetry"
        ]
    )

    # Group: OUTPUT (Y=11.5)
    draw_group_border(18.2, 11.2, 3.3, 2.5, "OUTPUT", colors['purple_border'])
    draw_box(
        18.4, 11.5, 2.9, 1.9, colors['purple_fill'], colors['purple_border'],
        "Visual HUD Overlay",
        [
            "• Joint angles overlay",
            "• Pain/Fatigue warnings",
            "• Performance scores"
        ]
    )

    # Main Flow Connections
    draw_arrow(3.7, 12.45, 4.1, 12.45)
    draw_arrow(10.8, 12.45, 11.2, 12.45)
    draw_arrow(17.8, 12.45, 18.2, 12.45)
    draw_arrow(7.3, 12.45, 7.5, 12.45, style='-', lw=1.0)


    # =========================================================================
    # ROW 1 OF DETAILED PANELS (Y = 5.8 to 10.0)
    # =========================================================================

    # Panel 1: Quaternion Kinematics
    draw_box(
        0.5, 5.8, 6.6, 4.2, colors['green_fill'], colors['green_border'],
        "1. QUATERNION KINEMATICS",
        [
            "SUBTITLE:Unit-Vector Normalization",
            "FORMULA:v_1 = \\frac{A-B}{\\lVert A-B \\rVert}, \\quad v_2 = \\frac{C-B}{\\lVert C-B \\rVert}",
            "",
            "SUBTITLE:Shortest Arc Quaternion (Melax 1998)",
            "FORMULA:d = v_1 \\cdot v_2, \\quad s = \\sqrt{(1+d)\\cdot 2.0}",
            "FORMULA:q = \\left[ s \\cdot 0.5, \\quad \\frac{v_1 \\times v_2}{s} \\right]",
            "",
            "SUBTITLE:Included & Clinical Angle",
            "FORMULA:\\theta = 2.0 \\cdot \\arccos(q.w)",
            "FORMULA:\\theta_{\\text{clin}} = 180.0 - \\theta \\quad \\text{(ISB flexion convention)}"
        ]
    )

    # Panel 2: Smoothness Analyzer
    draw_box(
        7.7, 5.8, 6.6, 4.2, colors['green_fill'], colors['green_border'],
        "2. MOVEMENT SMOOTHNESS (SPARC / LDLJ)",
        [
            "SUBTITLE:Spectral Arc Length (SPARC)",
            "Fourier transform of velocity: V(\\omega) = \\text{FFT}(v(t))",
            "Normalized spectrum: \\hat{M}(\\omega) = |V(\\omega)| / \\max(|V(\\omega)|)",
            "Adaptive cutoff frequency (threshold = 0.05):",
            "FORMULA:\\omega_c = \\max \\{\\omega \\mid \\hat{M}(\\omega) > 0.05\\}",
            "FORMULA:\\text{SPARC} = -\\int_0^{\\omega_c} \\sqrt{ \\left(\\frac{1}{\\omega_c}\\right)^2 + \\left(\\frac{d\\hat{M}}{d\\omega}\\right)^2 } d\\omega",
            "",
            "SUBTITLE:Log-Dimensionless Jerk (LDLJ)",
            "FORMULA:\\text{LDLJ} = -\\ln\\left(\\frac{T^5}{D^2} \\int_0^T |j(t)|^2 dt\\right)"
        ]
    )

    # Panel 3: Constrained DTW
    draw_box(
        14.9, 5.8, 6.6, 4.2, colors['yellow_fill'], colors['yellow_border'],
        "3. CONSTRAINED DTW ALIGNMENT",
        [
            "SUBTITLE:Amplitude Normalization",
            "FORMULA:u_{\\text{norm}} = \\frac{u - \\text{mean}(u)}{\\max(r_{\\text{centered}}) - \\min(r_{\\text{centered}})}",
            "",
            "SUBTITLE:Sakoe-Chiba Constraint Width",
            "FORMULA:w = \\max(\\text{int}(\\max(N,M) \\cdot 0.15), |N-M|, 5)",
            "",
            "SUBTITLE:Warping Recurrence (inside band)",
            "FORMULA:D_{i,j} = \\text{dist}(u_i, r_j) + \\min(D_{i-1,j}, D_{i,j-1}, D_{i-1,j-1})",
            "",
            "SUBTITLE:Fusion & Score Recovery",
            "FORMULA:d_{\\text{weighted}} = \\frac{\\sum w_k (D_{N,M} / \\text{path\\_len})}{\\sum w_k}",
            "FORMULA:\\text{Similarity (\\%)} = 100 \\cdot \\exp(-d_{\\text{weighted}} \\cdot 5.0)"
        ]
    )


    # =========================================================================
    # ROW 2 OF DETAILED PANELS (Y = 0.8 to 5.0)
    # =========================================================================

    # Panel 4: ROM Calibration
    draw_box(
        0.5, 0.8, 6.6, 4.2, colors['yellow_fill'], colors['yellow_border'],
        "4. SAFE-MAX ROM CALIBRATION",
        [
            "SUBTITLE:Median Noise Filter (window = 5)",
            "FORMULA:\\theta_{\\text{smooth}} = \\text{median}(\\theta[i-2 \\dots i+2])",
            "",
            "SUBTITLE:2-Sigma Outlier Box Filter",
            "FORMULA:\\text{Keep if } |\\theta_{\\text{smooth}} - \\mu| \\le 2.0 \\cdot \\sigma",
            "",
            "SUBTITLE:Percentile Limit Extraction",
            "FORMULA:\\theta_{\\text{user\\_max}} = \\text{P}_{95}(\\theta_{\\text{filtered}}), \\quad \\theta_{\\text{user\\_min}} = \\text{P}_{5}(\\theta_{\\text{filtered}})",
            "",
            "SUBTITLE:Calibration Confidence Metric",
            "FORMULA:\\text{Confidence} = \\max(0.0, \\ 1.0 - \\sigma / 30.0)",
            "",
            "SUBTITLE:Reference Exercise Scaling Factor",
            "FORMULA:\\text{Scale Factor} = \\theta_{\\text{user\\_max}} / \\theta_{\\text{ref\\_max}}"
        ]
    )

    # Panel 5: Clinical Behavior & Postural State Classifiers (Double Column Box)
    draw_box_multicol(
        7.7, 0.8, 13.8, 4.2, colors['red_fill'], colors['red_border'],
        "5. CLINICAL MULTIMODAL FACIAL & BODY BEHAVIOR STATE DETECTION",
        [
            "SUBTITLE:A. Action-Unit Based Facial States",
            "• Eye Closure Index (EAR):",
            "  FORMULA:EAR = \\frac{\\lVert p_{159} - p_{145} \\rVert + \\lVert p_{158} - p_{153} \\rVert}{2 \\cdot \\lVert p_{133} - p_{33} \\rVert}",
            "• Clinical Pain Score (PSPI):",
            "  FORMULA:\\text{PSPI} = AU_4 + 2 \\cdot AU_6 + AU_9 + 2 \\cdot AU_{43\\_approx}",
            "• PERCLOS Fatigue (60s Window):",
            "  FORMULA:\\text{PERCLOS (\\%)} = \\frac{\\text{Frames}(AU_{43} \\ge 3.0)}{\\text{Total Frames}} \\cdot 100\\%",
            "• Boredom Index:",
            "  FORMULA:\\text{Boredom} = 0.3 R_{\\text{neutral}} + 0.3 (1 - \\frac{\\sigma_{\\text{AUs}}}{2}) + 0.4 (1 - f_{AU12})"
        ],
        [
            "SUBTITLE:B. Kinematic Body States",
            "• Bilateral Asymmetry Index (Limping/Guarding):",
            "  FORMULA:AI(\\%) = \\frac{L_y - R_y}{0.5 \\cdot (|L_y| + |R_y|)} \\cdot 100\\%",
            "• Trunk Lean / Postural Collapse:",
            "  FORMULA:\\theta_{\\text{trunk}} = \\arccos\\left(\\frac{V_{\\text{trunk}} \\cdot [0, 1, 0]}{\\lVert V_{\\text{trunk}} \\rVert}\\right)",
            "• Monotonic Fatigue Trend (Mann-Kendall):",
            "  Significant decline if Z-score: |Z| > 1.96",
            "• Posture Compensation Penalties:",
            "  - Shoulder Hiking: \\max |ls_y - rs_y| > 0.05",
            "  - Trunk Lean: \\max |\\text{tilt}| > 15^\\circ",
            "  - Hip Shift: \\max |lh_y - rh_y| > 0.06"
        ]
    )

    # Dashed linkage arrows from Main Pipeline to Detailed Panels
    draw_arrow(12.9, 11.2, 3.8, 10.0, color='#718096', lw=1.2, dashed=True, style='-')
    draw_arrow(12.9, 11.2, 11.0, 10.0, color='#718096', lw=1.2, dashed=True, style='-')
    draw_arrow(12.9, 11.2, 18.2, 10.0, color='#718096', lw=1.2, dashed=True, style='-')
    draw_arrow(12.9, 11.2, 3.8, 5.0, color='#718096', lw=1.2, dashed=True, style='-')
    draw_arrow(12.9, 11.2, 14.6, 5.0, color='#718096', lw=1.2, dashed=True, style='-')

    # Label on dashed connections
    ax.text(12.9, 10.5, "Detailed Modules & Algorithms", fontsize=9.0, 
            fontstyle='italic', color='#4a5568', bbox=dict(facecolor='#ffffff', edgecolor='#cbd5e0', alpha=0.9, boxstyle='round,pad=0.2'),
            ha='center', va='center')

    # Save output image
    output_dir = "/home/haipd/ADAPT-Rehab/paper/figures"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "workflow_diagram.png")
    
    plt.savefig(output_path, dpi=250, bbox_inches='tight')
    plt.close()
    print(f"Successfully updated clean paper-oriented workflow diagram at: {output_path}")

if __name__ == "__main__":
    create_workflow_diagram()
