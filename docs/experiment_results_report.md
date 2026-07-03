# ADAPT-Rehab: Experiment Execution Report

**Generated**: 2026-06-17
**Experiment Runner**: `scripts/run_all_experiments.py` (v3.0)
**Datasets**: UI-PRMD (2,000 time-series), KIMORE (378 samples, 5 exercises)

---

## Executive Summary

The codebase **can and does** run experiments. The script `scripts/run_all_experiments.py` successfully executes all 5 experiments from `docs/proposed_experiments.md` and produces structured JSON results and a formatted paper-ready Markdown report in `evaluation/output/`.

However, there are **substantial gaps** between the experimental design described in `docs/proposed_experiments.md` and what the codebase actually implements and measures. The proposed document describes idealized experiments with aspirational results, while the codebase measures what is actually achievable with the current implementation. This report documents both the actual results and the gaps.

---

## 1. What the Codebase Measures vs. What the Paper Proposes

| Experiment | Paper Proposal | Codebase Implementation | Gap |
|---|---|---|---|
| **E1: Angle Accuracy** | Compare RTMW3D-L angles against Vicon ground truth; report MAE, RMSE, ICC(2,1), Bland-Altman | Measures **filter self-consistency** (raw Kinect SDK angles vs Butterworth-filtered); no Vicon comparison | **Major**: No ground-truth comparison. Paper claims "clinical-grade accuracy" but code only validates filter stability |
| **E2: Classification** | 6-D scorer on RTMW3D-L pipeline with per-dimension ablation; AUC > 0.92 expected | Computes 6-D scores from **single-joint Kinect SDK trajectory**; no ablation study; no RTMW3D-L pipeline | **Major**: Different data source (Kinect SDK vs RTMW3D-L). Paper claims AUC > 0.92, actual overall AUC = 0.532 |
| **E3: Compensation** | Temporal compensation detector with LSTM model; AUC > 0.85; synthetic perturbation analysis | Simple **threshold-based detection** from Kinect SDK joint rotations (trunk > 15°, shoulder > 12°) | **Major**: No temporal model. No synthetic perturbation protocol. No latency-to-detection measurement |
| **E4: Calibration** | Safe-Max P95 calibration with test-retest ICC > 0.85 | P95 calibration on KIMORE Kinect data; zero over-estimation achieved but **ICC ~0.12** (poor test-retest) | **Moderate**: Safety guarantee validated; test-retest reliability metric is far below target |
| **E5: Latency** | Per-stage profiling across 4 hardware configs; parallelization analysis | Micro-benchmark of analysis functions (Butterworth, SPARC, DTW, scoring); single hardware config; **no RTMW3D-L inference timing measured** | **Moderate**: Good analysis overhead measurement; missing multi-hardware comparison and actual pose estimation profiling |

---

## 2. Actual Experiment Results (Fresh Execution)

### Experiment 1: Joint Angle Pipeline Stability

**What was actually measured**: Self-consistency of the Butterworth filtering pipeline on Kinect SDK pre-computed joint angle trajectories. Raw angles are compared against their filtered counterparts to verify the filter preserves the underlying signal while removing noise.

**Results**:

| Metric | Value |
|---|---|
| Overall Filter MAE | **0.80°** |
| Per-exercise range | 0.04° (trunk rotation) – 2.74° (inline lunge) |
| Interpretation | Filter preserves clinical signal; MAE < 3° for all exercises |

**Per-Exercise Detail**:

| Exercise | Filter MAE (°) | ROM (°) | SPARC (raw→filt) | N |
|---|---|---|---|---|
| deep_squat | 2.14 | 36 ± 18 | -6.000 → -5.945 | 100 |
| hurdle_step | 0.18 | 6 ± 3 | -6.000 → -5.868 | 100 |
| inline_lunge | 2.74 | 42 ± 28 | -6.000 → -5.751 | 100 |
| side_lunge | 0.55 | 18 ± 13 | -5.980 → -5.637 | 100 |
| sit_to_stand | 0.65 | 25 ± 10 | -6.000 → -5.567 | 100 |
| standing_active_straight_leg_raise | 0.25 | 8 ± 5 | -5.968 → -5.403 | 100 |
| standing_shoulder_abduction | 0.86 | 16 ± 10 | -5.992 → -5.841 | 100 |
| standing_shoulder_extension | 0.50 | 10 ± 8 | -5.956 → -5.883 | 100 |
| standing_shoulder_int_ext_rotation | 0.05 | 3 ± 1 | -5.973 → -5.298 | 100 |
| standing_trunk_rotation | 0.04 | 2 ± 1 | -6.000 → -5.529 | 100 |

**Gap Analysis**: The paper proposes comparing RTMW3D-L angles against Vicon ground truth (gold standard) and Kinect v2 (clinical baseline) with per-joint MAE, ICC(2,1), and Bland-Altman analysis. The codebase only measures filter-induced signal change. This means the paper's Table 1.1 (showing per-joint MAE vs Vicon) and Figure 1.1 (time-series overlay with Vicon/Kinect) **cannot be generated** from current code. The paper claims "mean angular MAE of 3.8°" against Vicon — this is an **unsubstantiated projection**, not a measured result.

---

### Experiment 2: Repetition Quality Classification (6-D Scorer)

**What was actually measured**: 6-D clinical scores (ROM, Stability, Flow, Compensation, Smoothness, Total) computed from a single primary joint angle trajectory extracted from Kinect SDK data, compared between correct and incorrect execution groups.

**Results**:

| Metric | Value | Paper Target | Status |
|---|---|---|---|
| Overall AUC | **0.532** | ≥ 0.92 | ❌ Far below target |
| Overall Cohen's d | **0.020** | ≥ 1.5 | ❌ No effect |
| Best exercise AUC | 0.745 (shoulder abduction) | ≥ 0.90 | ❌ Below target |
| Exercises with AUC > 0.6 | 3/10 | 10/10 | ❌ Only 30% |

**Per-Dimension Results**:

| Dimension | AUC | Cohen's d | Correct Mean | Incorrect Mean |
|---|---|---|---|---|
| rom | 0.406 | -0.307 | 59.4 | 70.1 |
| stability | **0.628** | **0.441** | 77.6 | 65.3 |
| flow | 0.565 | 0.204 | 30.4 | 23.4 |
| compensation | 0.580 | 0.233 | 98.2 | 97.5 |
| smoothness | 0.411 | -0.401 | 5.5 | 10.9 |
| **total** | **0.532** | **0.020** | 60.6 | 60.4 |

**Per-Exercise AUC**:

| Exercise | AUC | Cohen's d | N (C/I) |
|---|---|---|---|
| deep_squat | 0.428 | -0.231 | 100/100 |
| hurdle_step | 0.601 | 0.403 | 100/100 |
| inline_lunge | 0.479 | 0.046 | 100/100 |
| side_lunge | 0.472 | -0.083 | 100/100 |
| **sit_to_stand** | **0.696** | **0.779** | 100/100 |
| standing_active_straight_leg_raise | 0.489 | 0.002 | 100/100 |
| **standing_shoulder_abduction** | **0.745** | **0.971** | 100/100 |
| standing_shoulder_extension | 0.539 | 0.077 | 100/100 |
| standing_shoulder_int_ext_rotation | 0.206 | -1.259 | 100/100 |
| standing_trunk_rotation | 0.495 | -0.103 | 100/100 |

**Why Results Are Low**: 
1. The 6-D scorer operates on a **single joint angle trajectory** (e.g., left knee flexion for squats), not the full-body kinematic profile
2. UI-PRMD's "incorrect" condition involves **biomechanically different** movements (e.g., trunk-leaning squat vs. upright squat), not necessarily lower-quality movements
3. A single-joint angle cannot capture multi-joint compensatory patterns — the "incorrect" squat may look identical to the "correct" squat in knee angle alone
4. The scoring dimensions (Symmetry, Flow) rely on multi-joint or bilateral data that isn't available from a single trajectory
5. **No ablation study was run** — the paper's Table 2.2 (leave-one-dimension-out) was not implemented

**Gap Analysis**: The paper's Table 2.1 (AUC 0.93, Cohen's d 2.4) and Table 2.2 (ablation with ΔAUC per dimension) cannot be reproduced from current results. The overall AUC of 0.532 is **no better than random guessing**. This is the most critical gap — the 6-D scorer's discriminative ability is not validated.

---

### Experiment 3: Compensation Detection Sensitivity

**What was actually measured**: Simple threshold-based detection (trunk lean > 15°, shoulder asymmetry > 12°) on Kinect SDK joint rotation trajectories. A composite score subtracts penalty points when thresholds are exceeded.

**Results**:

| Metric | Value | Paper Target | Status |
|---|---|---|---|
| Overall AUC | **0.604** | ≥ 0.85 | ❌ Below target |
| Cohen's d | 0.394 | — | Small effect |
| Correct mean score | 78.5 | — | — |
| Incorrect mean score | 70.2 | — | — |

| Compensation Type | Detection Rate (Incorrect) | False Positive Rate (Correct) |
|---|---|---|
| Trunk Lean (>15°) | 9.1% | 4.8% |
| Shoulder Hiking (>12°) | 76.2% | 58.8% |

**Analysis**:
- **Trunk lean** shows excellent specificity (FPR = 4.8%) but very low sensitivity (DR = 9.1%) — the 15° threshold may be too conservative
- **Shoulder hiking** has high sensitivity (76.2%) but poor specificity (FPR = 58.8%) — shoulder asymmetry is prevalent even in nominally "correct" movements, making it a non-specific marker
- The composite score's AUC of 0.604 suggests the compensation score provides some signal, but is not a reliable standalone discriminator

**What's Missing**:
- No synthetic perturbation protocol (MDC thresholds not established)
- No frame-level annotation or inter-annotator agreement measurement
- No temporal compensation detector (LSTM) — only static thresholding
- No latency-to-detection measurement
- No confusion matrix (cross-detection analysis)
- Hip shift compensation not implemented

**Gap Analysis**: The paper's Table 3.1 (AUC 0.87-0.91, MDC thresholds) and Table 3.2 (synthetic perturbation detection limits) cannot be generated from current code. The compensation detection is a **simplistic threshold-based system**, not the temporal model described in the architecture.

---

### Experiment 4: Calibration Safety & Personalization

**What was actually measured**: Safe-Max P95 calibration on KIMORE dataset — for each joint in each subject, the 95th percentile of filtered joint angles is compared against the absolute maximum to check for over-estimation.

**Results**:

| Metric | Value | Paper Target | Status |
|---|---|---|---|
| Overall Over-Estimation Rate | **0.00%** (0/1512) | 0% | ✅ Met |
| Test-Retest ICC (mean across joints) | **~0.06** | > 0.85 | ❌ Far below target |

**Per-Joint Detail**:

| Joint | P95 (°) | Safety Margin (°) | Over-Est. Rate | Personalization Ratio | ICC | N |
|---|---|---|---|---|---|---|
| left_knee | 108.2 ± 29.1 | 36.1 ± 24.0 | 0.00% | 8.13x | 0.121 | 378 |
| left_shoulder | 75.2 ± 17.9 | 24.3 ± 20.6 | 0.00% | 3.11x | -0.136 | 378 |
| right_knee | 111.2 ± 29.3 | 29.3 ± 21.2 | 0.00% | 4.38x | 0.137 | 378 |
| right_shoulder | 118.0 ± 22.9 | 13.9 ± 15.6 | 0.00% | 2.84x | 0.109 | 378 |

**Analysis**:
- **Safety guarantee validated**: P95 never exceeds true max across 1,512 joint calibrations — this is the most robust finding
- **ICC is very poor** (~0.1 vs target 0.85): The split-half method (first 5 vs last 5 samples per subject) produces inconsistent P95 estimates because KIMORE samples are sparse per subject and not necessarily from repeated calibration sessions
- **Personalization ratios (2.8–8.1×)**: Confirm substantial inter-subject variability, validating the need for personalization
- **P95 values are lower than paper claims**: The paper reports P95 of ~150° for shoulders; actual measurements are 75-118°. This may reflect different joint angle computation methods

**Why ICC is Low**:
1. The ICC computation uses split-half on KIMORE samples (first half vs second half per subject), but KIMORE provides only a few samples per subject from a single session — not true test-retest data
2. The angle extraction method (3-point angle from Kinect joint positions) may introduce noise
3. The paper proposed using "first 5 reps vs last 5 reps" but KIMORE doesn't provide per-rep segmentation

**Gap Analysis**: The safety guarantee (0% over-estimation) is well-supported, but the test-retest reliability claim (ICC > 0.85) is **not substantiated** by current data. This requires either a proper test-retest protocol with repeated calibration sessions, or acknowledgment of the limitation. The paper's Table 4.1 (per-joint safety analysis) can be partially populated but ICC values need revision.

---

### Experiment 5: System Latency & Throughput Profiling

**What was actually measured**: Micro-benchmark of individual analysis functions (Butterworth filter, SPARC, DTW, 6-D scoring) on synthetic data. Pose estimation time is estimated at 38 ms/frame (hardcoded, not measured).

**Hardware**: x86_64 (24 physical cores), 62.1 GB RAM, GPU: NVIDIA RTX 5880 Ada Generation (48 GB VRAM)

| Pipeline Stage | Latency (ms) |
|---|---|
| Butterworth filter | 0.110 |
| SPARC | 0.045 |
| DTW constrained | 20.742 |
| Full scoring pipeline | 0.302 |

| Metric | Value | Paper Target | Status |
|---|---|---|---|
| Analysis overhead | **0.30 ms/frame** | < 10 ms | ✅ Well within target |
| Est. total FPS (with pose) | **26.1 FPS** | > 10 FPS | ✅ Meets target |
| Analysis-only FPS | **3,306 FPS** | — | Negligible overhead |

**What's Missing**:
- **No actual RTMW3D-L inference timing** — pose estimation latency is hardcoded at 38 ms, not measured
- **No face analysis timing** (OpenFace 3.0) — not profiled
- **No multi-hardware comparison** — only tested on the current machine (which has an RTX 5880, much more powerful than the paper's H1-H4 configurations)
- **No parallelization analysis** (sequential vs. parallel vs. frame-skip) — not implemented
- **No GPU memory measurement** — not instrumented
- **No end-to-end pipeline run** with real video input — only synthetic micro-benchmarks
- **No LLM/TTS latency measurement** — LLM and TTS stages are not profiled

**Gap Analysis**: The analysis layer overhead is negligible (< 1 ms), which is the strongest latency finding. However, the paper's Table 5.1 (per-stage latency across 4 hardware configs), Table 5.2 (memory footprint), and Table 5.3 (pipeline configuration comparison) **cannot be populated** from current measurements. The paper claims "pose estimation 38.2 ms" — this is a hardcoded estimate, not a measurement. The current hardware (RTX 5880, 48 GB) far exceeds the paper's target configurations and is not representative of consumer hardware.

---

## 3. Summary of Gaps

| Paper Element | Status | Action Required |
|---|---|---|
| Table 1.1: Per-joint MAE vs Vicon | ❌ Not measurable | Need RTMW3D-L inference on UI-PRMD videos with Vicon alignment |
| Table 1.2: Per-exercise angle accuracy | ❌ Not measurable | Same as above |
| Figure 1.1: Time-series overlay | ❌ Not generatable | Need Vicon ground truth alignment |
| Figure 1.2: Bland-Altman plot | ❌ Not generatable | Need Vicon comparison data |
| Table 2.1: Classification AUC per exercise | ⚠️ Partially available | Results are poor (AUC 0.21-0.75); need multi-joint scoring |
| Table 2.2: Ablation study | ❌ Not run | Ablation infrastructure exists in `scripts/run_ablation.py` but not integrated with experiment runner |
| Table 2.3: KIMORE clinician correlation | ❌ Not run | KIMORE data loaded but clinical scores not used for correlation |
| Figure 2.1-2.5: ROC, raincloud, ablation, heatmap, scatter | ❌ Not generatable | No visualization code for these figures |
| Table 3.1: Per-type compensation AUC | ⚠️ Partial | Simple threshold detector only; no MDC |
| Table 3.2: Synthetic perturbation | ❌ Not run | No synthetic perturbation protocol implemented |
| Figure 3.1-3.4: PR curves, detection curves, confusion matrix, timeline | ❌ Not generatable | No visualization code |
| Table 4.1: Calibration safety per joint | ✅ Available | Zero over-estimation validated |
| Table 4.2: Personalization efficacy | ✅ Available | Ratios 2.8-8.1x confirmed |
| Table 4.3: Test-retest reliability | ⚠️ Poor results | ICC ~0.1, not 0.85 — needs proper test-retest data |
| Figure 4.1-4.3: Safety scatter, P95 convergence, ROM box plot | ❌ Not generatable | No visualization code |
| Table 5.1: Per-stage latency × 4 hardware | ❌ Not measurable | Single hardware; no pose/face timing |
| Table 5.2: Memory footprint | ❌ Not measured | No GPU/CPU memory instrumentation |
| Table 5.3: Pipeline config comparison | ❌ Not run | No parallel/skip-frame variants tested |
| Figure 5.1-5.4: Gantt, violin, throughput, GPU memory | ❌ Not generatable | No visualization or multi-config data |

**Legend**: ✅ = Available, ⚠️ = Partially available with caveats, ❌ = Not available

---

## 4. What the Codebase CAN Actually Demonstrate

Based on the experiment run, the codebase can credibly demonstrate:

1. **Pipeline stability**: The Butterworth filtering pipeline (fc=6 Hz, 4th-order) preserves clinical signal with < 1° mean absolute deviation across all exercises. This validates the signal processing chain, though it does not validate accuracy against ground truth.

2. **Calibration safety**: The P95-based Safe-Max calibration **never exceeds** the user's demonstrated maximum ROM (0% over-estimation across 1,512 joint calibrations). This is the strongest validated claim.

3. **Personalization necessity**: Inter-subject P95 variability is 2.8–8.1×, confirming that fixed targets are clinically inappropriate.

4. **Real-time feasibility**: The analysis layer adds < 1 ms overhead per frame. Even with estimated pose estimation latency of ~38 ms, the system achieves > 25 FPS, well above the 10 FPS rehabilitation threshold.

5. **Compensation signal**: Trunk lean and shoulder asymmetry provide some discriminative signal (AUC 0.604), with trunk lean showing excellent specificity (4.8% FPR).

6. **Exercise-dependent discrimination**: The 6-D scorer's ability to separate correct from incorrect movements varies substantially by exercise (AUC range: 0.21–0.75), with standing shoulder abduction (AUC 0.745) and sit-to-stand (AUC 0.696) showing the strongest discrimination.

---

## 5. Recommendations

### Immediate Actions (for honest paper writing)

1. **Revise E1 claims**: Replace "angular MAE of 3.8° vs Vicon" with "filter pipeline self-consistency MAE of 0.80°" until RTMW3D-L + Vicon comparison is implemented.

2. **Revise E2 claims**: Report actual AUC range (0.21-0.75) with the honest explanation that single-joint scoring cannot fully capture the multi-joint nature of compensatory movements. Frame as a **limitation** rather than a success.

3. **Revise E3 claims**: Report actual threshold-based detection results (AUC 0.604) rather than the claimed 0.87-0.91. Acknowledge that the temporal LSTM detector is not yet implemented.

4. **Revise E4 claims**: Report the validated zero over-estimation (strong result) but drop the ICC > 0.85 claim (not supported). Note that test-retest reliability requires a dedicated calibration study design.

5. **Revise E5 claims**: Report the measured analysis overhead (< 1 ms) but clarify that pose estimation timing is estimated, not measured, and that multi-hardware profiling is not yet done.

### Development Actions (to close gaps)

| Priority | Action | Effort |
|---|---|---|
| **P0** | Run RTMW3D-L inference on UI-PRMD videos to get actual pose estimates for E1 | High (requires video data, not just Kinect angles) |
| **P0** | Implement multi-joint 6-D scoring (not single-joint) for E2 | Medium |
| **P0** | Implement leave-one-dimension-out ablation for E2 | Low (infrastructure exists) |
| **P1** | Implement synthetic perturbation protocol for E3 | Medium |
| **P1** | Add GPU/CPU memory profiling instrumentation for E5 | Low |
| **P1** | Measure actual RTMW3D-L and OpenFace inference time for E5 | Medium |
| **P2** | Correlate ADAPT-Rehab scores with KIMORE clinical scores for E2 | Low (data already loaded) |
| **P2** | Generate publication-quality figures (Bland-Altman, ROC, raincloud, etc.) | Medium |
| **P3** | Multi-hardware latency profiling (laptop, CPU-only, Mac) | High (needs hardware access) |
| **P3** | Proper test-retest calibration protocol for E4 | High (needs study design) |

---

## 6. Reproducibility

To reproduce these results:

```bash
# Requirements
pip install numpy scipy scikit-learn

# Run all 5 experiments
cd /home/haipd/ADAPT-Rehab
python scripts/run_all_experiments.py --experiments E1,E2,E3,E4,E5

# Outputs
# - evaluation/output/experiment_results.json  (structured data)
# - evaluation/output/experiment_paper.md      (formatted report)
```

**Data requirements**: UI-PRMD dataset (Kinect angle files in `data/UI-PRMD/`) and KIMORE dataset (`data/KIMORE/kimore_exercise_dataset.pkl`). Both are present in the current environment.

---

*Report generated by re-running `scripts/run_all_experiments.py` on 2026-06-17. All metrics computed on actual dataset files. The gaps documented above reflect differences between the aspirational experimental design in `docs/proposed_experiments.md` and the currently implemented evaluation code.*
