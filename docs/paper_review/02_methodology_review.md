# Peer Review Report — Methodology (Reviewer 1)

## Manuscript Information

- **Title**: ADAPT-Rehab: A Multimodal AI System for Real-Time Elderly Rehabilitation with Whole-Body 3D Pose Estimation and Voice-Interactive LLM Coaching
- **Review Date**: 2026-06-30
- **Review Round**: Round 1

## Reviewer Information

### Reviewer Role
Peer Reviewer 1 (Methodology)

### Reviewer Identity
Measurement-and-evaluation scientist in human-motion analysis; publishes on 3D-pose benchmarking and reproducibility.

### Review Focus
Whether the metrics measure what their labels say, whether the KIMORE correlation is earned by the protocol, whether the FPS claim is well-defined and reproducible, and statistical reporting adequacy.

---

## Overall Assessment

### Recommendation
- [x] **Major Revision** — Substantial revisions needed, re-review required after revision

### Confidence Score
5 — Completely within my area of expertise, very confident in my assessment.

### Summary Assessment
The evaluation is the weakest part of an otherwise honest paper. Two of the three headline results rest on protocols that do not measure what their framing implies. (1) The KIMORE ρ=0.347 is produced by a *fixed-reference* protocol in which the expert templates are the top-10 recordings *by the very clinical score being predicted*; this is criterion contamination, and if those 10 recordings are included in the 378 scored, distance-to-self produces the maximum score by construction (direct leakage). The paper does not state they are held out. (2) The "0.18±0.22 mm temporal self-consistency" is computed on UI-PRMD's *Kinect v2 skeletons* (loaded, mapped), so it characterizes the cross-skeleton *mapping framework's* numerical stability, not RTMW3D-L tracking on RGB — yet it is reported under "Pose Estimation Temporal Stability," implying the latter. (3) 129.7 FPS is reported without hardware, resolution, batch size, or latency-vs-throughput definition, and is attributed inconsistently across sections. The statistical reporting omits confidence intervals, effect-size interpretation, and multiple-comparison correction across the five exercises. All of these are fixable, but the headline numbers cannot be trusted as reported.

---

## Strengths

### S1: Explicit no-claim boundary
Experiments §IV opens by disclaiming "metric-scale ground truth" and "clinical-grade joint-angle accuracy." This pre-emptive honesty is methodologically valuable — it tells the reader exactly what the numbers are *not*.

### S2: Two-benchmark design
Testing scoring on KIMORE (clinical-score correlation) *and* UCO (discriminability + cross-view) is a reasonable dual-pronged validation strategy; the cross-dataset pattern discussion (§V.D, position-DTW > angle-DTW on whole-body KIMORE but reversed on sparse-joint UCO) is the most scientifically interesting result in the paper.

### S3: Protocol transparency for DTW variants
Table II reports four variants (Euler, quaternion-cosine, rule-based, Procrustes-position) on the same data, enabling a like-for-like comparison — good experimental hygiene.

### S4: Cross-skeleton + ICC scaffold
§III.G explicitly defers MPJPE/PA-MPJPE/ICC against ground truth to future work rather than misreporting them — correct restraint.

---

## Weaknesses

### W1: Fixed-reference protocol risks criterion contamination (the central threat)
**Problem**: §IV.C: "the top-10 clinically-scored recordings per exercise serve as expert templates … DTW computes the similarity of each recording to the nearest expert template." The references are selected *on the outcome variable* (clinical score). If "each recording" includes the 10 templates, DTW-to-self yields distance 0 → maximal score → direct leakage inflating ρ. Even if held out, reference selection on the outcome induces range restriction that inflates correlation.
**Why it matters**: ρ=0.347 is the paper's headline validation result; if the protocol is contaminated, the headline is unsound.
**Suggestion**: report a *leave-one-subject-out* or *leave-one-recording-out* cross-validated ρ with references selected independently of the outcome (e.g., generic template or LOSO-mean template), and state explicitly that the 10 templates were excluded from the scored set.
**Severity**: Critical→Major (Critical if leakage is unaddressed; Major if shown held-out but un-cross-validated). *This is my highest-priority required revision.*

### W2: "Temporal stability" measures the mapping, not RTMW3D-L
**Problem**: §IV.B computes "mean per-joint deviation of each frame's 3D keypoints from the per-video temporal mean" on UI-PRMD, whose data are *Kinect v2 skeletons* (§III.Datasets). So the measured entity is the cross-skeleton *mapping framework* on Kinect data, not RTMW3D-L on RGB. Moreover, 0.18 mm is implausibly small for real depth-sensor data (Kinect v2 noise is ~1–2 mm+); if the statistic is the mean *signed* deviation it collapses to ~0 by construction, and if RMS, 0.18 mm needs justification.
**Why it matters**: the section title implies the pose estimator was validated for stability; it was not.
**Suggestion**: rename to "Cross-skeleton mapping numerical stability"; clarify the statistic (RMS vs signed); and don't present it as evidence about RTMW3D-L.
**Severity**: Major

### W3: FPS unmeasured/unspecified and internally inconsistent
**Problem**: No GPU model, input resolution, batch size, warm-up, measurement window, or latency vs throughput distinction. Abstract attributes 129.7 FPS to "the complete pipeline—…and coaching"; Discussion §V.A attributes it to "RTMW3D-L." A pipeline with an LLM API call every 5 s (§III.E) cannot sustain 129.7 FPS *as a pipeline* unless coaching is decoupled — which should be stated.
**Why it matters**: the central feasibility claim is under-defined.
**Suggestion**: specify hardware (GPU model, VRAM), resolution, batch=1 vs N, report mean±std and median FPS over a stated window, and a capture-to-feedback *latency* breakdown (pose, face, scoring, TTS) separately from async coaching.
**Severity**: Major

### W4: Statistical reporting gaps
**Problem**: (a) No 95% CI for any ρ or the AUC; (b) effect size not interpreted — ρ=0.347 is *weak* (r²≈0.12), yet framed as "significant clinical correlation"; (c) five per-exercise tests in Table I with no multiple-comparison correction; (d) the "overall" ρ pools exercises but the bootstrap/independence assumption across recordings within subjects is unjustified (clustered data).
**Why it matters**: significance (p<10⁻¹¹) is driven by N=378; the practical magnitude is weak.
**Suggestion**: add CIs, interpret magnitude against the field's ρ≈0.5 benchmark you yourselves cite, correct for 5 tests (Holm), and report whether recordings are nested by subject (and if so, use a mixed-effects or per-subject ρ).
**Severity**: Major

### W5: UCO "discrimination" is a low-bar task
**Problem**: AUC=0.974 for "same exercise vs different exercise" is near-trivially achievable — different rehabilitation exercises are very different movements.
**Why it matters**: presenting near-perfect AUC as "validation" overstates; the *clinically meaningful* task is quality *within* an exercise, which is the weaker UCO correlation (ρ≈0.08–0.18) you also report but don't headline.
**Suggestion**: reframe the AUC as a sanity-check baseline, and headline the within-exercise correlation instead.
**Severity**: Major

---

## Detailed Comments

### Research Questions & Hypotheses
Implicit RQ = "is real-time whole-body multimodal rehab feasible, and does the scoring correlate with clinical judgment?" The FPS answers (1), weakly; the KIMORE ρ answers (2), weakly and under protocol risk (W1).

### Research Design
Observational/correlational; appropriate for feasibility, but the fixed-reference design (W1) is the key threat to internal validity.

### Sampling Strategy
KIMORE 378 recordings, 78 subjects — adequate; but the per-exercise N≈75 with 10 templates means a meaningful fraction of each subset is reference-on-outcome.

### Data Collection
Datasets are public and well-chosen; good.

### Analysis Methods
Spearman ρ is appropriate for ordinal clinical scores; DTW variant comparison is fair. Missing: CIs, effect-size interpretation, multiple-comparison handling, subject-clustering check.

### Results Presentation
Tables I–III are clear; Figure 1 (scatter) and Figure 2 (histogram) are appropriate. But the "overall ρ" row in Table I pools exercises without a cluster-aware method.

### Reproducibility
Code "released upon acceptance" (blind — acceptable). But no LLM vendor/model, no RTMW3D-L checkpoint id, no OpenFace commit, no DTW band rationale (why 0.1?), no random seed. Cannot reproduce FPS or ρ as reported.

### Methodological Fallacies Detected
- *Criterion contamination* (reference-on-outcome) — W1.
- *Construct misalignment* — W2 (metric measures mapping, titled as pose stability).
- *Overgeneralization* — "significant clinical correlation" (abstract) given ρ=0.347 is weak and p-driven by N.

### Statistical Reporting Adequacy (Step 4a)
- **Effect sizes**: ρ reported but *magnitude not interpreted* against the field's ρ≈0.5 benchmark. → Needs Improvement.
- **Confidence intervals**: None reported for ρ or AUC. → Inadequate (add bootstrap CIs).
- **Power**: N/A (correlational, not hypothesis test), but the weak ρ with huge N means significance ≠ practical importance; this must be discussed.
- **Assumption testing**: Spearman is non-parametric (fine); but subject-clustering not addressed.
- **APA format**: ρ/p symbols fine; "p<10⁻¹¹" (abstract) vs "3.7e-12" (table) is a minor rounding inconsistency (3.7e-12 *is* < 1e-11, but pick one form).
- **Red flags**: *Selective reporting risk* — the fixed-reference protocol producing ρ=0.347 while a presumably fairer leave-one-out protocol may produce weaker results. The cross-dataset honesty in §V.D suggests the authors are *capable* of reporting weak results, which lowers my suspicion of intent, but the protocol choice still requires justification.

---

## Questions for Authors

1. **W1 (critical)**: Are the 10 KIMORE templates per exercise included in the 378 scored recordings? What is the leave-one-subject-out ρ under a reference-selection protocol independent of the clinical score?
2. **W2**: Is the 0.18 mm statistic RMS or mean-signed deviation, and on what entity (Kinect skeletons vs RTMW3D output)? Please report the same metric on RTMW3D-L running on RGB frames if feasible.
3. **W3**: Specify GPU, resolution, batch, window, and report capture-to-feedback latency; clarify the 129.7-FPS attribution.
4. **W4**: Add bootstrap 95% CIs for ρ and AUC and Holm-correct the 5 per-exercise tests; clarify subject clustering.

---

## Minor Issues

### Methodology / Equations
- Eq. 7 (DTW band) `w = max(⌊max(n,m)×0.1⌋, |n−m|)` — the `max(…, |n−m|)` term makes the constraint vacuous when sequences differ in length (the band becomes the full diagonal), defeating Sakoe-Chiba's purpose for unequal-length warping. State the intended band (e.g., `min(…)` or a fixed fraction) and justify 0.1.
- Butterworth cutoff 6.0 Hz is aggressive for some rehabilitation movements; justify (or cite) the cutoff choice.

### Reproducibility
- No GPU model, resolution, batch, checkpoint ids, or seeds reported anywhere in §IV or §V.A.

---

## Dimension Scores

| Dimension | Score (0-100) | Descriptor | Notes |
|-----------|--------------|------------|-------|
| Originality (20%) | 60 | Adequate | Integration novelty; cross-dataset insight is the defensible part |
| Methodological Rigor (25%) | 40 | Weak | Criterion contamination + FPS undefined + stats gaps |
| Evidence Sufficiency (25%) | 42 | Weak | Headline numbers not trustworthy as reported |
| Argument Coherence (15%) | 58 | Adequate | FPS internal contradiction |
| Writing Quality (15%) | 80 | Strong | Clear honest prose |
| **Weighted Average** | ~52 | **Major Revision** | |
