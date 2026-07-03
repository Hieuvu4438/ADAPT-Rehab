# ADAPT-Rehab — Dataset Verification & Evaluation Strategy

**Paper type clarification (read this first)**: ADAPT-Rehab is an **application / system paper**, not a method paper. We do **not** claim SOTA on pose estimation, pain detection, or any single sub-task. We compose existing models into an end-to-end multimodal coaching pipeline for elderly Vietnamese rehabilitation. The novelty is the **system** and the **scoring methodology**, not the backbone models. This changes the evaluation strategy entirely:
- We do **not** need to beat M3GYM's pose-estimation SOTA.
- We **do** need to show that (a) the integrated system works, (b) our proposed scoring system is discriminative and clinically meaningful, and (c) the system is more useful than comparable application-level baselines (Pose Tutor, OpenCap, STPT, etc.).

**Date**: 2026-06-28

---

## Part 1 — Verification of the 7 Candidate Dataset Papers

| # | Dataset | Venue | Reputation | Public? | RGB? | Verdict |
|---|---------|-------|-----------|---------|------|---------|
| 1 | **UCO Physical Rehabilitation** (Aguilar-Ortega 2023) | *Sensors* MDPI (Q2 SCI, IF ~3.4) | Reputable peer-reviewed | YES — `github.com/AVAuco/ucophyrehab` | **YES** — 5 RGB cams + OptiTrack GT ±0.5 mm | **Use**: scoring & pose |
| 2 | **AHA-3D** (Antunes 2018) | **BMVC 2018** (top-tier) | Highly reputable | YES — `vislab.isr.ist.utl.pt` | NO (Kinect v2 skeleton) | **Use**: senior-baseline |
| 3 | **3DYoga90** (Kim 2023) | arXiv preprint only | Not peer-reviewed | YES — `github.com/seonokkim/3DYoga90` | YES | Pretraining only; not citable as benchmark |
| 4 | **ExerGeneDB** (Pan 2025) | *J. Sport & Health Science* | Reputable but out of scope | YES | **NO** — gene-expression DB | **Drop** — not CV |
| 5 | **Riccio (2024)** | arXiv preprint | Not peer-reviewed | Uses Kaggle/InfiniteRep | YES | Method baseline (BiLSTM); not a benchmark |
| 6 | **MEx** (Wijekoon 2019) | arXiv preprint | Not peer-reviewed | YES — `github.com/anjanaw/MEx` | NO (depth + pressure + accelerometers) | Multi-modal fusion ablation only |
| 7 | **M3GYM** (Xu et al. 2024) | **CVPR 2024** (top-tier) | Highly reputable | YES | **YES** — 8 RGB cams + 2D/3D + mesh | **Use**: pose, real-world generalization |

### Selected for our experiments (per your choice)

**UCO + AHA-3D + M3GYM + KIMORE** (KIMORE added as the canonical clinical-score benchmark — see Part 2).

---

## Part 2 — SOTA Datasets to Use as Primary Evaluation Suite

| Dataset | Why for an application paper | GT we will use |
|---------|------------------------------|----------------|
| **UCO Physical Rehabilitation** | Only RGB+motion-capture rehab dataset → validates our pose + scoring on **real rehab patients** with sub-mm GT | OptiTrack joint positions → angle MAE |
| **AHA-3D** | Senior-specific, multi-generational → supports the "elderly-specific" claim of the paper | Frame-level action labels + age split |
| **M3GYM** (CVPR 2024) | Real-world gym, multi-view, occlusions → shows our pipeline generalizes outside the lab | 2D/3D keypoints, action labels, expert assessments |
| **KIMORE** (Capecci 2020) | Canonical clinical-score benchmark in rehab-AQA → required by reviewers in this subfield | Clinical 0–100 quality scores per trial |
| (Optional) **REHAB24-6** (2025) | Most recent RGB rehab benchmark, 6 exercises, cross-skeleton-format | 2D/3D skeleton + repetition annotations |

**Skip** Human3.6M / 3DPW / MPI-INF-3DHP / UNBC-McMaster as **primary** benchmarks — they are component-level SOTA suites, and our paper does not claim SOTA on any component. Mention them in a backbones table only.

---

## Part 3 — Evaluation Strategies for the Scoring System (the user's question)

This is the heart of the paper's empirical claim. Below are **7 strategies**. Your proposed "same-pose > different-pose score" is Strategy 1 and it IS academically valid.

### Strategy 1 — Discriminability / Verification Test (USER'S IDEA — VALIDATED) ✅

**What it tests**: Can our scoring system tell that two clips of the **same exercise** are more similar than two clips of **different exercises**?

**Protocol** (standard from face/speaker verification & pose similarity):
- Build a **pair list**: for each exercise class c ∈ {1..K}, form N positive pairs (same class) and N negative pairs (different class). Match difficulty by length & subject.
- Run our scoring function `s(V_i, V_j)` on every pair.
- Compute the **binary classification metrics** treating "same class" as positive:
  - **AUC-ROC** (target ≥ 0.95)
  - **EER** (Equal Error Rate — point where FAR = FRR; lower is better)
  - **Mann–Whitney U test** on `s_same` vs `s_diff` (report U, Z, p-value; target p < 0.001)
- Plot the score histogram: two well-separated distributions (bimodal) = strong discriminability.

**Why this is publishable**: This is exactly the verification paradigm used by:
- *Google Research — Recognizing Pose Similarity in Images and Videos* (pose-similarity model trained on contrastive pairs).
- *Group-Aware Contrastive Regression for AQA* (ICCV 2021) — uses same/different class contrastive learning.
- Face verification (FaceNet), speaker verification (x-vectors) — established methodology, widely accepted in reviewers' minds.

**Dataset mapping**: Use **AHA-3D** (4 senior-fitness exercise classes → 4-choose-2 = 6 class pairs) and **KIMORE** (5 exercises). M3GYM (500+ actions) is excellent for stress-testing with many confusable actions (e.g., "warrior I" vs "warrior II").

### Strategy 2 — Correlation with Clinical Ground Truth (THE standard in rehab-AQA)

**What it tests**: Does our numerical score agree with **expert clinicians' ratings**?

**Protocol**:
- On **KIMORE** (which has 0–100 clinical scores per trial), run our pipeline to produce a score per trial.
- Compute **Spearman ρ** (rank correlation — preferred, since clinical scores are ordinal) and **Pearson r** (linear).
- **Target**: ρ ≥ 0.65 (current SOTA on KIMORE is ~0.65; recent papers report 81.85% accuracy which translates to ρ ~ 0.7+).
- Scatter plot: our score (x) vs clinical score (y), with linear regression line and 95% CI.

**Why this is publishable**: Spearman ρ with expert scores is the universal evaluation metric in:
- Action Quality Assessment surveys (e.g., "A Decade of AQA" arXiv 2502.02817).
- All KIMORE / UI-PRMD papers (Capecci 2020, Bilić 2024, etc.).
- Clinical-rehab validation literature.

### Strategy 3 — Graded-Difficulty / Monotonicity Test (sensitivity)

**What it tests**: Does our score degrade **gracefully and monotonically** as the performer makes more errors?

**Protocol**:
- Take a clean reference trial. Synthetically perturb the keypoints by **controlled amounts**: ±5°, ±10°, ±15°, ±20°, ±25° on key joints (knee, shoulder, hip).
- For each perturbation level, compute our score.
- Plot: score vs perturbation magnitude.
- **Pass criterion**: monotonically decreasing, with a near-linear region in the clinically meaningful range. Report Spearman ρ between perturbation and score (target ρ ≤ −0.95).

**Why this is publishable**: Borrowed from psychometrics and biosignal validation — establishes **sensitivity** (small input changes produce measurable output changes), which is distinct from discriminability.

### Strategy 4 — Cross-Subject / Cross-View Robustness (generalization)

**What it tests**: Is the score stable across different **subjects** (body shapes, ages) and **camera views**?

**Protocol**:
- On **UCO** (5 viewpoints × 27 subjects) and **M3GYM** (8 views), compute scores for the same exercise performed by different subjects / from different views.
- Use **Leave-One-Subject-Out CV** (LOSOCV) and **Leave-One-View-Out CV** (LOVOCV).
- Report the **coefficient of variation** (CV = std/mean) across subjects and views. Target: CV ≤ 0.15 for the same exercise class.
- This is what UCO's authors themselves did (their Table 4 — viewpoint analysis).

### Strategy 5 — Comparison with Application-Level Baselines (NOT SOTA components)

**Important distinction**: We compare against **other application systems** that solve a similar problem, not against component-SOTA papers. This is the correct baseline level for a system paper.

| Baseline | Venue | Why compare |
|----------|-------|-------------|
| **Vanilla DTW on Euler angles** (Capecci 2020) | *Sensors* | Default template-matching baseline on KIMORE |
| **OpenCap** (Kidziński et al., *Nat. Commun.* 2020) | Nature | Single-camera biomechanics — engineering baseline |
| **Pose Tutor** (Bhat et al., CVPRW 2022) | CVPRW | Explainable pose-correction system |
| **STPT** (IEEE 2023) | IEEE | Elderly assessment via DTW — closest demographic match |
| **Real-Time Action Scoring (DTW+NCC)** (Nature 2025) | Nature Sci. Rep. | Most recent DTW-based rehab scoring system |
| **Wang et al. (2023) DTW+Classification for BaDuanJin** | (in SciDirect survey) | DTW-based exercise assessment — Asian clinical context |
| **MediaPipe BlazePose + cosine rule-based** | (system baseline) | De-facto open-source default — must beat this to justify complexity |

**Comparison metrics** (same across baselines, on KIMORE + UCO + AHA-3D):
- Spearman ρ with clinical scores
- Discriminability AUC (Strategy 1)
- Cross-subject CV (Strategy 4)

We do **not** need to win every cell — we need to be **competitive on the scoring metrics** while being the only system that adds **multimodal coaching + safety + Vietnamese language**. That is the application-paper story.

### Strategy 6 — Component Ablation (justify each design choice, not beat SOTA)

For each of our claimed novelties, ablate **on the same KIMORE/UCO benchmark**:

| Ablation axis | Variants | Expected finding |
|---------------|----------|------------------|
| Joint angle representation | Euler vs. **quaternion** | Quaternion better at multi-view / supine poses |
| Smoothness metric | Jerk vs. LDLJ vs. **SPARC** | SPARC most reliable across subjects |
| DTW variant | Vanilla vs. weighted vs. **constrained weighted** | Constrained is faster AND more accurate on bounded rehab motions |
| Pose backbone | MediaPipe vs. MeTRAbs vs. HybrIK | Direct-3D better for lying/seated rehab (justifies architecture) |
| Pain channel on/off | System with vs. without | Safety net; demonstrates multimodal value |

This is **internal ablation**, not external comparison. It justifies our system-design choices, which is the legitimate way to claim novelty in an application paper.

### Strategy 7 — User Study (application value)

For an application paper, a user study is often the highest-weighted evidence:

- **Participants**: N ≥ 20 Vietnamese seniors (within-subject AB design) + 3 clinicians for expert rating.
- **Conditions**: (A) ADAPT-Rehab full pipeline; (B) MediaPipe + rule-based (baseline system).
- **Metrics**:
  - **SUS** (System Usability Scale, target ≥ 75 for "good")
  - **Task completion rate** per session
  - **Expert clinician rating** (1–5 Likert; intra-class correlation for inter-rater reliability)
  - **Safety false-negative rate** on induced contraindications (must be < 1%)
  - **End-to-end latency** p95 < 800 ms

---

## Part 4 — Literature Review of Comparable Application Papers

These are system/application papers similar in scope to ADAPT-Rehab. They define the comparison group for the paper's "related work" section.

| Paper | Year / Venue | System summary | Evaluation they used |
|-------|--------------|----------------|----------------------|
| Capecci et al. | 2020, *Sensors* | DTW on Kinect skeleton for KIMORE rehab exercises | Spearman ρ with clinical score on KIMORE |
| Bhat et al. — **Pose Tutor** | 2022, CVPRW | Explainable pose-correction on in-the-wild video | Per-joint correctness + qualitative user eval |
| Kidziński et al. — **OpenCap** | 2020, *Nat. Commun.* | Single-camera biomechanics via OpenSim | Cross-validation against marker-based MoCap |
| **STPT** (spatio-temporal trajectory) | 2023, IEEE | DTW-based elderly rehab assessment | Score-vs-clinical, cross-subject |
| **Real-Time Action Scoring (DTW+NCC)** | 2025, *Sci. Rep.* | Real-time DTW + NCC scoring for PT | Discriminability + clinician correlation |
| **PosePilot** | 2025, arXiv | Edge-AI posture-correction with personalized feedback | User study + edge latency |
| AI Coaching Mobile App (Beckham et al.) | 2023, PMC/NIH | Mobile AI coach compared to expert PTs | Direct expert-vs-system comparison |
| Wang et al. — BaDuanJin | 2023 | DTW + classification for traditional Chinese exercise | Classification accuracy + DTW alignment |
| Antunes et al. — **AHA-3D** | 2018, BMVC | Senior fitness dataset, recognition + segmentation | Accuracy on action recognition + segmentation |
| Aguilar-Ortega et al. — **UCO** | 2023, *Sensors* | Multi-pose-estimator benchmark on RGB rehab | MPJPE / PA-MPJPE per viewpoint + per subject position |

**Gap that ADAPT-Rehab fills** (the paper's positioning):
1. **No existing system targets Vietnamese seniors** — language + cultural gap.
2. **No existing system integrates pain + fatigue + engagement detection into rehab coaching** — papers above are pose-only.
3. **No existing system uses direct-3D pose for elderly rehab** (most use MediaPipe 2D or Kinect skeleton).
4. **No existing system validates SPARC + quaternion + constrained-DTW as a combined scoring stack on RGB** — KIMORE papers use Euler + vanilla DTW on Kinect.

---

## Part 5 — Recommended Experimental Protocol (5 Phases)

### Phase 1 — Pose backbone selection (datasets: UCO + M3GYM)
- Compare MediaPipe, MeTRAbs, HybrIK, RTMPose-WholeBody on UCO (RGB + OptiTrack GT).
- Report joint-angle MAE (knee, shoulder, hip) stratified by viewpoint and patient position (supine/seated/standing).
- Outcome: justify why we chose direct-3D for the pipeline.

### Phase 2 — Scoring system validation (datasets: KIMORE + UCO + AHA-3D)
- Run **all 7 strategies** from Part 3 on our scoring stack.
- Run **Strategy 5** baselines (Capecci DTW, OpenCap, STPT, Real-Time Action Scoring).
- Produce the main result table: ρ, AUC, EER, CV per dataset, with ablation rows.

### Phase 3 — Multimodal / pain / fatigue sub-system (datasets: small elderly pilot + UNBC-McMaster for sanity only)
- Validate the pain/fatigue/engagement modules independently with **component-level** metrics on UNBC (sanity, not SOTA chase) + expert ratings on the elderly pilot.
- Frame as "system modules validated to a level suitable for the safety-critical feedback loop", not "we beat pain-detection SOTA".

### Phase 4 — End-to-end user study (Vietnamese seniors, N ≥ 20)
- Within-subject AB against rule-based baseline.
- SUS, completion, expert agreement (κ), latency, safety FN rate.

### Phase 5 — System paper integration
- Cross-cutting ablation table: each module on/off.
- Failure case analysis (3–5 qualitative examples).
- Deployment cost (FPS, GPU/CPU, memory).

---

## Part 6 — Concrete Experiment: The Verification Test on KIMORE (worked example)

This is the worked example for the user's proposed experiment, formalized.

**Setup**: KIMORE has 5 exercises × 78 subjects × 5 trials = ~1950 trials.

1. **Build pairs**:
   - Positive pairs: same exercise class, different subjects → ~50,000 pairs.
   - Negative pairs: different exercise classes → sample 50,000 to balance.
2. **Score with our pipeline**: `s_ij = our_scoring(V_i, V_j)`.
3. **Statistics**:
   - AUC of binary classifier "is this a same-exercise pair?" using only `s_ij`.
   - Mann–Whitney U test on positive vs negative score distributions.
   - Per-exercise-class confusion matrix (which exercise pairs are hardest to separate?).
4. **Expected figure**: histogram with two clearly separated modes (same-class scores centered near 0.85, different-class near 0.30) and the AUC printed on the plot.

**Repeat the same protocol** on **AHA-3D** (4 senior-fitness classes, multi-generational) and **UCO** (8 rehab exercises, multi-view). Triangulation across 3 datasets is what reviewers want.

**Augment with Strategy 2** (ρ vs clinical score on KIMORE) and **Strategy 4** (LOSOCV) — that's the standard evidence package for an AQA-style scoring paper.

---

## Part 7 — What to explicitly NOT do (to avoid reviewer pushback)

1. Do **not** claim SOTA on pose estimation. State: "We use off-the-shelf pretrained backbones (MeTRAbs, HybrIK, RTMPose) and do not retrain them; our contribution is the scoring and coaching layer on top."
2. Do **not** claim SOTA on pain detection. State: "We adopt the FACS-based PSPI formula (Prkachin-Solomon 2008) for its clinical interpretability, accepting lower raw accuracy than end-to-end CNN approaches in exchange for explainability to clinicians."
3. Do **not** compare against pure method papers (ViTPose, ErAS, etc.) on their own benchmarks. Compare against **application systems** (Strategy 5 baseline list).
4. Do **not** omit the user study — for an application paper targeting CHI / IJCAI / AAAI application tracks, the user study is the centerpiece, not the appendix.
5. Do **not** skip the failure case analysis — application papers are judged on honesty about failure modes.

---

## Sources (web search — for citation tracking)

**Datasets**:
- UCO Physical Rehabilitation — `mdpi.com/1424-8220/23/21/8862`
- AHA-3D (BMVC 2018) — `link.springer.com/article/10.1007/s00530-021-00815-4`
- M3GYM (CVPR 2024) — `openaccess.thecvf.com`
- KIMORE SOTA (2025, 81.85%) — `tandfonline.com/doi/full/10.1080/24751839.2025.2454053`
- KIMORE SOTA (2024) — `sciencedirect.com/science/article/pii/S0010482524016639`
- REHAB24-6 — `arxiv.org/abs/2505.18412`, `zenodo.org/records/13305826`

**Application baselines**:
- Pose Tutor (Bhat, CVPRW 2022)
- OpenCap (Kidziński, *Nat. Commun.* 2020) — `nature.com/articles/s41467-020-17907-9`
- Real-Time Action Scoring (2025, *Sci. Rep.*) — `nature.com/articles/s41598-025-29062-7` / `pmc.ncbi.nlm.nih.gov/articles/PMC12749503/`
- STPT elderly (IEEE 2023) — `ieeexplore.ieee.org/document/10098793`
- PMC AI Coaching Mobile App — `pmc.ncbi.nlm.nih.gov/articles/PMC10523222/`
- PosePilot (2025) — `arxiv.org/html/2505.19186v1`

**Evaluation methodology (verification / AQA)**:
- Group-Aware Contrastive Regression for AQA (ICCV 2021) — `openaccess.thecvf.com/content/ICCV2021/papers/Yu_Group-Aware_Contrastive_Regression_for_Action_Quality_Assessment_ICCV_2021_paper.pdf`
- A Decade of AQA survey — `arxiv.org/html/2502.02817v1`
- Pose similarity (Google Research) — `research.google/blog/recognizing-pose-similarity-in-images-and-videos/`
- Pose discrimination in similarity space (BMVC 1999) — `bmva-archive.org.uk/bmvc/1999/papers/52.pdf`
- Vision-Based Action Evaluation Survey (MDPI Sensors) — `mdpi.com/1424-8220/19/19/4129`

**Scoring methodology**:
- SPARC (Balasubramanian et al.) — `semanticscholar.org/paper/A-Robust-and-Sensitive-Metric-...`
- PSPI (Prkachin-Solomon 2008) — foundational pain-intensity formula
- PERCLOS (Wierwille 1994) — foundational fatigue metric
- Engagement Index (Whitehill 2014) — foundational boredom metric
