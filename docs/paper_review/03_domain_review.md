# Peer Review Report — Domain (Reviewer 2)

## Manuscript Information

- **Title**: ADAPT-Rehab: A Multimodal AI System for Real-Time Elderly Rehabilitation with Whole-Body 3D Pose Estimation and Voice-Interactive LLM Coaching
- **Review Date**: 2026-06-30
- **Review Round**: Round 1

## Reviewer Information

### Reviewer Role
Peer Reviewer 2 (Domain)

### Reviewer Identity
Senior computational-rehabilitation researcher; has built Kinect/RGB-based exercise-assessment systems; reviews for *J. NeuroEng. Rehabil.* and IEEE TBME.

### Review Focus
Domain fidelity of the "clinically-inspired" components (PSPI, SPARC), correct positioning against the rehab-scoring literature, and whether clinical-authority terms ("validated," "clinically-inspired") are honestly used.

---

## Overall Assessment

### Recommendation
- [x] **Major Revision** — Substantial revisions needed, re-review required after revision

### Confidence Score
5 — Completely within my area of expertise, very confident in my assessment.

### Summary Assessment
From a domain standpoint, the system is a credible integration of respectable components (RTMW3D-L, OpenFace AUs, SPARC, constrained DTW, LLM-RAG), and the cross-dataset finding that position-DTW beats angle-DTW on whole-body data (but reverses on sparse-joint data) is a genuinely useful methodological observation for the rehab-tech community. However, the domain fidelity of two "clinically-inspired" components is questionable: the "PSPI-inspired" pain formula substantially deviates from the Prkachin–Solomon PSPI it borrows authority from, and the SPARC normalization/composite (60% SPARC / 40% LDLJ) is a custom blend presented without the citation discipline the source metrics require. The KIMORE ρ=0.347 is correctly framed as below the ρ≈0.5 clinical benchmark in the limitations, but the abstract/intro/conclusion still say "validated on two public benchmarks" — correlation of a heuristic-scored output with clinical scores is *not* validation in the rehab-measurement sense. The literature coverage is adequate but misses several foundational and recent rehab-scoring references.

---

## Strengths

### S1: Correct choice of SPARC over jerk
§III.C adopts Balasubramanian et al.'s SPARC, which *is* the duration-independent, clinically-validated smoothness metric for stroke rehab — a better choice than raw jerk, and correctly cited.

### S2: ISB-convention awareness
§III.B notes clinical angles (0 = full extension) per ISB recommendation (Wu et al. 2005, ref_isb). Respecting ISB joint coordinate systems is the right domain discipline.

### S3: Whole-body rationale
The argument that 133 keypoints (incl. face + hands) matter for *rehab* (hand placement in weight-bearing poses; facial grimacing as compensation) is domain-literate and well-motivated (§V.A).

### S4: Fixed-reference-protocol honesty about magnitude
§V.H (Scoring maturity) explicitly admits ρ≈0.347 is below the ρ≈0.5 clinical benchmark. This is correct domain calibration.

---

## Weaknesses

### W1: "PSPI-inspired" formula deviates from Prkachin–Solomon without justification
**Problem**: Original PSPI (Prkachin & Solomon 2008) = AU4 + max(AU6,AU7) + max(AU9,AU10) + AU43, with *max-pairing* for brow/cheek/nose. The paper's Eq. 5 = AU4 + 2·AU6 + AU9 + 2·AU43_approx drops AU7 and AU10, doubles AU6, and doubles an EAR-*approximated* AU43.
**Why it matters**: "PSPI-inspired" borrows the clinical authority of a validated pain index while changing its structure; labeling pain levels "Mild/Moderate/Strong/Severe" (§III.D) implies clinical grounding the formula doesn't have.
**Suggestion**: rename it "an AU-based pain proxy" (not PSPI), or validate the adapted formula against UNBC-McMaster pain scores; cite the deviation explicitly.
**Severity**: Major

### W2: SPARC/LDLJ composite is unjustified
**Problem**: §III.C states smoothness = SPARC(60%)+LDLJ(40%), but Balasubramanian et al. (2012, ref_sparc) recommend SPARC *alone*; LDLJ (Rohrer 2002, ref_ldlj) is a separate, jerk-based measure with different units and sensitivity. Blending them with arbitrary weights is exactly the "heuristic weights unvalidated" problem the paper concedes for scoring but not for smoothness.
**Why it matters**: the composite may not be duration-independent (LDLJ isn't, strictly), undermining SPARC's stated advantage.
**Suggestion**: either justify the 60/40 blend with a citation/experiment, or report SPARC alone for the validated metric. Also, Eq. 4's notation (`dω̄/ωc`) is non-standard; the canonical SPARC arc length is ∫√[(dV̂/dω)² + (dω/ωc)²]dω — please correct.
**Severity**: Major

### W3: "Validated" overclaims correlation
**Problem**: Abstract, Contribution #5, and Conclusion say the scoring framework is "validated on two public benchmarks." In rehabilitation measurement, "validation" means demonstrating that the instrument measures the construct of interest against a gold standard with appropriate magnitude — ρ=0.347 (weak, r²≈0.12) is *correlation*, not validation, and the paper itself says the per-dimension weights are heuristic and uncalibrated.
**Why it matters**: clinical readers will read "validated" as clinic-ready.
**Suggestion**: replace "validated" with "benchmark-correlated" or "evaluated" throughout; reserve "validation" for the future clinician-calibrated work.
**Severity**: Major

### W4: Rehab-scoring literature is thin
**Problem**: The Related Work covers CV-rehab, 3D pose, pain, LLM, biomechanics — but omits foundational *exercise-quality scoring* work the rehab community measures against: e.g., the KIMORE paper's own baseline methods (Capecci et al. 2019 includes SVR/RF regression baselines for scoring — you should compare to those, since you use their dataset), the **UI-PRMD** original-scoring work (Vakanski et al. on DTW-based exercise quality), and recent **learned** exercise-scoring (e.g., Pose-based quality assessment with graph CNNs).
**Why it matters**: without these, the contribution's novelty ("scoring framework with benchmark validation") is unpositioned against the existing rehab-scoring literature on the *same datasets*.
**Suggestion**: add an "Exercise Quality Scoring" paragraph; report or compare to Capecci et al.'s KIMORE baselines (their regression ρ is the comparison your ρ=0.347 needs).
**Severity**: Major

### W5: Quaternion-vs-dot-product claim is asserted, not evaluated
**Problem**: §III.B and §V.C argue quaternion angles are "better for multi-plane" than dot-product, citing Aurand et al. 2024 (ref_quaternion_angles). But the paper *uses* quaternion angles and never *compares* them to dot-product on its own data — so the multi-plane-advantage claim is borrowed from the citation, not demonstrated.
**Why it matters**: a claimed advantage should be evidenced in the system that claims it.
**Suggestion**: add a within-system ablation (quaternion vs dot-product angle DTW ρ) to evidence the claim, or soften to "adopted per Aurand et al."
**Severity**: Minor→Major

---

## Detailed Comments

### Literature Review
- **Coverage**: missing foundational exercise-quality-scoring work on the *same datasets* used (W4).
- **Integration quality**: otherwise thematic and reasonably synthesized.
- **Research gap argument**: clear, though the "first to combine" framing is thin (see DA).

### Theoretical Framework
- "Clinically-inspired" is the de facto framework; its boundaries are honestly drawn in §V.H — but the abstract ignores its own limitation.

### Academic Argument Quality
- **Factual accuracy**: PSPI deviation (W1) and SPARC notation (W2) are domain-accuracy issues.
- **Argument logic**: the cross-dataset metric-selection argument (§V.D) is the strongest domain insight and is logically sound. The weakest is the "first to combine 5 things" novelty (see DA).
- **Terminology precision**: "validated" used loosely (W3).

### Contribution to the Field
- **Incremental contribution**: a legitimate systems-track integration *if* the benchmark correlation survives R1's protocol test and the unevaluated components are either evaluated or demoted.
- **Positioning**: needs comparison to Capecci et al.'s KIMORE baselines and Vakanski et al.'s UI-PRMD DTW scoring.
- **Overclaiming**: "validated" (W3), "PSPI" (W1), "robust guardrails" (Discussion §V.F) — three domain-credibility terms used loosely.

### Missing Key References
- **Capecci, M. et al. (2019), "KIMORE…" + their companion evaluation paper** — the KIMORE authors' own regression baselines; your ρ=0.347 must be compared to their reported scoring performance.
- **Vakanski, A. et al. — UI-PRMD exercise-quality / DTW scoring** — foundational rehab-DTW-scoring work on the dataset you cite.
- **A graph-CNN pose-based exercise quality assessment** (recent, in the 2022–2024 *J. NeuroEng. Rehabil.* / *Sensors* tradition) — to position against learned scoring.
- **Balasubramanian et al. (2015) "Movement smoothness … stroke"** — the clinical-validation companion to the SPARC metric paper you cite.
- For PSPI: **Hammal & Cohn 2012** on AU-based pain, which uses the original PSPI correctly — to anchor your deviation.

---

## Questions for Authors

1. Can you report KIMORE scoring performance against Capecci et al.'s own baselines (their regression ρ) using the same split?
2. Can you justify or rename the "PSPI-inspired" formula (Eq. 5), given it drops AU7/AU10 and doubles AU6/AU43 vs. the original?
3. Can you correct Eq. 4 (SPARC) to the canonical form and justify the 60/40 SPARC+LDLJ blend?
4. Is there an internal quaternion-vs-dot-product ablation, or is the multi-plane advantage adopted on citation alone?

---

## Minor Issues

### Terminology / Citations
- ref_quaternion_angles (Aurand et al. 2024) — verify this exists as cited; "IEEE Trans. Biomed. Eng., 2024" with no volume/pages is thin.
- ref_rtmw3d title in bib ("RTMW:…") omits the "3D"; minor.

---

## Dimension Scores

| Dimension | Score (0-100) | Descriptor | Notes |
|-----------|--------------|------------|-------|
| Originality (20%) | 58 | Adequate | Position-vs-baselines missing |
| Methodological Rigor (25%) | 50 | Adequate | (delegated to R1) |
| Evidence Sufficiency (25%) | 48 | Weak | PSPI/SPARC fidelity + missing baseline comparison |
| Argument Coherence (15%) | 62 | Adequate | Cross-dataset insight strong |
| Writing Quality (15%) | 80 | Strong | Clear domain prose |
| Literature Integration | 55 | Adequate | Rehab-scoring literature gap (W4) |
| **Weighted Average** | ~54 | **Major Revision** | |
