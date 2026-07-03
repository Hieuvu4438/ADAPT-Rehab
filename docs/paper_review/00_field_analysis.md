# Phase 0 — Field Analysis & Reviewer Configuration

## Paper Basic Information

- **Title**: *ADAPT-Rehab: A Multimodal AI System for Real-Time Elderly Rehabilitation with Whole-Body 3D Pose Estimation and Voice-Interactive LLM Coaching*
- **Venue (self-declared)**: ATC 2026 (IEEE, Intl. Conf. on Advanced Technologies for Communications), 6-page conference format, blind review
- **Full-text length**: ~5,200 words (current draft)
- **References**: 35 entries in `references.bib`
- **Figures referenced in text**: 2 (KIMORE scatter, UCO histogram). Note: `fig1_architecture*.tex`, `fig2_mpjpe_comparison.pdf`, `fig3_ablation_study.pdf`, `fig4_per_exercise.pdf` exist on disk but are **not** `\input` or `\includegraphics`'d anywhere in `main.tex`.

## Field Analysis

| Dimension | Analysis Result |
|-----------|-----------------|
| Primary Discipline | Applied computer vision / HCI for health (systems paper) |
| Secondary Disciplines | Biomechanics (kinematics, SPARC/DTW); Affective computing (AU/PSPI/PERCLOS); Speech/NLP (Whisper/Edge-TTS/LLM-RAG) |
| Research Paradigm | Quantitative + Engineering (systems feasibility) |
| Methodology Type | System integration + benchmark evaluation (correlational) |
| Target Journal Tier | Q2–Q3 conference (ATC is a solid mid-tier IEEE comms venue; not a top-tier CV/clinical journal) |
| Paper Maturity | Pre-submission draft — structure complete, language polished, *but* headline results have unaddressed methodological gaps and 1 missing core figure |

**Paper-maturity flags driving the review:** the abstract/intro foreground "elderly Vietnamese" and "validated scoring," but the evaluation covers neither Vietnamese language nor elderly participants nor most claimed components (facial state, LLM coaching). This framing-vs-evidence gap is the central tension the panel will examine.

## Recommended Target Venues (Top 3)

1. **ATC 2026** (self-declared) — fit: applied comms + AI-for-health track; the feasibility framing suits a systems track. **Acceptance bar is reachable with major revision.**
2. **IEEE EMBC / ICDSP** — better fit for the multimodal-health-systems angle if ATC's comms focus feels thin.
3. **IEEE Access / Sensors** (special issue on rehab tech) — if the authors later add the missing component evaluations; longer format would absorb the currently-unevaluated contributions.

## Reviewer Configuration Cards

### Reviewer Configuration Card #1 — EIC

**Role**: EIC
**Identity**: Program Chair of an IEEE ATC-track on "AI for Digital Health"; background in applied real-time multimedia systems; has accepted integration/feasibility papers but demands that *every* claimed contribution carry *some* evidence.
**Review Focus**:
1. Does the paper clear ATC's novelty bar for a *systems integration* paper, or is it assembly of off-the-shelf parts?
2. Are the abstract/title claims supported by what was actually evaluated?
3. Is the "first to combine X+Y+Z+W" framing defensible?

**Will particularly care about**: Whether headline numbers (129.7 FPS, ρ=0.347, AUC=0.974) are reported with the rigor a feasibility claim requires, and whether "validated" language is earned.
**Possible blind spots**: Won't deeply audit the DTW protocol or SPARC formula (that's R1/R2's job).

### Reviewer Configuration Card #2 — Peer Reviewer 1 (Methodology)

**Role**: Methodology Reviewer
**Identity**: Measurement-and-evaluation scientist in human-motion analysis; publishes on 3D pose benchmarking and reproducibility; ruthlessly checks whether a metric measures what its label says and whether a correlation is earned by the protocol.
**Review Focus**:
1. The KIMORE fixed-reference protocol (top-10-by-clinical-score as templates) — is the headline ρ earned or inflated by criterion contamination?
2. The 0.18 mm "temporal self-consistency" — what does it actually measure?
3. The 129.7 FPS claim — measurement protocol, hardware, latency vs throughput, and the internal contradiction between "RTMW3D-L sustains 129.7 FPS" (Discussion) and "complete pipeline … sustains 129.7 FPS" (Abstract).
4. Statistical reporting (CI, effect size, multiple comparisons).

**Will particularly care about**: Whether the central validation number survives a leave-one-out / cross-validated protocol with references selected independently of the outcome.
**Possible blind spots**: May under-weight the engineering-integration value; a pure-methods reviewer can miss that a feasibility paper need not hit clinical-grade rigor if framed honestly.

### Reviewer Configuration Card #3 — Peer Reviewer 2 (Domain)

**Role**: Domain Reviewer (rehabilitation technology + biomechanics)
**Identity**: Senior researcher in computational rehabilitation; has built Kinect/RGB-based exercise-assessment systems; knows PSPI/SPARC/ISB conventions and the KIMORE/UCO literature firsthand; reviews for *J. NeuroEng. Rehabil.* and IEEE TBME.
**Review Focus**:
1. PSPI "inspired" formula fidelity (AU4+2·AU6+AU9+2·AU43 vs. the Prkachin–Solomon original) — is the adaptation defensible or a misappropriated label?
2. SPARC formula notation (Eq. 4) and the SPARC/LDLJ 60/40 composite — is it used per Balasubramanian et al.?
3. Are KIMORE/UCO results positioned correctly against the field (ρ≈0.5 clinical benchmarks)?
4. Missing foundational references in rehab scoring.

**Will particularly care about**: Domain accuracy of the kinematic/affective claims and whether "clinically-inspired" is honest shorthand or borrowed-clinical-authority laundering.
**Possible blind spots**: May not push on the FPS/throughput reproducibility (R1's turf).

### Reviewer Configuration Card #4 — Peer Reviewer 3 (Cross-disciplinary / Practical)

**Role**: Perspective Reviewer
**Identity**: HCI / accessibility researcher working on voice interfaces for low-literacy and aging populations; familiar with Vietnamese-language ASR/TTS and the deployment realities of LLM-API coaching in low-connectivity rural settings.
**Review Focus**:
1. The "elderly Vietnamese" framing is the #1 contribution but is *entirely* unevaluated — what would an HCI reviewer demand?
2. Keyword-based LLM "guardrails" and "safety architecture" — are these real safety mechanisms or aspirational?
3. Stakeholder voices missing (actual elderly users, physiotherapists, caregivers).
4. Deployment feasibility: GPU dependency + internet-for-API + rural Vietnam.

**Will particularly care about**: Whether the title's promise to a vulnerable population is matched by any evidence involving that population, and whether safety claims rest on unvalidated pain detection.
**Possible blind spots**: May discount that feasibility papers legitimately defer clinical validation; risk of demanding a user study that's out of scope.

### Reviewer Configuration Card #5 — Devil's Advocate

**Role**: Devil's Advocate (stress-test, no scoring)
**Identity**: Adversarial reviewer whose job is to construct the strongest case *against* acceptance by attacking the paper's load-bearing claims and the honesty of its headline framing.
**Review Focus**:
1. Is the headline ρ=0.347 an artifact of reference-on-outcome selection?
2. Does "order of magnitude" (0.400 vs 0.042) rhetorically inflate a null-vs-weak comparison?
3. Is "complete pipeline 129.7 FPS" contradicted by the 5-second LLM cooldown and the Discussion's attribution to RTMW3D-L alone?
4. Does the title overclaim a population (Vietnamese elderly) that was never studied?
5. Is the "first to combine 5 things" novelty a low-conjunction bar?

**Will particularly care about**: Whether the paper's *strongest-sounding* claims (FPS, ρ, "first," "validated," "order of magnitude") survive skeptical scrutiny.
**Possible blind spots**: Must not nitpick; will be required to acknowledge the paper's genuine strengths before attacking.

## Quality Gates (self-check)

- [x] All 6 analysis dimensions completed, none omitted
- [x] All 5 Reviewer Configuration Cards produced
- [x] Review focus areas of 5 reviewers do not overlap
- [x] Reviewer 3's angle is genuinely cross-disciplinary (HCI/accessibility), distinct from R1/R2
- [x] Recommended target journals match the paper's discipline and quality
- [x] Identity descriptions are specific (not "a methodology expert" but a measurement-and-evaluation scientist in human-motion analysis)
