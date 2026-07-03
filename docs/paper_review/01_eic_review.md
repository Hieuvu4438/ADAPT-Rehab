# Peer Review Report — EIC

## Manuscript Information

- **Title**: ADAPT-Rehab: A Multimodal AI System for Real-Time Elderly Rehabilitation with Whole-Body 3D Pose Estimation and Voice-Interactive LLM Coaching
- **Review Date**: 2026-06-30
- **Review Round**: Round 1

## Reviewer Information

### Reviewer Role
EIC

### Reviewer Identity
Program Chair, IEEE ATC "AI for Digital Health" track; applied real-time multimedia systems background.

### Review Focus
Whether the paper clears a systems-integration novelty bar; whether abstract/title claims are supported by what was actually evaluated; and whether the "first to combine X+Y+Z+W" framing is defensible.

---

## Overall Assessment

### Recommendation
- [x] **Major Revision** — Substantial revisions needed, re-review required after revision

### Confidence Score
4 — Mostly within my area of expertise, high confidence.

### Summary Assessment
This is a well-structured, honestly-written systems-feasibility paper integrating whole-body 3D pose (RTMW3D-L), AU-based facial-state detection, voice-interactive LLM-RAG coaching, and a SPARC/DTW scoring framework — running, per the authors, at 129.7 FPS on a consumer GPU. As an integration/feasibility contribution for an ATC systems track, the *premise* is reasonable and timely. However, three load-bearing claims are not yet supported at the level the abstract and title advertise: (i) the title's "Elderly Vietnamese" framing is backed by zero Vietnamese-language and zero elderly-participant evaluation; (ii) two of the four claimed component contributions (facial-state detection, LLM coaching) have *no* quantitative evaluation at all; and (iii) the headline feasibility number (129.7 FPS) is attributed to "the complete pipeline" in the Abstract but to "RTMW3D-L" alone in the Discussion (§V.A). These are fixable, but until fixed the paper over-promises relative to what the Experiments section delivers. The honesty of the Limitations section is a genuine strength and partially redeems the framing gaps, but abstracts must stand on their own.

---

## Strengths

### S1: Genuinely honest scoping
The paper repeatedly draws a clear line between "feasibility" and "clinical validation" — e.g., Experiments opens with "We explicitly do *not* claim clinical-grade joint-angle accuracy," and Limitations names "No user study with elderly participants…has been conducted." This candor is above average for the venue.

### S2: Self-contained systems narrative
The five-stage architecture (Methodology §III.A) and graceful-degradation design (LLM-unreachable → rule-based; OpenFace-unavailable → geometric fallback) show real engineering judgment, not a toy demo.

### S3: Public-benchmark grounding
Scoring is tested on KIMORE and UCO rather than only on authors' own recordings — appropriate and strengthens credibility.

### S4: Complete ethics/COI/data statements
The Ethics Statement concretely bounds what was *not* done (no human-subjects experiment, skeletal-only data) — this satisfies ATC's required statements cleanly.

---

## Weaknesses

### W1: Title/abstract foreground an unevaluated population
**Problem**: The title promises "Elderly Rehabilitation," and Contribution #1 is "Vietnamese-language." Yet no Vietnamese ASR/TTS WER, no Vietnamese LLM evaluation, no elderly participants (acknowledged in §V.H) appear.
**Why it matters**: a reader selects this paper *for* the Vietnamese/elderly angle and finds it absent.
**Suggestion**: either rename to "…Toward Elderly Vietnamese Rehabilitation…" with explicit future-work framing in the abstract, or add at least one Vietnamese-language result (e.g., Whisper-v3 vn WER on a public VN corpus) so the framing is non-empty.
**Severity**: Major

### W2: Two of four component "contributions" are described but never evaluated
**Problem**: The facial-state pipeline (§III.D) and LLM coaching (§III.E) have no quantitative results.
**Why it matters**: a contribution claim needs evidence; otherwise these are system *features*, not research contributions.
**Suggestion**: either (a) add a focused evaluation (e.g., PSPI-inspired pain correlation on UNBC-McMaster; RAG retrieval precision; guardrail false-negative rate on a contraindication test set) or (b) demote them to "integrated components, evaluation deferred to future work" in the contribution list.
**Severity**: Major

### W3: Headline FPS attribution is internally inconsistent
**Problem**: Abstract says "The complete pipeline—pose estimation, facial analysis, scoring, and coaching—sustains 129.7 FPS." Discussion §V.A says "RTMW3D-L sustains 129.7 FPS on average." Coaching includes LLM API calls with a "5-second cooldown" (§III.E), which is incompatible with a 129.7 FPS *pipeline*.
**Why it matters**: the central feasibility result is ambiguous.
**Suggestion**: state explicitly that 129.7 FPS is the perception+analysis loop (pose/face/scoring) measured on [specify GPU], with coaching on an async non-blocking thread; report end-to-end capture-to-feedback *latency* separately.
**Severity**: Major

### W4: No architecture figure is included
**Problem**: `fig1_architecture*.tex` exist on disk but are never `\input` in `main.tex`; the experiments section also references no MPJPE or ablation figure even though `fig2/3/4` PDFs exist. A "system architecture" paper with no architecture figure is a presentation gap.
**Why it matters**: readers cannot visualize the pipeline the paper is built around.
**Suggestion**: include the architecture diagram and at least the DTW-variant ablation as a figure.
**Severity**: Major (presentation)

---

## Detailed Comments

### Title & Abstract
- The abstract's final sentence ("first integrated rehabilitation system combining …") is the boldest claim and the most exposed to the panel (see DA's strongest counter-argument). The headline "129.7 FPS" framing needs the resolution in W3.

### Introduction
- Contributions are well-organized; the "explicit about scope" sentence (¶3) is a strength. But Contribution #1 (Vietnamese) and #3/#4 (facial-state, LLM) overstate their evaluated status.

### Methodology / Research Design
- (Primarily R1's role.) EIC-level observation: the architecture is coherent; the graceful-degradation story is a real plus.

### Results / Findings
- (Primarily R1's role.) EIC-level observation: the headline ρ and AUC need the methodological backing R1 will check.

### Discussion
- Limitations section is exemplary and should be preserved. The cross-dataset metric-selection insight (§V.D) is the strongest scientific finding and deserves to *lead* the contributions rather than be buried.

### Conclusion
- Appropriately modest ("This paper is a feasibility study…"). Good.

### References
- `references_posture_correction.bib`, `references.txt` exist alongside `references.bib` — confirm the bibliography intended for submission.

---

## Questions for Authors

1. Is 129.7 FPS the pose-estimator throughput, the perception+analysis loop, or end-to-end capture-to-feedback latency? On which GPU, at what input resolution and batch size?
2. Were any of the 10 KIMORE expert templates themselves included in the 378 recordings scored in Table I? (i.e., is the reference set held out?)
3. Why are "facial-state detection" and "LLM coaching" listed as *contributions* rather than integrated components awaiting evaluation?
4. Is there any Vietnamese-language or elderly-user evidence at all (even a pilot)? If not, will the title be adjusted?

---

## Minor Issues

### Figures and Tables
- `fig1_architecture*.tex`, `fig2_mpjpe_comparison.pdf`, `fig3_ablation_study.pdf`, `fig4_per_exercise.pdf` exist on disk but are not referenced from `main.tex`.
- The figure manifest (`figures/README.md`) describes Figure 1–3 with content (MPJPE bar chart, pain examples) that does not match the figures actually referenced; update or remove the README.

---

## Dimension Scores

| Dimension | Score (0-100) | Descriptor | Notes |
|-----------|--------------|------------|-------|
| Originality (20%) | 62 | Adequate | Integration novelty; "first to combine" is a low conjunction bar |
| Methodological Rigor (25%) | 48 | Weak | Protocol/FPS gaps (delegated to R1) |
| Evidence Sufficiency (25%) | 45 | Weak | Two contributions unevaluated; titled population unevaluated |
| Argument Coherence (15%) | 60 | Adequate | Abstract vs Discussion FPS contradiction |
| Writing Quality (15%) | 82 | Strong | Clear, honest prose |
| Significance & Impact (optional) | 55 | Adequate | High if claims survive revision |
| **Weighted Average** | ~55 | **Major Revision** | |

### Recommendation to Peer Reviewers
- **R1**: please determine whether the KIMORE ρ survives a fair (cross-validated, reference-not-on-outcome) protocol — this is my single highest-priority question.
- **R2**: please audit the PSPI and SPARC formula fidelity.
- **R3**: please judge whether the elderly/Vietnamese framing is defensible as written.
