# ADAPT-Rehab Paper Review — Editorial Package

Multi-perspective peer review of the ATC 2026 submission:

> **ADAPT-Rehab: A Multimodal AI System for Real-Time Elderly Rehabilitation with Whole-Body 3D Pose Estimation and Voice-Interactive LLM Coaching**

Produced by the `academic-paper-reviewer` skill (v1.9.1), `full` mode, `balanced` spectrum. Review is **read-only** — no manuscript file was modified (per the skill's READ-ONLY iron rule).

## Decision

**Major Revision** (re-review required after revision).

Two Devil's Advocate CRITICAL findings prevent any Accept:
- **C1** — Headline KIMORE ρ=0.347 is produced by a fixed-reference protocol whose templates are selected on the outcome variable (clinical score), and the paper does not state the templates are held out.
- **C2** — The 129.7 FPS headline is internally inconsistent (Abstract: "complete pipeline … and coaching"; Discussion: "RTMW3D-L" alone), while coaching imposes a 5-second LLM cooldown.

All four reviewers (EIC, Methodology, Domain, Perspective) independently converge on Major Revision.

## Files

| # | File | Phase | Content |
|---|------|-------|---------|
| 0 | [`00_field_analysis.md`](00_field_analysis.md) | Phase 0 | Field analysis + 5 reviewer configuration cards |
| 1 | [`01_eic_review.md`](01_eic_review.md) | Phase 1 | Editor-in-Chief review |
| 2 | [`02_methodology_review.md`](02_methodology_review.md) | Phase 1 | Peer Reviewer 1 — methodology |
| 3 | [`03_domain_review.md`](03_domain_review.md) | Phase 1 | Peer Reviewer 2 — domain |
| 4 | [`04_perspective_review.md`](04_perspective_review.md) | Phase 1 | Peer Reviewer 3 — cross-disciplinary |
| 5 | [`05_devils_advocate.md`](05_devils_advocate.md) | Phase 1 | Devil's Advocate stress-test |
| 6 | [`06_editorial_decision.md`](06_editorial_decision.md) | Phase 2 | Editorial decision letter + revision roadmap |

## Highest-leverage revisions (gateway trio)

1. **R1** — Re-run KIMORE scoring cross-validated (LOSO), references independent of outcome; report bootstrap CI; compare to Capecci et al. baselines.
2. **R2 / R3** — Specify FPS/latency measurement + resolve Abstract-vs-Discussion contradiction; soften "validated" / "order of magnitude" / "robust" language.
3. **R5 / R6** — Evaluate-or-demote the unevaluated components (facial-state, LLM coaching); resolve the "Elderly Vietnamese" title-vs-evidence gap.

Gating item is **R1** (~4–6 weeks total revision window).

## How to use

- Read `06_editorial_decision.md` first for the decision and prioritized roadmap.
- Each Phase 1 file is an independent reviewer report (they did not see each other's reports).
- The roadmap in `06` is structured to feed directly back into `academic-paper` revision mode (R→A→C response format).

### Suggested next steps

- **Socratic revision coaching** — walk through the roadmap one issue at a time.
- **Response-letter skeleton** — draft R→A→C entries for R1–R10.
- **Codebase triage for R1** — `scripts/scoring_stack.py` and `scripts/run_kimore_experiments.py` likely contain the fixed-reference implementation that needs to become LOSO.
