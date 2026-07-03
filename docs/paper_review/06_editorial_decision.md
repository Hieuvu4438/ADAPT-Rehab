# Editorial Decision

## Manuscript Information

- **Title**: ADAPT-Rehab: A Multimodal AI System for Real-Time Elderly Rehabilitation with Whole-Body 3D Pose Estimation and Voice-Interactive LLM Coaching
- **Submission Date**: 2026-06-30
- **Decision Date**: 2026-06-30
- **Review Round**: Round 1

---

## Decision

### Major Revision

*(Re-review required after revision.)*

---

## Reviewer Summary

| Reviewer | Role | Recommendation | Confidence |
|----------|------|---------------|------------|
| EIC | ATC "AI for Digital Health" PC chair | Major Revision | 4 |
| Reviewer 1 | Measurement-and-evaluation scientist (motion analysis) | Major Revision | 5 |
| Reviewer 2 | Computational-rehabilitation domain expert | Major Revision | 5 |
| Reviewer 3 | HCI / accessibility researcher | Major Revision | 4 |
| Devil's Advocate | Adversarial stress-test | (flagged 2 CRITICAL) | n/a |

---

## Consensus Analysis

### Points of Agreement (Consensus)

**[CONSENSUS-4]** (all four non-DA reviewers + DA agree):
1. **The KIMORE fixed-reference protocol must be re-run cross-validated with references independent of the outcome** (R1-W1; DA-C1; corroborated EIC Q2). The single most important required revision.
2. **The 129.7 FPS result must be re-specified** (hardware, resolution, latency-vs-throughput, async-coaching) and the Abstract/Discussion contradiction resolved (R1-W3; EIC-W3; DA-C2).
3. **"Validated" / "order of magnitude" / "robust guardrails" language must be softened to match evidence** (R2-W3; DA-M1/M4; EIC recommendation).
4. **The facial-state and LLM-coaching contributions need at least minimal evaluation or demotion to integrated components** (EIC-W2; R3-W2/W3; DA-M6; partial R1).
5. **Missing architecture figure must be included** (EIC-W4).

**[CONSENSUS-3]** (3/4 reviewers agree):
- **The "Elderly Vietnamese" title/Contribution #1 overpromises relative to evaluation** (EIC-W1, R3-W1, DA-M2 agree; detailed differently — EIC on title, R3 on population+language+stakeholders, DA on title-evidence mismatch). **Dissent: R2**, whose domain focus is biomechanics/formula fidelity rather than population framing. **Editor's Resolution**: adopt; the title/framing adjustment is required (also an ethics-of-claim issue per R3), but the depth of the change is the authors' choice (rename *or* add Vietnamese evidence).
- **Effect-size magnitude must be interpreted against the ρ≈0.5 benchmark the paper itself cites** (R1-W4, R2-W4, DA-M4). EIC did not raise effect-size interpretation explicitly but recommended "honest framing."

### Points of Disagreement

**Disagreement 1 — Is ρ=0.347 "weak" (R1/R2/DA) or a "feasibility signal" (paper)?**
- R1/R2/DA: ρ=0.347 is *weak* (r²≈0.12) and p<10⁻¹¹ is N-driven.
- Paper position (§V.H): acknowledges below the ρ≈0.5 benchmark but still leads with "significant clinical correlation."
- **Disagreement type**: Severity disagreement.
- **Editor's Resolution**: Both are right — the correlation is *statistically significant* and *clinically weak*. The paper must hold both facts simultaneously: "significant but weak; below the ρ≈0.5 clinical benchmark; a feasibility signal, not a validated measure." Reserve "validated" for the future work.

**Disagreement 2 — Demote unevaluated components or evaluate them?**
- R3/EIC: at least minimal evaluation (WER, pain-correlation, guardrail FNR) preferred.
- R1: would accept honest demotion to "components, evaluation deferred."
- DA: demands they not be called contributions without evidence.
- **Disagreement type**: Direction disagreement.
- **Editor's Resolution**: Either path is acceptable; *not* acceptable is the status quo (full contribution claims, zero evidence). Given the 6-page ATC limit, **demotion + one targeted mini-evaluation** (the highest-value being a Vietnamese-language check per R3-W1, since it's the titled claim) is the most efficient path; the authors may substitute if they prefer.

**Disagreement 3 — Is the "order of magnitude" framing salvageable?**
- DA (M1): pure inflation; remove.
- R1/R2: not raised specifically.
- **Disagreement type**: Existence disagreement.
- **Editor's Resolution**: The *relative* finding (position-DTW correlates where angle-DTW does not) is sound and should be stated plainly; the "order of magnitude" multiplier of a null result must go. DA's framing prevails.

### DA-CRITICAL Handling

- **DA-C1** (protocol contamination): corroborated by R1-W1 (Score 5). **Required author response** even though the paper is honest overall — the authors must report a cross-validated, reference-independent ρ or the headline is unsound. Decision **cannot** be Accept — consistent with Major Revision (Checkpoint Rule #4).
- **DA-C2** (FPS contradiction): corroborated by R1-W3 + EIC-W3. **Required author response**: specify GPU/protocol, resolve Abstract-vs-Discussion attribution, report capture-to-feedback latency. Both criticals are tractable.

---

## Decision Rationale

All four reviewers and the Devil's Advocate converge on **Major Revision** rather than Reject — and the panel is explicit that this is not a quality ceiling but an honesty gap between the paper's headline framing and its evidence. Three findings force this decision. **First (DA-C1 / R1-W1, Critical):** the headline KIMORE correlation (ρ=0.347) is produced by a fixed-reference protocol whose templates are selected on the very clinical score being predicted; the paper does not state the templates are held out, so the central validation result is not shown to survive a fair, cross-validated protocol. **Second (DA-C2 / R1-W3 / EIC-W3, Critical):** the headline feasibility number (129.7 FPS) is internally inconsistent — attributed to "the complete pipeline … and coaching" in the Abstract but to "RTMW3D-L" alone in the Discussion, while coaching imposes a 5-second LLM cooldown — so the central feasibility claim is ill-defined. **Third (consensus, Major):** the title's "Elderly Vietnamese" promise and two of four claimed contributions (facial-state detection, LLM coaching) are unevaluated, and "validated"/"order of magnitude"/"robust guardrails" overclaim. These are fixable: the paper's own Limitations and cross-dataset honesty show the authors *can* report weak results and frame honestly — the revision is to extend that honesty to the headline claims and add a minimal, targeted evaluation. Reject is not warranted: the integration is non-trivial, the honesty is above venue norms, the cross-dataset metric-selection insight is a genuine finding, and no flaw is unfixable. Major Revision (not Minor) because resolving the protocol, FPS, and overclaim issues requires re-analysis and new measurements, not phrasing tweaks.

---

## Required Revisions (Must Fix)

| # | Revision Item | Source | Severity | Section | Estimated Effort |
|---|--------------|--------|----------|---------|-----------------|
| R1 | Re-run KIMORE scoring under a **leave-one-subject-out (or leave-one-recording-out)** protocol with references selected **independently of clinical score**; confirm templates excluded from scored set; report cross-validated ρ + 95% bootstrap CI. Compare to Capecci et al.'s KIMORE baselines. | R1-W1 / DA-C1 | Critical | §IV.C, Table I | 5–8 days |
| R2 | Specify **FPS measurement**: GPU model/VRAM, input resolution, batch size, warm-up, window; report mean±std + median; report **capture-to-feedback latency breakdown** (pose/face/scoring/TTS) separately from async coaching; resolve Abstract-"complete pipeline" vs Discussion-"RTMW3D-L" attribution. | R1-W3 / DA-C2 / EIC-W3 | Critical | Abstract, §IV.B, §V.A | 2–4 days |
| R3 | **Soften headline language**: "validated" → "benchmark-correlated/evaluated"; remove "order of magnitude" (replace with "angle-DTW shows no correlation; position-DTW shows weak significant correlation"); "robust guardrails" → "baseline keyword filter (future work: learned guardrails)." | R2-W3 / DA-M1,M4 / EIC | Critical | Abstract, Intro #5, §V.F, Conclusion | 1 day |
| R4 | **Direct the cross-dataset metric-selection insight (§V.D)** as the leading scientific finding; demote "first to combine A∧B∧C∧D∧E" conjunction-novelty to secondary. | DA-M5 / R2 | Major | Abstract, Contributions, Conclusion | 1 day |
| R5 | **Facial-state + LLM-coaching contributions**: either add a targeted mini-evaluation (PSPI-proxy vs UNBC-McMaster pain; RAG retrieval precision; guardrail FNR on a contraindication test set) **or** demote to "integrated components; evaluation deferred." | EIC-W2 / R3-W2,W3 / DA-M6 | Major | Contribution list, §III.D/E | 4–7 days |
| R6 | **Title/framing of "Elderly Vietnamese"**: rename to "Toward …" **or** add Vietnamese-language evidence (Whisper-v3 vn-WER on VIVOS/Common-Voice; intelligibility check on RAG KB + Edge-TTS). | EIC-W1 / R3-W1 / DA-M2 | Major | Title, Abstract, Intro #1 | 3–5 days |
| R7 | **Fix DTW band Eq. 7** (the `max(…, |n−m|)` term makes Sakoe-Chiba vacuous for unequal-length sequences); justify the 0.1 fraction. | R1-minor | Major | Eq. 7, §III.G | 0.5 day |
| R8 | **PSPI formula**: rename Eq. 5 "AU-based pain proxy" (not PSPI) or validate against UNBC-McMaster; explicitly document deviation from Prkachin–Solomon (dropped AU7/AU10, doubled AU6/AU43). **SPARC**: correct Eq. 4 to canonical form; justify or drop the 60/40 SPARC+LDLJ blend (LDLJ isn't duration-independent). | R2-W1, W2 | Major | Eq. 4, Eq. 5, §III.C/D | 2–3 days |
| R9 | **Statistical reporting**: add 95% CIs for all ρ/AUC; Holm-correct the 5 per-exercise tests; interpret effect-size magnitude vs the ρ≈0.5 benchmark; address subject clustering (per-subject ρ or mixed-effects). | R1-W4 / R2-W4 | Major | Tables I–III | 1–2 days |
| R10 | **Include the architecture figure** (`fig1_architecture*`) and at least the DTW-variant ablation as figures; remove orphan-figure references from `figures/README.md`. | EIC-W4 | Major | Figures | 0.5 day |

### Required Item Details

**R1: Cross-validated KIMORE protocol**
- **Problem**: Headline ρ=0.347 produced by selecting top-10-by-clinical-score recordings as templates; if not held out, distance-to-self inflates the result.
- **Source**: R1-W1 (Confidence 5); DA-C1; EIC Q2.
- **Requirement**: run LOSO (or leave-one-recording-out) with a reference-selection protocol independent of the clinical score (e.g., LOSO-mean template or generic reference); explicitly exclude templates from the scored set; report cross-validated ρ + 95% bootstrap CI; compare against Capecci et al.'s KIMORE regression baselines on the same split.
- **Acceptance criteria**: a cross-validated ρ with CI is reported; the paper states the reference set is disjoint from the scored set; the result is discussed relative to Capecci et al.

**R2: FPS / latency specification**
- **Problem**: 129.7 FPS attributed inconsistently (Abstract: full pipeline incl. coaching; Discussion: RTMW3D-L alone); coaching has a 5-second LLM cooldown incompatible with a 129.7 FPS *pipeline*.
- **Source**: R1-W3 (Confidence 5); EIC-W3; DA-C2.
- **Requirement**: state GPU model/VRAM, input resolution, batch size, warm-up, measurement window; report mean±std and median FPS; report a capture-to-feedback latency breakdown (pose, face, scoring, TTS) separately from async LLM coaching; resolve the Abstract-vs-Discussion attribution.
- **Acceptance criteria**: a single, consistent FPS/latency statement appears in both Abstract and Discussion; the Async-coaching caveat is explicit.

**R3: Honest headline language**
- **Problem**: "validated," "order of magnitude," "robust guardrails" overclaim relative to evidence.
- **Source**: R2-W3; DA-M1/M4; EIC.
- **Requirement**: replace "validated" with "benchmark-correlated/evaluated"; replace "order of magnitude" with the literal contrast; replace "robust guardrails" with "baseline keyword filter (future work: learned guardrails)."
- **Acceptance criteria**: no instance of the three overclaiming phrases in Abstract/Intro/Conclusion/§V.F without the qualifying language.

---

## Suggested Revisions (Should Fix)

| # | Revision Item | Source | Priority | Section | Expected Improvement |
|---|--------------|--------|----------|---------|---------------------|
| S1 | Add **rehab-scoring literature** (Capecci KIMORE baselines; Vakanski UI-PRMD DTW scoring; recent graph-CNN exercise-quality); compare your ρ to their reported numbers. | R2-W4 | P2 | §II, §V.D | Positions contribution vs field |
| S2 | Internal **quaternion-vs-dot-product ablation** to evidence the multi-plane advantage (or soften to "adopted per Aurand et al."). | R2-W5 | P2 | §III.B, §V.C | Evidences a borrowed claim |
| S3 | Reframe **UCO AUC=0.974** as a sanity-check baseline; headline the within-exercise (clinical) correlation instead; report UCO CIs. | R1-W5 / DA-M3 | P2 | §IV.D, §V.D | Restores honesty of UCO framing |
| S4 | **Safety-interlock fail-safe**: state that pain-pause defaults to conservative pause when the (unvalidated) detector is uncertain; explicitly mark the safety mechanism "prototype, not clinically validated." | R3-W2 | P2 | §V.G | De-risks the highest-stakes overclaim |
| S5 | **Deployment-feasibility paragraph**: quantify GPU/API cost/availability for the target low-/middle-income rural setting; elevate edge-LLM beyond a one-line future-work item. | R3-W5 / DA alt-path | P2 | §V.F, §V.H, Conclusion | Grounds the population framing |
| S6 | Clarify the **0.18 mm "temporal stability"** metric (RMS vs signed; entity measured = mapping framework on Kinect data, not RTMW3D on RGB); retitle the subsection accordingly. | R1-W2 | P2 | §IV.B | Removes construct misalignment |
| S7 | Even a **small stakeholder consultation** (3–5 physios on scoring dimensions; 2–3 elderly think-alouds) to convert heuristic choices to evidence. | R3-W4 | P2/P3 | §III.F, §V | Converts assertions to evidence |
| S8 | **Reproducibility appendix**: RTMW3D-L checkpoint id, OpenFace commit, LLM model/vendor, seeds, filter/butterworth cutoff justification. | R1-reproducibility | P3 | §IV or appendix | Enables replication |

---

## Revision Roadmap

### Priority 1 — Structural / Evidence (Estimated total effort: ~16–26 days)
- [ ] R1: Cross-validated, reference-independent KIMORE ρ + bootstrap CI + Capecci baseline comparison.
- [ ] R2: FPS/latency specification; resolve Abstract-vs-Discussion contradiction.
- [ ] R3: Soften "validated"/"order of magnitude"/"robust" across Abstract, Intro, §V.F, Conclusion.
- [ ] R6: Title change ("Toward …") **or** add Vietnamese-language evidence.
- [ ] R5: Evaluate or demote facial-state + LLM-coaching contributions.
- [ ] R8: Rename/fix PSPI (Eq. 5) and SPARC (Eq. 4); justify 60/40 blend.
- [ ] R10: Include architecture figure.

### Priority 2 — Domain depth (Estimated total effort: ~6–10 days)
- [ ] R7: Fix DTW band Eq. 7.
- [ ] R9: CIs + Holm correction + effect-size interpretation + clustering.
- [ ] R4: Re-lead with cross-dataset metric-selection insight.
- [ ] S1: Add rehab-scoring literature + baseline comparison.
- [ ] S2: Quaternion-vs-dot-product ablation.
- [ ] S3: Reframe UCO AUC.

### Priority 3 — Honesty / deployment / reproducibility (Estimated total effort: ~3–5 days)
- [ ] S4: Safety-interlock fail-safe statement.
- [ ] S5: Deployment-feasibility paragraph.
- [ ] S6: Clarify/retitle 0.18 mm metric.
- [ ] S7: Stakeholder consultation (optional but high-value).
- [ ] S8: Reproducibility appendix.

### Total Estimated Effort
- **Major Revision: ~4–6 weeks** (the gating item is R1 — re-running KIMORE cross-validated and locating/comparing Capecci baselines).

---

## Revision Deadline

- **Recommended deadline**: 6 weeks from decision.
- **Basis**: Major Revision (6–8 weeks guidance); tightened to 6 weeks because most items are specification/framing work and only R1 requires fresh experiments.
- **Extension policy**: notify the chairs 1 week before the deadline if an extension is needed.

---

## Response Letter Instructions

Please use the R→A→C format (Recall → Address → Change) to respond to every reviewer comment item by item.

**Must include**:
1. Response and revision description for each Required Revision (R1–R10).
2. Response for each Suggested Revision (S1–S8) — adopted, or reason for not adopting.
3. Change markup (mark all changes in the revised manuscript with color or track changes).
4. Cross-reference table of new section/line/equation numbers.

For R1 and R2 specifically, include the *new* cross-validated table and the *new* FPS/latency table as evidence of the fix.

---

## Closing

We encourage you to carefully consider the reviewers' comments and submit a substantially revised manuscript. The panel found the integration genuinely valuable and the honesty of the Limitations section above venue norms; the required revisions focus on extending that honesty to the headline claims and specifying the two central numbers (ρ, FPS) under fair, reproducible protocols. Please note that the revised manuscript will undergo another round of review.

---

## Appendix: Reviewer Report Summary

### EIC Report Summary
- Recommendation: Major Revision | Confidence: 4
- Key Point: Title/abstract overclaim; FPS attribution contradictory; missing architecture figure.

### Reviewer 1 (Methodology) Summary
- Recommendation: Major Revision | Confidence: 5
- Key Point: KIMORE protocol contamination (critical); 0.18 mm measures mapping not pose; FPS unspecified.

### Reviewer 2 (Domain) Summary
- Recommendation: Major Revision | Confidence: 5
- Key Point: PSPI/SPARC fidelity; "validated" overclaim; missing rehab-scoring baseline comparison.

### Reviewer 3 (Perspective) Summary
- Recommendation: Major Revision | Confidence: 4
- Key Point: Titled population unevaluated; safety interlock rests on unvalidated detector; keyword guardrails weak.

### Devil's Advocate Summary
- Findings: 2 CRITICAL (C1 protocol contamination; C2 FPS contradiction), 7 MAJOR, 4 MINOR.
- Key Point: Three inflations (ρ-protocol, FPS-attribution, conjunction-novelty + title) deflate the headline claims without denying a single number.

---

*This review package was produced read-only — no file under `paper/` was modified.*
