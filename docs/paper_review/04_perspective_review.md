# Peer Review Report — Perspective (Reviewer 3)

## Manuscript Information

- **Title**: ADAPT-Rehab: A Multimodal AI System for Real-Time Elderly Rehabilitation with Whole-Body 3D Pose Estimation and Voice-Interactive LLM Coaching
- **Review Date**: 2026-06-30
- **Review Round**: Round 1

## Reviewer Information

### Reviewer Role
Peer Reviewer 3 (Cross-disciplinary / Practical)

### Reviewer Identity
HCI / accessibility researcher in voice interfaces for low-literacy and aging populations; familiar with Vietnamese-language ASR/TTS and rural-deployment realities.

### Review Focus
Whether the titled vulnerable population is matched by any evidence involving that population, whether the safety architecture rests on validated mechanisms, whether stakeholder voices are present, and deployment feasibility in the target low-/middle-income setting.

---

## Overall Assessment

### Recommendation
- [x] **Major Revision** — Substantial revisions needed, re-review required after revision

### Confidence Score
4 — Mostly within my area of expertise, high confidence.

### Summary Assessment
As a cross-disciplinary reviewer from HCI/accessibility, I find the *intent* of this paper valuable — a voice-first, elder-respectful rehabilitation tool for a low-resource language is a genuinely underserved direction. But the paper's strongest population-facing claims rest on no evidence from that population: there are no Vietnamese-language ASR/TTS results, no elderly users, no low-connectivity deployment study, and no participatory input from physiotherapists or caregivers. The "safety architecture" rests on a pain detector (PSPI-inspired, Eq. 5) and keyword-based LLM guardrails that are themselves unevaluated — so a load-bearing safety claim ("automatic exercise pause on pain detection," §V.G) inherits the unreliability of an unvalidated detector. The implicit assumption that "integration + real-time FPS = a usable elderly system" is an HCI non-sequitur. These are fixable by honest reframing + at least one mini evaluation, but as written the paper promises more to vulnerable users than it has shown.

---

## Strengths

### S1: Genuine accessibility lens
Contribution #1 (elderly-oriented, voice-first, respectful address forms, wait-for-user synchronization) shows real HCI awareness — "Wait-for-User synchronization allows unlimited time at each checkpoint" (§V.E) is precisely the kind of accommodation aging users need.

### S2: Safety-first posture
The design "defaults to caution when uncertainty is detected" (§V.F) and pauses on pain — the *stance* is correct even though the detector is unvalidated (W2 below).

### S3: Graceful degradation is an accessibility feature
§V.A notes LLM coaching can be disabled on lower-end hardware while retaining core pose+scoring — this matters for the rural-Vietnam deployment context the title implies.

### S4: Privacy framing
"Skeletal-only analysis option (no raw video storage)" (Ethics Statement) is a thoughtful privacy choice that matters for elder/data-protection contexts.

---

## Weaknesses

### W1: The titled population is studied by proxy only
**Problem**: No Vietnamese-language evaluation of Whisper-v3 / Edge-TTS / the LLM (no WER, no intelligibility, no Vietnamese-medical-term retrieval), and no elderly participants (acknowledged in §V.H).
**Why it matters**: the title is a promise to a specific, vulnerable population; delivering an English-corpus, adult-demonstrator system under that title risks tokenizing the population for novelty.
**Suggestion**: at minimum add (a) a Whisper-v3-vn WER on a public Vietnamese corpus, (b) a small qualitative check that the RAG KB and Edge-TTS voice handle Vietnamese rehab instructions intelligibly, and (c) soften the title to "…Toward Elderly Vietnamese…".
**Severity**: Major

### W2: Safety architecture rests on unvalidated detectors
**Problem**: §V.G calls pain-triggered pause an "unsupervised safety mechanism essential for home-based rehabilitation." But the pain detector (Eq. 5) is an unvalidated AU-proxy, and the facial-state pipeline has no quantitative evaluation.
**Why it matters**: deploying an unvalidated pain detector as the *safety interlock* for unsupervised home use is the single highest-risk overclaim in the paper — a false negative means a user in pain is told to continue.
**Suggestion**: explicitly label the pain-pause as a *fail-safe prototype not yet validated for unsupervised use*; add a fail-safe default ("if pain detector uncertain → conservative pause") and evaluate the detector's sensitivity (false-negative rate) on a pain corpus (UNBC-McMaster) before any deployment claim.
**Severity**: Major (ethical/safety)

### W3: LLM "guardrails" are keyword-based and easily bypassed
**Problem**: §V.F describes "keyword-based guardrails" blocking phrases like "push through." Paraphrase defeats keyword filters trivially, and LLM safety research has moved well beyond blocklists.
**Why it matters**: calling this a "robust" safety mechanism overclaims reliability for a health-coaching LLM.
**Suggestion**: either (a) evaluate the guardrail's false-negative rate on a contraindication/advice test set, or (b) downgrade "robust guardrails" to "baseline keyword filter; learned-classifier guardrails are future work."
**Severity**: Major

### W4: Stakeholder voices missing
**Problem**: No input from physiotherapists (to validate the six scoring dimensions/weights), no elderly-or-caregiver requirements elicitation, no report of a single interaction with the target user.
**Why it matters**: the scoring *weights* are "clinically-inspired" with no clinician input shown, and the UX is asserted rather than elicited.
**Suggestion**: even a small structured consultation (3–5 physios rating the dimensions; 2–3 elderly think-alouds) would convert several heuristic choices into evidence-backed ones and would materially strengthen every contribution.
**Severity**: Major

### W5: Deployment feasibility is asserted, not stress-tested
**Problem**: The system needs a consumer GPU *and* internet for the LLM API in a low-/middle-income rural-Vietnam setting (you cite the rural-elderly access problem in the Intro).
**Why it matters**: the same population framing that motivates the paper makes the GPU+connectivity stack implausible at scale.
**Suggestion**: add a deployment-feasibility paragraph quantifying the cost/availability of the required GPU and API access in the target setting; revisit edge-LLM as more than a one-line future-work item.
**Severity**: Major

---

## Detailed Comments

### Assumption Audit
- **Explicit assumptions**: "integration is computationally practical and architecturally sound" (Intro, ¶3). Reasonable.
- **Implicit assumptions**: "if it runs at 129.7 FPS, it is usable by elderly Vietnamese users." This does not follow — usability is not throughput. (HCI lens.)
- **Implicit assumptions**: "an AU-proxy pain detector is safe enough to act as an unsupervised pause interlock." Unsupported (W2).
- **Paradigmatic assumptions**: the paper treats feasibility+benchmark-correlation as sufficient for a population-facing system; an HCI paradigm would require evidence *with* that population before the title-level claim.

### Cross-Disciplinary Connections
- **Parallel research**: The LLM-safety literature (Constitutional AI, red-teaming, refusal-classifiers) is more mature than keyword blocklists and could strengthen W3 cheaply.
- **Borrowing opportunities**: Vietnamese ASR evaluation has public benchmarks (e.g., VIVOS, Common Voice vi) — Whisper-v3 vn-WER is a small, high-value addition.
- **Methodological borrowing**: Accessibility UX for aging users (e.g., Czaja/Lee's design-for-aging principles) would let you ground Contribution #1 in a framework rather than assertion.

### Practical Impact
- **Real-world application**: High *if* the safety detector is validated and the language stack is shown on Vietnamese. Currently the impact is aspirational.
- **Implementation feasibility**: GPU + API + rural Vietnam = low (W5).
- **Stakeholders**: Missing — physios, elderly users, caregivers, rural-clinic implementers (W4).

### Broader Implications
- **Ethical dimensions**: deploying an unvalidated pain detector as a safety interlock for unsupervised home use by a vulnerable population (W2) — the highest-impact ethics issue in the paper.
- **Social impact**: positive *if* honest; risk of eroding trust in AI-rehab if the safety claim fails in the field.
- **Future directions**: edge-LLM, learned guardrails, participatory design with Vietnamese elderly and physios.

---

## Cross-Disciplinary Reading Recommendations

1. **Czaja, S. J. et al. (2019), "Designing for Older Adults"** — anchors the elderly-UX claims in a framework.
2. **OpenAI / Anthropic safety papers on refusal-trained classifiers** — supersedes keyword guardrails (W3).
3. **VIVOS / Common Voice Vietnamese** — to report Whisper-v3 vn-WER (W1).
4. **Capecci et al. KIMORE companion + Vakanski et al.** — rehab-scoring grounding (overlaps R2's W4).

---

## Questions for Authors

1. What is the *youngest* piece of Vietnamese-language evidence you could add in revision, and would you consider a title change until it exists?
2. What is the false-negative rate of your pain detector on a labeled pain corpus, and is the pause interlock fail-safe when the detector is uncertain?
3. Have any physiotherapists seen the six scoring dimensions/weights?
4. Is the keyword guardrail's robustness quantified against paraphrased harmful advice?

---

## Minor Issues

- "neural voice" (§III.A item 5) — Edge-TTS is cloud; framing it as a local "neural voice" for rural use understates connectivity dependence.

---

## Dimension Scores

| Dimension | Score (0-100) | Descriptor | Notes |
|-----------|--------------|------------|-------|
| Originality (20%) | 60 | Adequate | Population framing valuable but unevidenced |
| Methodological Rigor (25%) | 50 | Adequate | (delegated to R1) |
| Evidence Sufficiency (25%) | 42 | Weak | No VN/elderly evidence; safety on unvalidated detector |
| Argument Coherence (15%) | 58 | Adequate | FPS→usability non-sequitur |
| Writing Quality (15%) | 80 | Strong | Clear, humane prose |
| Significance & Impact | 55 | Adequate | High if safety + language validated |
| **Weighted Average** | ~52 | **Major Revision** | |
