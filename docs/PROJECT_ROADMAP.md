# ADAPT-Rehab: Project Roadmap

> Phased development plan for turning ADAPT-Rehab into a research conference paper.
> Each phase has clear deliverables, dependencies, and estimated effort.

---

## Overview

```
Phase 0: Foundation & Setup          [Week 1]
Phase 1: 3D Pose Estimation          [Week 2-3]
Phase 2: Advanced Kinematics         [Week 3-4]
Phase 3: Face Analysis (Pain/Emotion)[Week 4-5]
Phase 4: Scoring & Compensation      [Week 5-6]
Phase 5: LLM Coaching & Voice        [Week 6-7]
Phase 6: Integration & Testing       [Week 7-8]
Phase 7: Benchmark Evaluation        [Week 8-10]
Phase 8: Clinical Validation         [Week 10-12]
Phase 9: Paper Writing               [Week 12-14]
```

```
┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐
│ Phase 0 │──▶│ Phase 1 │──▶│ Phase 2 │──▶│ Phase 3 │
│ Setup   │   │ 3D Pose │   │ Kinemat │   │  Face   │
└─────────┘   └─────────┘   └─────────┘   └─────────┘
                  │               │               │
                  ▼               ▼               ▼
              ┌─────────┐   ┌─────────┐   ┌─────────┐
              │ Phase 4 │──▶│ Phase 5 │──▶│ Phase 6 │
              │ Scoring │   │ LLM+Voice│  │Integrate│
              └─────────┘   └─────────┘   └─────────┘
                                                │
                  ┌─────────────────────────────┘
                  ▼
              ┌─────────┐   ┌─────────┐   ┌─────────┐
              │ Phase 7 │──▶│ Phase 8 │──▶│ Phase 9 │
              │Benchmark│   │ Clinical│   │  Paper  │
              └─────────┘   └─────────┘   └─────────┘
```

---

## Phase 0: Foundation & Setup

**Goal**: Get the development environment working and understand the codebase.

### Tasks

| # | Task | Deliverable | Status |
|---|------|-------------|--------|
| 0.1 | Set up Python environment | `requirements.txt` installed, tests pass | ☐ |
| 0.2 | Download MediaPipe models | `models/pose_landmarker_lite.task`, `models/face_landmarker.task` | ☐ |
| 0.3 | Run existing v2 code | `python main_v2.py --mode test` works | ☐ |
| 0.4 | Review all existing code | Understand core/, modules/, utils/ | ☐ |
| 0.5 | Set up Git workflow | Branch strategy, commit messages | ☐ |

### Deliverable
- Working v2 system running on webcam
- All existing tests passing
- Understanding of current architecture

### Dependencies
- None (starting point)

---

## Phase 1: 3D Pose Estimation

**Goal**: Replace MediaPipe with direct 3D pose estimation for ~50% accuracy improvement.

### Tasks

| # | Task | Deliverable | Status |
|---|------|-------------|--------|
| 1.1 | Implement `core/pose3d/base.py` | Abstract interface + factory | ✅ Done |
| 1.2 | Implement `core/pose3d/metrab.py` | MeTRAbs integration | ☐ |
| 1.3 | Implement `core/pose3d/mediapipe_fallback.py` | CPU fallback | ☐ |
| 1.4 | Download MeTRAbs model | `models/metrib_384.bin` | ☐ |
| 1.5 | Test MeTRAbs on sample video | Verify 3D keypoints output | ☐ |
| 1.6 | Compare MeTRAbs vs MediaPipe | Joint angle MAE comparison | ☐ |
| 1.7 | Implement TensorRT optimization | 30+ FPS on consumer GPU | ☐ |
| 1.8 | Update `main_v3.py` pose pipeline | Use new 3D estimator | ☐ |

### Technical Details

**MeTRAbs Setup**:
```bash
pip install tensorflow tensorflow-hub
# Model auto-downloads from TF Hub on first use
```

**Expected Results**:
- MPJPE: 63mm (MediaPipe) → 25-35mm (MeTRAbs)
- Joint angle MAE: 10-15° → 3-7°
- FPS: 300+ (MediaPipe) → 25-30 (MeTRAbs)

### Deliverable
- `core/pose3d/` module working
- MeTRAbs producing metric-scale 3D keypoints
- Comparison table: MediaPipe vs MeTRAbs accuracy

### Dependencies
- Phase 0 complete

---

## Phase 2: Advanced Kinematics

**Goal**: Upgrade angle computation and add clinically validated smoothness metrics.

### Tasks

| # | Task | Deliverable | Status |
|---|------|-------------|--------|
| 2.1 | Implement `core/kinematics_quaternion.py` | Quaternion angle computation | ✅ Done |
| 2.2 | Implement `core/smoothness.py` | SPARC + LDLJ metrics | ✅ Done |
| 2.3 | Validate quaternion vs dot-product | Compare on multi-plane movements | ☐ |
| 2.4 | Validate SPARC on different speeds | Prove duration-independence | ☐ |
| 2.5 | Integrate into pose3d pipeline | Auto-compute both angle types | ☐ |
| 2.6 | Add constrained DTW | Sakoe-Chiba band constraint | ☐ |

### Technical Details

**Quaternion Advantage**:
```python
# Dot product: fails at 90° (gimbal lock)
# Quaternion: no gimbal lock, works at any angle

# Test: shoulder abduction at 90°
# Dot product: ~85-95° (inaccurate)
# Quaternion: ~90° (accurate)
```

**SPARC Advantage**:
```
Patient A: slow exercise (20 sec)  → SPARC = -0.8
Patient B: fast exercise (10 sec)  → SPARC = -0.9
→ Both are equally smooth (SPARC is duration-independent)

# Jerk would falsely show Patient A as smoother
```

### Deliverable
- Quaternion angles validated against goniometer
- SPARC metric validated on varying speeds
- Updated `core/pose3d/base.py` computing both angle types

### Dependencies
- Phase 1 complete (needs 3D keypoints)

---

## Phase 3: Face Analysis (Pain/Emotion)

**Goal**: Replace rule-based FACS with deep learning pain/emotion detection.

### Tasks

| # | Task | Deliverable | Status |
|---|------|-------------|--------|
| 3.1 | Implement `modules/perception/face_detector.py` | MediaPipe Face Mesh wrapper | ✅ Done |
| 3.2 | Implement `modules/perception/au_detector.py` | py-feat JAANet integration | ✅ Done |
| 3.3 | Implement `modules/perception/emotion_classifier.py` | MobileNetV3 emotion | ✅ Done |
| 3.4 | Install py-feat and test | Verify AU detection works | ☐ |
| 3.5 | Test emotion classifier on FER2013 | Benchmark accuracy | ☐ |
| 3.6 | Compare new vs old pain detection | AU-based vs rule-based | ☐ |
| 3.7 | Fine-tune on elderly faces (if data available) | Domain adaptation | ☐ |

### Architecture

```
Face Frame (224x224)
  │
  ├─→ py-feat JAANet → AU4, AU6, AU7, AU9, AU10, AU43 → PSPI score
  │
  └─→ MobileNetV3-Large → 7 emotions (softmax)
  
Combined: pain_level + emotion_state
```

### PSPI Formula
```
PSPI = AU4 + max(AU6, AU7) + max(AU9, AU10) + max(AU20, AU25, AU26)

Pain Level:
  0-4:   NONE
  5-8:   MILD
  9-12:  MODERATE
  13-16: SEVERE
```

### Deliverable
- AU detection working via py-feat
- Emotion classification working
- PSPI pain score computed
- Comparison: new vs old pain detection accuracy

### Dependencies
- Phase 0 complete

---

## Phase 4: Scoring & Compensation

**Goal**: Enhanced 6-dimension scoring with temporal compensation detection.

### Tasks

| # | Task | Deliverable | Status |
|---|------|-------------|--------|
| 4.1 | Implement `modules/compensation.py` | Temporal compensation detector | ✅ Done |
| 4.2 | Implement `modules/fatigue.py` | Multi-indicator fatigue | ✅ Done |
| 4.3 | Implement `modules/scoring_v2.py` | 6-dimension scorer | ✅ Done |
| 4.4 | Test compensation detection | Verify shoulder hiking, trunk lean detection | ☐ |
| 4.5 | Test fatigue detection | Verify jerk + ROM + velocity indicators | ☐ |
| 4.6 | Compare v1 vs v2 scoring | Show improvement from new metrics | ☐ |
| 4.7 | Tune scoring weights | Optimize via ablation study | ☐ |

### Scoring Dimensions (v2)

| Dimension | Weight | Metric | v1 Method | v2 Method |
|-----------|--------|--------|-----------|-----------|
| ROM | 25% | Angle accuracy | Max angle / target | Same |
| Stability | 15% | Angle variability | Std in HOLD phase | Same |
| Flow | 20% | Motion smoothness | DTW similarity | Same |
| Symmetry | 15% | Left-right balance | Angle difference | Same |
| Compensation | 15% | Compensatory movements | Frame-by-frame thresholds | Temporal LSTM |
| **Smoothness** | **10%** | Movement quality | Not measured | **SPARC metric** |

### Deliverable
- Compensation detection working
- Fatigue detection working
- 6-dimension scoring integrated
- Ablation study showing each dimension's contribution

### Dependencies
- Phase 2 complete (needs SPARC)
- Phase 3 complete (needs pain detection for fatigue)

---

## Phase 5: LLM Coaching & Voice

**Goal**: Add voice-interactive LLM coaching for personalized exercise guidance.

### Tasks

| # | Task | Deliverable | Status |
|---|------|-------------|--------|
| 5.1 | Implement `modules/intelligence/llm/client.py` | OpenAI/Anthropic API client | ✅ Done |
| 5.2 | Implement `modules/intelligence/llm/prompts.py` | Vietnamese prompt templates | ✅ Done |
| 5.3 | Implement `modules/intelligence/llm/safety.py` | Safety guardrails | ✅ Done |
| 5.4 | Implement `modules/intelligence/voice/tts.py` | Edge-TTS Vietnamese | ✅ Done |
| 5.5 | Implement `modules/intelligence/voice/asr.py` | Whisper ASR | ✅ Done |
| 5.6 | Implement `modules/intelligence/coach/rehab_coach.py` | Coaching orchestrator | ✅ Done |
| 5.7 | Set up OpenAI/Claude API key | API access working | ☐ |
| 5.8 | Test LLM feedback quality | Vietnamese output review | ☐ |
| 5.9 | Test safety guardrails | Verify harmful advice blocked | ☐ |
| 5.10 | Test TTS output | Vietnamese voice quality | ☐ |
| 5.11 | Build RAG knowledge base | Clinical guidelines in vector store | ☐ |

### Architecture

```
Exercise State
  │
  ├─→ Prompt Builder (Vietnamese templates)
  │     ├─ Exercise name, phase, angles
  │     ├─ ROM/stability scores
  │     └─ Pain/fatigue levels
  │
  ├─→ Safety Guardrails
  │     ├─ Block harmful advice
  │     └─ Check contraindications
  │
  ├─→ LLM API (GPT-4o / Claude)
  │     └─ System prompt: Vietnamese rehab coach
  │
  └─→ Edge-TTS (Vietnamese voice)
        └─ Audio output
```

### Safety Rules
```
BLOCKED:
- "ignore pain", "push through"
- "take medication", "diagnosis"
- "you need surgery"

ELDERLY RISKY:
- "heavy lifting", "high impact"
- "jumping", "deep squat"

PAIN OVERRIDE:
- If pain_level = MODERATE/SEVERE
  → Block "continue", "keep going"
  → Force rest message
```

### Deliverable
- LLM coaching working via API
- Vietnamese voice output working
- Safety guardrails verified
- RAG knowledge base built

### Dependencies
- Phase 0 complete (for API setup)
- Phase 3-4 complete (needs pain/fatigue data for prompts)

---

## Phase 6: Integration & Testing

**Goal**: Wire all components together into a working v3 system.

### Tasks

| # | Task | Deliverable | Status |
|---|------|-------------|--------|
| 6.1 | Update `main_v3.py` | Full pipeline integration | ☐ |
| 6.2 | Wire pose3d → kinematics → scoring | End-to-end pose pipeline | ☐ |
| 6.3 | Wire face → AU → emotion → pain | End-to-end face pipeline | ☐ |
| 6.4 | Wire scoring → LLM → TTS | End-to-end feedback pipeline | ☐ |
| 6.5 | Test full system on webcam | Real-time demo working | ☐ |
| 6.6 | Fix integration bugs | All components working together | ☐ |
| 6.7 | Performance profiling | FPS per component | ☐ |
| 6.8 | Write unit tests for integration | `tests/test_integration.py` | ☐ |

### Integration Flow

```
Webcam Frame (30 FPS)
  │
  ├─→ [Perception]
  │     ├─ MeTRAbs → 3D keypoints → joint angles
  │     ├─ Face Mesh → 468 landmarks
  │     └─ py-feat → AU activations → PSPI
  │
  ├─→ [Analysis]
  │     ├─ Quaternion angles
  │     ├─ SPARC smoothness
  │     ├─ Compensation detection
  │     ├─ Fatigue detection
  │     └─ 6-dim scoring
  │
  ├─→ [Intelligence]
  │     ├─ LLM feedback generation
  │     ├─ Safety check
  │     └─ TTS voice output
  │
  └─→ [Output]
        ├─ Visual: skeleton overlay + scores
        └─ Audio: Vietnamese voice feedback
```

### Deliverable
- Working v3 system on webcam
- All components integrated
- Performance acceptable (15+ FPS end-to-end)

### Dependencies
- Phases 1-5 all complete

---

## Phase 7: Benchmark Evaluation

**Goal**: Prove system accuracy on public datasets and ablation studies.

### Tasks

| # | Task | Deliverable | Status |
|---|------|-------------|--------|
| 7.1 | Download UI-PRMD dataset | 10 rehab exercises, Kinect ground truth | ☐ |
| 7.2 | Implement `evaluation/benchmarks/uiprmd.py` | Dataset loader | ☐ |
| 7.3 | Run pose estimation on UI-PRMD | MPJPE computation | ☐ |
| 7.4 | Run joint angle evaluation | MAE vs Kinect ground truth | ☐ |
| 7.5 | Compare vs MediaPipe baseline | Accuracy improvement table | ☐ |
| 7.6 | Compare vs published methods | Literature comparison | ☐ |
| 7.7 | Run ablation study | Each component's contribution | ☐ |
| 7.8 | FPS benchmark | Different hardware configs | ☐ |

### Evaluation Metrics

| Metric | Formula | Target |
|--------|---------|--------|
| MPJPE | Mean Per Joint Position Error | <35mm |
| P-MPJPE | Procrustes-aligned MPJPE | <28mm |
| Angle MAE | Joint angle error (degrees) | <7° |
| ICC | Intraclass Correlation Coefficient | >0.85 |
| FPS | Frames per second | >25 |

### Ablation Configurations

| Config | pose3d | quaternion | sparc | compensation | llm |
|--------|--------|-----------|-------|-------------|-----|
| Full | ✓ | ✓ | ✓ | ✓ | ✓ |
| No 3D pose | ✗ | ✓ | ✓ | ✓ | ✓ |
| No quaternion | ✓ | ✗ | ✓ | ✓ | ✓ |
| No SPARC | ✓ | ✓ | ✗ | ✓ | ✓ |
| No compensation | ✓ | ✓ | ✓ | ✗ | ✓ |
| No LLM | ✓ | ✓ | ✓ | ✓ | ✗ |

### Deliverable
- UI-PRMD benchmark results
- Comparison table vs baselines
- Ablation study results
- FPS benchmark on 3 hardware configs

### Dependencies
- Phase 6 complete (needs working system)

---

## Phase 8: Clinical Validation

**Goal**: Validate system with real elderly Vietnamese users.

### Tasks

| # | Task | Deliverable | Status |
|---|------|-------------|--------|
| 8.1 | Write ethics protocol | IRB submission document | ☐ |
| 8.2 | Recruit participants | 15-30 elderly Vietnamese adults | ☐ |
| 8.3 | Design study protocol | Pre-post design, 4-week intervention | ☐ |
| 8.4 | Collect baseline data | Pre-intervention ROM, pain, SUS | ☐ |
| 8.5 | Run intervention | 4-week guided exercise with system | ☐ |
| 8.6 | Collect post data | Post-intervention ROM, pain, SUS | ☐ |
| 8.7 | Statistical analysis | Paired t-test, effect size | ☐ |
| 8.8 | Write results | Tables, figures, narrative | ☐ |

### Study Design

```
Week 0: Baseline assessment
  ├─ ROM measurement (goniometer)
  ├─ Pain assessment (NPRS)
  ├─ System Usability Scale (SUS)
  └─ Demographics

Week 1-4: Intervention
  ├─ 3 sessions/week with ADAPT-Rehab
  ├─ Each session: 15-20 minutes
  ├─ Exercises: arm raise, bicep curl, squat
  └─ System logs: angles, scores, feedback

Week 4: Post-assessment
  ├─ ROM measurement (goniometer)
  ├─ Pain assessment (NPRS)
  ├─ SUS questionnaire
  └─ Qualitative feedback
```

### Metrics

| Category | Metric | Tool |
|----------|--------|------|
| Usability | System Usability Scale | SUS questionnaire |
| Engagement | Session completion rate | System logs |
| Clinical | ROM improvement | Goniometer |
| Clinical | Pain level change | NPRS |
| Safety | Adverse events | Incident log |

### Deliverable
- Ethics approval
- 15-30 participants enrolled
- Pre-post data collected
- Statistical analysis complete

### Dependencies
- Phase 7 complete (benchmarks first)
- Ethics board approval

---

## Phase 9: Paper Writing

**Goal**: Write and submit the research paper.

### Tasks

| # | Task | Deliverable | Status |
|---|------|-------------|--------|
| 9.1 | Choose target venue | AAAI, IJCAI, CHI, EMBC, etc. | ☐ |
| 9.2 | Write Abstract | 150-250 words | ☐ |
| 9.3 | Write Introduction | Problem, motivation, contributions | ☐ |
| 9.4 | Write Related Work | Literature review | ☐ |
| 9.5 | Write Methodology | System architecture, algorithms | ☐ |
| 9.6 | Write Experiments | Benchmarks, ablation, clinical | ☐ |
| 9.7 | Write Discussion | Interpretation, limitations | ☐ |
| 9.8 | Create Figures | Architecture diagram, results plots | ☐ |
| 9.9 | Create Tables | Comparison tables, ablation tables | ☐ |
| 9.10 | Internal review | Co-author feedback | ☐ |
| 9.11 | Submit | Before deadline | ☐ |

### Paper Structure

```
1. Abstract (250 words)
2. Introduction (1.5 pages)
   - Problem: elderly rehab access in Vietnam
   - Motivation: CV + LLM can help
   - Contributions: 3-5 bullet points
3. Related Work (1.5 pages)
   - CV-based rehabilitation
   - 3D pose estimation
   - Pain/emotion detection
   - LLM in healthcare
4. Methodology (3 pages)
   - System architecture (Figure 1)
   - 3D pose estimation (MeTRAbs)
   - Quaternion kinematics
   - SPARC smoothness
   - Multi-task pain/emotion
   - LLM coaching
5. Experiments (3 pages)
   - Benchmarks (UI-PRMD)
   - Ablation study
   - Clinical validation
   - Comparison vs baselines
6. Discussion (1 page)
   - Key findings
   - Limitations
   - Future work
7. References
```

### Key Claims to Prove

| Claim | Evidence Needed |
|-------|-----------------|
| 50% better 3D accuracy | MPJPE comparison on UI-PRMD |
| SPARC is duration-independent | Varying speed experiment |
| Multi-task improves detection | Ablation: single vs multi-task |
| LLM coaching improves engagement | SUS + session completion |
| System is usable for elderly | SUS > 68 (above average) |

### Deliverable
- Complete paper draft
- All figures and tables
- Submitted before deadline

### Dependencies
- Phases 7-8 complete (needs results)

---

## Quick Reference: What to Do When

### "I have 1 hour"
- Run existing tests: `pytest tests/ -v`
- Read `core/pose3d/base.py` to understand the interface
- Review `docs/RESEARCH_STRATEGY.md` for context

### "I have 1 day"
- Complete Phase 0 (setup)
- Start Phase 1 (download MeTRAbs, test on sample)

### "I have 1 week"
- Complete Phases 0-2 (setup + 3D pose + kinematics)
- Start Phase 3 (face analysis)

### "I have 1 month"
- Complete Phases 0-6 (full working system)
- Start Phase 7 (benchmarks)

### "I have 3 months"
- Complete all phases
- Paper submitted

---

## Risk Mitigation

| Risk | Impact | Mitigation |
|------|--------|------------|
| MeTRAbs too slow | Can't do real-time | Use MediaPipe fallback + TensorRT |
| No elderly face dataset | Can't train emotion model | Use AU-based approach (no training needed) |
| LLM API too expensive | Can't run many experiments | Cache responses, use smaller models |
| Can't recruit elderly | No clinical validation | Use proxy validation (goniometer comparison) |
| Paper rejected | Lost time | Target multiple venues, have backup plan |

---

## File Reference

| File | Purpose |
|------|---------|
| `.claude/CLAUDE.md` | Project context for Claude |
| `docs/RESEARCH_STRATEGY.md` | Literature review + strategies |
| `docs/PROJECT_ROADMAP.md` | This file — phased plan |
| `core/pose3d/` | 3D pose estimation backends |
| `core/kinematics_quaternion.py` | Quaternion angles |
| `core/smoothness.py` | SPARC metric |
| `modules/perception/` | Face analysis (AU, emotion) |
| `modules/intelligence/` | LLM + voice coaching |
| `modules/scoring_v2.py` | Enhanced 6-dim scoring |
| `modules/compensation.py` | Temporal compensation |
| `modules/fatigue.py` | Multi-indicator fatigue |
| `evaluation/` | Benchmarks + metrics |
| `tests/` | Unit tests |
| `main_v3.py` | v3 main application |
