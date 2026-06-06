# ADAPT-Rehab: Literature Review & Research Strategy

> Document for turning ADAPT-Rehab (MEMOTION) into a research conference paper.
> Last updated: 2026-06-06

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Literature Review](#2-literature-review)
   - 2.1 Computer Vision-Based Rehabilitation Systems
   - 2.2 3D Pose Estimation for Rehabilitation
   - 2.3 Pain and Emotion Detection for Elderly
   - 2.4 Multimodal AI Rehabilitation Systems
   - 2.5 Advanced Kinematics & Biomechanics Algorithms
3. [Gap Analysis of Current Implementation](#3-gap-analysis)
4. [Improvement Strategies](#4-improvement-strategies)
5. [Proposed Research Contributions](#5-proposed-research-contributions)
6. [References](#6-references)

---

## 1. Executive Summary

ADAPT-Rehab is a real-time physical rehabilitation support system for elderly Vietnamese users, using webcam-based pose estimation to track exercises, compare against reference videos, and provide personalized feedback. The system currently uses MediaPipe for pose/face detection and implements a 4-phase pipeline: Pose Detection → Safe-Max Calibration → Motion Synchronization → Scoring & Analysis.

After deep research across 5 dimensions (3D pose estimation, emotion/pain detection, multimodal LLM integration, advanced kinematics, and rehabilitation technology landscape), we identify **6 major improvement areas** that can elevate this project to a publishable research contribution:

1. **3D Pose Estimation**: Upgrade from MediaPipe to a hybrid MediaPipe 2D → MotionAGFormer/MotionBERT 3D lifting pipeline (~40% accuracy improvement)
2. **Pain/Emotion Detection**: Replace basic FACS with deep learning AU detection + multi-task pain/emotion model
3. **Multimodal Integration**: Add voice instruction + LLM-based coaching (Whisper + GPT-4o/Claude + Edge-TTS)
4. **Advanced Kinematics**: Upgrade from dot-product angles to quaternion-based joint computation + SPARC smoothness metric
5. **Compensation Detection**: Enhance with GCN-based temporal models for compensatory movement detection
6. **Vietnamese Elderly Context**: Position as the first Vietnamese-language, elderly-specific rehabilitation system

**Key Differentiators for Paper:**
- First webcam-based rehabilitation system specifically designed for Vietnamese elderly
- Novel hybrid 3D pose estimation pipeline optimized for rehabilitation
- Multimodal integration (vision + voice + LLM) for elderly exercise guidance
- Real-world deployment consideration (low-resource, privacy-preserving)

---

## 2. Literature Review

### 2.1 Computer Vision-Based Rehabilitation Systems

#### Commercial Systems

| Platform | Technology | Validation | Key Feature |
|----------|-----------|------------|-------------|
| **Sword Health** | Motion sensors + AI | Multiple RCTs | Outcomes comparable to in-person PT |
| **Hinge Health** | Wearable sensors | Published clinical data | Reduced surgeries, ER visits |
| **Kaia Health** | AI-guided digital therapy | Peer-reviewed | Chronic back pain management |
| **Kemtai (Physitrack)** | Pure webcam CV | Internal validation | No special sensors needed |

**Industry Trend**: The field is moving from proprietary sensors toward standard webcam/RGB camera solutions, driven by accessibility and cost reduction. ADAPT-Rehab aligns with this trend.

#### Academic Systems

Notable academic contributions:
- **Stroke rehabilitation systems** — Kinect/depth camera-based upper-limb assessment (Capecci et al., Oña et al.)
- **Automated TUG (Timed Up and Go) test** — Multiple systems using monocular RGB + pose estimation for fall-risk assessment
- **Home-based telerehabilitation platforms** — Integrated systems with real-time feedback

#### Clinical Validation Approaches

**Study Designs Used:**
- Randomized Controlled Trials (RCTs) — gold standard
- Prospective cohort designs
- Pre-post comparisons with control groups
- Crossover designs

**Technical Validation Metrics:**
- Mean Absolute Error (MAE) for joint angles (typically 5-15 degrees)
- Comparison against gold-standard motion capture (Vicon, Qualisys)
- Comparison against clinical goniometer

**Clinical Outcome Metrics:**
- Numeric Pain Rating Scale (NPRS)
- Range of Motion (ROM) measurements
- Exercise adherence rates
- System Usability Scale (SUS)

#### MediaPipe Validation Results

| Joint/Movement | MAE vs Motion Capture | Source |
|---|---|---|
| Shoulder flexion & abduction | 5-8 degrees | PMC 2024 |
| Most movements (general) | Within 10 degrees | MDPI Sensors 2024 |
| Overall vs gold standard | 83.3% no significant difference | PMC 2024 |
| Shoulder rotation | Larger errors (beyond 10 degrees) | PMC 2024 |
| Complex multi-joint movements | Reduced accuracy | IEEE 2023 |

#### Vietnam-Specific Context

- Vietnam projected to become "aged society" by 2030 (20%+ aged 60+)
- "Aging before affluence" — aging faster than economic development
- **Severe shortage**: ~0.4-0.5 physiotherapists per 100,000 population
- Rural-urban disparity: 5-10x gap between cities and rural provinces
- National Digital Transformation Program targeting healthcare modernization
- **No published Vietnamese-specific CV rehabilitation system exists** — clear research gap

---

### 2.2 3D Pose Estimation for Rehabilitation

#### Current State: MediaPipe Limitations

MediaPipe BlazePose achieves ~63mm MPJPE on Human3.6M benchmark. While fast (300+ FPS on mobile), it has significant limitations:
- Z-coordinates are NOT metric (normalized relative to torso size)
- Single-frame processing with no temporal context
- Degrades significantly under self-occlusion
- Poor accuracy for frontal/transverse plane rotations
- Struggles with lower limb tracking during functional exercises

#### State-of-the-Art 3D Pose Estimation (2024)

| Method | MPJPE (mm) | P-MPJPE (mm) | Real-time | Architecture |
|--------|-----------|--------------|-----------|-------------|
| **GLA-GCN** | ~35.7 | ~28.0 | Yes (GPU) | Graph Conv (Global-Local Adaptive) |
| **MotionAGFormer** | ~35.5-38.9 | ~29.5 | ~35 FPS (GPU) | Graph Transformer hybrid |
| **MotionBERT** | ~37.8 | ~27.4 | ~33 FPS (RTX 3090) | DSTformer (dual-stream spatiotemporal) |
| **PoseFormerV2** | ~38-40 | ~30 | ~30+ FPS (GPU) | Transformer (frequency-domain FFT) |
| **VideoPose3D** | ~46.8 | ~36.5 | ~65 FPS (GPU) | Temporal conv (dilated) |
| **BlazePose (MediaPipe)** | ~63.0 | ~63.0 | 300+ FPS mobile | Lightweight CNN |

**Key Finding**: SOTA methods (2024) achieve ~35-39mm MPJPE, roughly **40% better** than MediaPipe's 63mm.

#### Practical Upgrade Path for ADAPT-Rehab

The most practical approach is a **hybrid pipeline**:
1. Use MediaPipe as the 2D keypoint detector (fast and robust for 2D)
2. Feed 2D keypoints into a temporal lifting model (MotionAGFormer or PoseFormerV2) for 3D reconstruction
3. With TensorRT optimization on consumer GPU (RTX 3060+): 30+ FPS with ~35-40mm MPJPE

**Clinical Impact of Accuracy Improvement:**
- Multi-plane joint angle estimation (frontal/transverse planes)
- Occluded joint recovery during exercises
- Lower limb tracking where current solutions fall short

#### Key Papers

- MotionBERT: Zhu et al., "MotionBERT: A Unified Perspective on Learning Human Motion Representations," ICCV 2023
- MotionAGFormer: "MotionAGFormer: Multi-Attention with Graph Transformer for 3D Human Pose Estimation," WACV 2024
- PoseFormerV2: "PoseFormerV2: Exploring Frequency Domain for Efficient 3D Human Pose Estimation," ICCV 2023
- Clinical validation: "Skeletal tracking as an alternative to motion capture for rehabilitation," JNER 2024

---

### 2.3 Pain and Emotion Detection for Elderly

#### Current State: Basic FACS in ADAPT-Rehab

The current `PainDetector` uses rule-based FACS with fixed thresholds for 6 Action Units (AU4, AU6, AU7, AU9, AU10, AU43). Limitations:
- Error-prone landmark-based AU estimation
- No deep learning feature extraction
- Fixed thresholds don't adapt to individual faces
- Not validated on elderly faces

#### Pain Detection Datasets

| Dataset | Description | Modality | Key Feature |
|---------|-------------|----------|-------------|
| **UNBC-McMaster** | 200 videos, 25 participants, shoulder pain | Video + AU codes | Primary benchmark for pain estimation |
| **BioVid Heat Pain** | Multimodal (face + EMG + ECG + EDA) | Video + Physiological | Multi-intensity pain levels |
| **EmoPain** | Chronic pain in rehabilitation | Video + Body posture | Both posed and spontaneous pain |

**PSPI Formula**: PSPI = AU4 + max(AU6, AU7) + max(AU9, AU10) + max(AU20, AU25, AU26)

#### State-of-the-Art Pain Detection Models

| Approach | Architecture | Performance (UNBC) | Real-time |
|----------|-------------|-------------------|-----------|
| **AU-based (PSPI)** | OpenFace 2.0 + PSPI | PCC ~0.70-0.80 | Yes |
| **CNN Regression** | ResNet-50/EfficientNet | MAE ~1.0-1.5, PCC ~0.80-0.88 | Yes |
| **Transformer** | ViT/Swin + temporal | MAE ~0.8-0.9, PCC >0.90 | With GPU |
| **Multi-task** | Joint AU + pain regression | Best overall | With GPU |

#### Emotion Recognition for Elderly

**Current FER Benchmarks (2024 SOTA):**

| Model | FER2013 | RAF-DB | AffectNet-7 |
|-------|---------|--------|-------------|
| ViT-based (best) | ~76-77% | ~92-93% | ~67-70% |
| CLIP-finetuned | ~75% | ~91% | ~66% |
| DINOv2-based | ~74% | ~92% | ~68% |
| MobileNetV3-Large | ~71% | ~87% | ~60% |

**Challenges for Elderly Faces:**
1. Reduced facial expressiveness — subtler muscle movements
2. Skin texture confusion — wrinkles vs AU activations
3. Dataset bias — models trained on young faces show 10-20% lower accuracy on elderly
4. Medication effects — facial masking from common elderly medications
5. Comorbidities — stroke, Bell's palsy causing facial asymmetry

**Critical Gap**: No large-scale public dataset specifically for elderly facial emotion recognition (60-90+ years).

#### AU Detection Models

| Model | Architecture | Key Strength |
|-------|-------------|--------------|
| **OpenFace 2.0** | SVM/NN on geometric+appearance | Widely used baseline, 35+ AUs |
| **JAA-Net** | Multi-task (landmark + AU) | Joint alignment improves localization |
| **ME-GraphAU** | Graph Neural Network | Models AU co-occurrence |
| **py-feat** | Multiple backends (JAANet, DRML) | Python-native, open source |

#### Recommended Upgrade for ADAPT-Rehab

**Face Analysis Pipeline:**
```
Camera → Face Detection (MediaPipe) → Face Alignment (468 landmarks)
  → Feature Extraction (MobileNetV3-Large or EfficientNet-B0)
  → AU Detection (py-feat/JAANet)
  → Emotion Classification (multi-task head)
  → Pain Intensity Regression
  → Output at 15-30 FPS
```

**For Elderly Adaptation:**
- Use AU-based approaches (more robust to age-related appearance changes)
- Domain adaptation from general models to elderly faces
- Fine-tune on any available elderly data
- Consider synthetic data augmentation using GANs/diffusion models

---

### 2.4 Multimodal AI Rehabilitation Systems

#### LLM-Based Rehabilitation Assistants

**Key Papers:**

| Paper | Year | Key Finding |
|-------|------|-------------|
| "Rehabilitation Chatbot on Top of GPT-4o for Post-Stroke Patients" | 2024 | RAG chatbot for post-stroke exercise guidance |
| "ChatGPT for Physiotherapy: Is It Ready for Use?" | 2024 | LLMs can generate plausible programs but lack individualization |
| "ChatGPT in Physical and Rehabilitation Medicine: Not Yet" | 2024 | Not ready for independent clinical use; generic recommendations |
| "A Framework of Real-Time AIGC Multimodal Feedback for Fitness" | 2024 | Pose estimation + LLMs + real-time multimodal feedback |
| "Pose Trainer: Correcting Exercise Posture using Pose Estimation" | 2020 | Foundational pose estimation exercise correction (32 citations) |

**Key Finding**: The specific combination of **voice-interactive LLM + real-time pose estimation + TTS feedback for elderly rehabilitation** is genuinely novel. No paper integrates all three modalities into a unified system targeting elderly rehabilitation.

#### Voice Instruction Systems

**Speech Recognition (ASR):**

| System | Strengths | Limitations |
|--------|-----------|-------------|
| **Whisper large-v3** | Best open-source ASR, 99 languages | Higher WER for elderly speech (10-30% degradation) |
| **Whisper + LoRA fine-tuning** | Domain adaptation | Requires training data |

**Text-to-Speech (TTS):**

| System | Offline | Quality | Voice Cloning | Latency |
|--------|---------|---------|--------------|---------|
| **Edge-TTS** | No | High | No | ~200ms (network) |
| **Coqui XTTS** | Yes | High | Yes (6s sample) | ~100ms (local GPU) |
| **Piper TTS** | Yes | Good | No | ~50ms (local CPU) |

**Recommendation**: Edge-TTS for prototyping, Coqui XTTS for production (offline, privacy, voice customization).

#### Multimodal Architecture Pattern

The emerging pattern in literature:
1. **Vision Module**: MediaPipe Pose extracts 33 body landmarks in real-time
2. **Analysis Module**: Joint angles computed; exercise type detected
3. **LLM Module**: Pose data + exercise context fed to LLM for natural language feedback
4. **Output Module**: Feedback delivered via TTS (voice), visual overlays, or text
5. **Adaptation Module**: Difficulty adjusted based on performance history

#### RAG for Rehabilitation Knowledge

- Key benefit: reduces hallucinations by grounding LLM outputs in verified clinical literature
- Knowledge base: exercise libraries, clinical guidelines, contraindications
- Frameworks: LangChain, LlamaIndex for RAG pipeline
- Safety: Guardrails AI for output filtering, contraindication checking

---

### 2.5 Advanced Kinematics & Biomechanics Algorithms

#### Joint Angle Computation Methods

| Method | Parameters | Gimbal Lock | 3D Accuracy | Clinical Use |
|--------|-----------|-------------|-------------|-------------|
| **Dot Product (PAM)** | 1 (scalar) | N/A | Sagittal plane only | Simple ROM |
| **Rotation Matrix (FHAM)** | 9 (3x3) | No | Good | Research |
| **Quaternion** | 4 | No | Good | Growing |
| **Euler/Cardan (EAM)** | 3 | Yes | Sequence-dependent | Clinical standard (ISB) |
| **Grood-Suntay JCS** | 3 | No | Anatomically aligned | ISB recommended |

**Key Paper**: Sangeux & Polak (2020), "Joint Angle Calculation Methods for Gait Analysis: A Systematic Review," Gait & Posture.

**Recommendation for ADAPT-Rehab**: Implement quaternion-based computation as an alternative to the current dot-product method, especially for multi-plane movements.

#### Movement Smoothness Metrics

| Metric | Robust to Duration | Bounded | Best For |
|--------|-------------------|---------|----------|
| **Number of Velocity Peaks** | No | No | Simple assessment |
| **Log-Dimensionless Jerk (LDLJ)** | Partially | Yes | Stroke rehabilitation |
| **SPARC (Spectral Arc Length)** | Yes | Yes | Cross-speed comparison |

**Key Finding**: SPARC is preferred when movement speed varies between sessions or patients — critical for elderly rehabilitation where each person moves at different speeds.

**Current ADAPT-Rehab**: Uses squared jerk for fatigue detection. Should add SPARC as a complementary smoothness metric.

#### DTW Variants

| Variant | Key Feature | Best For |
|---------|------------|----------|
| **Standard DTW** | Optimal warping path | Basic comparison |
| **Weighted DTW** | Per-joint weighting | Rehabilitation (current) |
| **Constrained DTW** | Sakoe-Chiba band | Preventing pathological alignments |
| **Soft-DTW** | Differentiable | Integration with deep learning |

**Current ADAPT-Rehab**: Uses weighted DTW (good). Could benefit from constrained DTW to prevent pathological alignments.

#### Compensation Detection

**Current Approach**: Rule-based thresholds for shoulder hiking, trunk lean, hip asymmetry.

**Advanced Approaches (Literature):**
- Graph Convolutional Networks modeling body joint dependencies
- Temporal models (LSTM/Transformer) for sequential motion analysis
- Explainable AI (XAI) for clinician-interpretable feedback
- Few-shot learning for limited clinical data

#### Balance and Stability Metrics

**Nonlinear Dynamics Metrics (more sensitive than linear measures):**
- Sample Entropy (SampEn): Measures regularity of movement signals
- Recurrence Quantification Analysis (RQA): Captures fatigue-related changes
- Multi-scale sample entropy: Complexity across time scales

---

## 3. Gap Analysis

### 3.1 Technical Gaps

| Component | Current State | Gap | Impact |
|-----------|--------------|-----|--------|
| **Pose Estimation** | MediaPipe 2D/3D (63mm MPJPE) | No temporal modeling, poor depth accuracy | ~50-60% improvement with direct 3D model |
| **Pain Detection** | Rule-based FACS (6 AUs) | No deep learning, fixed thresholds | Misses subtle pain, false positives |
| **Emotion Recognition** | Not implemented | Missing entirely | Cannot detect distress/discomfort |
| **Voice Interaction** | Not implemented | No voice guidance | Less accessible for elderly |
| **LLM Integration** | Not implemented | No adaptive coaching | Generic, non-personalized feedback |
| **Joint Angles** | Dot product only | No quaternion/rotation matrix | Poor multi-plane accuracy |
| **Smoothness Metric** | Jerk only | No SPARC | Cannot compare across speeds |
| **Compensation** | Rule-based thresholds | No temporal/ML models | Brittle, high false positive rate |

### 3.2 Clinical Validation Gaps

| Gap | Current State | Required |
|-----|--------------|----------|
| Validation against motion capture | None | Vicon/OptiTrack comparison |
| Clinical population testing | None | Elderly patient studies |
| Longitudinal tracking | Session-level only | Weeks/months progress |
| Clinician feedback integration | None | Therapist dashboard |
| Standardized outcome metrics | None | NPRS, ROM, SUS |

### 3.3 Research Contribution Gaps

| Gap | Opportunity |
|-----|------------|
| No Vietnamese-specific CV rehab system | First-mover advantage |
| No elderly-specific pose estimation validation | Novel validation study |
| No multimodal (vision+voice+LLM) rehab for elderly | Genuinely novel integration |
| No compensatory movement detection for elderly | Under-studied population |
| No privacy-preserving edge deployment for rehab | Practical contribution |

---

## 4. Improvement Strategies

### Strategy 1: Direct 3D Pose Estimation (No MediaPipe Dependency)

**Goal**: Replace MediaPipe entirely with a direct image-to-3D pose estimation model for better accuracy and cleaner architecture.

**Why NOT MediaPipe + Lifting Hybrid?**
- Hybrid (MediaPipe 2D → lifting model) is unnecessarily complex — two models to maintain
- MediaPipe's 2D detection is optimized for speed, not accuracy
- Direct end-to-end models are now fast enough for real-time (25-30 FPS)
- For a paper, direct 3D is a cleaner, more novel contribution

**Direct 3D Pose Estimation Models (Image → 3D in one step):**

| Model | Input | MPJPE (mm) | FPS | Key Advantage |
|-------|-------|-----------|-----|---------------|
| **MeTRAbs** | Raw image | ~20-30 | 20-30 | Metric-scale (real cm/mm) — critical for rehab |
| **HybrIK** | Raw image | ~30 | 25-30 | Analytical IK, physically plausible joints |
| **ROMP** | Raw image | ~35 | 25-30 | Full SMPL body mesh |
| **MotionBERT** | Raw video clip | ~37.8 | 33 | Self-supervised pre-training, temporal |

**Recommended: MeTRAbs** for rehabilitation because:
- Produces **metric-scale** 3D coordinates (real centimeters, not normalized)
- No need for separate calibration to get real-world measurements
- Joint angles directly comparable to clinical goniometer
- Handles multiple people (useful for group rehab settings)

**Implementation**:
```
Webcam Frame (30 FPS)
  → MeTRAbs / HybrIK (direct image → 3D keypoints, metric scale)
  → Kalman Filter (temporal smoothing)
  → Quaternion-Based Joint Angle Computation
  → Output at 25-30 FPS
```

**Fallback Strategy**:
- If GPU unavailable: fall back to MediaPipe 2D + simple depth estimation
- If FPS too low: use lighter model variant or reduce input resolution

**Expected Improvement**:
- MPJPE: 63mm → 20-30mm (~50-60% improvement over MediaPipe)
- Metric-scale coordinates (real centimeters) vs MediaPipe's normalized values
- Multi-plane angle accuracy: 10-15° → 3-7°
- Physically plausible joint rotations (no impossible poses)

**Papers to Cite**:
- MeTRAbs: Sarandi et al., "Metric-Scale Truncated-Barrel Human Body Shape Estimation," WACV 2021
- HybrIK: Li et al., "HybrIK: A Hybrid Analytical-Neural Inverse Kinematics Solution for 3D Human Pose and Shape Estimation," CVPR 2021
- MotionBERT: Zhu et al., ICCV 2023
- "Skeletal tracking as alternative to MoCap for rehabilitation," JNER 2024

---

### Strategy 2: Deep Learning Pain/Emotion Detection

**Goal**: Replace rule-based FACS with deep learning multi-task model for pain and emotion detection.

**Implementation**:
```
Face Frame
  → MediaPipe Face Detection + Alignment
  → MobileNetV3-Large Feature Extraction
  → Multi-task Head:
      ├─ AU Detection (py-feat/JAANet)
      ├─ Emotion Classification (7 classes)
      └─ Pain Intensity Regression (0-16 PSPI)
  → Temporal Smoothing (moving average)
  → Clinical Alert Generation
```

**Technical Details**:
- Replace current `PainDetector` with deep learning model
- Use py-feat library for AU detection (Python-native, open source)
- Fine-tune MobileNetV3-Large on FER benchmarks
- Add domain adaptation for elderly faces
- Implement multi-task learning (joint AU + emotion + pain)

**Expected Improvement**:
- Pain detection accuracy: ~70% → ~85-90%
- Emotion recognition: new capability
- Better robustness to elderly facial characteristics

**Papers to Cite**:
- UNBC-McMaster dataset and PSPI scoring
- JAA-Net for AU detection
- ViT-based FER (SOTA on RAF-DB)

---

### Strategy 3: Multimodal Voice + LLM Integration

**Goal**: Add voice instruction and LLM-based coaching for elderly exercise guidance.

**Implementation**:
```
Exercise Session
  → Pose Analysis (from Strategy 1)
  → Pain/Emotion Analysis (from Strategy 2)
  → Context Builder:
      ├─ Current exercise phase
      ├─ Joint angle deviations
      ├─ Pain/emotion state
      └─ User profile (age, ROM limits)
  → LLM (GPT-4o/Claude) with RAG:
      ├─ Clinical rehabilitation knowledge base
      ├─ Exercise library with parameters
      └─ Safety guardrails (contraindications)
  → Natural Language Feedback
  → Edge-TTS (Vietnamese voice)
  → Audio Output + Visual Overlay
```

**Technical Details**:

**LLM via API (NOT self-hosted):**
- Use GPT-4o or Claude API for reasoning and feedback generation
- No need to run LLM locally — saves GPU for vision models
- API calls are fast (~200-500ms) and high quality
- Standard practice in research papers — reproducible with documented prompts
- Cost: ~$0.01-0.05 per exercise session (very affordable)

**Voice Pipeline:**
- ASR: Whisper large-v3 (open-source, runs locally)
- TTS: Edge-TTS for Vietnamese (free, high quality, cloud-based)
  - Alternative: Coqui XTTS for offline/privacy-sensitive deployment
- Vietnamese-language prompts and responses

**RAG Architecture:**
- Knowledge base: Clinical rehabilitation guidelines, exercise libraries, contraindications
- Framework: LangChain or LlamaIndex for RAG pipeline
- Safety: Guardrails AI for output filtering, contraindication checking
- Prompt engineering: Vietnamese-language system prompts with rehabilitation context

**Why API is Better for Research:**
- No GPU overhead for LLM inference
- Better quality than any self-hosted alternative
- Easy to reproduce (just document API version + prompts)
- Focus compute budget on vision models (which actually need GPU)
- Standard practice in 2024-2025 AI research papers

**Expected Improvement**:
- Accessibility: voice guidance for visually impaired elderly
- Personalization: adaptive coaching based on real-time performance
- Engagement: conversational interaction increases adherence
- Safety: RAG-grounded clinical knowledge prevents harmful advice

**Papers to Cite**:
- "Rehabilitation Chatbot on Top of GPT-4o" (2024)
- "A Framework of Real-Time AIGC Multimodal Feedback for Fitness" (2024)
- "Pose Trainer" (arXiv:2006.11718)

---

### Strategy 4: Advanced Kinematics Algorithms

**Goal**: Improve motion assessment accuracy and add clinically validated metrics.

**Implementation**:

**4a. Quaternion-Based Joint Angles**:
```python
# Replace current dot-product method with quaternion rotation
def calculate_joint_angle_quaternion(landmarks, joint_type):
    # Get segment vectors
    proximal_vec = landmarks[vertex] - landmarks[proximal]
    distal_vec = landmarks[vertex] - landmarks[distal]
    
    # Compute rotation quaternion between segments
    q = quaternion_rotation_between(proximal_vec, distal_vec)
    
    # Extract angle from quaternion
    angle = 2 * np.arccos(q.w) * (180 / np.pi)
    return angle
```

**4b. SPARC Smoothness Metric**:
```python
def compute_sparc(velocity_profile, fs=30):
    """Spectral Arc Length - robust to movement duration."""
    # Compute Fourier magnitude spectrum
    freq = np.fft.rfftfreq(len(velocity_profile), d=1/fs)
    mag = np.abs(np.fft.rfft(velocity_profile))
    
    # Normalize
    mag_norm = mag / np.max(mag)
    
    # Compute arc length of spectrum
    arc_length = -np.sum(np.sqrt(
        (np.diff(freq) ** 2) + (np.diff(mag_norm) ** 2)
    ))
    
    return arc_length
```

**4c. Constrained DTW**:
```python
def constrained_dtw(seq1, seq2, window_percent=0.1):
    """DTW with Sakoe-Chiba band constraint."""
    n, m = len(seq1), len(seq2)
    w = max(int(max(n, m) * window_percent), abs(n - m))
    # ... standard DTW with window constraint
```

**4d. Enhanced Compensation Detection**:
- Add temporal modeling (LSTM) for compensation pattern detection
- Implement trunk lean angle computation from spine landmarks
- Add hip shift detection using hip landmark asymmetry over time

**Expected Improvement**:
- Multi-plane angle accuracy: 5-8° (from 10-15°)
- Smoothness metric: clinically validated SPARC
- Compensation: temporal pattern detection vs frame-by-frame thresholds

---

### Strategy 5: Enhanced Scoring System

**Goal**: Add clinically validated metrics and improve scoring methodology.

**New Metrics to Add**:

| Metric | Formula | Clinical Meaning |
|--------|---------|-----------------|
| **SPARC** | Spectral arc length of velocity | Movement smoothness (duration-independent) |
| **Sample Entropy** | SampEn of joint angle time series | Movement regularity/complexity |
| **Compensation Index** | Weighted sum of detected compensations | Movement quality |
| **Fatigue Index** | Jerk ratio + ROM degradation + velocity decline | Multi-indicator fatigue |
| **Symmetry Index** | \|L - R\| / (0.5 * (L + R)) * 100 | Left-right balance |

**Updated Scoring Weights**:

| Component | Current Weight | Proposed Weight | Rationale |
|-----------|---------------|-----------------|-----------|
| ROM Score | 30% | 25% | Reduced to accommodate new metrics |
| Stability Score | 20% | 15% | Combined with smoothness |
| Flow Score | 20% | 20% | Keep (now using SPARC) |
| Symmetry Score | 15% | 15% | Keep |
| Compensation Score | 15% | 15% | Enhanced with temporal models |
| **Smoothness (SPARC)** | 0% | **10%** | New: clinically validated |

---

### Strategy 6: System Architecture Improvements

**Goal**: Prepare for production deployment and research validation.

**Architecture Changes**:

```
┌─────────────────────────────────────────────────────────┐
│                    ADAPT-Rehab v3.0                      │
├─────────────────────────────────────────────────────────┤
│  Input Layer                                            │
│  ├─ Webcam (30 FPS)                                     │
│  ├─ Microphone (Whisper ASR)                            │
│  └─ User Profile (JSON)                                 │
├─────────────────────────────────────────────────────────┤
│  Perception Layer                                       │
│  ├─ 3D Pose: MeTRAbs/HybrIK (direct image→3D, metric)  │
│  ├─ Face: MediaPipe Face Mesh (468 landmarks)           │
│  ├─ AU Detection: py-feat/JAANet                        │
│  └─ Emotion: MobileNetV3-Large (fine-tuned)             │
├─────────────────────────────────────────────────────────┤
│  Analysis Layer                                         │
│  ├─ Kinematics: Quaternion-based angles                 │
│  ├─ Smoothness: SPARC + Jerk                            │
│  ├─ DTW: Weighted + Constrained                         │
│  ├─ Compensation: Temporal LSTM model                   │
│  ├─ Fatigue: Multi-indicator (Jerk + ROM + velocity)    │
│  └─ Pain: Multi-task AU + emotion + pain regression     │
├─────────────────────────────────────────────────────────┤
│  Intelligence Layer                                     │
│  ├─ LLM: GPT-4o/Claude API (not self-hosted)           │
│  ├─ RAG: LangChain + clinical knowledge base            │
│  ├─ Safety: Contraindication checker                    │
│  ├─ Personalization: User profile + history             │
│  └─ Voice: Whisper (ASR) + Edge-TTS (Vietnamese)        │
├─────────────────────────────────────────────────────────┤
│  Output Layer                                           │
│  ├─ Visual: Skeleton overlay + angle arcs               │
│  ├─ Audio: Voice instructions + feedback                │
│  ├─ Dashboard: Real-time metrics                        │
│  └─ Reports: Session summary + recommendations          │
├─────────────────────────────────────────────────────────┤
│  Data Layer                                             │
│  ├─ Session Logger (JSON/CSV)                           │
│  ├─ User Profiles (JSON)                                │
│  ├─ Exercise Library                                    │
│  └─ Clinical Knowledge Base (RAG)                       │
└─────────────────────────────────────────────────────────┘
```

---

## 5. Proposed Research Contributions

### Contribution 1: First Vietnamese-Language Elderly Rehabilitation System

**Novelty**: No published CV-based rehabilitation system exists specifically for Vietnamese elderly users.

**Paper Angle**: "ADAPT-Rehab: A Real-Time Vision-Based Rehabilitation System for Elderly Vietnamese Users with Personalized Exercise Guidance"

**Key Claims**:
- Vietnamese-language interface and voice guidance
- Culturally adapted encouragement messages
- Safe-Max calibration for elderly ROM limitations
- Wait-for-User synchronization (never rushes elderly)

**Evaluation**:
- Usability study with Vietnamese elderly participants (n=20-30)
- System Usability Scale (SUS) scores
- Exercise adherence rates
- Qualitative feedback on cultural appropriateness

---

### Contribution 2: Direct 3D Pose Estimation for Rehabilitation

**Novelty**: First application of direct image-to-3D metric-scale pose estimation (MeTRAbs/HybrIK) specifically for elderly rehabilitation exercise assessment.

**Paper Angle**: "Direct 3D Pose Estimation for Real-Time Rehabilitation Exercise Assessment: A Metric-Scale Approach"

**Key Claims**:
- 50-60% improvement in 3D accuracy over MediaPipe
- Metric-scale coordinates (real centimeters) for clinical-comparable measurements
- Real-time performance (25-30 FPS) on consumer hardware
- Validated against motion capture for rehabilitation exercises

**Evaluation**:
- Comparison against Vicon/OptiTrack motion capture (joint angle MAE)
- Comparison against MediaPipe and OpenPose baselines
- UI-PRMD dataset benchmark
- FPS benchmark on different hardware (RTX 4090, 3060, GTX 1660)
- Ablation study: direct 3D vs MediaPipe vs OpenPose

---

### Contribution 3: Multimodal AI Coaching for Elderly Rehabilitation

**Novelty**: First integration of vision + voice + LLM for elderly rehabilitation exercise guidance.

**Paper Angle**: "Multimodal AI Coaching for Elderly Rehabilitation: Integrating Computer Vision, Voice Instruction, and Large Language Models"

**Key Claims**:
- Real-time pose analysis + LLM-generated personalized feedback
- Voice-interactive guidance in Vietnamese
- RAG-grounded clinical knowledge for safe recommendations
- Adaptive difficulty based on detected fatigue/pain

**Evaluation**:
- Comparison against non-LLM feedback (rule-based messages)
- User engagement metrics (session duration, adherence)
- Clinical safety audit (no harmful recommendations)
- Feedback quality assessment by physiotherapists

---

### Contribution 4: Elderly-Specific Pain/Emotion Detection

**Novelty**: Multi-task deep learning model for joint AU detection, emotion recognition, and pain estimation specifically adapted for elderly faces.

**Paper Angle**: "Multi-Task Pain and Emotion Detection for Elderly Rehabilitation: An AU-Aware Approach"

**Key Claims**:
- Joint AU + emotion + pain estimation in single forward pass
- Domain adaptation for elderly facial characteristics
- Interpretable AU-based features for clinical trust
- Real-time performance (15-30 FPS)

**Evaluation**:
- UNBC-McMaster benchmark (pain estimation)
- RAF-DB/AffectNet benchmark (emotion recognition)
- Custom elderly validation set (if collected)
- Comparison against OpenFace 2.0 baseline

---

### Contribution 5: Advanced Kinematics for Rehabilitation Scoring

**Novelty**: Integration of clinically validated biomechanics metrics (SPARC, quaternion angles, constrained DTW) into a real-time rehabilitation scoring system.

**Paper Angle**: "Clinically-Informed Real-Time Exercise Scoring for Rehabilitation: Integrating SPARC, Quaternion Kinematics, and Compensatory Movement Detection"

**Key Claims**:
- SPARC-based smoothness metric (duration-independent)
- Quaternion-based joint angles for multi-plane accuracy
- Temporal compensation detection using LSTM
- Validated against clinical assessment scores

**Evaluation**:
- Correlation with clinical outcome measures (Fugl-Meyer, Berg Balance)
- Comparison against current scoring methods
- Sensitivity to change over rehabilitation progress
- Inter-rater reliability with physiotherapist assessments

---

## 6. Benchmark Evaluation Strategy

### Do We Need Benchmarks?

**Yes.** For a conference paper, you need evaluation. The question is what kind:

| Paper Type | Minimum Evaluation | Ideal Evaluation |
|-----------|-------------------|------------------|
| **Systems paper** (focus on architecture) | FPS, latency, ablation study | + accuracy on public dataset |
| **Algorithm paper** (focus on novel method) | Comparison vs baselines on public dataset | + clinical validation |
| **Application paper** (focus on real-world impact) | User study + case studies | + technical accuracy benchmark |

**For ADAPT-Rehab, we should aim for a hybrid**: technical accuracy benchmark + small clinical user study.

### Recommended Evaluation Plan

#### Evaluation 1: Joint Angle Accuracy (Technical)

**Goal**: Prove our 3D pose estimation produces accurate joint angles.

**Method**:
- Record 10-15 elderly participants performing rehabilitation exercises
- Simultaneously capture with our system AND gold-standard motion capture (Vicon/OptiTrack)
- Compute Mean Absolute Error (MAE) for each joint

**Metrics**:
| Joint | Expected MAE | Clinical Threshold |
|-------|-------------|-------------------|
| Shoulder flexion | 3-7° | <10° acceptable |
| Shoulder abduction | 5-8° | <10° acceptable |
| Elbow flexion | 2-5° | <5° excellent |
| Knee flexion | 3-7° | <10° acceptable |

**Comparison Baselines**:
- MediaPipe BlazePose (current state)
- OpenPose
- Our system (MeTRAbs/HybrIK)

**Statistical Tests**:
- Bland-Altman plots for agreement analysis
- Intraclass Correlation Coefficient (ICC) for reliability
- Paired t-test or Wilcoxon signed-rank for significance

#### Evaluation 2: Public Dataset Benchmark

**Goal**: Compare against published methods on a standard benchmark.

**Recommended Datasets**:

| Dataset | Exercises | Ground Truth | Why Use It |
|---------|-----------|-------------|------------|
| **UI-PRMD** | 10 rehab exercises | Kinect + clinical scores | Standard rehab benchmark |
| **KIMORE** | 5 rehab exercises | Kinect + therapist scores | Clinical quality labels |
| **Human3.6M** | General actions | Vicon MoCap | Universal 3D HPE benchmark |

**Metrics to Report**:
- MPJPE (Mean Per Joint Position Error) in mm
- P-MPJPE (Procrustes-aligned MPJPE) in mm
- Joint angle MAE in degrees
- FPS (frames per second)

#### Evaluation 3: Clinical User Study

**Goal**: Validate the system is usable and beneficial for elderly Vietnamese users.

**Study Design**:
- N = 15-30 elderly participants (age 60+)
- Pre-post design: measure ROM before and after 4-week intervention
- Control: compare against paper-based exercise instructions

**Metrics**:

| Category | Metric | Tool |
|----------|--------|------|
| **Usability** | System Usability Scale (SUS) | Standard questionnaire |
| **Engagement** | Session completion rate | System logs |
| **Engagement** | Average session duration | System logs |
| **Clinical** | ROM improvement (pre vs post) | Goniometer |
| **Clinical** | Pain level change | NPRS (Numeric Pain Rating Scale) |
| **Satisfaction** | User satisfaction | Custom Likert scale |
| **Safety** | Adverse events | Incident reporting |

**Ethical Considerations**:
- IRB/Ethics committee approval required
- Informed consent from all participants
- Right to withdraw at any time
- Data privacy (no video storage, skeleton-only analysis)

#### Evaluation 4: Ablation Study

**Goal**: Show each component contributes to overall performance.

**Components to Ablate**:

| Ablation | What Removed | Expected Impact |
|----------|-------------|-----------------|
| w/o 3D lifting | Use MediaPipe 3D directly | ~40% worse MPJPE |
| w/o quaternion angles | Use dot-product angles | Multi-plane accuracy drops |
| w/o SPARC | Use jerk only | Cannot compare across speeds |
| w/o temporal smoothing | Raw frame-by-frame | Noisy joint angles |
| w/o compensation detection | Remove compensation scoring | Miss compensatory movements |
| w/o LLM feedback | Use rule-based messages | Less personalized feedback |

**Metric**: Overall system score vs each ablated variant.

#### Evaluation 5: Real-Time Performance Benchmark

**Goal**: Prove the system runs in real-time on consumer hardware.

**Hardware Configurations**:

| Config | GPU | Expected FPS |
|--------|-----|-------------|
| High-end | RTX 4090 | 40-60 |
| Mid-range | RTX 3060 | 25-35 |
| Budget | GTX 1660 | 15-25 |
| CPU-only | Intel i7 | 5-10 (fallback to MediaPipe) |

**Metrics**:
- FPS per component (pose, face, scoring, LLM)
- End-to-end latency (camera → feedback)
- GPU memory usage
- CPU usage

### Summary: Minimum Viable Evaluation for Conference Paper

If resources are limited, prioritize in this order:

1. **Joint angle accuracy** on 5-10 participants vs goniometer (essential)
2. **Ablation study** showing each component matters (essential)
3. **Real-time FPS benchmark** on 2-3 hardware configs (essential)
4. **UI-PRMD benchmark** comparison (highly recommended)
5. **Clinical user study** with SUS + ROM improvement (highly recommended)
6. **Full Vicon validation** (ideal but expensive)

---

## 7. References

### Pose Estimation & Computer Vision

1. Zhu et al., "MotionBERT: A Unified Perspective on Learning Human Motion Representations," ICCV 2023. arXiv:2210.06551
2. "MotionAGFormer: Multi-Attention with Graph Transformer for 3D Human Pose Estimation," WACV 2024. arXiv:2310.14391
3. "PoseFormerV2: Exploring Frequency Domain for Efficient 3D Human Pose Estimation," ICCV 2023. arXiv:2303.14080
4. Bazarevsky et al., "BlazePose: On-device Real-time Body Pose Tracking," Google Research, 2020. arXiv:2006.10204
5. "Skeletal tracking as an alternative to motion capture for rehabilitation," J NeuroEngineering Rehab, 2024.
6. "Accuracy and Reliability of 2D and 3D Pose Estimation for Post-Stroke Exercise," MDPI Applied Sciences 14(8):3200, 2024.
7. "AI-Based Joint Angle Extraction From Monocular Video for Rehabilitation," PMC-10868734, 2024.
8. "Markerless Pose Estimation in Telerehabilitation: A Systematic Review," MDPI Applied Sciences 14(10):4050, 2024.
9. "Scoping Review of Pose Estimation in Exercise Rehabilitation," PubMed PMID 39133197, 2024.

### Pain & Emotion Detection

10. Lucey et al., "The Extended Cohn-Kanade Dataset (CK+): A complete dataset for action unit and emotion-specified expression," CVPR 2010.
11. "UNBC-McMaster Shoulder Pain Expression Archive Database." https://www.pitt.edu/~jeffcohn/biased_data/
12. "Facial Action Unit Detection Using Deep Learning: A Survey," IEEE TPAMI, 2023.
13. "JAA-Net: Joint Facial Action Unit Detection and Face Alignment," ECCV 2018.
14. "ME-GraphAU: Graph Neural Network for Facial Action Unit Detection," IEEE FG 2023.
15. "Multi-Task Learning for Pain Estimation," IEEE Transactions on Affective Computing, 2024.

### Rehabilitation Technology

16. "A Comprehensive Survey on Automated Rehabilitation Systems," ScienceDirect, 2025.
17. "Pose Trainer: Correcting Exercise Posture using Pose Estimation," arXiv:2006.11718, 2020.
18. "A Framework of Real-Time AIGC Multimodal Feedback for Fitness," MDPI Information 15(9):526, 2024.
19. "Rehabilitation Chatbot on Top of GPT-4o for Post-Stroke Patients," arXiv:2405.10665, 2024.
20. "ChatGPT in Physical and Rehabilitation Medicine: Not Yet," PM&R Journal, 2024.
21. "Applications of Large Language Models in Post-Stroke Rehabilitation," Frontiers in Medicine, 2024.

### Kinematics & Biomechanics

22. Sangeux & Polak, "Joint Angle Calculation Methods for Gait Analysis: A Systematic Review," Gait & Posture, 2020.
23. Balasubramanian et al., "The Spectral Arc Length (SPARC) Metric," Journal of Biomechanics, 2012.
24. "Comparative Evaluation of Kinematic Smoothness Metrics," ScienceDirect, 2024.
25. "Validity and Reliability of Smoothness Metrics in Stroke," J NeuroEngineering Rehab, 2021.
26. "Generalized Procrustes Analysis of Biomechanical Movement Data," Springer, 2021.
27. Aurand et al., "Euler Angles vs. Quaternions for Joint Angles in Gait Analysis," IEEE, 2024.

### Vietnam Context

28. Vietnam National Digital Transformation Program (2021-2025).
29. Ministry of Health Decision No. 5316/QD-BYT (2020) on health sector digital transformation.
30. National Action Plan for the Elderly (2021-2025).
31. WHO Vietnam Country Profile: Rehabilitation Workforce.
