# ADAPT-Rehab — Project Context

## Project Overview

ADAPT-Rehab (internally "MEMOTION") is a **real-time physical rehabilitation support system** for **elderly Vietnamese users**. It uses computer vision to track body movements, compare against reference exercises, and provide personalized, safe feedback.

**Goal**: Turn this into a research conference paper with 5 key contributions:
1. First Vietnamese-language elderly rehabilitation system
2. Direct 3D pose estimation for rehabilitation (MeTRAbs/HybrIK)
3. Multimodal AI coaching (vision + voice + LLM API)
4. Elderly-specific pain/emotion detection (deep learning)
5. Advanced kinematics scoring (SPARC, quaternion, constrained DTW)

## Architecture (v3.0 Target)

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
│  ├─ 3D Pose: RTMW3D (133 keypoints, whole-body)        │
│  ├─ Face: OpenFace 3.0 (8 AUs + Emotion + Gaze)        │
│  ├─ AU Detection: OpenFace 3.0 GNN (AFG + FineGrain)   │
│  └─ State: AU-based formulas (PSPI, PERCLOS, etc.)      │
├─────────────────────────────────────────────────────────┤
│  Analysis Layer                                         │
│  ├─ Kinematics: Quaternion-based angles                 │
│  ├─ Smoothness: SPARC + Jerk                            │
│  ├─ DTW: Weighted + Constrained                         │
│  ├─ Compensation: Temporal LSTM model                   │
│  ├─ Fatigue: PERCLOS + Blink + Yawn + Velocity Loss    │
│  ├─ Pain: PSPI (Prkachin-Solomon, 2008) via AU        │
│  ├─ Boredom: Engagement Index (Whitehill, 2014)        │
│  └─ Body: RTMW3D behavioral (ROM decline, asymmetry)   │
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

## Technology Stack

| Component | Technology | Notes |
|-----------|-----------|-------|
| Language | Python 3.10+ | Type hints, dataclasses |
| 3D Pose | RTMW3D (MMPose) | 133 keypoints, whole-body, real-time |
| Face Detection | OpenFace 3.0 (RetinaFace) | Better than MediaPipe for AU |
| AU Detection | OpenFace 3.0 (GNN) | 8 AUs: AU1,2,4,6,9,12,25,26 |
| Emotion | OpenFace 3.0 (EfficientNet-B0) | 8 classes (AffectNet) |
| Pain Detection | PSPI formula (Prkachin-Solomon, 2008) | AU4 + max(AU6,7) + max(AU9,10) + AU43 |
| Fatigue Detection | PERCLOS + Blink + Yawn (Wierwille, 1994) | Multi-indicator composite |
| Boredom Detection | Engagement Index (Whitehill, 2014) | AU12 - AU1 - AU15 |
| Body Behavior | RTMW3D kinematics | Velocity loss, ROM decline, asymmetry |
| ASR | Whisper large-v3 | Open source, multilingual |
| TTS | Edge-TTS | Vietnamese voice, cloud-based |
| LLM | GPT-4o / Claude API | Via API, not self-hosted |
| Numerics | NumPy, SciPy | Butterworth filter, FFT, SPARC |
| DTW | fastdtw | Weighted DTW implementation |
| Visualization | OpenCV, PIL | Vietnamese text rendering |
| Testing | pytest, mypy | Type checking, unit tests |

## Code Conventions

- **Language**: English code, Vietnamese comments/messages for user-facing content
- **Type Hints**: Use Python 3.10+ type hints everywhere
- **Dataclasses**: Use `@dataclass` for data structures (designed for Flutter/Dart portability)
- **Docstrings**: Google-style docstrings with Args/Returns/Example sections
- **Naming**: snake_case for functions/variables, PascalCase for classes, UPPER_SNAKE for constants
- **Error Handling**: Safe wrappers (e.g., `calculate_angle_safe`) that return defaults instead of crashing
- **Logging**: Use `SessionLogger` for structured JSON/CSV logging
- **Versioning**: Each module has its own version in `__init__.py`

## File Structure

```
ADAPT-Rehab/
├── .claude/
│   └── CLAUDE.md              # This file — project context
├── docs/
│   └── RESEARCH_STRATEGY.md   # Literature review + improvement strategies
├── core/                      # Core algorithmic components
│   ├── __init__.py
│   ├── data_types.py          # Data classes (Point3D, LandmarkSet, etc.)
│   ├── detector.py            # MediaPipe wrapper (being replaced)
│   ├── procrustes.py          # Skeleton normalization
│   ├── kinematics.py          # Joint angle computation
│   ├── synchronizer.py        # FSM-based motion sync
│   └── dtw_analysis.py        # Weighted DTW
├── modules/                   # Functional modules
│   ├── __init__.py
│   ├── calibration.py         # Safe-Max calibration
│   ├── target_generator.py    # Personalized target rescaling
│   ├── video_engine.py        # Video player with checkpoints
│   ├── pain_detection.py      # FACS-based pain detection
│   └── scoring.py             # Multi-dimensional scoring
├── utils/                     # Utilities
│   ├── __init__.py
│   ├── logger.py              # Session logging
│   └── visualization.py       # OpenCV/PIL drawing helpers
├── perception/                # Perception layer
│   ├── __init__.py
│   ├── pose3d/                # Direct 3D pose estimation
│   │   ├── __init__.py
│   │   ├── base.py            # Abstract pose estimator interface
│   │   ├── metrab.py          # MeTRAbs implementation
│   │   ├── hybrik.py          # HybrIK implementation
│   │   └── mediapipe_fallback.py  # Fallback for CPU-only
│   ├── face_detector.py       # MediaPipe face detection (468 landmarks)
│   ├── openface_analyzer.py   # OpenFace 3.0 wrapper (AU + Emotion + Gaze)
│   └── facial_state_detector.py  # AU-based state detection (PSPI, PERCLOS, etc.)
├── analysis/                  # Analysis layer
│   ├── __init__.py
│   ├── kinematics.py          # Joint angle computation
│   ├── kinematics_quaternion.py  # Quaternion-based angles
│   ├── smoothness.py          # SPARC + LDLJ metrics
│   ├── compensation.py        # Compensatory movement detection
│   ├── fatigue.py             # Multi-indicator fatigue
│   ├── scoring_v2.py          # Enhanced scoring system
│   └── body_state_detector.py # RTMW3D behavioral state detection
├── intelligence/              # NEW: Intelligence layer
│   ├── __init__.py
│   ├── voice/                 # Voice interaction
│   │   ├── __init__.py
│   │   ├── asr.py             # Whisper speech recognition
│   │   └── tts.py             # Edge-TTS / Coqui XTTS
│   ├── llm/                   # LLM integration
│   │   ├── __init__.py
│   │   ├── client.py          # API client (GPT-4o/Claude)
│   │   ├── prompts.py         # Prompt templates
│   │   ├── rag.py             # RAG pipeline
│   │   └── safety.py          # Safety guardrails
│   └── coach/                 # Exercise coaching
│       ├── __init__.py
│       └── rehab_coach.py     # Main coaching orchestrator
├── analysis/                  # NEW: Enhanced analysis
│   ├── __init__.py
│   ├── kinematics_v2.py       # Quaternion-based angles
│   ├── smoothness.py          # SPARC + LDLJ metrics
│   ├── compensation.py        # Temporal compensation detection
│   ├── fatigue.py             # Multi-indicator fatigue
│   └── scoring_v2.py          # Enhanced scoring system
├── evaluation/                # NEW: Benchmark evaluation
│   ├── __init__.py
│   ├── benchmarks/            # Dataset loaders
│   │   ├── __init__.py
│   │   ├── uiprmd.py          # UI-PRMD dataset
│   │   └── kimore.py          # KIMORE dataset
│   ├── metrics/               # Evaluation metrics
│   │   ├── __init__.py
│   │   ├── mpjpe.py           # MPJPE computation
│   │   └── angle_mae.py       # Joint angle MAE
│   ├── ablation.py            # Ablation study runner
│   └── clinical_study.py      # Clinical user study tools
├── assets/                    # Media files (gitignored)
│   ├── audio_feedbacks/
│   ├── models/
│   └── reference_videos/
├── data/                      # Data files (gitignored)
│   ├── logs/
│   ├── user_profiles/
│   └── knowledge_base/        # RAG knowledge base
├── models/                    # ML model files (gitignored)
├── tests/                     # Unit tests
│   ├── __init__.py
│   ├── test_kinematics.py
│   ├── test_scoring.py
│   ├── test_perception.py
│   └── test_analysis.py
├── scripts/                   # Usage examples
├── main_v2.py                 # Current main app
├── main_v3.py                 # NEW: v3 main app (target)
├── requirements.txt
└── README.md
```

## Key Design Decisions

1. **Direct 3D pose estimation** (not MediaPipe + lifting hybrid) — cleaner architecture, better accuracy
2. **LLM via API** (not self-hosted) — no GPU overhead, better quality, standard for papers
3. **MediaPipe only for face** (not pose) — fast face detection is still valuable
4. **AU-based pain detection** (not end-to-end) — more interpretable for clinicians, robust to elderly faces
5. **SPARC over jerk for smoothness** — duration-independent, clinically validated
6. **Quaternion over dot-product for angles** — no gimbal lock, better multi-plane accuracy

## Research Paper Targets

| Contribution | Paper Title | Venue Target |
|-------------|-------------|-------------|
| Main system | "ADAPT-Rehab: Multimodal AI Rehabilitation for Elderly Vietnamese Users" | AAAI / IJCAI / CHI |
| 3D pose | "Direct 3D Pose Estimation for Rehabilitation Exercise Assessment" | CVPR Workshop / EMBC |
| Pain/emotion | "Multi-Task Pain and Emotion Detection for Elderly Rehabilitation" | IEEE FG / ACII |
| Multimodal | "Voice-Interactive LLM Coaching for Elderly Rehabilitation" | ACL Workshop / EMNLP |
| Kinematics | "Clinically-Informed Real-Time Exercise Scoring" | JNER / IEEE TBME |

## Evaluation Benchmarks

- **UI-PRMD**: 10 rehabilitation exercises, Kinect ground truth
- **KIMORE**: 5 rehabilitation exercises, clinical quality scores
- **Human3.6M**: General 3D HPE benchmark (for pose estimation comparison)
- **UNBC-McMaster**: Pain expression dataset (for pain detection evaluation)
- **Custom**: Vietnamese elderly participants (clinical validation)

## Important Notes

- All user-facing messages in Vietnamese (encouragement, instructions, alerts)
- Safety-first: never exceed user's calibrated ROM, always detect pain
- Wait-for-User: reference video pauses at checkpoints, never rushes elderly
- Privacy: skeleton-only analysis option (no raw video storage)
- Designed for easy Flutter/Dart portability (data classes)
