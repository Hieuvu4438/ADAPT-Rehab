%%
%% Fig. 1 — ADAPT-Rehab System Architecture (conference-ready)
%% Block names: functional, ≤4 words. Details deferred to caption.
%%

flowchart LR
    %% ── INPUT ──────────────────────────────────────────────────
    subgraph IN["① Input"]
        CAM["Video Stream"]
        REF["Exercise Reference"]
    end

    %% ── PERCEPTION ─────────────────────────────────────────────`
    subgraph PERC["② Perception"]
        POSE["3D Pose Estimation"]
        FACE["Facial Behavior Analysis"]
        RPOSE["Reference Pose Extraction"]
    end

    %% ── ANALYSIS ───────────────────────────────────────────────
    subgraph ANA["③ Analysis"]
        KIN["Kinematic Analysis"]
        DTW["DTW Alignment"]
        STATE["Multimodal State Detection"]
        SCORE["6-D Performance Scoring"]
    end

    %% ── INTELLIGENCE ────────────────────────────────────────────
    subgraph INTEL["④ Coaching"]
        LLM["LLM-based Coach"]
        TTS["Speech Synthesis"]
    end

    %% ── OUTPUT + KB ─────────────────────────────────────────────
    OUT["⑤ Multimodal Feedback"]
    KB["Clinical Knowledge Base"]

    %% ── EDGES ───────────────────────────────────────────────────
    CAM --> POSE & FACE
    REF --> RPOSE

    POSE --> KIN & DTW
    RPOSE --> DTW
    FACE --> STATE
    KIN --> DTW & STATE & SCORE
    DTW --> SCORE
    STATE --> SCORE & LLM

    KB --> LLM
    LLM --> TTS & OUT
    SCORE --> OUT
    TTS --> OUT

%% ─────────────────────────────────────────────────────────────────
%% FIGURE CAPTION (copy vào paper):
%%
%% Fig. 1. Overview of ADAPT-Rehab, a multimodal AI-assisted
%% rehabilitation system for elderly users. A single webcam provides
%% the Video Stream for two parallel perception branches: (i) 3D Pose
%% Estimation via RTMW3D-L (133 COCO-WholeBody keypoints) and
%% (ii) Facial Behavior Analysis combining MediaPipe Face Mesh with
%% OpenFace 3.0 GNN to produce Action Units (AU1/2/4/6/9/12/25/26),
%% emotion logits, and gaze. The Exercise Reference branch extracts a
%% Reference Pose Sequence with the same model. In the Analysis stage,
%% Kinematic Analysis computes quaternion-based joint angles, ROM, and
%% velocity; DTW Alignment measures temporal similarity between the
%% user and reference trajectories; Multimodal State Detection fuses
%% facial AUs (PSPI pain, PERCLOS fatigue) with kinematic cues
%% (asymmetry index, velocity loss) into a unified patient state;
%% and 6-D Performance Scoring evaluates ROM, Movement Flow,
%% Bilateral Symmetry, Postural Stability, Compensation, and
%% Smoothness (SPARC). The Coaching stage passes all signals to an
%% LLM-based Coach (Gemini 2.0 / GPT-4o) augmented with a RAG
%% pipeline over clinical guidelines (ACSM, APTA, WHO) and a
%% contraindication safety filter. Corrective feedback is delivered
%% via Speech Synthesis (Edge-TTS, Vietnamese) and visual/dashboard
%% overlays as Multimodal Feedback, which is also persisted as a
%% structured session log.
%% ─────────────────────────────────────────────────────────────────
