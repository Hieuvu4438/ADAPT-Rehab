flowchart TD
    subgraph Input["Input Layer"]
        subgraph UserInput["User Input"]
            UCAM[Webcam<br/>30 FPS]
            UMIC[Microphone]
            UPRO[User Profile<br/>JSON & Calib]
        end

        subgraph RefInput["Reference Input"]
            REF[Reference Video<br/>Exercise Library]
        end
    end

    subgraph Perception["Perception Layer"]
        subgraph UserPerception["User Perception"]
            UPOSE[3D Body Pose<br/>RTMW3D-L]
            UFACE[Face Mesh<br/>MediaPipe]
            UAU[AU Detection<br/>OpenFace 3.0 GNN]
            USTATE[Facial State Detection<br/>AU-based]
        end

        subgraph RefPerception["Reference Perception"]
            RPOSE[3D Reference Pose<br/>RTMW3D-L]
        end
        
        UASR[Whisper ASR<br/>Speech-to-Text]
    end

    subgraph Analysis["Analysis Layer"]
        KIN[Kinematics<br/>Quaternion angles]
        SMO[Smoothness<br/>SPARC + Jerk]
        DTW[DTW<br/>Compare User ↔ Ref]
        COMP[Compensation<br/>Heuristics / LSTM]
        
        subgraph BodyState["Body State Detection (Kinematic)"]
            ASYM[Asymmetry Index<br/>Pain/Guarding]
            VLOSS[Velocity Loss & ROM Decline<br/>Fatigue/Exhaustion]
            BVAR[Velocity Variability<br/>Boredom]
        end
        
        SCORE[Enhanced Scorer<br/>6-Dimension Stack]
    end

    subgraph Intelligence["Intelligence Layer"]
        LLM[LLM Coach<br/>GPT-4o / Claude]
        RAG[RAG Pipeline<br/>Clinical KB]
        SAFETY[Safety<br/>Contraindication]
        PERS[Personalization<br/>Profile + History]
        VOICE[Voice Engine<br/>Edge-TTS]
    end

    subgraph Output["Output Layer"]
        VIS[Visual Overlay<br/>Skeleton + ROM arcs]
        AUD[Audio Feedback<br/>Vietnamese Voice]
        DASH[Real-time HUD]
        REPORT[Session Report<br/>JSON/PDF]
    end

    subgraph Data["Data Layer"]
        LOG[Session Logger<br/>JSON/CSV]
        USERS[User Profiles<br/>JSON]
        EXERC[Exercise Library]
        KB[Clinical KB]
    end

    %% Input to Perception
    UCAM --> UPOSE
    UCAM --> UFACE
    UMIC --> UASR
    REF --> RPOSE

    %% Perception internal
    UFACE --> UAU
    UAU --> USTATE

    %% Perception to Analysis / Intelligence
    UPOSE --> KIN
    UPOSE --> COMP
    RPOSE --> DTW
    KIN --> DTW
    KIN --> SMO
    KIN --> ASYM
    KIN --> VLOSS
    
    UASR --> LLM

    %% Analysis Internal & Scoring
    DTW --> SCORE
    SMO --> SCORE
    COMP --> SCORE
    ASYM --> SCORE
    VLOSS --> SCORE

    %% Dual State Cues to LLM / Scoring
    USTATE --> LLM
    BodyState --> LLM
    SCORE --> LLM

    %% Intelligence Layer
    UPRO --> PERS
    UPRO --> SCORE
    LLM --> RAG
    LLM --> SAFETY
    LLM --> PERS
    RAG --> SAFETY
    LLM --> VOICE

    %% Intelligence to Output
    SAFETY --> VIS
    SAFETY --> AUD
    SAFETY --> DASH
    PERS --> VIS
    PERS --> AUD
    VOICE --> AUD
    SCORE --> DASH
    SCORE --> REPORT

    %% Output to Data
    VIS --> LOG
    AUD --> LOG
    DASH --> LOG
    REPORT --> LOG
    LOG --> USERS
    EXERC --> REF
    KB --> RAG
