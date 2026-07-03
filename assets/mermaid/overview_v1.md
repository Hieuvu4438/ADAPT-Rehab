flowchart TD
    subgraph Input["Input Layer"]
        subgraph UserInput["User Input"]
            UCAM[Webcam<br/>30 FPS]
            UMIC[Microphone<br/>Whisper ASR]
            UPRO[User Profile<br/>JSON]
        end

        subgraph RefInput["Reference Input"]
            REF[Reference Video<br/>Exercise Library]
        end
    end

    subgraph Perception["Perception Layer"]
        subgraph UserPerception["User Perception"]
            UPOSE[3D Pose<br/>RTMW3D]
            UFACE[Face<br/>OpenFace 3.0]
            UAU[AU Detection<br/>GNN]
            USTATE[State Detection<br/>AU-based]
        end

        subgraph RefPerception["Reference Perception"]
            RPOSE[3D Pose<br/>RTMW3D]
        end
    end

    subgraph Analysis["Analysis Layer"]
        KIN[Kinematics<br/>Quaternion angles]
        SMO[Smoothness<br/>SPARC + Jerk]
        DTW[DTW<br/>Compare User ↔ Ref]
        COMP[Compensation<br/>Temporal LSTM]
        FAT[Fatigue<br/>PERCLOS + Blink]
        PAIN[Pain<br/>PSPI]
        BORED[Boredom<br/>Engagement Index]
        BODY[Body<br/>ROM + Asymmetry]
    end

    subgraph Intelligence["Intelligence Layer"]
        LLM[LLM<br/>GPT-4o / Claude]
        RAG[RAG<br/>LangChain]
        SAFETY[Safety<br/>Contraindication]
        PERS[Personalization<br/>Profile + History]
        VOICE[Voice<br/>Whisper + Edge-TTS]
    end

    subgraph Output["Output Layer"]
        VIS[Visual<br/>Skeleton overlay]
        AUD[Audio<br/>Voice feedback]
        DASH[Dashboard<br/>Real-time metrics]
        REPORT[Reports<br/>Session summary]
    end

    subgraph Data["Data Layer"]
        LOG[Session Logger<br/>JSON/CSV]
        USERS[User Profiles<br/>JSON]
        EXERC[Exercise Library<br/>Reference videos]
        KB[Clinical KB<br/>RAG]
    end

    UCAM --> UPOSE
    UMIC --> UAU
    UPRO --> USTATE
    REF --> RPOSE

    UPOSE --> KIN
    RPOSE --> DTW
    KIN --> DTW
    DTW --> SMO
    UPOSE --> UAU
    UAU --> USTATE
    USTATE --> FAT
    USTATE --> PAIN
    USTATE --> BORED

    SMO --> COMP
    COMP --> LLM
    FAT --> LLM
    PAIN --> LLM
    BORED --> LLM
    BODY --> LLM

    KIN --> BODY

    LLM --> RAG
    LLM --> SAFETY
    LLM --> PERS
    RAG --> SAFETY

    SAFETY --> VIS
    SAFETY --> AUD
    SAFETY --> DASH
    SAFETY --> REPORT
    PERS --> VIS
    PERS --> AUD

    VIS --> LOG
    AUD --> LOG
    DASH --> LOG
    REPORT --> LOG
    LOG --> USERS
    EXERC --> REF
    KB --> RAG