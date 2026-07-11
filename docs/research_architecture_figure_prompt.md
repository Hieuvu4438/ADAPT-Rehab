# System Architecture Diagram Generation Prompt (A* Conference Standard)

Copy and paste the prompt below into a generative AI (like Claude 3.5 Sonnet or GPT-4o) to generate a professional SVG or LaTeX TikZ diagram.

---

```markdown
You are an expert academic illustrator and senior designer specializing in creating system architecture diagrams for top-tier computer science/AI conferences (e.g., CVPR, ICCV, NeurIPS, AAAI, CHI).

Your task is to write clean, valid, and fully-responsive SVG code (or LaTeX TikZ code) for a system architecture diagram based on the detailed specifications below. The diagram must look professional, use a clean layout, align perfectly to a grid, have high contrast, and be readable in both color and black-and-white print.

### 1. General Style & Design Principles
- **Aesthetic**: Minimalist, clean, modern academic style. No gradients or shadows unless necessary. Clean shapes with precise borders and solid fills.
- **Grid & Alignment**: The layout consists of 5 columns aligned horizontally (left to right), representing the main pipeline backbone. There are also 3 detailed call-out panels located near their respective columns.
- **Typography**: Use a single clean Sans-Serif font (Helvetica or Arial). Text must not overlap or overflow. Use precise text anchoring (`text-anchor="middle"` or left/right alignment as appropriate).
- **Legibility**: High contrast. Font sizes and weights are specified below.
- **Print-Friendliness**: The color palette must distinguish elements clearly even when photocopied or printed in grayscale.

### 2. Color Palette Specification
Use the exact hex codes below:
- **Primary Accent** (for Perception headers & borders): `#1F4E79` (Dark Blue)
- **Primary Fill** (for Perception column background): `#D9E2F3` (Light Blue)
- **Secondary Accent** (for Safety/Pain components & borders): `#C0392B` (Warm Red)
- **Secondary Fill** (for Safety/Pain components backgrounds): `#FADBD8` (Light Red)
- **Neutral Border** (for standard blocks): `#2C3E50` (Deep Slate Blue)
- **Neutral Fill (Body)** (for standard column backgrounds): `#F4F6F7` (Very Light Gray)
- **Neutral Box Fill** (for block backgrounds inside columns): `#FFFFFF` (Pure White)
- **Main Text**: `#1B2631` (Near Black) for titles and primary labels.
- **Secondary Text**: `#566573` (Medium Gray) for subtitles, parameters, and arrow labels.

### 3. Layout & Geometry (5 Columns)
Draw 5 vertical columns from left to right. Each column should have a colored background (using `Neutral Fill` or `Primary Fill`), a border, and a bold title at the top (14pt, Bold, `#1B2631`).
Inside each column, arrange the blocks vertically with equal spacing:

#### Column 1: SENSING (Input Layer)
- Column Background: `#F4F6F7`, Border: `#566573` (1.25px)
- **B1.1: RGB Camera / Webcam**
  - Text: **RGB Camera / Webcam** (12pt Bold, `#1B2631`) \n 30 FPS, 1080p Video Input (9.5pt Regular, `#566573`)
  - Box Fill: `#FFFFFF`, Box Border: `#2C3E50` (1.25px)

#### Column 2: PERCEPTION (Perception Layer - PRIMARY ACCENT)
- Column Background: `#D9E2F3`, Border: `#1F4E79` (2px)
- **B2.1: RTMW3D-L**
  - Text: **RTMW3D-L** (12pt Bold, `#1B2631`) \n Whole-body 3D Pose, 133 KP (9.5pt Regular, `#566573`)
  - Box Fill: `#FFFFFF`, Box Border: `#1F4E79` (2px)
- **B2.2: OpenFace 3.0**
  - Text: **OpenFace 3.0** (12pt Bold, `#1B2631`) \n 8 Action Units, Emotion, Gaze (9.5pt Regular, `#566573`)
  - Box Fill: `#FFFFFF`, Box Border: `#1F4E79` (2px)

#### Column 3: ANALYSIS (Analysis Layer)
- Column Background: `#F4F6F7`, Border: `#566573` (1.25px)
- **B3.1: Quaternion Kinematics**
  - Text: **Quaternion Kinematics** (12pt Bold, `#1B2631`) \n 8 Bilateral Joints, ISB standard (9.5pt Regular, `#566573`)
  - Box Fill: `#FFFFFF`, Box Border: `#2C3E50` (1.25px)
- **B3.2: Constrained DTW**
  - Text: **Constrained DTW** (12pt Bold, `#1B2631`) \n Sakoe-Chiba Band Alignment (9.5pt Regular, `#566573`)
  - Box Fill: `#FFFFFF`, Box Border: `#2C3E50` (1.25px)
- **B3.3: Compensation & Fatigue**
  - Text: **Compensation & Fatigue** (12pt Bold, `#1B2631`) \n Joint Jitter & Body Drift (9.5pt Regular, `#566573`)
  - Box Fill: `#FFFFFF`, Box Border: `#2C3E50` (1.25px)

#### Column 4: INTELLIGENCE (Decision/Reasoning Layer)
- Column Background: `#F4F6F7`, Border: `#566573` (1.25px)
- **B4.1: LLM Coach**
  - Text: **LLM Coach** (12pt Bold, `#1B2631`) \n GPT-4o / Claude Agent (9.5pt Regular, `#566573`)
  - Box Fill: `#FFFFFF`, Box Border: `#2C3E50` (1.25px)
- **B4.2: Safety Guardrails (SECONDARY ACCENT)**
  - Text: **Safety Guardrails** (12pt Bold, `#1B2631`) \n Contraindication Interceptor (9.5pt Regular, `#566573`)
  - Box Fill: `#FADBD8`, Box Border: `#C0392B` (2px)
- **B4.3: Edge-TTS**
  - Text: **Edge-TTS** (12pt Bold, `#1B2631`) \n vi-VN Voice Synthesizer (9.5pt Regular, `#566573`)
  - Box Fill: `#FFFFFF`, Box Border: `#2C3E50` (1.25px)

#### Column 5: INTERFACE (Output Layer)
- Column Background: `#F4F6F7`, Border: `#566573` (1.25px)
- **B5.1: Visual Overlay**
  - Text: **Visual Overlay** (12pt Bold, `#1B2631`) \n 3D Skeleton + Dynamic ROM Arcs (9.5pt Regular, `#566573`)
  - Box Fill: `#FFFFFF`, Box Border: `#2C3E50` (1.25px)
- **B5.2: Voice Feedback**
  - Text: **Voice Feedback** (12pt Bold, `#1B2631`) \n Natural Audio Instructions (9.5pt Regular, `#566573`)
  - Box Fill: `#FFFFFF`, Box Border: `#2C3E50` (1.25px)
- **B5.3: Real-time HUD (Small Chip/Box)**
  - Text: **Real-time HUD** (12pt Bold, `#1B2631`) \n Rep Counter & Session Scores (9.5pt Regular, `#566573`)
  - Box Fill: `#FFFFFF`, Box Border: `#2C3E50` (1.25px)

---

### 4. Detailed Call-out Panels
To detail complex logic without cluttering the main flow, draw 3 call-out panels around the main columns, each connected to its source block via a thin dashed connection line:

1. **PANEL A: Detailed Face Parser** (Position: above Column 2/Perception)
   - Title: **PANEL A: Detailed Face Parser** (14pt Bold, `#1B2631`)
   - Inside Items (10pt Regular, `#1B2631`):
     - 1. `EAR Calculator` (Eyelid aspect ratio)
     - 2. `PSPI Pain`: AU4 + 2*AU6 + AU9 + 2*AU43
     - 3. `PERCLOS`: % frames eye closure >= 80%
   - Box Style: Fill: `#FFFFFF`, Border: `#1F4E79` (2px, dashed)
   - Connection: A thin dashed line (1px, color `#1F4E79`) from B2.2 (OpenFace 3.0) to Panel A.

2. **PANEL B: Kinematic & Smoothness** (Position: below Column 3/Analysis)
   - Title: **PANEL B: Kinematic & Smoothness** (14pt Bold, `#1B2631`)
   - Inside Items (10pt Regular, `#1B2631`):
     - 1. `Butterworth Filter` (Order 4, Cutoff 6Hz)
     - 2. `SPARC Smoothness`: Arc length of Fourier spectrum
     - 3. `Melax Quaternion`: Vector rotation w/o gimbal lock
   - Box Style: Fill: `#FFFFFF`, Border: `#2C3E50` (1.5px, dashed)
   - Connection: Thin dashed lines (1px, color `#566573`) from both B3.1 and B3.3 to Panel B.

3. **PANEL C: Safety Guardrail Loop** (Position: below Column 4/Intelligence)
   - Title: **PANEL C: Safety Guardrail Loop** (14pt Bold, `#1B2631`)
   - Inside Items (10pt Regular, `#1B2631`):
     - 1. `Context Serializer` (JSON builder)
     - 2. `Safety Guardrail`: Threshold monitor
     - 3. `Emergency Interceptor`: Hijacks output on high Pain/Fatigue
   - Box Style: Fill: `#FADBD8`, Border: `#C0392B` (2px, dashed)
   - Connection: A thin dashed line (1px, color `#C0392B`) from B4.2 (Safety Guardrails) to Panel C.

---

### 5. Connections & Data Flow Map
Draw clear, solid directed arrows (1.5px thick, color `#2C3E50`) with small arrowheads for standard forward paths. Write arrow labels (9pt, Regular, `#1B2631` or `#566573`) centered above or alongside the paths:

- **L1.1**: B1.1 (Camera) -> B2.1 (RTMW3D). Label: `RGB Frame`
- **L1.2**: B1.1 (Camera) -> B2.2 (OpenFace). Label: `RGB Frame`
- **L2.1**: B2.1 (RTMW3D) -> B3.1 (Kinematics). Label: `3D Keypoints`
- **L2.2**: B2.1 (RTMW3D) -> B3.3 (Comp & Fatigue). Label: `3D Keypoints`
- **L2.3**: B2.2 (OpenFace) -> B3.3 (Comp & Fatigue). Label: `AU intensities`
- **L3.1**: B3.1 (Kinematics) -> B3.2 (DTW). Label: `Filtered Angles`
- **L3.2**: B3.1 (Kinematics) -> B3.3 (Comp & Fatigue). Label: `Angular Velocity`
- **L3.3**: B3.2 (DTW) -> B4.1 (LLM Coach). Label: `Similarity Score`
- **L3.4**: B3.3 (Comp & Fatigue) -> B4.2 (Safety). Label: `Pain/Fatigue/Comp`
- **L4.1**: B4.1 (LLM Coach) -> B4.2 (Safety). Label: `Raw Text Output`
- **L4.2**: B4.2 (Safety) -> B4.3 (Edge-TTS). Label: `Safe Text`
- **L4.3**: B4.3 (Edge-TTS) -> B5.2 (Voice). Label: `Audio stream`
- **L4.4**: B4.2 (Safety) -> B5.1 (Visual). Label: `Control signals`
- **L_FB (Feedback Loop)**: An arc/dashed arrow originating from B5.3 (HUD/Scores) and looping backward along the bottom to connect to B4.1 (LLM Coach).
  - Style: Warm Red `#C0392B`, dashed (2px), curved.
  - Label: `Pain/Fatigue Feedback Loop` (9pt, Bold, color `#C0392B`).

### 6. Legend (Bottom-Left Corner)
Provide a neat, small legend (8pt Regular, `#566573`) explaining the key color accents:
- **Blue Border/Fill**: Primary Perception Pipeline
- **Red Border/Fill**: Safety & Pain Guardrails
- **Solid Arrow**: Forward Data Stream
- **Dashed Arrow**: Zoom-in Detail / Feedback Loop

### 7. Output Requirements
Please output the SVG code (fully self-contained, responsive via viewBox, clean grouping structure, proper text tags) inside a single code block. Ensure the diagram is visually balanced, leaves enough white space, and doesn't overlap text.
```
