# Citation Audit Report — ADAPT-Rehab Paper

**Audit date:** 2026-06-30
**Scope:** `paper/` directory — all `.tex` files + both `.bib` files
**Method:** Extracted every `\cite`-family command from all 14 `.tex` files; cross-referenced against `@entry` keys defined in `references.bib` (the only bib in the build) and `references_posture_correction.bib` (a separate file, *not* in the build). Static analysis only — LaTeX could not be compiled in this environment (see §5).

---

## 1. Headline findings

| Metric | Count | Status |
|---|---|---|
| Unique citation keys used in the paper | **41** | — |
| Keys defined in `references.bib` (in the build) | **41** | — |
| Keys defined in `references_posture_correction.bib` (NOT in the build) | 4 | orphan file |
| **Cited but undefined** (will break compile) | **0** | ✅ **FIXED** (`ref_uco` entry added 2026-06-30) |
| Defined in `references.bib` but never cited (orphan) | **0** | ✅ clean |
| Duplicate keys across the two bib files | **0** | ✅ clean |

**Bottom line:** All 41 citation keys used in the paper now resolve to a `@entry` in `references.bib`, and every defined entry is actually cited — no undefined citations, no dead entries. The previously-missing `ref_uco` entry (cited in `experiments.tex:10`) was added with bibliographic metadata verified against the publisher PDF.

---

## 2. `ref_uco` — RESOLVED (2026-06-30)

- **Citation site:** `paper/sections/experiments.tex:10`
  ```latex
  \item \textbf{UCO Physical Rehabilitation}~\cite{ref_uco}: 393 recordings of 16 exercises ...
  ```
- **Source:** Aguilar-Ortega et al. (2023), *"UCO Physical Rehabilitation: New Dataset and Study of Human Pose Estimation Methods on Physical Rehabilitation Exercises"*, **Sensors** 2023, 23(21), 8862. MDPI, SCI/PubMed indexed.
- **Action taken:** Added a complete `@article{ref_uco, ...}` entry to `paper/references.bib` (Datasets block, after `ref_kimore`), with all 9 authors and DOI `10.3390/s23218862` verified directly from the publisher PDF (`data/dataset_papers/Aguilar-Ortega et al. - 2023 - UCO Physical Rehabilitation....pdf`, page 1 citation block).

Entry as written:
```bibtex
@article{ref_uco,
    author = {Aguilar-Ortega, Rafael and Berral-Soler, Rafael and Jim{\'e}nez-Velasco, Isabel and Romero-Ram{\'i}rez, Francisco J. and Garc{\'i}a-Mar{\'i}n, Manuel and Zafra-Palma, Jorge and Mu{\~n}oz-Salinas, Rafael and Medina-Carnicer, Rafael and Mar{\'i}n-Jim{\'e}nez, Manuel J.},
    title = {UCO Physical Rehabilitation: New Dataset and Study of Human Pose Estimation Methods on Physical Rehabilitation Exercises},
    journal = {Sensors},
    volume = {23},
    number = {21},
    pages = {8862},
    year = {2023},
    publisher = {MDPI},
    doi = {10.3390/s23218862},
    url = {https://www.mdpi.com/1424-8220/23/21/8862}
}
```

---

## 3. Orphaned/stray bibliography files (housekeeping)

These two files in `paper/` are **not included in the build** (`main.tex:111` is `\bibliography{references}` only) and their keys are cited **nowhere**. Their content was already merged into `references.bib` under the `ref_*` convention during the posture-correction integration.

| File | Keys | Status | Recommendation |
|---|---|---|---|
| `paper/references_posture_correction.bib` | `chen2020posetrainer`, `pinero2021posture`, `simoes2024mediapipe`, `turner2024mobile` | Duplicate-content, uncited, not in build | **Safe to delete** — content is in `references.bib` as `ref_pose_trainer` / `ref_pinero` / `ref_simoes` / `ref_turner` |
| `paper/references.txt` | 76-line free-form bib dump (scratch from pre-curation) | Untracked, not in build | **Safe to delete** — appears to be an earlier working dump |

Neither file currently breaks the build, but keeping them invites future confusion (someone running `\bibliography{references,references_posture_correction}` would create silent duplicate-reference rows since the *content* overlaps even though the *keys* differ).

---

## 4. Complete list of citations currently used (41 keys)

All keys below are cited at least once in the paper and resolve to a `@entry` in `references.bib` (except `ref_uco`). Grouped by topic, with `file:line` of every citation site.

### 4.1 Posture / exercise-correction (the 4 newly curated)
| Key | Reference | Cited at |
|---|---|---|
| `ref_pose_trainer` | Chen & Yang 2020 — *Pose Trainer* (arXiv:2006.11718) | introduction.tex L7; related_work.tex L6 |
| `ref_pinero` | Piñero-Fuentes et al. 2021 — *Sensors* (MDPI) | related_work.tex L6 |
| `ref_simoes` | Simoes et al. 2024 — *Procedia CS* (ICTH) | related_work.tex L6 |
| `ref_turner` | Turner et al. 2024 — *VISIGRAPP/Springer CCIS* | related_work.tex L6 |

### 4.2 Vietnam / system context (2)
| Key | Reference | Cited at |
|---|---|---|
| `ref_vietnam_aging` | National Action Plan for the Elderly 2021–2025 | introduction.tex L5 |
| `ref_vietnam_health` | WHO Vietnam Rehab Workforce Profile | introduction.tex L5 |

### 4.3 3D pose estimation (7)
| Key | Reference | Cited at |
|---|---|---|
| `ref_rtmw3d` | RTMW3D whole-body pose (arXiv:2407.08791) | abstract L3; introduction L14; methodology L12, L23; experiments L39, L51, L57; discussion L8; related_work L9 |
| `ref_mmpose` | OpenMMLab MMPose toolbox | abstract L3; introduction L14; methodology L23; related_work L9 |
| `ref_metrabs` | MeTRAbs metric-scale body shape (WACV 2021) | related_work L9 |
| `ref_motionbert` | MotionBERT (ICCV 2023) | experiments L49; related_work L9 |
| `ref_motionagformer` | MotionAGFormer (WACV 2024) | experiments L50; related_work L9 |
| `ref_blazepose` | MediaPipe BlazePose (arXiv:2006.10204) | experiments L53; related_work L9 |
| `ref_mediapipe_accuracy` | Lugaresi et al., MediaPipe framework | related_work L9 |

### 4.4 Pain / emotion / fatigue / engagement (8)
| Key | Reference | Cited at |
|---|---|---|
| `ref_pspi` | Prkachin & Solomon 2008 — PSPI pain index | abstract L3; introduction L16; methodology L80; related_work L12 |
| `ref_jaanet` | JAA-Net AU detection (ECCV 2018) | related_work L12 |
| `ref_me_graphau` | ME-GraphAU (IEEE FG 2023) | related_work L12 |
| `ref_rafdb` | RAF-DB emotion dataset (CVPR 2017) | related_work L12 |
| `ref_elderly_faces` | Dibeklioğlu — facial expressions of elderly | discussion L49; related_work L12 |
| `ref_perclos` | Wierwille 1994 — PERCLOS | abstract L3; introduction L16; methodology L92; related_work L14 |
| `ref_ear` | Eye Aspect Ratio — blink detection | methodology L85; related_work L14 |
| `ref_engagement` | Whitehill et al. 2014 — engagement | abstract L3; introduction L16; methodology L94; related_work L14 |

### 4.5 LLM / healthcare / safety (4)
| Key | Reference | Cited at |
|---|---|---|
| `ref_whisper` | Radford et al. — Whisper (ICML 2023) | introduction L18 |
| `ref_rehab_chatbot` | Rehabilitation chatbot on GPT-4o | related_work L17 |
| `ref_rag` | Lewis et al. — RAG (NeurIPS 2020) | related_work L17 |
| `ref_llm_rehab` | ChatGPT in Physical & Rehab Medicine | related_work L17 |
| `ref_llm_medical_safety` | LLMs in Medicine — comprehensive review | related_work L17 |
| `ref_boredom` | D'Mello & Graesser 2010 — affect detection | related_work L14 |
| `ref_aigc_fitness` | AIGC multimodal fitness feedback | introduction L7; related_work L6 |

### 4.6 Biomechanics / kinematics (8)
| Key | Reference | Cited at |
|---|---|---|
| `ref_sparc` | Balasubramanian et al. 2012 — SPARC | abstract L3; introduction L20; methodology L59; related_work L20 |
| `ref_quaternion_angles` | Euler vs. quaternions for joint angles | methodology L45; related_work L20 |
| `ref_isb` | ISB joint coordinate system (Wu et al. 2005) | methodology L53; related_work L20 |
| `ref_ldlj` | Rohrer et al. — movement smoothness / stroke | methodology L68; related_work L20 |
| `ref_dtw` | Müller — DTW (Springer, 2007) | related_work L20 |
| `ref_dtw_rehab` | Constrained DTW for rehab (IEEE BHI 2023) | related_work L20 |
| `ref_icc` | Koo & Li 2016 — ICC reliability | methodology L154 |
| `ref_hybrik` | HybrIK inverse kinematics (CVPR 2021) | experiments L52 |

### 4.7 Datasets & commercial systems (5)
| Key | Reference | Cited at |
|---|---|---|
| `ref_kimore` | Capecci et al. 2019 — KIMORE dataset | experiments L9 |
| `ref_uco` | Aguilar-Ortega et al. 2023 — UCO Physical Rehab | experiments L10 — ✅ **entry added** |
| `ref_uiprmd` | UI-PRMD rehab movement data | experiments L11; methodology L150 |
| `ref_sword_health` | Sword Health (commercial) | related_work L6 |
| `ref_pose_rehab_review` | Islam et al. 2024 — vision pose rehab survey | related_work L6 |

**Total cited keys: 41 / Resolved in `references.bib`: 41 / Undefined: 0.**

---

## 5. Notable δ from last commit (`dcc2cb0`, 2026-06-17)

Git diff of `paper/references.bib` vs. HEAD (working tree):

- **Added** (this session, posture-correction integration): `ref_pinero`, `ref_simoes`, `ref_turner`
- **Corrected**: `ref_pose_trainer` author field was `Tang, J. and others` → now `Chen, Steven and Yang, Richard R.` (+ eprint/url fields)
- **Removed**: none
- The `ref_uco` citation was added to `experiments.tex` but the matching bib entry was **never created** — this is the one outstanding fix.

---

## 6. Compile-blocker note

A live `pdflatex`/`bibtex` compile could not be run in this environment:
- `pdflatex` / `bibtex` / `latexmk` are not installed on PATH.
- `tectonic` is present at `/home/haipd/.local/bin/tectonic` but requires GLIBC ≥ 2.36; this host has older glibc, so the binary fails to start.
- Docker is installed but the daemon socket needs root (`permission denied`).

All findings above are from static parsing of `.tex` and `.bib`. The `ref_uco` undefined-citation issue is certain from static analysis; a live compile would surface it as a `Citation 'ref_uco' undefined` warning and render an empty `[?]` in the bibliography. Fix §2 before compiling.
