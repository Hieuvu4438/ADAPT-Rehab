#!/usr/bin/env python3
"""
End-to-End Pipeline Runner for ADAPT-Rehab.

Processes a single rehab/exercise video and logs the output of every stage in detail.
Outputs all files (logs, scores, visual frames) to a target subdirectory under
/home/haipd/ADAPT-Rehab/evaluation/output/.
"""

import os
import sys
import time
import json
import numpy as np
import cv2

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# Core imports
from core.pose3d.base import create_estimator
from core.kinematics_quaternion import QuaternionKinematics
from core.smoothness import SmoothnessAnalyzer
from core.dtw_constrained import constrained_dtw

# Module imports
from modules.perception.openface_analyzer import OpenFaceAnalyzer
from modules.perception.facial_state_detector import FacialState
from modules.analysis.body_state_detector import BodyStateDetector
from modules.compensation import CompensationDetector
from modules.fatigue import FatigueAnalyzer
from modules.scoring_v2 import EnhancedScorer
from modules.intelligence.coach.rehab_coach import RehabCoach
from modules.intelligence.llm.client import LLMClient
from modules.intelligence.llm.safety import SafetyGuardrails
from modules.intelligence.voice.tts import TextToSpeech


def run_pipeline(video_path: str, output_dir: str):
    """Runs the full pipeline end-to-end and logs details at every step."""
    os.makedirs(output_dir, exist_ok=True)
    log_file_path = os.path.join(output_dir, "e2e_run.log")
    
    # Simple logger that prints to stdout and writes to a file
    class Logger:
        def __init__(self, filepath):
            self.terminal = sys.stdout
            self.log = open(filepath, 'w', encoding='utf-8')
        def info(self, message):
            self.terminal.write(message + '\n')
            self.log.write(message + '\n')
            self.log.flush()
        def close(self):
            self.log.close()

    logger = Logger(log_file_path)

    logger.info("=" * 80)
    logger.info("ADAPT-REHAB END-TO-END PIPELINE RUN")
    logger.info("=" * 80)
    logger.info(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Input Video: {video_path}")
    logger.info(f"Output Directory: {output_dir}")
    logger.info("-" * 80)

    # -------------------------------------------------------------------------
    # STAGE 1: Video Input Reading
    # -------------------------------------------------------------------------
    logger.info("\n>>> STAGE 1: VIDEO INPUT PROCESSING")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.info(f"[ERROR] Cannot open input video: {video_path}")
        return
    
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    logger.info(f"  - Algorithm/Reader: OpenCV VideoCapture (cv2.VideoCapture)")
    logger.info(f"  - Video Resolution: {w}x{h} pixels")
    logger.info(f"  - Frame Rate: {fps:.2f} FPS")
    logger.info(f"  - Total Frames in Video: {total_frames}")

    # -------------------------------------------------------------------------
    # STAGE 2: Perception Module Initialization
    # -------------------------------------------------------------------------
    logger.info("\n>>> STAGE 2: INITIALIZING PERCEPTION MODELS")
    
    logger.info("  1. 3D Pose Estimator Model:")
    logger.info("     - Model: RTMW3D-L (MMPose 2024) — auto-fallback to MediaPipe")
    logger.info("     - Keypoints: 133 whole-body joints (body, hands, face)")
    pose_estimator = None
    for backend in ["rtmw3d", "mediapipe_fallback"]:
        try:
            candidate = create_estimator(backend)
            if candidate.initialize():
                pose_estimator = candidate
                logger.info(f"     - Active backend: {backend}")
                logger.info(f"     - Initialization Status: SUCCESS")
                break
            else:
                logger.info(f"     - {backend} init returned False")
        except Exception as e:
            logger.info(f"     - {backend} failed: {e}")
    if pose_estimator is None:
        logger.info("     - Initialization Status: FAILED (no backend available)")
    pose_init = pose_estimator is not None

    logger.info("  2. Face & Emotion Analysis Model:")
    logger.info("     - Models: OpenFace 3.0 (AU + Emotion + Gaze)")
    logger.info("     - Target AUs: AU1, AU2, AU4, AU6, AU9, AU12, AU25, AU26 (8 AUs)")
    logger.info("     - State Detection: PSPI (pain), PERCLOS (fatigue), Engagement (boredom)")
    logger.info("     - Emotions: 8 classes (AffectNet: neutral, happy, sad, surprise, fear, disgust, anger, contempt)")
    openface_analyzer = OpenFaceAnalyzer(device="cpu")
    face_init = openface_analyzer.initialize()
    logger.info(f"     - Initialization Status: {'SUCCESS' if face_init else 'FAILED'}")

    # -------------------------------------------------------------------------
    # STAGE 3: Analysis & Intelligence Initialization
    # -------------------------------------------------------------------------
    logger.info("\n>>> STAGE 3: INITIALIZING ANALYSIS & INTELLIGENCE")
    
    logger.info("  1. Biomechanical Kinematics & Smoothness:")
    logger.info("     - Algorithm: Quaternion Joint Calculations (no gimbal lock)")
    logger.info("     - Algorithm: SPARC (Spectral Arc Length) duration-invariant smoothness")
    logger.info("     - Algorithm: Log-Dimensionless Jerk (LDLJ) for consistency")
    smoothness_analyzer = SmoothnessAnalyzer(fs=fps)

    logger.info("  2. Compensation & Fatigue Models:")
    logger.info("     - Model: Geometric rule-based compensation detection")
    logger.info("     - Model: Multi-indicator fatigue detection (ROM reduction, jerk ratio, speed)")
    compensation_detector = CompensationDetector()
    logger.info(f"     - Compensation Backend: Geometric (threshold-based)")
    fatigue_analyzer = FatigueAnalyzer()

    logger.info("  3. Scoring Engine:")
    logger.info("     - Model: EnhancedScorer (6-dimension weighted clinical scoring)")
    scorer = EnhancedScorer()
    scorer.start_session("e2e_evaluation")

    logger.info("  4. Generative AI Coach (LLM & Voice):")
    logger.info("     - Models: RAG + GPT-4o / Claude API + Safety Guardrails")
    logger.info("     - Model: Edge-TTS (Vietnamese Voice Synthesis)")
    llm = LLMClient(provider="openai", api_key=os.environ.get("OPENAI_API_KEY"))
    llm_init = llm.initialize()
    safety = SafetyGuardrails()
    coach = RehabCoach(llm_client=llm if llm_init else None, safety=safety)
    logger.info(f"     - LLM API Status: {'ACTIVE' if llm_init else 'INACTIVE (Running rule-based local fallback)'}")

    # -------------------------------------------------------------------------
    # STAGE 4: Processing Frames (End-to-End Pipeline Loop)
    # -------------------------------------------------------------------------
    logger.info("\n>>> STAGE 4: RUNNING FRAME-BY-FRAME PIPELINE")
    
    # We will process up to 60 frames (~2 seconds of exercise) for demonstration
    frames_to_process = min(60, total_frames)
    pose_history = []
    angles_history = []
    timestamps = []
    
    start_time = time.time()
    
    for idx in range(frames_to_process):
        ret, frame = cap.read()
        if not ret:
            break
            
        ts_ms = int(idx * (1000.0 / fps))
        ts_s = idx / fps
        
        # 1. Perception
        pose_res = pose_estimator.estimate(frame, ts_ms)
        face_res = openface_analyzer.analyze(frame, ts_ms) if face_init else None
        
        # 2. Accumulate history
        if pose_res.is_valid and pose_res.keypoints_3d is not None:
            pose_history.append(pose_res.keypoints_3d)
            angles_history.append(pose_res.joint_angles_quaternion)
            timestamps.append(ts_s)
            
        # Logging frame metrics periodically
        if idx % 15 == 0:
            logger.info(f"  Frame {idx:02d}/{frames_to_process:02d}:")
            if pose_res.is_valid:
                logger.info(f"    - Pose 3D: DETECTED ({len(pose_res.keypoints_3d)} keypoints)")
                if pose_res.joint_angles_quaternion:
                    j_str = ", ".join([f"{k}: {v:.1f}°" for k, v in list(pose_res.joint_angles_quaternion.items())[:3]])
                    logger.info(f"    - Joint Angles (Quaternion): {j_str}")
            else:
                logger.info(f"    - Pose 3D: NOT DETECTED (Reason: {pose_res.error_message})")
                
            if face_res and face_res.is_valid:
                state_info = ""
                if face_res.state_result:
                    sr = face_res.state_result
                    state_info = f", State={sr.state.value} (Pain={sr.pain_score:.2f}, Fatigue={sr.fatigue_score:.2f})"
                logger.info(f"    - Face: Emotion={face_res.emotion_label} (conf={face_res.emotion_confidence:.1%}){state_info}")

    cap.release()
    pose_estimator.close()
    
    elapsed = time.time() - start_time
    logger.info(f"\nProcessing Completed. Total Elapsed Time: {elapsed:.2f}s ({frames_to_process / elapsed:.1f} FPS)")

    # -------------------------------------------------------------------------
    # STAGE 5: Scoring and Clinical Analysis
    # -------------------------------------------------------------------------
    logger.info("\n>>> STAGE 5: SCORING & CLINICAL ANALYSIS")
    
    if len(angles_history) < 5:
        logger.info("[ERROR] Not enough frames with valid poses detected to score the exercise.")
        return
        
    # Extract left shoulder angle for score calculation
    l_shoulder_angles = np.array([a.get("left_shoulder", 0) for a in angles_history])
    ts_arr = np.array(timestamps)
    target_angle = float(np.max(l_shoulder_angles)) * 1.1 if len(l_shoulder_angles) > 0 else 150.0

    # 1. Smoothness Calculation (SPARC)
    smooth_res = smoothness_analyzer.analyze(l_shoulder_angles, ts_arr)
    logger.info(f"  - Smoothness Algorithm: SPARC (Spectral Arc Length)")
    logger.info(f"    * SPARC Score: {smooth_res.sparc:.4f} (More negative = jerky, typical: [-2, 0])")
    logger.info(f"    * Log-Dimensionless Jerk (LDLJ): {smooth_res.ldjl:.4f}")
    logger.info(f"    * Number of Velocity Peaks (NVP): {smooth_res.nvp}")

    # 2. Dynamic Time Warping (DTW) vs Reference
    # Mock a reference sequence
    ref_angles = np.linspace(30, 140, len(l_shoulder_angles))
    dtw_dist, dtw_path = constrained_dtw(l_shoulder_angles, ref_angles, window_percent=0.1)
    logger.info(f"  - Temporal Sync Algorithm: Constrained DTW (Sakoe-Chiba band constraint)")
    logger.info(f"    * Warping Distance: {dtw_dist:.3f}")
    logger.info(f"    * Alignment Path Length: {len(dtw_path)}")

    # 3. Compensation Detection (analyze full pose history at once)
    comp_res = compensation_detector.analyze(pose_history)
    logger.info(f"  - Compensation Detection:")
    logger.info(f"    * Compensation Score: {comp_res.score:.1f}/100 "
                f"(100 = no compensation)")
    logger.info(f"    * Shoulder Height Diff (avg): {comp_res.shoulder_diff_avg:.3f}")
    logger.info(f"    * Trunk Tilt (avg): {comp_res.trunk_tilt_avg:.1f}°")
    logger.info(f"    * Hip Diff (avg): {comp_res.hip_diff_avg:.3f}")
    if comp_res.detected_types:
        logger.info(f"    * Detected: {', '.join(comp_res.detected_types)}")
    
    # 4. Final Scoring Call
    score = scorer.score_rep(
        angles=l_shoulder_angles,
        timestamps=ts_arr,
        target_angle=target_angle,
        pose_sequence=pose_history
    )
    
    logger.info(f"  - Enhanced Scoring Engine Output (v2, 6 Dimensions):")
    logger.info(f"    * ROM Score (25%):        {score.rom_score:.1f}/100")
    logger.info(f"    * Stability Score (15%):  {score.stability_score:.1f}/100")
    logger.info(f"    * Flow Score (20%):       {score.flow_score:.1f}/100")
    logger.info(f"    * Symmetry Score (15%):   {score.symmetry_score:.1f}/100")
    logger.info(f"    * Compensation Score (15%):{score.compensation_score:.1f}/100")
    logger.info(f"    * Smoothness Score (10%):  {score.smoothness_score:.1f}/100")
    logger.info(f"    =========================================")
    logger.info(f"    * FINAL TOTAL SCORE:      {score.total_score:.1f}/100")
    logger.info(f"    * Fatigue Assessment:     {score.fatigue.name}")

    # -------------------------------------------------------------------------
    # STAGE 6: Generative Feedback & Voice Guidance
    # -------------------------------------------------------------------------
    logger.info("\n>>> STAGE 6: GENERATIVE COACHING FEEDBACK")
    
    # Run RehabCoach to generate the friendly Vietnamese advice
    feedback = coach.update(
        current_angle=float(np.mean(l_shoulder_angles)),
        target_angle=target_angle,
        phase="ECCENTRIC",
        rom_score=score.rom_score,
        stability_score=score.stability_score,
        fatigue_level=score.fatigue.name,
        pain_level="NONE",
        rep_count=1
    )
    
    logger.info(f"  - AI Rehab Coach Output:")
    logger.info(f"    * Coach Prompt Type: {feedback.feedback_type}")
    logger.info(f"    * Vietnamese Feedback Text:")
    logger.info(f"      \"{feedback.message}\"")
    logger.info(f"    * Text-To-Speech Script (Edge-TTS HoaiMy Neural):")
    logger.info(f"      \"{feedback.voice_text}\"")

    # Save outputs to JSON files
    run_summary = {
        "metadata": {
            "video": os.path.basename(video_path),
            "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
            "duration_s": float(total_frames / fps)
        },
        "perception": {
            "pose_model": "RTMW3D-L",
            "face_model": "MediaPipe Face Mesh",
            "face_detection_successful": face_init
        },
        "kinematics": {
            "primary_angle_mean": float(np.mean(l_shoulder_angles)),
            "sparc": float(smooth_res.sparc),
            "ldjl": float(smooth_res.ldjl),
            "nvp": int(smooth_res.nvp),
            "dtw_distance": float(dtw_dist)
        },
        "scoring": score.to_dict(),
        "ai_coach": {
            "feedback_text": feedback.message,
            "voice_script": feedback.voice_text,
            "type": feedback.feedback_type
        }
    }
    
    summary_path = os.path.join(output_dir, "run_summary.json")
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(run_summary, f, indent=2, ensure_ascii=False)
        
    logger.info("-" * 80)
    logger.info(f"Saved run summary JSON: {summary_path}")
    logger.info(f"Saved log file: {log_file_path}")
    logger.info("=" * 80)
    
    logger.close()


if __name__ == "__main__":
    video = os.path.join(PROJECT_ROOT, "data", "yoga_datasets", "Yoga_Vid_Collected", "Abhay_Bhujangasana.mp4")
    out = os.path.join(PROJECT_ROOT, "evaluation", "output", "e2e_run_results")
    run_pipeline(video, out)
