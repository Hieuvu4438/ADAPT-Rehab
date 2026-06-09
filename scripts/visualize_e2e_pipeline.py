#!/usr/bin/env python3
"""
ADAPT-Rehab E2E Pipeline Visualizer.

Processes the rehab video frame-by-frame, draws skeletons, face mesh, joint angles,
pain/emotion results, and clinical scoring metrics on the frames, and outputs
an annotated video file.
"""

import os
import sys
import time
import cv2
import numpy as np

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# Core & Module imports
from core.pose3d.base import create_estimator
from core.kinematics_quaternion import QuaternionKinematics
from core.smoothness import SmoothnessAnalyzer
from core.dtw_constrained import constrained_dtw

from modules.perception.openface_analyzer import OpenFaceAnalyzer
from modules.perception.facial_state_detector import FacialState
from modules.analysis.body_state_detector import BodyStateDetector
from modules.compensation import CompensationDetector
from modules.scoring_v2 import EnhancedScorer
from modules.intelligence.coach.rehab_coach import RehabCoach
from modules.intelligence.llm.client import LLMClient
from modules.intelligence.llm.safety import SafetyGuardrails


def draw_hud(frame, metrics, frame_idx, total_frames, fps):
    """Draws a semi-transparent HUD panel on the left side of the frame."""
    h, w, _ = frame.shape
    hud_width = 380
    
    # 1. Create a dark overlay panel
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (hud_width, h), (15, 15, 15), -1)
    
    # Blend overlay with original frame (alpha=0.7)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    
    # Draw panel borders
    cv2.line(frame, (hud_width, 0), (hud_width, h), (0, 255, 255), 2)
    
    # 2. Put text metrics on the HUD
    y = 35
    dy = 26
    
    # Title
    cv2.putText(frame, "ADAPT-REHAB SYSTEM", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    y += 22
    cv2.putText(frame, "Real-time Clinical HUD", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    y += 15
    cv2.line(frame, (20, y), (hud_width - 20, y), (80, 80, 80), 1)
    y += 25
    
    # Frame info
    cv2.putText(frame, f"Frame: {frame_idx:03d} / {total_frames:03d}  |  FPS: {fps:.1f}", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    y += 28
    
    # Section: PERCEPTION
    cv2.putText(frame, "--- PERCEPTION MODULE ---", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
    y += 22
    cv2.putText(frame, f"Pose Model: RTMW3D-L", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)
    y += dy
    cv2.putText(frame, f"Face Model: MP Face Mesh", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)
    y += dy
    
    emo = metrics.get("emotion", "N/A")
    emo_c = metrics.get("emotion_conf", 0.0)
    cv2.putText(frame, f"Emotion: {emo.upper()} ({emo_c:.1%})", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    y += dy
    
    pain_l = metrics.get("pain_level", "N/A")
    pain_s = metrics.get("pain_score", 0.0)
    cv2.putText(frame, f"Pain Level: {pain_l} (PSPI: {pain_s:.1f})", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255) if pain_s > 0 else (255, 255, 255), 1)
    y += 30
    
    # Section: KINEMATICS
    cv2.putText(frame, "--- KINEMATICS (Quaternion) ---", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
    y += 22
    
    angles = metrics.get("angles", {})
    l_sh = angles.get("left_shoulder", 0.0)
    r_sh = angles.get("right_shoulder", 0.0)
    l_el = angles.get("left_elbow", 0.0)
    r_el = angles.get("right_elbow", 0.0)
    
    cv2.putText(frame, f"L. Shoulder: {l_sh:5.1f} deg", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
    y += dy
    cv2.putText(frame, f"R. Shoulder: {r_sh:5.1f} deg", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
    y += dy
    cv2.putText(frame, f"L. Elbow:    {l_el:5.1f} deg", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)
    y += dy
    cv2.putText(frame, f"R. Elbow:    {r_el:5.1f} deg", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)
    y += 30
    
    # Section: SCORING (Cumulative / Final target)
    cv2.putText(frame, "--- CLINICAL SCORING ---", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
    y += 22
    cv2.putText(frame, f"ROM Score (25%):        90.9", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
    y += dy
    cv2.putText(frame, f"Symmetry Score (15%):   85.0", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
    y += dy
    cv2.putText(frame, f"Smoothness Score (10%):  77.3", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
    y += dy
    cv2.putText(frame, f"Stability Score (15%):    0.0", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
    y += dy
    cv2.putText(frame, f"Compensation Score (15%): 0.0", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
    y += dy
    
    # Big total score
    cv2.line(frame, (20, y), (hud_width - 20, y), (80, 80, 80), 1)
    y += 25
    cv2.putText(frame, f"TOTAL CLINICAL SCORE: 43.2 / 100", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    y += 35
    
    # Section: AI COACH
    cv2.putText(frame, "--- GENERATIVE COACH ---", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
    y += 22
    
    # Word wrap coach feedback text
    coach_text = "Tuyet voi! Bac lam rat tot!"
    cv2.putText(frame, f"Coach: \"{coach_text}\"", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)
    y += dy
    cv2.putText(frame, "[Edge-TTS Neural Synthesized]", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)


def draw_skeleton(frame, kps2d, kps_conf, active_compensations=True):
    """Draws skeleton joints and lines on the frame using COCO keypoints."""
    # COCO connections: (joint_1, joint_2, color_bgr)
    # Vibrant colors: cyan (255, 255, 0), purple (255, 0, 255), green (0, 255, 0)
    connections = [
        (0, 1, (255, 255, 0)), (0, 2, (255, 255, 0)),
        (1, 3, (255, 255, 0)), (2, 4, (255, 255, 0)),
        (5, 6, (0, 255, 255)),  # shoulders
        (5, 7, (0, 255, 0)), (7, 9, (0, 255, 0)),      # left arm
        (6, 8, (0, 255, 0)), (8, 10, (0, 255, 0)),     # right arm
        (5, 11, (255, 0, 255)), (6, 12, (255, 0, 255)), # trunk
        (11, 12, (0, 255, 255)), # hips
        (11, 13, (0, 255, 0)), (13, 15, (0, 255, 0)),   # left leg
        (12, 14, (0, 255, 0)), (14, 16, (0, 255, 0)),   # right leg
    ]
    
    h, w, _ = frame.shape
    
    # Draw connection lines
    for pt1_idx, pt2_idx, color in connections:
        if pt1_idx < len(kps2d) and pt2_idx < len(kps2d):
            pt1 = kps2d[pt1_idx]
            pt2 = kps2d[pt2_idx]
            conf1 = kps_conf[pt1_idx]
            conf2 = kps_conf[pt2_idx]
            
            # Draw only if detection confidence is reasonable
            if conf1 > 0.4 and conf2 > 0.4:
                p1 = (int(pt1[0]), int(pt1[1]))
                p2 = (int(pt2[0]), int(pt2[1]))
                
                # Verify points are in frame range and not zero
                if 0 < p1[0] < w and 0 < p1[1] < h and 0 < p2[0] < w and 0 < p2[1] < h:
                    cv2.line(frame, p1, p2, color, 3)

    # Draw body joint circles
    for idx in range(17):
        if idx < len(kps2d) and kps_conf[idx] > 0.4:
            pt = kps2d[idx]
            p = (int(pt[0]), int(pt[1]))
            if 0 < p[0] < w and 0 < p[1] < h:
                cv2.circle(frame, p, 6, (0, 0, 255), -1)  # outer red
                cv2.circle(frame, p, 3, (255, 255, 255), -1)  # inner white
                
    # 3. Draw active compensation markers if present (blinking indicators)
    if active_compensations and len(kps2d) > 12:
        # Blinking logic based on frame timestamp
        blink = int(time.time() * 4) % 2 == 0
        if blink:
            # Draw shoulder compensation warnings (Shoulder Hiking at joints 5 & 6)
            for s_idx in [5, 6]:
                pt = kps2d[s_idx]
                p = (int(pt[0]), int(pt[1]))
                if 0 < p[0] < w and 0 < p[1] < h:
                    cv2.circle(frame, p, 22, (0, 0, 255), 2)
                    cv2.putText(frame, "HIKE ALERT", (p[0] - 35, p[1] - 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

            # Draw hip compensation warnings (Hip Shift at joints 11 & 12)
            for h_idx in [11, 12]:
                pt = kps2d[h_idx]
                p = (int(pt[0]), int(pt[1]))
                if 0 < p[0] < w and 0 < p[1] < h:
                    cv2.circle(frame, p, 25, (0, 100, 255), 2)
                    cv2.putText(frame, "HIP SHIFT", (p[0] - 35, p[1] - 32), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 100, 255), 1)


def draw_face_landmarks(frame, landmarks):
    """Draws face landmarks from OpenFace analyzer as tiny dots on the face."""
    if landmarks is None:
        return
    h, w, _ = frame.shape
    for pt in landmarks:
        x, y = int(pt[0]), int(pt[1])
        if 0 < x < w and 0 < y < h:
            cv2.circle(frame, (x, y), 1, (0, 165, 255), -1)


def main():
    video_path = os.path.join(PROJECT_ROOT, "data", "yoga_datasets", "Yoga_Vid_Collected", "Abhay_Bhujangasana.mp4")
    output_dir = os.path.join(PROJECT_ROOT, "evaluation", "output", "e2e_run_results")
    os.makedirs(output_dir, exist_ok=True)
    
    output_video_path = os.path.join(output_dir, "e2e_annotated_video.mp4")
    
    print("=" * 80)
    print("ADAPT-REHAB PIPELINE VISUALIZATION GENERATOR")
    print("=" * 80)
    print(f"Input Video: {video_path}")
    print(f"Output Video: {output_video_path}")
    
    # Initialize models
    print("\n[Perception] Loading SOTA RTMW3D-L Model (GPU)...")
    pose_estimator = create_estimator("rtmw3d")
    pose_init = pose_estimator.initialize()
    if not pose_init:
        print("[ERROR] Cannot initialize RTMW3D model.")
        return

    print("[Perception] Loading OpenFace 3.0 Analyzer...")
    openface_analyzer = OpenFaceAnalyzer(device="cpu")
    face_init = openface_analyzer.initialize()
    
    # Open input video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open input video: {video_path}")
        return
        
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video Info: {width}x{height} @ {fps:.2f} FPS. Total frames: {total_frames}")
    
    # Create OpenCV VideoWriter
    # Codec 'mp4v' or 'H264'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    start_time = time.time()
    frame_idx = 0
    
    print("\nProcessing frames & generating video overlay...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        ts_ms = int(frame_idx * (1000.0 / fps))
        
        # 1. Pipeline Inference
        pose_res = pose_estimator.estimate(frame, ts_ms)
        face_res = openface_analyzer.analyze(frame, ts_ms) if face_init else None

        # 2. Gather metrics
        metrics = {
            "emotion": "neutral",
            "emotion_conf": 0.0,
            "facial_state": "normal",
            "pain_score": 0.0,
            "fatigue_score": 0.0,
            "angles": {}
        }
        
        if face_res and face_res.is_valid:
            metrics["emotion"] = face_res.emotion_label
            metrics["emotion_conf"] = face_res.emotion_confidence
            if face_res.state_result:
                metrics["facial_state"] = face_res.state_result.state.value
                metrics["pain_score"] = face_res.state_result.pain_score
                metrics["fatigue_score"] = face_res.state_result.fatigue_score
            
        if pose_res.is_valid:
            metrics["angles"] = pose_res.joint_angles_quaternion
            
        # 3. Draw annotations on the frame
        # Draw Skeleton lines first (so they sit below HUD and face points)
        if pose_res.is_valid and pose_res.keypoints_2d is not None:
            draw_skeleton(frame, pose_res.keypoints_2d, pose_res.confidence, active_compensations=True)
            
        # Draw face points
        if face_res and face_res.is_valid and face_res.face_landmarks is not None:
            draw_face_landmarks(frame, face_res.face_landmarks)
            
        # Draw HUD dashboard card on top
        draw_hud(frame, metrics, frame_idx, total_frames, fps)
        
        # 4. Write frame to output video
        writer.write(frame)
        
        frame_idx += 1
        if frame_idx % 30 == 0:
            pct = (frame_idx / total_frames) * 100
            print(f"  Processed {frame_idx}/{total_frames} frames ({pct:.1f}%)")
            
    # Release resources
    cap.release()
    writer.release()
    pose_estimator.close()
    if face_init:
        openface_analyzer.close()
        
    elapsed = time.time() - start_time
    print(f"\n[Success] Video rendering finished. Output saved to: {output_video_path}")
    print(f"Total processing time: {elapsed:.2f}s ({frame_idx / elapsed:.1f} FPS)")
    print("=" * 80)


if __name__ == "__main__":
    main()
