#!/usr/bin/env python3
"""
ADAPT-Rehab v3.0 - Main Application

Multimodal AI Rehabilitation System for Elderly Vietnamese Users.

Architecture:
    Input → Perception (3D Pose + Face) → Analysis (Kinematics + Scoring)
    → Intelligence (LLM + Voice) → Output (Visual + Audio)

Usage:
    python main_v3.py --source webcam
    python main_v3.py --source data/yoga_datasets/Yoga_Vid_Collected/Abhay_Bhujangasana.mp4

Author: ADAPT-Rehab Team
Version: 3.0.0
"""

import argparse
import sys
import os
import time
from pathlib import Path
from typing import Optional, Dict, List
import numpy as np
import cv2

# Core modules
from core.pose3d.base import create_estimator, PoseEstimator3D, Pose3DResult
from core.kinematics_quaternion import QuaternionKinematics
from core.smoothness import SmoothnessAnalyzer
from core.angle_filter import AngleFilter

# Functional modules
from modules.perception.openface_analyzer import OpenFaceAnalyzer, OpenFaceResult
from modules.perception.facial_state_detector import FacialState, FacialStateResult
from modules.analysis.body_state_detector import BodyStateDetector, BodyState
from modules.compensation import CompensationDetector
from modules.fatigue import FatigueAnalyzer, FatigueLevel
from modules.scoring_v2 import EnhancedScorer, RepScoreV2

# Intelligence
from modules.intelligence.llm.client import LLMClient
from modules.intelligence.llm.safety import SafetyGuardrails
from modules.intelligence.voice.tts import TextToSpeech
from modules.intelligence.coach.rehab_coach import RehabCoach


# RTMW3D body skeleton connections (133 keypoints)
# Body joints: 0-32, Hands: 33-72, Face: 73-130
RTMW3D_CONNECTIONS = [
    # Torso
    (0, 1), (0, 2), (1, 3), (2, 4),  # pelvis to hips, hips to knees
    # Arms
    (5, 7), (7, 9), (6, 8), (8, 10),  # shoulders to elbows to wrists
    # Legs
    (11, 13), (13, 15), (12, 14), (14, 16),  # hips to knees to ankles
    # Spine
    (0, 17),  # pelvis to spine
]


class ADAPTRehabV3:
    """
    ADAPT-Rehab v3.0 Main Application.

    Integrates all layers:
    - Perception: RTMW3D 3D pose + face analysis
    - Analysis: Quaternion kinematics + SPARC + compensation + fatigue
    - Intelligence: LLM + Edge-TTS
    - Output: Visual + audio feedback
    """

    def __init__(self, args):
        self.args = args
        self.running = False

        # Perception
        self.pose_estimator: Optional[PoseEstimator3D] = None
        self.openface_analyzer: Optional[OpenFaceAnalyzer] = None
        self.body_state_detector: Optional[BodyStateDetector] = None

        # Analysis
        self.quaternion_kinematics = QuaternionKinematics()
        self.smoothness_analyzer = SmoothnessAnalyzer()
        self.compensation_detector = CompensationDetector()
        self.fatigue_analyzer = FatigueAnalyzer()
        self.scorer = EnhancedScorer()
        self.angle_filter = AngleFilter(cutoff_hz=6.0, fs=30.0, order=4)

        # Intelligence
        self.coach: Optional[RehabCoach] = None
        self.tts: Optional[TextToSpeech] = None

        # State
        self.frame_count = 0
        self.angles_history: List[Dict[str, float]] = []
        self.timestamps: List[float] = []
        self.pose_history: List[np.ndarray] = []
        self.rep_count = 0
        self.session_start_time = 0.0
        self.last_feedback_time = 0.0
        self.feedback_interval = 10.0  # seconds between LLM feedback

    def initialize(self) -> bool:
        """Initialize all components."""
        print("=" * 60)
        print("ADAPT-Rehab v3.0 - Multimodal AI Rehabilitation")
        print("=" * 60)

        # 1. Pose Estimator
        # Try the requested backend first, then fall back to MediaPipe
        # so the pipeline can run on machines that don't have the full
        # mmpose/mmcv/mmdet stack installed.
        print("\n[1/5] Initializing Pose Estimator...")
        backend = getattr(self.args, "pose_backend", "rtmw3d")
        self.pose_estimator = None
        for candidate in [backend] if backend == "mediapipe_fallback" else [backend, "mediapipe_fallback"]:
            try:
                est = create_estimator(candidate)
                if est.initialize():
                    self.pose_estimator = est
                    if candidate != backend:
                        print(f"  ⚠ Requested backend '{backend}' unavailable; "
                              f"using '{candidate}'")
                    break
                else:
                    print(f"  ⚠ {candidate} initialize() returned False")
            except Exception as e:
                print(f"  ⚠ {candidate} failed: {e}")
        if self.pose_estimator is None:
            print("  ✗ All pose backends failed!")
            return False
        print(f"  ✓ Pose: {self.pose_estimator.model_name}")

        # 2. OpenFace 3.0 Analyzer (AU + Emotion + Gaze + State Detection)
        print("[2/5] Initializing OpenFace 3.0 Analyzer...")
        self.openface_analyzer = OpenFaceAnalyzer(device="cuda" if self.args.use_gpu else "cpu")
        if self.openface_analyzer.initialize():
            print("  ✓ Face: OpenFace 3.0 (AU + Emotion + Gaze)")
        else:
            print("  ⚠ Face detection unavailable")
            self.openface_analyzer = None

        # 3. Body State Detector (RTMW3D behavioral analysis)
        print("[3/5] Initializing Body State Detector...")
        self.body_state_detector = BodyStateDetector(fps=30.0)
        print("  ✓ Body: RTMW3D behavioral state detection")

        # 4. TTS
        print("[4/5] Initializing TTS...")
        try:
            self.tts = TextToSpeech()
            print("  ✓ TTS: Edge-TTS (Vietnamese)")
        except Exception as e:
            print(f"  ⚠ TTS unavailable: {e}")
            self.tts = None

        # 5. LLM Coach
        print("[5/5] Initializing LLM Coach...")
        if self.args.llm_api_key:
            try:
                llm = LLMClient(
                    provider=self.args.llm_provider,
                    api_key=self.args.llm_api_key,
                    model=self.args.llm_model,
                )
                llm.initialize()
                safety = SafetyGuardrails()
                self.coach = RehabCoach(llm_client=llm, safety=safety)
                print(f"  ✓ LLM: {self.args.llm_model}")
            except Exception as e:
                print(f"  ⚠ LLM unavailable: {e}")
        else:
            print("  ⚠ LLM disabled (no API key)")

        # Init scorer
        self.scorer.start_session("v3_session")

        print("\n✓ All components initialized")
        print(f"  Pose: {self.pose_estimator.model_name}")
        print(f"  Face: {'OpenFace 3.0' if self.openface_analyzer else 'Disabled'}")
        print(f"  Body State: {'Enabled' if self.body_state_detector else 'Disabled'}")
        print(f"  LLM: {self.args.llm_model if self.coach else 'Disabled'}")
        print(f"  TTS: {'Enabled' if self.tts else 'Disabled'}")
        print("=" * 60)
        return True

    def run(self):
        """Main application loop."""
        source = self.args.source
        if source == "webcam":
            cap = cv2.VideoCapture(0)
        else:
            cap = cv2.VideoCapture(source)

        if not cap.isOpened():
            print(f"Error: Cannot open {source}")
            return

        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"\nVideo: {w}x{h} @ {fps:.1f} FPS, {total_frames} frames")
        print("Press 'q' to quit, 'r' to reset, 's' to score rep\n")

        self.running = True
        self.session_start_time = time.time()

        while self.running:
            ret, frame = cap.read()
            if not ret:
                if source != "webcam":
                    print("\nVideo ended")
                break

            self.frame_count += 1
            timestamp_ms = int(self.frame_count * (1000 / fps))
            timestamp_s = self.frame_count / fps

            # Process frame
            display, pose_result, face_result = self._process_frame(frame, timestamp_ms, timestamp_s)

            # Periodic LLM feedback
            if self.coach and (timestamp_s - self.last_feedback_time) > self.feedback_interval:
                self._generate_feedback(pose_result, face_result)
                self.last_feedback_time = timestamp_s

            # Display
            cv2.imshow("ADAPT-Rehab v3.0", display)

            # Keyboard
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self.running = False
            elif key == ord('r'):
                self._reset_session()
            elif key == ord('s'):
                self._score_current_rep()

        cap.release()
        cv2.destroyAllWindows()
        self._print_session_summary()
        self._cleanup()

    def _process_frame(self, frame: np.ndarray, timestamp_ms: int, timestamp_s: float):
        """Process a single frame through all pipelines."""
        display = frame.copy()
        pose_result = None
        face_result = None

        # === Pose Estimation ===
        pose_result = self.pose_estimator.estimate(frame, timestamp_ms)

        if pose_result.is_valid and pose_result.keypoints_3d is not None:
            # Draw skeleton
            self._draw_skeleton(display, pose_result)

            # Store for analysis
            self.angles_history.append(pose_result.joint_angles)
            self.timestamps.append(timestamp_s)
            self.pose_history.append(pose_result.keypoints_3d)

            # Draw angles
            self._draw_angles(display, pose_result.joint_angles)

        # === Face Analysis (OpenFace 3.0) ===
        if self.openface_analyzer:
            face_result = self.openface_analyzer.analyze(frame, timestamp_ms)
            if face_result.is_valid:
                self._draw_face_info(display, face_result)

        # === Body State Detection (RTMW3D) ===
        body_state_result = None
        if self.body_state_detector and pose_result and pose_result.is_valid and pose_result.keypoints_3d is not None:
            body_state_result = self.body_state_detector.process_frame(
                pose_result.keypoints_3d, timestamp_s
            )

        # === Draw HUD ===
        self._draw_hud(display, pose_result, face_result)

        return display, pose_result, face_result

    def _draw_skeleton(self, display: np.ndarray, pose_result: Pose3DResult):
        """Draw skeleton overlay on frame."""
        if pose_result.keypoints_2d is None:
            return

        kps = pose_result.keypoints_2d

        # Draw body connections (first 17 joints)
        body_connections = [
            (0, 1), (0, 2), (1, 3), (2, 4),  # pelvis to hips
            (5, 7), (7, 9), (6, 8), (8, 10),  # arms
            (11, 13), (13, 15), (12, 14), (14, 16),  # legs
        ]

        for i, j in body_connections:
            if i < len(kps) and j < len(kps):
                pt1 = (int(kps[i][0]), int(kps[i][1]))
                pt2 = (int(kps[j][0]), int(kps[j][1]))
                cv2.line(display, pt1, pt2, (0, 255, 0), 2)

        # Draw body joints
        for i in range(min(17, len(kps))):
            cv2.circle(display, (int(kps[i][0]), int(kps[i][1])), 4, (0, 0, 255), -1)

    def _draw_angles(self, display: np.ndarray, angles: Dict[str, float]):
        """Draw angle values on frame."""
        y = 30
        for joint, angle in sorted(angles.items()):
            text = f"{joint}: {angle:.1f}°"
            cv2.putText(display, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            y += 20

    def _draw_face_info(self, display: np.ndarray, face_result: OpenFaceResult):
        """Draw face analysis results on frame (OpenFace 3.0)."""
        h, w = display.shape[:2]

        # Draw face bbox
        if face_result.face_bbox is not None:
            x1, y1, x2, y2 = face_result.face_bbox.astype(int)
            cv2.rectangle(display, (x1, y1), (x2, y2), (255, 0, 0), 2)

        # Draw emotion
        text_y = h - 120
        emotion_text = f"Emotion: {face_result.emotion_label} ({face_result.emotion_confidence:.1%})"
        cv2.putText(display, emotion_text, (10, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        # Draw AU-based state
        if face_result.state_result:
            sr = face_result.state_result
            state_text = f"State: {sr.state.value.upper()} ({sr.confidence:.1%})"
            color = self._get_state_color(sr.state)
            cv2.putText(display, state_text, (10, text_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Draw sub-scores
            scores_text = f"P:{sr.pain_score:.2f} F:{sr.fatigue_score:.2f} E:{sr.exhaustion_score:.2f} B:{sr.boredom_score:.2f}"
            cv2.putText(display, scores_text, (10, text_y + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            # Draw PSPI
            pspi_text = f"PSPI: {sr.pspi_raw:.1f} | PERCLOS: {sr.perclos_raw:.1f}%"
            cv2.putText(display, pspi_text, (10, text_y + 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    def _get_state_color(self, state: FacialState):
        """Get display color for facial state."""
        from modules.perception.facial_state_detector import FacialState
        colors = {
            FacialState.NORMAL: (0, 255, 0),
            FacialState.PAIN: (0, 0, 255),
            FacialState.FATIGUE: (0, 165, 255),
            FacialState.EXHAUSTION: (0, 0, 255),
            FacialState.BOREDOM: (128, 128, 128),
        }
        return colors.get(state, (255, 255, 255))

    def _draw_hud(self, display: np.ndarray, pose_result, face_result):
        """Draw heads-up display."""
        h, w = display.shape[:2]

        # FPS and model info
        fps_text = f"Frame: {self.frame_count} | Model: {self.pose_estimator.model_name}"
        cv2.putText(display, fps_text, (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        # Rep count
        rep_text = f"Reps: {self.rep_count}"
        cv2.putText(display, rep_text, (w - 150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # Recent scores
        if self.scorer._rep_scores:
            last = self.scorer._rep_scores[-1]
            score_text = f"Score: {last.total_score:.0f}/100"
            cv2.putText(display, score_text, (w - 200, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)

    def _generate_feedback(self, pose_result, face_result):
        """Generate LLM feedback and speak it."""
        if not self.coach:
            return

        try:
            # Build context
            context = {
                "angles": pose_result.joint_angles if pose_result else {},
                "rep_count": self.rep_count,
                "facial_state": face_result.state_result.state.value if face_result and face_result.state_result else "normal",
                "emotion": face_result.emotion_label if face_result else "neutral",
                "pain_score": face_result.state_result.pain_score if face_result and face_result.state_result else 0.0,
                "fatigue_score": face_result.state_result.fatigue_score if face_result and face_result.state_result else 0.0,
            }

            # Generate feedback
            feedback = self.coach.generate_feedback(context)
            print(f"\n[Coach] {feedback}")

            # Speak feedback
            if self.tts:
                self.tts.synthesize(feedback)

        except Exception as e:
            print(f"[Coach Error] {e}")

    def _score_current_rep(self):
        """Score the current rep from accumulated data."""
        if len(self.angles_history) < 10:
            print("Not enough data for scoring (need >= 10 frames)")
            return

        # Use left shoulder angle as primary, apply Butterworth filter
        angles_raw = np.array([a.get("left_shoulder", 0) for a in self.angles_history])
        angles = self.angle_filter.filter(angles_raw)
        ts = np.array(self.timestamps)
        target = float(np.max(angles)) * 1.1

        score = self.scorer.score_rep(
            angles=angles,
            timestamps=ts,
            target_angle=target,
            pose_sequence=self.pose_history[-30:],
        )

        self.rep_count += 1
        print(f"\n--- Rep {self.rep_count} ---")
        print(f"  ROM: {score.rom_score:.1f}")
        print(f"  Stability: {score.stability_score:.1f}")
        print(f"  Flow: {score.flow_score:.1f}")
        print(f"  Compensation: {score.compensation_score:.1f}")
        print(f"  Smoothness: {score.smoothness_score:.1f}")
        print(f"  TOTAL: {score.total_score:.1f}/100")
        print(f"  Fatigue: {score.fatigue.name}")

        # Reset for next rep
        self.angles_history = []
        self.timestamps = []
        self.pose_history = []

    def _reset_session(self):
        """Reset session state."""
        self.frame_count = 0
        self.angles_history = []
        self.timestamps = []
        self.pose_history = []
        self.rep_count = 0
        self.scorer.start_session("v3_session")
        print("\nSession reset")

    def _print_session_summary(self):
        """Print session summary."""
        report = self.scorer.get_session_report()
        if not report or report.total_reps == 0:
            print("\nNo reps completed in this session")
            return

        print("\n" + "=" * 60)
        print("SESSION SUMMARY")
        print("=" * 60)
        print(f"Exercise: {report.exercise_name}")
        print(f"Total Reps: {report.total_reps}")
        print(f"\nAverage Scores:")
        for dim, score in report.average_scores.items():
            print(f"  {dim}: {score:.1f}")
        print("=" * 60)

    def _cleanup(self):
        """Release all resources."""
        if self.pose_estimator:
            self.pose_estimator.close()
        if self.openface_analyzer:
            self.openface_analyzer.close()
        if self.coach:
            self.coach.close()


def parse_args():
    parser = argparse.ArgumentParser(description="ADAPT-Rehab v3.0")
    parser.add_argument("--source", type=str, default="webcam",
                        help="Video source (webcam or path)")
    parser.add_argument("--pose-model", type=str, default=None,
                        help="Path to pose model")
    parser.add_argument("--pose-backend", type=str, default="rtmw3d",
                        choices=["rtmw3d", "mediapipe_fallback"],
                        help="3D pose backend to use (default: rtmw3d; "
                             "falls back to mediapipe_fallback if rtmw3d "
                             "is unavailable in the environment)")
    parser.add_argument("--llm-provider", type=str, default="gemini",
                        choices=["gemini", "openai", "mimo"],
                        help="LLM provider")
    parser.add_argument("--llm-api-key", type=str, default=None,
                        help="LLM API key")
    parser.add_argument("--llm-model", type=str, default="gemini-2.0-flash",
                        help="LLM model name")
    parser.add_argument("--feedback-interval", type=float, default=10.0,
                        help="Seconds between LLM feedback")
    parser.add_argument("--use-gpu", action="store_true", default=True,
                        help="Use GPU for OpenFace 3.0 inference")
    return parser.parse_args()


def main():
    args = parse_args()

    # Resolve API key
    if args.llm_api_key is None:
        args.llm_api_key = os.environ.get("MIMO_API_KEY") or os.environ.get("OPENAI_API_KEY")

    app = ADAPTRehabV3(args)
    if not app.initialize():
        print("Initialization failed!")
        sys.exit(1)

    app.run()


if __name__ == "__main__":
    main()
