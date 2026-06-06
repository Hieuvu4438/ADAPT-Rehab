#!/usr/bin/env python3
"""
ADAPT-Rehab v3.0 - Main Application

Multimodal AI Rehabilitation System for Elderly Vietnamese Users.

Architecture:
    Input → Perception (3D Pose + Face) → Analysis (Kinematics + Scoring)
    → Intelligence (LLM + Voice) → Output (Visual + Audio)

Key Improvements from v2:
    - Direct 3D pose estimation (MeTRAbs/HybrIK) instead of MediaPipe
    - Deep learning pain/emotion detection
    - LLM-based exercise coaching (API, not self-hosted)
    - Quaternion-based joint angles
    - SPARC smoothness metric
    - Temporal compensation detection

Usage:
    python main_v3.py --source webcam --ref-video exercise.mp4
    python main_v3.py --mode demo

Author: ADAPT-Rehab Team
Version: 3.0.0
"""

import argparse
import sys
import os
import time
from pathlib import Path
from typing import Optional, Dict
import numpy as np

try:
    import cv2
except ImportError:
    print("OpenCV required. Install: pip install opencv-python")
    sys.exit(1)

# Perception layer
from perception.pose3d.base import create_estimator, PoseEstimator3D, Pose3DResult
from perception.face.face_detector import FaceDetector
from perception.face.au_detector import ActionUnitDetector
from perception.face.emotion_classifier import EmotionClassifier

# Analysis layer
from analysis.kinematics_v2 import QuaternionKinematics
from analysis.smoothness import SmoothnessAnalyzer
from analysis.compensation import CompensationDetector
from analysis.fatigue import FatigueAnalyzer
from analysis.scoring_v2 import EnhancedScorer

# Intelligence layer (optional - requires API keys)
try:
    from intelligence.llm.client import LLMClient
    from intelligence.voice.tts import TextToSpeech
    from intelligence.coach.rehab_coach import RehabCoach
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False

# Legacy modules (still used)
from core.synchronizer import (
    MotionSyncController, MotionPhase, SyncStatus,
    create_arm_raise_exercise, create_elbow_flex_exercise,
)
from modules.calibration import SafeMaxCalibrator, CalibrationState, UserProfile
from modules.video_engine import SyncedVideoPlayer
from utils.logger import SessionLogger
from utils.visualization import (
    put_vietnamese_text, draw_skeleton, draw_panel,
    draw_progress_bar, draw_phase_indicator, COLORS,
)


class ADAPTRehabV3:
    """
    ADAPT-Rehab v3.0 Main Application.

    Integrates all layers:
    - Perception: 3D pose + face analysis
    - Analysis: Kinematics + scoring
    - Intelligence: LLM coaching + voice
    - Output: Visual + audio feedback
    """

    def __init__(self, args):
        """Initialize application with command-line arguments."""
        self.args = args
        self.running = False

        # Perception
        self.pose_estimator: Optional[PoseEstimator3D] = None
        self.face_detector: Optional[FaceDetector] = None
        self.au_detector: Optional[ActionUnitDetector] = None
        self.emotion_classifier: Optional[EmotionClassifier] = None

        # Analysis
        self.quaternion_kinematics = QuaternionKinematics()
        self.smoothness_analyzer = SmoothnessAnalyzer()
        self.compensation_detector = CompensationDetector()
        self.fatigue_analyzer = FatigueAnalyzer()
        self.scorer = EnhancedScorer()

        # Intelligence
        self.llm_client: Optional[LLMClient] = None
        self.tts: Optional[TextToSpeech] = None
        self.coach: Optional[RehabCoach] = None

        # Legacy
        self.calibrator = SafeMaxCalibrator()
        self.logger = SessionLogger()
        self.sync_controller: Optional[MotionSyncController] = None

        # State
        self.user_profile: Optional[UserProfile] = None
        self.current_phase = "detection"
        self.frame_count = 0

    def initialize(self) -> bool:
        """Initialize all components."""
        print("=" * 60)
        print("ADAPT-Rehab v3.0 - Multimodal AI Rehabilitation")
        print("=" * 60)

        # 1. Initialize 3D Pose Estimator
        print("\n[1/5] Initializing 3D Pose Estimator...")
        pose_backend = self.args.pose_backend  # "metrab", "hybrik", "mediapipe_fallback"
        self.pose_estimator = create_estimator(pose_backend)
        if not self.pose_estimator.initialize(model_path=self.args.pose_model):
            print(f"  ⚠ Failed to initialize {pose_backend}, falling back to MediaPipe")
            self.pose_estimator = create_estimator("mediapipe_fallback")
            self.pose_estimator.initialize()

        # 2. Initialize Face Analysis
        print("[2/5] Initializing Face Analysis...")
        self.face_detector = FaceDetector()
        self.face_detector.initialize()

        if self.args.enable_au:
            self.au_detector = ActionUnitDetector()
            self.au_detector.initialize()

        if self.args.enable_emotion:
            self.emotion_classifier = EmotionClassifier()
            self.emotion_classifier.initialize(model_path=self.args.emotion_model)

        # 3. Initialize Intelligence Layer
        print("[3/5] Initializing Intelligence Layer...")
        if LLM_AVAILABLE and self.args.llm_api_key:
            self.llm_client = LLMClient(
                provider=self.args.llm_provider,
                api_key=self.args.llm_api_key,
            )
            self.llm_client.initialize()

            self.tts = TextToSpeech(voice=self.args.tts_voice)
            self.tts.initialize()

            self.coach = RehabCoach(
                llm_client=self.llm_client,
                tts=self.tts,
            )
            print("  ✓ LLM coaching enabled")
        else:
            print("  ⚠ LLM coaching disabled (no API key)")

        # 4. Initialize Calibration
        print("[4/5] Initializing Calibration...")
        if self.args.user_profile:
            self.user_profile = self.calibrator.load_profile(self.args.user_profile)
            if self.user_profile:
                print(f"  ✓ Loaded profile: {self.user_profile.user_id}")

        # 5. Initialize Logger
        print("[5/5] Initializing Logger...")
        self.logger.start_session("v3_session")

        print("\n✓ All components initialized successfully!")
        print(f"  Pose Backend: {self.pose_estimator.model_name}")
        print(f"  LLM: {'Enabled' if self.llm_client else 'Disabled'}")
        print("=" * 60)

        return True

    def run(self):
        """Main application loop."""
        # Open video source
        if self.args.source == "webcam":
            cap = cv2.VideoCapture(0)
        else:
            cap = cv2.VideoCapture(self.args.source)

        if not cap.isOpened():
            print("Error: Cannot open video source")
            return

        self.running = True
        print("\nPress 'q' to quit, 'c' to calibrate, 's' to start exercise")

        while self.running:
            ret, frame = cap.read()
            if not ret:
                break

            self.frame_count += 1
            timestamp_ms = int(self.frame_count * (1000 / 30))

            # Process frame
            display_frame = self._process_frame(frame, timestamp_ms)

            # Display
            cv2.imshow("ADAPT-Rehab v3.0", display_frame)

            # Handle keyboard
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self.running = False
            elif key == ord('c'):
                self._start_calibration()
            elif key == ord('s'):
                self._start_exercise()

        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        self._cleanup()

    def _process_frame(self, frame: np.ndarray, timestamp_ms: int) -> np.ndarray:
        """Process a single frame through all layers."""
        display = frame.copy()

        # Perception: 3D Pose
        pose_result = self.pose_estimator.estimate(frame, timestamp_ms)

        if pose_result.is_valid:
            # Draw skeleton
            if pose_result.keypoints_2d is not None:
                display = draw_skeleton(display, pose_result.keypoints_2d)

            # Display joint angles
            y_offset = 30
            for joint, angle in pose_result.joint_angles.items():
                text = f"{joint}: {angle:.1f}°"
                cv2.putText(display, text, (10, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                y_offset += 20

        # Perception: Face Analysis
        face_result = self.face_detector.detect(frame)
        if face_result.is_valid and face_result.bbox is not None:
            x1, y1, x2, y2 = face_result.bbox.astype(int)
            cv2.rectangle(display, (x1, y1), (x2, y2), (255, 0, 0), 2)

        # Display FPS
        fps_text = f"FPS: {pose_result.fps:.0f} | Model: {pose_result.model_name}"
        cv2.putText(display, fps_text, (10, display.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return display

    def _start_calibration(self):
        """Start ROM calibration."""
        print("\n[Calibration] Starting Safe-Max Calibration...")
        self.current_phase = "calibration"

    def _start_exercise(self):
        """Start exercise session."""
        print("\n[Exercise] Starting exercise session...")
        self.current_phase = "exercise"

        if self.coach:
            self.coach.start_exercise("Arm Raise", {"age": 70})

    def _cleanup(self):
        """Release all resources."""
        print("\n[Cleanup] Releasing resources...")
        if self.pose_estimator:
            self.pose_estimator.close()
        if self.face_detector:
            self.face_detector.close()
        if self.llm_client:
            self.llm_client.close()
        self.logger.end_session()
        print("[Cleanup] Done!")


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="ADAPT-Rehab v3.0")

    # Input
    parser.add_argument("--source", type=str, default="webcam",
                        help="Video source (webcam, path, or RTSP URL)")
    parser.add_argument("--ref-video", type=str, default=None,
                        help="Reference exercise video path")

    # Pose estimation
    parser.add_argument("--pose-backend", type=str, default="mediapipe_fallback",
                        choices=["metrab", "hybrik", "mediapipe_fallback"],
                        help="3D pose estimation backend")
    parser.add_argument("--pose-model", type=str, default=None,
                        help="Path to pose model file")

    # Face analysis
    parser.add_argument("--enable-au", action="store_true",
                        help="Enable Action Unit detection")
    parser.add_argument("--enable-emotion", action="store_true",
                        help="Enable emotion classification")
    parser.add_argument("--emotion-model", type=str, default=None,
                        help="Path to emotion model file")

    # LLM
    parser.add_argument("--llm-provider", type=str, default="openai",
                        choices=["openai", "anthropic"],
                        help="LLM API provider")
    parser.add_argument("--llm-api-key", type=str, default=None,
                        help="LLM API key (or set OPENAI_API_KEY/ANTHROPIC_API_KEY env)")

    # TTS
    parser.add_argument("--tts-voice", type=str, default="vi-VN-HoaiMyNeural",
                        help="Edge-TTS voice name")

    # User
    parser.add_argument("--user-profile", type=str, default=None,
                        help="Path to user profile JSON")

    # Mode
    parser.add_argument("--mode", type=str, default="full",
                        choices=["full", "demo", "calibration", "test"],
                        help="Application mode")

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # Resolve API key from environment if not provided
    if args.llm_api_key is None:
        args.llm_api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")

    # Create and run application
    app = ADAPTRehabV3(args)

    if not app.initialize():
        print("Initialization failed!")
        sys.exit(1)

    app.run()


if __name__ == "__main__":
    main()
