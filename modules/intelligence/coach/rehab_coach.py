"""
LLM-Based Rehabilitation Coach.

Orchestrates real-time exercise coaching by combining:
- Pose analysis results (joint angles, phase)
- Pain/emotion detection
- Fatigue detection
- LLM-generated feedback
- TTS voice output

Safety-first: all LLM output is validated before delivery.

Usage:
    coach = RehabCoach(llm_client=client, tts=tts)
    coach.start_exercise("Arm Raise", user_profile={"age": 70})
    feedback = coach.update(
        current_angle=120, target_angle=150, phase="ECCENTRIC",
        rom_score=80, stability_score=75,
        fatigue_level="LIGHT", pain_level="NONE", rep_count=3
    )
    if feedback.is_valid:
        tts.synthesize(feedback.voice_text)
"""

from dataclasses import dataclass
from typing import Optional, Dict
import time


@dataclass
class CoachingFeedback:
    """Feedback from the rehabilitation coach."""
    message: str = ""           # Text message (Vietnamese)
    voice_text: str = ""        # Text for TTS output
    feedback_type: str = "general"  # general, correction, encouragement, warning
    priority: int = 0           # 0=low, 1=medium, 2=high (pain/fatigue)
    is_valid: bool = False


class RehabCoach:
    """
    LLM-based rehabilitation coach.

    Generates personalized, contextual feedback for elderly users
    based on their real-time exercise performance.

    Priority system:
        1. Pain alerts (highest priority) — immediate feedback
        2. Fatigue warnings — encourage rest
        3. Performance feedback — general coaching
        4. Encouragement — positive reinforcement

    Cooldown: minimum 5 seconds between LLM calls to avoid
    excessive API usage and overwhelming the user.

    Example:
        >>> coach = RehabCoach(llm_client=client, tts=tts)
        >>> coach.start_exercise("Arm Raise", {"age": 70, "condition": "shoulder pain"})
        >>> while exercising:
        ...     feedback = coach.update(...)
        ...     if feedback.is_valid:
        ...         speak(feedback.voice_text)
    """

    MIN_FEEDBACK_INTERVAL = 5.0  # seconds between LLM calls
    PAIN_PRIORITY = 2
    FATIGUE_PRIORITY = 1
    GENERAL_PRIORITY = 0

    def __init__(self, llm_client=None, tts=None, safety=None):
        """
        Initialize rehabilitation coach.

        Args:
            llm_client: LLMClient instance for API calls.
            tts: TextToSpeech instance for voice output.
            safety: SafetyGuardrails instance for output validation.
        """
        self._llm = llm_client
        self._tts = tts
        self._safety = safety
        self._exercise = ""
        self._user_profile: Dict = {}
        self._conversation_history = []
        self._last_feedback_time = 0.0
        self._last_feedback_type = ""
        self._active = False

    def start_exercise(self, exercise_name: str, user_profile: Dict = None) -> CoachingFeedback:
        """
        Start a new exercise session with initial greeting.

        Args:
            exercise_name: Name of the exercise.
            user_profile: User info (age, condition, ROM limits).

        Returns:
            CoachingFeedback with greeting message.
        """
        self._exercise = exercise_name
        self._user_profile = user_profile or {}
        self._conversation_history = []
        self._last_feedback_time = 0.0
        self._active = True

        age = self._user_profile.get("age", 70)
        condition = self._user_profile.get("condition", "")

        if self._llm:
            prompt = (
                f"Bác {age} tuổi bắt đầu bài tập {exercise_name}. "
                f"{'Tình trạng: ' + condition + '. ' if condition else ''}"
                "Hãy chào bác ngắn gọn và hướng dẫn chuẩn bị. "
                "Gọi 'Bác', sử dụng tiếng Việt thân thiện."
            )
            response = self._llm.chat(
                prompt,
                system_prompt=self._get_system_prompt(),
            )
            if response.is_valid:
                msg = self._validate(response.content)
                self._conversation_history.append({"role": "assistant", "content": msg})
                self._last_feedback_time = time.time()
                return CoachingFeedback(
                    message=msg, voice_text=msg,
                    feedback_type="encouragement", priority=self.GENERAL_PRIORITY,
                    is_valid=True,
                )

        # Fallback greeting
        msg = f"Bác ơi, mình bắt đầu bài tập {exercise_name} nhé! Chuẩn bị sẵn sàng nào!"
        return CoachingFeedback(message=msg, voice_text=msg, feedback_type="encouragement", is_valid=True)

    def update(
        self,
        current_angle: float,
        target_angle: float,
        phase: str,
        rom_score: float = 0.0,
        stability_score: float = 0.0,
        fatigue_level: str = "FRESH",
        pain_level: str = "NONE",
        rep_count: int = 0,
    ) -> CoachingFeedback:
        """
        Update coaching state and generate feedback.

        Feedback is generated based on priority:
            1. Pain detected → warning
            2. Fatigue detected → encourage rest
            3. Time since last feedback > threshold → performance feedback

        Args:
            current_angle: Current joint angle (degrees).
            target_angle: Target angle (degrees).
            phase: Current exercise phase (IDLE, ECCENTRIC, HOLD, CONCENTRIC).
            rom_score: ROM score (0-100).
            stability_score: Stability score (0-100).
            fatigue_level: Fatigue level (FRESH, LIGHT, MODERATE, HEAVY).
            pain_level: Pain level (NONE, MILD, MODERATE, SEVERE).
            rep_count: Number of completed reps.

        Returns:
            CoachingFeedback with message and voice text.
        """
        now = time.time()

        # Priority 1: Pain alert (always generate)
        if pain_level in ["MODERATE", "SEVERE"]:
            return self._generate_pain_feedback(pain_level)

        # Priority 2: Fatigue warning (always generate if level increased)
        if fatigue_level in ["MODERATE", "HEAVY"]:
            return self._generate_fatigue_feedback(fatigue_level, rep_count)

        # Priority 3: Performance feedback (with cooldown)
        if now - self._last_feedback_time >= self.MIN_FEEDBACK_INTERVAL:
            return self._generate_performance_feedback(
                current_angle, target_angle, phase,
                rom_score, stability_score, rep_count
            )

        return CoachingFeedback()

    def _generate_pain_feedback(self, pain_level: str) -> CoachingFeedback:
        """Generate pain-related warning feedback."""
        if self._llm:
            prompt = (
                f"CẢNH BÁO: Phát hiện đau mức {pain_level} ở bác. "
                "Hãy cảnh báo nhẹ nhàng bằng tiếng Việt và nhắc bác nghỉ ngơi. "
                "Ngắn gọn (1-2 câu)."
            )
            response = self._llm.chat(prompt, system_prompt=self._get_system_prompt())
            if response.is_valid:
                msg = self._validate(response.content)
                self._last_feedback_time = time.time()
                self._last_feedback_type = "warning"
                return CoachingFeedback(
                    message=msg, voice_text=msg,
                    feedback_type="warning", priority=self.PAIN_PRIORITY,
                    is_valid=True,
                )

        # Fallback
        msg = "Bác ơi, có vẻ bác đang đau. Mình nghỉ ngơi một chút nhé! Sức khỏe là quan trọng nhất!"
        self._last_feedback_time = time.time()
        return CoachingFeedback(message=msg, voice_text=msg, feedback_type="warning", priority=self.PAIN_PRIORITY, is_valid=True)

    def _generate_fatigue_feedback(self, fatigue_level: str, rep_count: int) -> CoachingFeedback:
        """Generate fatigue warning feedback."""
        if self._llm:
            prompt = (
                f"Bác đang mệt mức {fatigue_level}. Đã tập {rep_count} rep. "
                "Hãy động viên bác nghỉ ngơi bằng tiếng Việt. Ngắn gọn (1-2 câu)."
            )
            response = self._llm.chat(prompt, system_prompt=self._get_system_prompt())
            if response.is_valid:
                msg = self._validate(response.content)
                self._last_feedback_time = time.time()
                self._last_feedback_type = "encouragement"
                return CoachingFeedback(
                    message=msg, voice_text=msg,
                    feedback_type="encouragement", priority=self.FATIGUE_PRIORITY,
                    is_valid=True,
                )

        # Fallback
        msg = f"Bác làm tốt lắm! Đã tập {rep_count} rep. Mình nghỉ một chút rồi tiếp nhé!"
        self._last_feedback_time = time.time()
        return CoachingFeedback(message=msg, voice_text=msg, feedback_type="encouragement", priority=self.FATIGUE_PRIORITY, is_valid=True)

    def _generate_performance_feedback(
        self, current_angle: float, target_angle: float,
        phase: str, rom_score: float, stability_score: float, rep_count: int,
    ) -> CoachingFeedback:
        """Generate performance feedback."""
        if self._llm:
            angle_diff = current_angle - target_angle
            status = "đạt mục tiêu" if abs(angle_diff) < 10 else (
                f"thiếu {abs(angle_diff):.0f}°" if angle_diff < 0 else f"vượt {angle_diff:.0f}°"
            )

            prompt = (
                f"Bài tập: {self._exercise}\n"
                f"Giai đoạn: {phase}\n"
                f"Góc hiện tại: {current_angle:.1f}° ({status})\n"
                f"Góc mục tiêu: {target_angle:.1f}°\n"
                f"Điểm ROM: {rom_score:.0f}/100\n"
                f"Điểm ổn định: {stability_score:.0f}/100\n"
                f"Đã tập: {rep_count} rep\n\n"
                "Hãy đưa ra feedback ngắn gọn (1-2 câu) bằng tiếng Việt. "
                "Động viên bác nếu tốt, hướng dẫn sửa nếu chưa đúng."
            )
            response = self._llm.chat(prompt, system_prompt=self._get_system_prompt())
            if response.is_valid:
                msg = self._validate(response.content)
                self._last_feedback_time = time.time()
                self._last_feedback_type = "general"
                return CoachingFeedback(
                    message=msg, voice_text=msg,
                    feedback_type="general", priority=self.GENERAL_PRIORITY,
                    is_valid=True,
                )

        # Fallback
        if rom_score >= 90:
            msg = "Tuyệt vời! Bác làm rất tốt!"
        elif rom_score >= 70:
            msg = "Tốt lắm! Cố thêm một chút nữa nhé!"
        else:
            msg = "Bác cố gắng thêm một chút. Gần đạt rồi!"

        self._last_feedback_time = time.time()
        return CoachingFeedback(message=msg, voice_text=msg, feedback_type="general", priority=self.GENERAL_PRIORITY, is_valid=True)

    def _get_system_prompt(self) -> str:
        """Get system prompt for LLM."""
        age = self._user_profile.get("age", 70)
        return (
            f"Bạn là trợ lý AI chuyên về phục hồi chức năng cho người cao tuổi Việt Nam. "
            f"Người tập {age} tuổi. "
            "Hướng dẫn bài tập an toàn, dễ hiểu. Động viên bác. "
            "Gọi 'Bác', sử dụng tiếng Việt thân thiện, ngắn gọn. "
            "KHÔNG khuyến khích vượt quá giới hạn. DỪNG ngay khi đau. "
            "KHÔNG đưa ra lời khuyên y tế chung chung."
        )

    def _validate(self, message: str) -> str:
        """Validate LLM output through safety guardrails."""
        if self._safety:
            check = self._safety.validate(message, user_age=self._user_profile.get("age", 70))
            if not check.is_safe:
                return self._safety.filter(message)
        return message

    def stop(self) -> None:
        """Stop coaching session."""
        self._active = False

    def close(self) -> None:
        """Release resources."""
        self._active = False
        self._conversation_history = []
