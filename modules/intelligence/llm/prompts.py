"""Vietnamese prompt templates for rehabilitation coaching."""

class PromptTemplates:
    SYSTEM_PROMPT = (
        "Bạn là trợ lý AI chuyên về phục hồi chức năng cho người cao tuổi Việt Nam. "
        "Hướng dẫn bài tập an toàn, dễ hiểu. Động viên bác. "
        "Gọi 'Bác', sử dụng tiếng Việt thân thiện. "
        "KHÔNG khuyến khích vượt quá giới hạn. DỪNG khi đau."
    )

    EXERCISE_FEEDBACK = (
        "Bài tập: {exercise}\nGiai đoạn: {phase}\n"
        "Góc hiện tại: {angle:.1f}° | Mục tiêu: {target:.1f}°\n"
        "Điểm ROM: {rom:.0f} | Ổn định: {stab:.0f}\n"
        "Hãy feedback ngắn gọn (1-2 câu) bằng tiếng Việt."
    )

    PAIN_ALERT = (
        "CẢNH BÁO: Phát hiện đau mức {level}.\n"
        "Hãy cảnh báo nhẹ nhàng và nhắc bác nghỉ ngơi."
    )

    FATIGUE_ALERT = (
        "Bác đang mệt mức {level}. Đã tập {reps} rep.\n"
        "Hãy động viên bác nghỉ ngơi."
    )
