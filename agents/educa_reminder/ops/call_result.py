"""call_result — CRM mapping: determine ARId, comment, report_result."""


# ── ARId mapping: (state, intent) → ARId ──
# Exact match first, then wildcard (*) on intent, then wildcard on state
ARID_MAP = {
    # Success cases
    ("REMINDER", "student_joining"): "SUCCESS_JOINING",
    ("TALK_WITH_STUDENT", "student_joining"): "SUCCESS_JOINING",
    ("REMINDER", "already_absent"): "SUCCESS_ABSENT_NOTIFIED",
    ("TALK_WITH_STUDENT", "already_absent"): "SUCCESS_ABSENT_NOTIFIED",
    ("REMINDER", "request_absent"): "SUCCESS_ABSENT_REQUESTED",
    ("TALK_WITH_STUDENT", "request_absent"): "SUCCESS_ABSENT_REQUESTED",
    # Transfer
    ("TRANSFER_HOTLINE", "*"): "TRANSFER_HOTLINE",
    # Intent-based (any state)
    ("*", "technical_issue"): "TECHNICAL_ISSUE",
    ("*", "wrong_schedule"): "WRONG_SCHEDULE",
    ("*", "cancel_course"): "CANCEL_COURSE",
    ("*", "ask_about_program"): "ASK_ABOUT_PROGRAM",
    ("*", "wrong_number"): "WRONG_NUMBER",
    ("*", "wrong_student_name"): "WRONG_STUDENT_NAME",
    ("*", "wrong_support_number"): "WRONG_SUPPORT_NUMBER",
    ("*", "student_pickup"): "STUDENT_PICKUP",
    ("*", "busy"): "CUSTOMER_BUSY",
    ("*", "deny"): "CUSTOMER_DENY",
    # Finish fallback
    ("FINISH", "*"): "CALL_COMPLETED",
}


# ── Comment mapping: intent → comment ──
COMMENT_MAP = {
    "student_joining": "Phụ huynh/Học sinh xác nhận đang vào lớp",
    "already_absent": "Phụ huynh/Học sinh thông báo đã xin nghỉ",
    "request_absent": "Phụ huynh/Học sinh xin nghỉ",
    "technical_issue": "Học sinh gặp lỗi kỹ thuật",
    "wrong_schedule": "Sai lịch học",
    "cancel_course": "Yêu cầu hủy/bảo lưu khóa học",
    "wrong_number": "Nhầm số điện thoại",
    "wrong_student_name": "Sai tên học sinh",
    "wrong_support_number": "Số đăng ký hộ",
    "need_support": "Cần hỗ trợ",
    "busy": "Khách hàng bận",
    "deny": "Khách hàng từ chối",
    "confirm": "Xác nhận đúng phụ huynh",
    "fallback": "Không hiểu/Không nghe rõ",
    "silent": "Không hiểu/Không nghe rõ",
    "unclear": "KH chưa phản hồi",
    "student_pickup": "Học sinh nghe máy hộ bố mẹ",
    "ask_about_program": "Muốn hỏi thêm về chương trình",
}


# ── Report result mapping: intent → result string ──
REPORT_RESULTS = {
    "student_joining": "Đồng ý vào học",
    "confirm": "Xác nhận phụ huynh",
    "already_absent": "Xin nghỉ",
    "request_absent": "Xin nghỉ",
    "need_support": "Lỗi kỹ thuật",
    "technical_issue": "Lỗi kỹ thuật",
    "wrong_schedule": "Đã đổi lịch",
    "ask_about_program": "Tư vấn chương trình",
    "cancel_course": "Huỷ khoá học",
    "wrong_number": "Nhầm máy / Lừa đảo",
    "wrong_student_name": "Sai tên",
    "wrong_support_number": "Sai số hỗ trợ",
    "busy": "Bận",
    "unclear": "KH chưa phản hồi",
    "fallback": "Chưa rõ phản hồi KH",
    "silent": "Chưa rõ phản hồi KH",
    "this_number": "Giữ số hiện tại",
    "read_phone": "Đổi số điện thoại",
    "other_number": "Đổi số điện thoại",
    "student_pickup": "Con nghe máy",
}


def _determine_arid(state: str, intent: str) -> str:
    """Determine ARId from state + intent. Exact match → wildcard intent → wildcard state."""
    # Exact match
    key = (state, intent)
    if key in ARID_MAP:
        return ARID_MAP[key]
    # Wildcard on state (any state, specific intent)
    key = ("*", intent)
    if key in ARID_MAP:
        return ARID_MAP[key]
    # Wildcard on intent (specific state, any intent)
    key = (state, "*")
    if key in ARID_MAP:
        return ARID_MAP[key]
    return "UNKNOWN"


def _determine_final_comment(state: str, intent: str, student_name: str) -> str:
    """Generate context-aware final comment when call ends."""
    if state == "FINISH":
        if intent == "student_joining":
            return f"Đã nhắc học thành công - {student_name} đã vào lớp"
        elif intent in ("already_absent", "request_absent"):
            return f"Ghi nhận xin nghỉ - {student_name}"
        elif intent == "wrong_number":
            return "Nhầm số điện thoại"
        elif intent == "deny":
            return "Khách hàng từ chối"
        elif intent == "busy":
            return "Khách hàng bận - không tiện nghe máy"
        else:
            return f"Kết thúc cuộc gọi - {intent}"
    elif state == "TRANSFER_HOTLINE":
        return "Chuyển hotline hỗ trợ"
    else:
        return f"Cuộc gọi kết thúc ở state {state}"


def build_call_result(
    current_state: str,
    intent: str,
    previous_state: str,
    customer_speech: str,
    customer_confirmed: bool,
    new_phone_number: str,
    script_data: dict,
) -> dict:
    """Build call_result dict for CRM reporting.

    Called after state_transition to build CRM data per turn.
    Final comment generated when state is FINISH or TRANSFER_HOTLINE.
    """
    student_name = script_data.get("student_name", "học sinh")

    # ARId based on state BEFORE transition (matches old behavior)
    arid = _determine_arid(previous_state, intent)
    # But final_comment based on state AFTER transition
    final_state = current_state
    comment = COMMENT_MAP.get(intent, f"State: {current_state}, Intent: {intent}")
    report_result = REPORT_RESULTS.get(intent)

    call_result = {
        "current_state": current_state,
        "current_intent": intent,
        "previous_state": previous_state,
        "last_customer_speech": customer_speech,
        "customer_confirmed": customer_confirmed,
        "new_phone_number": new_phone_number or None,
        "ARId": arid,
        "Comment": comment,
        "report_result": report_result,
    }

    # Final comment when call ends
    if final_state in ("FINISH", "TRANSFER_HOTLINE"):
        call_result["final_comment"] = _determine_final_comment(
            final_state, intent, student_name
        )
        call_result["final_state"] = final_state
        call_result["final_intent"] = intent

    return {"call_result": call_result}
