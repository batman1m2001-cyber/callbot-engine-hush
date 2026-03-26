"""State machine config — pure dicts, no enums."""

IMMEDIATE_END_INTENTS = {"student_joining", "already_absent", "request_absent", "cancel_course"}
TRANSFER_HOTLINE_INTENTS = {"technical_issue", "wrong_schedule", "ask_about_program"}
RETRY_INTENTS = {"need_support", "busy", "unclear", "fallback", "other_number", "invalid_phone"}
TERMINAL_STATES = {"FINISH", "TRANSFER_HOTLINE"}

MAX_RETRIES = {
    "need_support": 2,
    "busy": 2,
    "unclear": 3,
    "fallback": 2,
    "other_number": 2,
    "invalid_phone": 2,
}

# STATE_TRANSITIONS[current_state][intent] = next_state
# "default" is fallback when intent not in table
STATE_TRANSITIONS = {
    "REMINDER": {
        "student_joining": "FINISH",
        "already_absent": "FINISH",
        "request_absent": "FINISH",
        "cancel_course": "FINISH",
        "technical_issue": "TRANSFER_HOTLINE",
        "wrong_schedule": "TRANSFER_HOTLINE",
        "ask_about_program": "TRANSFER_HOTLINE",
        "wrong_student_name": "CONFIRM_CUSTOMER",
        "wrong_support_number": "ASK_PHONE",
        "wrong_number": "CONFIRM_CUSTOMER",
        "student_pickup": "TALK_WITH_STUDENT",
        "need_support": "REMINDER",
        "busy": "REMINDER",
        "unclear": "REMINDER",
        "fallback": "REMINDER",
        "confirm": "REMINDER",
        "deny": "FINISH",
        "default": "CONFIRM_CUSTOMER",
    },
    "CONFIRM_CUSTOMER": {
        "confirm": "REMINDER",
        "wrong_number": "FINISH",
        "busy": "REMINDER",
        "deny": "FINISH",
        "unclear": "CONFIRM_CUSTOMER",
        "fallback": "CONFIRM_CUSTOMER",
        "default": "CONFIRM_CUSTOMER",
    },
    "ASK_PHONE": {
        "this_number": "REMINDER",
        "read_phone": "FINISH",
        "other_number": "ASK_PHONE",
        "invalid_phone": "ASK_PHONE",
        "fallback": "ASK_PHONE",
        "default": "ASK_PHONE",
    },
    "TALK_WITH_STUDENT": {
        "student_joining": "FINISH",
        "already_absent": "FINISH",
        "request_absent": "FINISH",
        "technical_issue": "TRANSFER_HOTLINE",
        "wrong_schedule": "TRANSFER_HOTLINE",
        "need_support": "TALK_WITH_STUDENT",
        "unclear": "TALK_WITH_STUDENT",
        "fallback": "TALK_WITH_STUDENT",
        "default": "TALK_WITH_STUDENT",
    },
    "TRANSFER_HOTLINE": {"default": "TRANSFER_HOTLINE"},
    "FINISH": {"default": "FINISH"},
}

# Allowed intents per state (for LLM validators)
STATE_ALLOWED_INTENTS = {
    "REMINDER": [
        "need_support", "student_joining", "already_absent", "request_absent",
        "wrong_schedule", "cancel_course", "technical_issue", "wrong_student_name",
        "wrong_support_number", "wrong_number", "student_pickup", "busy", "deny",
        "unclear", "fallback", "ask_about_program", "confirm",
    ],
    "CONFIRM_CUSTOMER": [
        "confirm", "wrong_number", "busy", "deny", "unclear", "fallback",
    ],
    "ASK_PHONE": [
        "this_number", "read_phone", "other_number", "invalid_phone", "fallback",
    ],
    "TALK_WITH_STUDENT": [
        "need_support", "student_joining", "already_absent", "request_absent",
        "wrong_schedule", "technical_issue", "unclear", "fallback",
    ],
}

FALLBACK_RESPONSES = {
    "REMINDER": "Dạ em chưa nghe rõ ạ. Anh chị có thể nói lại được không ạ?",
    "CONFIRM_CUSTOMER": "Dạ anh chị vui lòng xác nhận giúp em, anh chị có phải phụ huynh của bé không ạ?",
    "ASK_PHONE": "Dạ anh chị vui lòng cho em xin số điện thoại ạ.",
    "TALK_WITH_STUDENT": "Dạ cô chưa nghe rõ. Con vào lớp học giúp cô nhé.",
    "TRANSFER_HOTLINE": "Dạ anh chị vui lòng liên hệ hotline để được hỗ trợ ạ.",
    "FINISH": "Dạ em cảm ơn anh chị ạ. Chào anh chị.",
}
