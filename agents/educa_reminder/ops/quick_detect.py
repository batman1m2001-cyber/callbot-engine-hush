"""quick_detect op — rule-based intent detection (no LLM)."""

import re

from hush.core.ops import op

PHONE_PATTERN = re.compile(r'(?:\+?84|0)(\d{9})')


def _normalize_phone(raw: str) -> str:
    """Convert +84/84 prefix to 0."""
    digits = re.sub(r'[^\d]', '', raw)
    if digits.startswith('84') and len(digits) == 11:
        digits = '0' + digits[2:]
    elif not digits.startswith('0'):
        digits = '0' + digits
    return digits


def _is_valid_phone(phone: str) -> bool:
    return bool(re.match(r'^0\d{9}$', phone))


@op
def quick_detect(normalized_text: str, current_state: str):
    """Detect intent from rules. Returns needs_llm=True if LLM needed."""
    # Silent
    if not normalized_text:
        return {
            "intent": "silent",
            "confidence": 0.95,
            "extraction_data": {},
            "needs_llm": False,
        }

    # Phone detection (only in ASK_PHONE state)
    if current_state == "ASK_PHONE":
        match = PHONE_PATTERN.search(normalized_text)
        if match:
            phone = _normalize_phone(match.group(0))
            if _is_valid_phone(phone):
                return {
                    "intent": "read_phone",
                    "confidence": 0.95,
                    "extraction_data": {"phone_number": phone, "phone_valid": True},
                    "needs_llm": False,
                }
            else:
                return {
                    "intent": "invalid_phone",
                    "confidence": 0.85,
                    "extraction_data": {"phone_number": phone, "phone_valid": False},
                    "needs_llm": False,
                }

    # No quick match — need LLM
    return {
        "intent": None,
        "confidence": 0.0,
        "extraction_data": {},
        "needs_llm": True,
    }
