"""state_transition op — pure state machine logic."""

from hush.core.ops import op

from data.state_config import (
    IMMEDIATE_END_INTENTS,
    MAX_RETRIES,
    RETRY_INTENTS,
    STATE_TRANSITIONS,
    TRANSFER_HOTLINE_INTENTS,
)


@op
def state_transition(
    intent: str,
    current_state: str,
    intent_retry_counts: dict,
    extraction_data: dict,
):
    """Determine next state from intent + current state. Pure logic, no LLM."""
    counts = dict(intent_retry_counts)  # copy
    previous_state = current_state
    should_transfer = False
    should_hangup = False

    # Track retries
    if intent in RETRY_INTENTS:
        counts[intent] = counts.get(intent, 0) + 1

    # Immediate end
    if intent in IMMEDIATE_END_INTENTS:
        new_state = "FINISH"
        should_hangup = True

    # Transfer hotline
    elif intent in TRANSFER_HOTLINE_INTENTS:
        new_state = "TRANSFER_HOTLINE"
        should_transfer = True

    # Deny
    elif intent == "deny":
        new_state = "FINISH"
        should_hangup = True

    # Confirm
    elif intent == "confirm":
        if current_state == "CONFIRM_CUSTOMER":
            new_state = "REMINDER"
        else:
            new_state = current_state

    # Student pickup
    elif intent == "student_pickup":
        new_state = "TALK_WITH_STUDENT"

    # Wrong number — different behavior per state
    elif intent == "wrong_number":
        if current_state == "REMINDER":
            new_state = "CONFIRM_CUSTOMER"
        else:
            new_state = "FINISH"
            should_hangup = True

    # Wrong student name
    elif intent == "wrong_student_name":
        new_state = "CONFIRM_CUSTOMER"

    # Wrong support number
    elif intent == "wrong_support_number":
        new_state = "ASK_PHONE"

    # Phone intents
    elif intent == "this_number":
        new_state = "REMINDER"
    elif intent == "read_phone":
        new_state = "FINISH"
        should_hangup = True
    elif intent in ("other_number", "invalid_phone"):
        max_r = MAX_RETRIES.get(intent, 2)
        if counts.get(intent, 0) >= max_r:
            new_state = "FINISH"
            should_hangup = True
        else:
            new_state = "ASK_PHONE"

    # Retry intents (need_support, busy, unclear, fallback)
    elif intent in RETRY_INTENTS:
        max_r = MAX_RETRIES.get(intent, 2)
        retry_count = counts.get(intent, 0)

        if retry_count >= max_r:
            # Exceeded retries
            if intent == "need_support":
                new_state = "TRANSFER_HOTLINE"
                should_transfer = True
            elif current_state == "TALK_WITH_STUDENT" and intent == "fallback":
                new_state = "TRANSFER_HOTLINE"
                should_transfer = True
            else:
                new_state = "FINISH"
                should_hangup = True
        else:
            # Loopback
            new_state = current_state

    # Default from transition table
    else:
        transitions = STATE_TRANSITIONS.get(current_state, {})
        new_state = transitions.get(intent, transitions.get("default", current_state))

    return {
        "new_state": new_state,
        "previous_state": previous_state,
        "should_transfer": should_transfer,
        "should_hangup": should_hangup,
        "intent_retry_counts": counts,
    }
