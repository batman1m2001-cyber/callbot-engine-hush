"""Conversation manager — update shared state after each turn."""

from hush.core.ops import op


@op
def update_conversation(
    transcript: str,
    response: str,
    intent: str,
    new_state: str,
    conversation_history: list,
    intent_retry_counts: dict,
) -> dict:
    """Update conversation state after a turn.

    Takes current shared state + turn results → returns updated state
    to be pushed back to PARENT shared vars.
    """
    # Update history
    new_history = list(conversation_history)
    if transcript:
        new_history.append({"role": "user", "content": transcript})
    if response:
        new_history.append({"role": "assistant", "content": response})

    # Update retry counts
    new_retry_counts = dict(intent_retry_counts)
    if intent:
        new_retry_counts[intent] = new_retry_counts.get(intent, 0) + 1

    return {
        "updated_state": new_state,
        "updated_history": new_history,
        "updated_retry_counts": new_retry_counts,
        "updated_response": response or "",
    }
