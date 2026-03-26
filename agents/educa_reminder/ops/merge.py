"""merge ops — combine outputs from branched paths."""

from hush.core.ops import op

from agents.educa_reminder.data.state_config import FALLBACK_RESPONSES


@op
def merge_intent(
    quick_intent=None,
    quick_confidence=0.0,
    quick_extraction_data=None,
    llm_result=None,
):
    """Merge quick_detect and classify_intent results."""
    if quick_intent is not None:
        return {
            "intent": quick_intent,
            "confidence": quick_confidence,
            "extraction_data": quick_extraction_data or {},
        }
    return {
        "intent": llm_result or "fallback",
        "confidence": 0.85 if llm_result and llm_result != "fallback" else 0.5,
        "extraction_data": {},
    }


@op
def merge_response(
    rule_response=None,
    llm_content=None,
    new_state="REMINDER",
    script_data=None,
):
    """Merge rule-based and LLM response."""
    if rule_response:
        return {"response": rule_response}
    if llm_content:
        return {"response": llm_content}
    # Ultimate fallback
    fallback = FALLBACK_RESPONSES.get(new_state, "Dạ em chưa nghe rõ ạ.")
    if script_data:
        for key, val in (script_data or {}).items():
            if val:
                fallback = fallback.replace(f"[{key}]", str(val))
    return {"response": fallback}
