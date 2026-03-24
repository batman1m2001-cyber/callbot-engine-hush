"""skip ops — pass-through for if_() branches that bypass LLM."""

from hush.core.ops import op


@op
def skip_classify(intent, confidence, extraction_data):
    """Forward quick_detect results as-is (skip LLM classification)."""
    return {
        "quick_intent": intent,
        "quick_confidence": confidence,
        "quick_extraction_data": extraction_data,
    }


@op
def skip_generate(rule_response):
    """Forward rule response as-is (skip LLM generation)."""
    return {"rule_response": rule_response}
