"""generate_rule op — rule-based response generation from templates."""

import random
from pathlib import Path

import yaml

from hush.core.ops import op

# Load response templates
from agents.educa_reminder.data.response_templates import (
    STATE_GREETINGS,
    RESPONSE_TEMPLATES,
    FINISH_RESPONSE_TEMPLATES,
    TRANSFER_HOTLINE_RESPONSE_TEMPLATES,
    refactor_sentence,
)

# Load state guidance + openings from prompts.yaml
_YAML_PATH = Path(__file__).parent.parent / "data" / "prompts.yaml"
with open(_YAML_PATH, "r", encoding="utf-8") as f:
    _CONFIG = yaml.safe_load(f)

_STATE_GUIDANCE = _CONFIG.get("state_guidance", {})
_RESPONSE_OPENINGS = _CONFIG.get("response_openings", ["Dạ"])
_PROMPTS = _CONFIG["prompts"]


@op
def generate_rule(
    new_state: str,
    intent: str,
    previous_state: str,
    state_retry_count: int,
    script_data: dict,
    conversation_history: list,
):
    """Try rule-based response. Returns needs_llm_generate=True if no template found."""
    response = None

    # FINISH state — depends on previous_state
    if new_state == "FINISH":
        from_state_templates = FINISH_RESPONSE_TEMPLATES.get(previous_state, {})
        templates = from_state_templates.get(intent.upper(), [])
        if templates:
            idx = min(state_retry_count, len(templates) - 1)
            response = refactor_sentence(templates[idx], script_data)

    # TRANSFER_HOTLINE state — depends on previous_state
    elif new_state == "TRANSFER_HOTLINE":
        from_state_templates = TRANSFER_HOTLINE_RESPONSE_TEMPLATES.get(previous_state, {})
        templates = from_state_templates.get(intent.upper(), [])
        if templates:
            idx = min(state_retry_count, len(templates) - 1)
            response = refactor_sentence(templates[idx], script_data)

    # Normal states — state greeting (retry_count=0) or state+intent response
    else:
        if state_retry_count == 0 and new_state != previous_state:
            # Entering new state — use greeting
            greetings = STATE_GREETINGS.get(new_state, [])
            if greetings:
                response = refactor_sentence(random.choice(greetings), script_data)
        else:
            # State+intent response
            state_templates = RESPONSE_TEMPLATES.get(new_state, {})
            templates = state_templates.get(intent.upper(), [])
            if templates:
                idx = min(state_retry_count, len(templates) - 1)
                response = refactor_sentence(templates[idx], script_data)

    # Build LLM prompt context (used by generate_llm if needed)
    llm_prompt_context = None
    if response is None:
        last_agent_msg = ""
        last_customer_msg = ""
        for msg in reversed(conversation_history or []):
            if msg.get("role") == "agent" and not last_agent_msg:
                last_agent_msg = msg.get("content", "")
            elif msg.get("role") == "customer" and not last_customer_msg:
                last_customer_msg = msg.get("content", "")
            if last_agent_msg and last_customer_msg:
                break

        selected_opening = random.choice(_RESPONSE_OPENINGS)
        guidance = _STATE_GUIDANCE.get(new_state, "")

        llm_prompt_context = _PROMPTS["best_response_prompt"].format(
            agent_name=script_data.get("agent_name", ""),
            student_name=script_data.get("student_name", ""),
            class_time=script_data.get("class_time", ""),
            program_name=script_data.get("program_name", ""),
            hotline=script_data.get("hotline", ""),
            current_state=new_state,
            guidance=guidance,
            last_agent_msg=last_agent_msg,
            last_customer_msg=last_customer_msg,
            selected_opening=selected_opening,
        ).strip()

    return {
        "rule_response": response,
        "needs_llm_generate": response is None,
        "llm_prompt_context": llm_prompt_context,
    }
