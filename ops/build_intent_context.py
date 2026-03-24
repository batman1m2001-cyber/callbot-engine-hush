"""build_intent_context op — prepare LLM prompt data for intent classification."""

from pathlib import Path

import yaml

from hush.core.ops import op

# Load prompts config once
_YAML_PATH = Path(__file__).parent.parent / "data" / "prompts.yaml"
with open(_YAML_PATH, "r", encoding="utf-8") as f:
    _CONFIG = yaml.safe_load(f)

_INTENT_DEFS = _CONFIG["intent_definitions"]
_FEW_SHOT = _CONFIG.get("few_shot_examples", {})
_STATE_ALLOWED = _CONFIG["state_allowed_intents"]
_PROMPTS = _CONFIG["prompts"]


def _build_intent_definitions_xml(allowed_intents: list) -> str:
    """Build XML block of intent definitions for allowed intents."""
    parts = []
    for intent_name in allowed_intents:
        info = _INTENT_DEFS.get(intent_name)
        if not info:
            continue
        examples_str = " | ".join(info["examples"])
        parts.append(
            f'<intent name="{intent_name}">\n'
            f'  <description>{info["description"]}</description>\n'
            f'  <examples>{examples_str}</examples>\n'
            f'</intent>'
        )
    return "\n".join(parts)


def _build_few_shot_xml(state: str, max_examples: int = 3) -> str:
    """Build few-shot examples XML block for a state."""
    examples = _FEW_SHOT.get(state, [])
    if not examples:
        return ""
    lines = []
    for ex in examples[:max_examples]:
        lines.append(
            f'<example>\n'
            f'  <agent>"{ex["agent"]}"</agent>\n'
            f'  <customer>"{ex["customer"]}"</customer>\n'
            f'  <result>{ex["intent"]}</result>\n'
            f'</example>'
        )
    return "<examples_to_learn>\n" + "\n".join(lines) + "\n</examples_to_learn>\n"


@op
def build_intent_context(
    current_state: str,
    normalized_text: str,
    agent_speech: str,
    script_data: dict,
):
    """Build all prompt data needed for LLM intent classification."""
    allowed_intents = _STATE_ALLOWED.get(current_state, [])
    intent_definitions_xml = _build_intent_definitions_xml(allowed_intents)
    few_shot_xml = _build_few_shot_xml(current_state)
    allowed_intents_csv = ", ".join(allowed_intents)

    # Build the user prompt from intent_prompt template
    intent_prompt = _PROMPTS["intent_prompt"].format(
        program_name=script_data.get("program_name", ""),
        student_name=script_data.get("student_name", ""),
        class_time=script_data.get("class_time", ""),
        state=current_state,
        intent_definitions_str=intent_definitions_xml,
        allowed_intents_str=allowed_intents_csv,
        few_shot_str=few_shot_xml,
        agent_speech=agent_speech or "",
        customer_speech=normalized_text,
    ).strip()

    # Validators dict ready for ParserOp — @fallback is default when all retries exhausted
    validators = {"result": [i for i in allowed_intents if i != "fallback"] + ["@fallback"]}

    return {
        "analyzer_system_prompt": _PROMPTS["analyzer_system_prompt"].strip(),
        "intent_prompt": intent_prompt,
        "validators": validators,
    }
