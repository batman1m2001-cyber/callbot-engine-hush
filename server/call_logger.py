"""Call logger — builds log_summary post-call and writes to JSONL."""

import json
import logging
from datetime import datetime
from pathlib import Path

from server import config
from agents.educa_reminder.ops.call_result import build_call_result

LOGGER = logging.getLogger(__name__)


def build_log_summary(
    handle_state,
    script_data: dict,
    call_id: str,
    phone_number: str = "",
    customer_id: str = "",
    start_time: float = 0.0,
    end_time: float = 0.0,
) -> dict:
    """Build log_summary from pipeline state after call ends.

    Reads final shared vars from handle_state, calls build_call_result()
    to get ARId/Comment/report_result, then assembles the summary.
    """
    graph_name = "ws_callbot_pipeline"

    # Extract shared vars from pipeline state
    try:
        current_state = handle_state[graph_name, "current_state"]
    except (KeyError, IndexError):
        current_state = "UNKNOWN"

    try:
        conversation_history = handle_state[graph_name, "conversation_history"]
    except (KeyError, IndexError):
        conversation_history = []

    # Build transcript from conversation history
    transcript = []
    for entry in (conversation_history or []):
        speaker = "agent" if entry.get("role") == "assistant" else "customer"
        transcript.append({
            "speaker": speaker,
            "text": entry.get("content", ""),
        })

    # Extract last turn's intent/state from the final educa_agent frame
    # These are the last values written to shared state
    last_intent = ""
    last_previous_state = ""
    last_customer_speech = ""
    customer_confirmed = False
    new_phone_number = ""

    # Try to get values from the last turn's state
    try:
        last_intent = handle_state[graph_name, "intent"]
        if isinstance(last_intent, list):
            last_intent = last_intent[-1] if last_intent else ""
    except (KeyError, IndexError):
        pass

    try:
        last_customer_speech = handle_state[graph_name, "last_agent_response"]
    except (KeyError, IndexError):
        pass

    # Build CRM call_result using the plain function
    cr = build_call_result(
        current_state=current_state,
        intent=last_intent,
        previous_state=last_previous_state,
        customer_speech=last_customer_speech,
        customer_confirmed=customer_confirmed,
        new_phone_number=new_phone_number,
        script_data=script_data,
    )
    call_result = cr["call_result"]

    duration = (end_time - start_time) if (start_time and end_time) else 0.0

    return {
        "call_id": call_id,
        "customer_id": customer_id,
        "phone_number": phone_number,
        "action_code": call_result.get("ARId", "UNKNOWN"),
        "end_reason": current_state,
        "duration_seconds": round(duration, 2),
        "transcript": transcript,
        "call_result": call_result,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }


def write(call_id: str, log_summary: dict):
    """Append log_summary to logs/educa_reminder/{YYYYMMDD}/calls.jsonl."""
    date_dir = datetime.utcnow().strftime("%Y%m%d")
    path = Path(config.LOG_BASE_DIR) / "educa_reminder" / date_dir / "calls.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(log_summary, ensure_ascii=False, default=str)
    with open(path, "a") as f:
        f.write(line + "\n")
    LOGGER.info(f"[{call_id}] Call log written to {path}")
