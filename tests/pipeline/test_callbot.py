"""Test full callbot pipeline — single turn and multi-turn."""

import asyncio
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "../../.env"))

from hush.core import Hush
from hush.core.tracing import get_flush_worker
from hush.telemetry import LangfuseTracer

from pipeline.callbot import callbot_pipeline

AUDIO_DIR = os.path.join(os.path.dirname(__file__), "../speech/audio")

SCRIPT_DATA = {
    "student_name": "Minh",
    "class_time": "19:00",
    "program_name": "AI CLASS",
    "agent_name": "Linh",
    "company": "Edupia",
    "hotline": "1900636464",
    "phone_number": "0912345678",
}


async def run_single_turn(name, wav_file, expected_intent=None):
    """Test single turn: 1 WAV → full pipeline."""
    wav_path = os.path.join(AUDIO_DIR, wav_file)
    if not os.path.exists(wav_path):
        print(f"  SKIP: {wav_file} not found")
        return None

    print(f"\n{'='*60}")
    print(f"TEST: {name}")
    print(f"  WAV: {wav_file}")

    wf = callbot_pipeline(wav_path=wav_path, script_data=SCRIPT_DATA)
    tracer = LangfuseTracer(resource="langfuse:default")
    engine = Hush(wf, env=os.path.join(os.path.dirname(__file__), "../../.env"),
                  resources=os.path.join(os.path.dirname(__file__), "../../resources.yaml"),
                  tracer=tracer)

    t0 = time.time()
    result = await engine.run(inputs={})
    elapsed = (time.time() - t0) * 1000

    response = result.get("response")
    intent = result.get("intent")
    new_state = result.get("new_state")

    if isinstance(response, list):
        response = response[0] if response else None
    if isinstance(intent, list):
        intent = intent[0] if intent else None
    if isinstance(new_state, list):
        new_state = new_state[0] if new_state else None

    print(f"  Intent: {intent}")
    print(f"  New State: {new_state}")
    print(f"  Response: \"{(response or '')[:80]}\"")
    print(f"  Time: {elapsed:.0f}ms")

    if expected_intent:
        match = intent == expected_intent
        print(f"  Intent check: {'OK' if match else 'FAIL'} (expected {expected_intent})")

    return result


async def main():
    print("Full Callbot Pipeline Tests")
    print("=" * 60)

    await run_single_turn("Confirm customer", "03_confirm.wav", "confirm")
    await run_single_turn("Student joining", "01_student_joining.wav", "student_joining")
    await run_single_turn("Busy", "04_busy.wav", "busy")

    print(f"\n{'='*60}")
    print("Waiting for Langfuse flush...")
    get_flush_worker().wait(timeout=10)
    print("Done.")


if __name__ == "__main__":
    asyncio.run(main())
