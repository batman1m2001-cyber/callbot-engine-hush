"""Test full callbot pipeline — single turn and multi-turn."""

import asyncio
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "../../.env"))

from hush.core import Hush

from pipeline.callbot import callbot_pipeline, callbot_pipeline_multi

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


async def test_single_turn(name, wav_file, expected_intent=None):
    """Test single turn: 1 WAV → full pipeline."""
    wav_path = os.path.join(AUDIO_DIR, wav_file)
    if not os.path.exists(wav_path):
        print(f"  SKIP: {wav_file} not found")
        return None

    print(f"\n{'='*60}")
    print(f"TEST: {name}")
    print(f"  WAV: {wav_file}")

    wf = callbot_pipeline(wav_path=wav_path, script_data=SCRIPT_DATA)
    engine = Hush(wf, env=os.path.join(os.path.dirname(__file__), "../../.env"),
                  resources=os.path.join(os.path.dirname(__file__), "../../resources.yaml"))

    t0 = time.time()
    result = await engine.run(inputs={})
    elapsed = (time.time() - t0) * 1000

    response = result.get("response")
    intent = result.get("intent")
    new_state = result.get("new_state")

    # Unwrap lists (streaming output collection)
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
        print(f"  Intent check: {'✓' if match else '✗'} (expected {expected_intent})")

    return result


async def test_multi_turn():
    """Test multi-turn: multiple WAVs → shared state carries across turns."""
    wav_files = [
        "03_confirm.wav",       # Turn 1: CONFIRM_CUSTOMER → confirm
        "01_student_joining.wav",  # Turn 2: REMINDER → student_joining
    ]
    wav_paths = [os.path.join(AUDIO_DIR, f) for f in wav_files]

    # Check files exist
    for p in wav_paths:
        if not os.path.exists(p):
            print(f"  SKIP: {p} not found")
            return

    print(f"\n{'='*60}")
    print(f"TEST: Multi-turn conversation")
    print(f"  WAVs: {wav_files}")

    wf = callbot_pipeline_multi(wav_paths=wav_paths, script_data=SCRIPT_DATA)
    engine = Hush(wf, env=os.path.join(os.path.dirname(__file__), "../../.env"),
                  resources=os.path.join(os.path.dirname(__file__), "../../resources.yaml"))

    t0 = time.time()
    result = await engine.run(inputs={})
    elapsed = (time.time() - t0) * 1000

    responses = result.get("response", [])
    intents = result.get("intent", [])
    states = result.get("new_state", [])

    if not isinstance(responses, list):
        responses = [responses]
    if not isinstance(intents, list):
        intents = [intents]
    if not isinstance(states, list):
        states = [states]

    print(f"  Turns: {len(intents)}")
    for i in range(len(intents)):
        r = responses[i] if i < len(responses) else "?"
        print(f"    Turn {i+1}: intent={intents[i]}, state={states[i] if i < len(states) else '?'}")
        print(f"             response=\"{(r or '')[:60]}\"")

    print(f"  Total time: {elapsed:.0f}ms")
    print(f"  Turns detected: {len(intents)} (expected {len(wav_files)})")


async def main():
    print("Full Callbot Pipeline Tests")
    print("=" * 60)

    # Single turn tests
    await test_single_turn("Confirm customer", "03_confirm.wav", "confirm")
    await test_single_turn("Student joining", "01_student_joining.wav", "student_joining")
    await test_single_turn("Busy", "04_busy.wav", "busy")

    # Multi-turn test
    await test_multi_turn()

    print(f"\n{'='*60}")
    print("All tests completed!")


if __name__ == "__main__":
    asyncio.run(main())
