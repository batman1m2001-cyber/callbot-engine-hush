"""Benchmark: concurrent callbot pipeline runs (simulate CCU).

Each "call" = 1 WAV file → full pipeline (audio → VAD → STT → LLM → TTS).
Runs N calls concurrently, measures per-call and total time.
"""

import asyncio
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "../.env"))

from hush.core import Hush
from pipeline.callbot import callbot_pipeline

AUDIO_DIR = os.path.join(os.path.dirname(__file__), "../tests/speech/audio")

SCRIPT_DATA = {
    "student_name": "Minh",
    "class_time": "19:00",
    "program_name": "AI CLASS",
    "agent_name": "Linh",
    "hotline": "1900636464",
    "parent_name": "anh chị",
}

# WAV files to cycle through for each call
WAV_FILES = [
    "01_student_joining.wav",
    "03_confirm.wav",
    "04_busy.wav",
    "06_read_phone.wav",
    "07_fallback.wav",
]


async def run_single_call(call_id: int, wav_file: str) -> dict:
    """Run one callbot pipeline."""
    wav_path = os.path.join(AUDIO_DIR, wav_file)
    t0 = time.perf_counter()

    wf = callbot_pipeline(wav_path=wav_path, script_data=SCRIPT_DATA)
    engine = Hush(wf, env=os.path.join(os.path.dirname(__file__), "../.env"),
                  resources=os.path.join(os.path.dirname(__file__), "../resources.yaml"))
    result = await engine.run(inputs={})

    elapsed = (time.perf_counter() - t0) * 1000

    intent = result.get("intent")
    if isinstance(intent, list):
        intent = intent[0] if intent else None

    return {
        "call_id": call_id,
        "wav": wav_file,
        "intent": intent,
        "time_ms": elapsed,
    }


async def bench(ccu: int):
    """Run N concurrent calls."""
    print(f"\n{'='*60}")
    print(f"CCU={ccu}: Running {ccu} concurrent callbot pipelines")
    print(f"{'='*60}")

    # Assign WAV files round-robin
    tasks = []
    for i in range(ccu):
        wav = WAV_FILES[i % len(WAV_FILES)]
        tasks.append(run_single_call(i, wav))

    t0 = time.perf_counter()
    results = await asyncio.gather(*tasks, return_exceptions=True)
    total_ms = (time.perf_counter() - t0) * 1000

    # Report
    successes = [r for r in results if isinstance(r, dict)]
    errors = [r for r in results if isinstance(r, Exception)]

    times = [r["time_ms"] for r in successes]
    if times:
        avg = sum(times) / len(times)
        p50 = sorted(times)[len(times) // 2]
        p99 = sorted(times)[int(len(times) * 0.99)]
        slowest = max(times)
        fastest = min(times)
    else:
        avg = p50 = p99 = slowest = fastest = 0

    print(f"\n  Results: {len(successes)} ok, {len(errors)} errors")
    print(f"  Total wall time: {total_ms:.0f}ms")
    print(f"  Per-call: avg={avg:.0f}ms  p50={p50:.0f}ms  p99={p99:.0f}ms")
    print(f"            fastest={fastest:.0f}ms  slowest={slowest:.0f}ms")
    print(f"  Throughput: {len(successes) / (total_ms / 1000):.1f} calls/sec")

    if successes:
        print(f"\n  Per-call breakdown:")
        for r in sorted(successes, key=lambda x: x["call_id"]):
            print(f"    Call {r['call_id']:2d}: {r['wav']:<30s} intent={r['intent']:<20s} {r['time_ms']:.0f}ms")

    if errors:
        print(f"\n  Errors:")
        for e in errors:
            print(f"    {e}")


async def warmup():
    """Warmup: run 1 call to load models, init connections."""
    print("Warming up (1 call to load models + connections)...")
    wav_path = os.path.join(AUDIO_DIR, WAV_FILES[0])
    wf = callbot_pipeline(wav_path=wav_path, script_data=SCRIPT_DATA)
    engine = Hush(wf, env=os.path.join(os.path.dirname(__file__), "../.env"),
                  resources=os.path.join(os.path.dirname(__file__), "../resources.yaml"))
    t0 = time.perf_counter()
    await engine.run(inputs={})
    print(f"Warmup done: {(time.perf_counter() - t0) * 1000:.0f}ms\n")


async def main():
    await warmup()

    for ccu in [1, 2, 4, 8]:
        await bench(ccu)

    print(f"\n{'='*60}")
    print("Benchmark complete!")


if __name__ == "__main__":
    asyncio.run(main())
