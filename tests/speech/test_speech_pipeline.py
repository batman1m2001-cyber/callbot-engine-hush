"""Test speech pipeline end-to-end via Hush engine.

Uses generated WAV files as input. Streaming: async source yields chunks → graph processes.
"""

import asyncio
import os
import sys
import time

import numpy as np
from scipy.io import wavfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "../../.env"))

from hush.core import Hush, GraphOp, graph, START, END, PARENT, op
from hush.providers.ops import TritonOp

from speech.audio_processor import AudioProcessor
from speech.vad_detector import VadDetector

AUDIO_DIR = os.path.join(os.path.dirname(__file__), "audio")
CHUNK_SIZE = 320  # 40ms at 8kHz


@op
async def wav_source(wav_path: str):
    """Async generator: read WAV file, yield 320-sample chunks like telco.

    Appends 1s silence tail to trigger VAD end detection.
    """
    sr, audio = wavfile.read(wav_path)
    # Append silence tail
    silence = np.zeros(int(1.0 * sr), dtype=audio.dtype)
    audio = np.concatenate([audio, silence])

    for i in range(0, len(audio), CHUNK_SIZE):
        chunk = audio[i:i + CHUNK_SIZE]
        if len(chunk) < CHUNK_SIZE:
            chunk = np.pad(chunk, (0, CHUNK_SIZE - len(chunk)))
        yield {
            "raw_chunk": chunk.tobytes(),
            "cmc_time": int(time.time() * 1000),
        }
        await asyncio.sleep(0)  # yield control to event loop


@graph
def test_pipeline(wav_path):
    """Audio source → AudioProcessor → VadDetector → STT."""
    source = wav_source(wav_path=wav_path)

    audio = AudioProcessor(
        name="audio",
        inputs={
            "raw_chunk": source["raw_chunk"],
            "cmc_time": source["cmc_time"],
        },
    )

    vad = VadDetector(
        name="vad",
        inputs={
            "audio": audio["audio"],
            "cmc_time": audio["cmc_time"],
            "recv_time": audio["recv_time"],
        },
    )

    stt = TritonOp(
        name="stt",
        url="192.168.1.212:8001",
        model_name="fastconformer_asr",
        inputs_map={"AUDIO_SIGNAL": "speech_audio"},
        outputs_map={"TRANSCRIPT": "transcript"},
        inputs={"speech_audio": vad["speech_audio"]},
    )

    vad["speech_duration_ms"] >> PARENT["speech_duration_ms"]
    vad["cmc_start_time"] >> PARENT["cmc_start_time"]

    START >> source >> audio >> vad >> stt >> END


async def run_test(name: str, filename: str, expected_segments: int, expected_words: list = None):
    """Run one test case."""
    filepath = os.path.join(AUDIO_DIR, filename)
    if not os.path.exists(filepath):
        print(f"  SKIP: {filename} (not found)")
        return

    print(f"\n{'='*60}")
    print(f"TEST: {name}")
    print(f"  File: {filename}")

    wf = test_pipeline(wav_path=filepath)
    engine = Hush(wf)

    start = time.time()
    result = await engine.run(inputs={})
    elapsed = time.time() - start

    transcripts = result.get("transcript", [])
    if isinstance(transcripts, str):
        transcripts = [transcripts]
    durations = result.get("speech_duration_ms", [])
    if isinstance(durations, (int, float)):
        durations = [durations]

    n_segments = len(transcripts) if transcripts else 0
    print(f"  Segments: {n_segments} (expected {expected_segments})")

    if transcripts:
        for i, t in enumerate(transcripts):
            dur = durations[i] if i < len(durations) else "?"
            print(f"    [{i}] \"{t}\" ({dur}ms)")

    if expected_words and transcripts:
        transcript_lower = " ".join(str(t) for t in transcripts).lower()
        for word in expected_words:
            found = word.lower() in transcript_lower
            print(f"    Check '{word}': {'✓' if found else '✗'}")

    status = "✓" if n_segments == expected_segments else "✗"
    print(f"  Time: {elapsed*1000:.0f}ms {status}")


async def main():
    print("Speech Pipeline E2E Test (Hush Engine)")
    print("=" * 60)

    await run_test("Silence", "02_silence.wav", 0)
    await run_test("Student joining", "01_student_joining.wav", 1, ["vào"])
    await run_test("Confirm", "03_confirm.wav", 1, ["đúng"])
    await run_test("Busy", "04_busy.wav", 1, ["bận"])
    await run_test("Read phone", "06_read_phone.wav", 1, ["chín", "tám"])

    print(f"\n{'='*60}")
    print("All tests completed!")


if __name__ == "__main__":
    asyncio.run(main())
