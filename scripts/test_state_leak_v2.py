"""Test: audio_processor_v2 (@graph + PARENT.shared) — no state leak."""

import asyncio
import os
import sys
import time

import numpy as np
from scipy.io import wavfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ["LOG_LEVEL"] = "WARNING"

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "../.env"))

from hush.core import Hush, graph, START, END, PARENT, op
from speech.audio_processor_v2 import audio_processor
from speech.vad_detector_v2 import vad_detector

AUDIO_DIR = os.path.join(os.path.dirname(__file__), "../tests/speech/audio")


@op
async def wav_source(wav_path: str):
    sr, audio = wavfile.read(wav_path)
    silence = np.zeros(int(1.0 * sr), dtype=audio.dtype)
    audio = np.concatenate([audio, silence])
    for i in range(0, len(audio), 320):
        chunk = audio[i:i + 320]
        if len(chunk) < 320:
            chunk = np.pad(chunk, (0, 320 - len(chunk)))
        yield {"raw_chunk": chunk.tobytes(), "cmc_time": int(time.time() * 1000)}
        await asyncio.sleep(0)


@graph
def pipeline(wav_path):
    source = wav_source(wav_path=wav_path)
    audio = audio_processor(
        raw_chunk=source["raw_chunk"],
        cmc_time=source["cmc_time"],
    )
    vad = vad_detector(
        audio=audio["audio"],
        cmc_time=audio["cmc_time"],
        recv_time=audio["recv_time"],
    )
    START >> source >> audio >> vad >> END


async def main():
    wav = os.path.join(AUDIO_DIR, "03_confirm.wav")

    # Request 1
    print("Request 1:")
    wf = pipeline(wav_path=wav)
    engine = Hush(wf)
    result1 = await engine.run(inputs={})
    segments1 = result1.get("speech_audio", [])
    n1 = len(segments1) if isinstance(segments1, list) else (1 if segments1 is not None else 0)
    print(f"  Segments: {n1}")

    # Request 2 — reuse same engine
    print("\nRequest 2 (reuse engine):")
    result2 = await engine.run(inputs={})
    segments2 = result2.get("speech_audio", [])
    n2 = len(segments2) if isinstance(segments2, list) else (1 if segments2 is not None else 0)
    print(f"  Segments: {n2}")

    # Compare
    if n1 == n2 and n1 > 0:
        print(f"\n✓ Both requests: {n1} segments — state isolated")
    elif n1 != n2:
        print(f"\n✗ STATE LEAK — req1={n1}, req2={n2} segments (should be equal)")
    else:
        print(f"\n? No segments detected in either request")


if __name__ == "__main__":
    asyncio.run(main())
