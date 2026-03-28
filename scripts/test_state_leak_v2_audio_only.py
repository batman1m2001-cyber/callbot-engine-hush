"""Test: audio_processor_v2 only — check state isolation."""

import asyncio
import os
import sys
import time

import numpy as np
from scipy.io import wavfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ["LOG_LEVEL"] = "WARNING"

from hush.core import Hush, graph, START, END, PARENT, op
from speech.audio_processor_v2 import audio_processor


@op
async def wav_source(wav_path: str):
    sr, audio = wavfile.read(wav_path)
    silence = np.zeros(int(0.5 * sr), dtype=audio.dtype)
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
    audio = audio_processor(raw_chunk=source["raw_chunk"], cmc_time=source["cmc_time"])
    START >> source >> audio >> END


AUDIO_DIR = os.path.join(os.path.dirname(__file__), "../tests/speech/audio")


async def main():
    wav = os.path.join(AUDIO_DIR, "03_confirm.wav")

    wf = pipeline(wav_path=wav)
    engine = Hush(wf)

    # Request 1
    result1 = await engine.run(inputs={})
    audio1 = result1.get("audio", [])
    n1 = len(audio1) if isinstance(audio1, list) else 0
    print(f"Request 1: {n1} chunks")

    # Request 2
    result2 = await engine.run(inputs={})
    audio2 = result2.get("audio", [])
    n2 = len(audio2) if isinstance(audio2, list) else 0
    print(f"Request 2: {n2} chunks")

    # Request 3
    result3 = await engine.run(inputs={})
    audio3 = result3.get("audio", [])
    n3 = len(audio3) if isinstance(audio3, list) else 0
    print(f"Request 3: {n3} chunks")

    if n1 == n2 == n3 and n1 > 0:
        print(f"\n✓ All 3 requests: {n1} chunks — state isolated")
    elif n1 > 0 and (n2 != n1 or n3 != n1):
        print(f"\n✗ STATE LEAK — {n1}, {n2}, {n3} (should be equal)")
    else:
        print(f"\n? Results: {n1}, {n2}, {n3}")


if __name__ == "__main__":
    asyncio.run(main())
