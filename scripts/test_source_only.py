"""Test 1: wav_source → END (no downstream ops)."""

import asyncio
import os
import sys
import time

import numpy as np
from scipy.io import wavfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ["LOG_LEVEL"] = "DEBUG"

from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "../.env"))

from hush.core import Hush, graph, START, END, PARENT, op


@op
async def wav_source(wav_path: str):
    sr, audio = wavfile.read(wav_path)
    silence = np.zeros(int(0.5 * sr), dtype=audio.dtype)
    audio = np.concatenate([audio, silence])
    count = 0
    for i in range(0, len(audio), 320):
        chunk = audio[i : i + 320]
        if len(chunk) < 320:
            chunk = np.pad(chunk, (0, 320 - len(chunk)))
        yield {"raw_chunk": chunk.tobytes(), "cmc_time": 1000 + i}
        count += 1
        await asyncio.sleep(0)
    print(f"  wav_source done: {count} chunks")


@graph
def test_graph(wav_path):
    s = wav_source(wav_path=wav_path)
    START >> s >> END


async def main():
    wav = os.path.join(os.path.dirname(__file__), "../tests/speech/audio/03_confirm.wav")
    print(f"Test: wav_source only → END")
    wf = test_graph(wav_path=wav)
    engine = Hush(wf)
    t0 = time.time()
    result = await engine.run(inputs={})
    print(f"  Time: {(time.time()-t0)*1000:.0f}ms")
    print(f"  Output keys: {[k for k in result if k != '$state']}")
    print("  PASS" if result else "  FAIL")


if __name__ == "__main__":
    asyncio.run(main())
