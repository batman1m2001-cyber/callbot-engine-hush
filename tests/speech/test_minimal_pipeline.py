"""Minimal pipeline test: wav_source → AudioProcessor → VadDetector."""

import asyncio
import os
import sys
import time

import numpy as np
from scipy.io import wavfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "../../.env"))

from hush.core import Hush, graph, START, END, PARENT, op
from speech.audio_processor import AudioProcessor
from speech.vad_detector import VadDetector

AUDIO_DIR = os.path.join(os.path.dirname(__file__), "audio")


@op
async def wav_source(wav_path: str):
    """Read WAV, yield 320-sample chunks + 1s silence tail."""
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
def minimal_pipeline(wav_path):
    source = wav_source(wav_path=wav_path)
    audio = AudioProcessor(
        name="audio",
        inputs={"raw_chunk": source["raw_chunk"], "cmc_time": source["cmc_time"]},
    )
    vad = VadDetector(
        name="vad",
        inputs={
            "audio": audio["audio"],
            "cmc_time": audio["cmc_time"],
            "recv_time": audio["recv_time"],
        },
    )
    START >> source >> audio >> vad >> END


async def main():
    filepath = os.path.join(AUDIO_DIR, "03_confirm.wav")
    print(f"Testing: {filepath}")

    wf = minimal_pipeline(wav_path=filepath)
    engine = Hush(wf)

    start = time.time()
    result = await engine.run(inputs={})
    elapsed = time.time() - start

    print(f"Time: {elapsed*1000:.0f}ms")
    for k, v in result.items():
        if k != "$state":
            if hasattr(v, "shape"):
                print(f"  {k}: shape={v.shape}")
            elif isinstance(v, list):
                print(f"  {k}: len={len(v)}")
            else:
                print(f"  {k}: {v}")


if __name__ == "__main__":
    asyncio.run(main())
