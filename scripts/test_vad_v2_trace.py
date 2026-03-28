"""Trace: what exactly does vad_infer receive as audio input?"""

import asyncio
import os
import sys
import time

import numpy as np
from scipy.io import wavfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from hush.core import Hush, graph, START, END, PARENT, op
from speech.audio_processor_v2 import audio_processor

# Intercept: replace vad_infer with tracer
@op
def trace_vad_input(audio, cmc_time: int, recv_time: int) -> dict:
    """Just log what we receive."""
    if isinstance(audio, np.ndarray):
        print(f"  audio: ndarray shape={audio.shape} dtype={audio.dtype}")
    elif isinstance(audio, list):
        print(f"  audio: list len={len(audio)} type[0]={type(audio[0]).__name__ if audio else '?'}")
    else:
        print(f"  audio: type={type(audio).__name__}")
    return {"logged": True}


@op
async def wav_source(wav_path: str):
    sr, data = wavfile.read(wav_path)
    silence = np.zeros(int(0.3 * sr), dtype=data.dtype)  # short silence
    data = np.concatenate([data[:2400], silence])  # only first 300ms + silence
    for i in range(0, len(data), 320):
        chunk = data[i:i+320]
        if len(chunk) < 320:
            chunk = np.pad(chunk, (0, 320 - len(chunk)))
        yield {"raw_chunk": chunk.tobytes(), "cmc_time": int(time.time() * 1000)}
        await asyncio.sleep(0)


@graph
def pipeline(wav_path):
    source = wav_source(wav_path=wav_path)
    audio = audio_processor(raw_chunk=source["raw_chunk"], cmc_time=source["cmc_time"])
    trace = trace_vad_input(audio=audio["audio"], cmc_time=audio["cmc_time"], recv_time=audio["recv_time"])
    START >> source >> audio >> trace >> END


AUDIO_DIR = os.path.join(os.path.dirname(__file__), "../tests/speech/audio")


async def main():
    wav = os.path.join(AUDIO_DIR, "03_confirm.wav")
    wf = pipeline(wav_path=wav)
    engine = Hush(wf)
    result = await engine.run(inputs={})
    print(f"\nDone. logged count: {result.get('logged')}")


asyncio.run(main())
