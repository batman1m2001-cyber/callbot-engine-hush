"""Debug vad_detector_v2 — trace each op."""

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
from speech.audio_processor_v2 import audio_processor
from speech.vad_detector_v2 import vad_detector

AUDIO_DIR = os.path.join(os.path.dirname(__file__), "../tests/speech/audio")


@op
async def wav_source(wav_path: str):
    sr, audio_data = wavfile.read(wav_path)
    silence = np.zeros(int(1.0 * sr), dtype=audio_data.dtype)
    audio_data = np.concatenate([audio_data, silence])
    for i in range(0, len(audio_data), 320):
        chunk = audio_data[i:i + 320]
        if len(chunk) < 320:
            chunk = np.pad(chunk, (0, 320 - len(chunk)))
        yield {"raw_chunk": chunk.tobytes(), "cmc_time": int(time.time() * 1000)}
        await asyncio.sleep(0)


@graph
def pipeline(wav_path):
    source = wav_source(wav_path=wav_path)
    audio = audio_processor(raw_chunk=source["raw_chunk"], cmc_time=source["cmc_time"])
    vad = vad_detector(audio=audio["audio"], cmc_time=audio["cmc_time"], recv_time=audio["recv_time"])
    START >> source >> audio >> vad >> END


async def main():
    wav = os.path.join(AUDIO_DIR, "03_confirm.wav")
    wf = pipeline(wav_path=wav)
    engine = Hush(wf)
    result = await engine.run(inputs={})

    segments = result.get("speech_audio", [])
    n = len(segments) if isinstance(segments, list) else (1 if segments is not None else 0)
    print(f"\nSegments: {n}")


if __name__ == "__main__":
    asyncio.run(main())
