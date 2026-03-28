"""Reproduce: AudioProcessor/VadDetector state leak across requests.

Build graph once → run twice → check if 2nd request has clean state.
"""

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
from speech.audio_processor import AudioProcessor
from speech.vad_detector import VadDetector

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


# Build graph ONCE
@graph
def pipeline(wav_path):
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
    wav = os.path.join(AUDIO_DIR, "03_confirm.wav")

    # Request 1
    wf = pipeline(wav_path=wav)
    engine = Hush(wf)
    result1 = await engine.run(inputs={})
    segments1 = result1.get("speech_audio", [])
    n1 = len(segments1) if isinstance(segments1, list) else (1 if segments1 is not None else 0)
    print(f"Request 1: {n1} segments")

    # Check internal state BEFORE request 2
    audio_op = wf._ops.get("audio")
    vad_op = wf._ops.get("vad")
    print(f"  AudioProcessor buffer_len after req1: {audio_op._buffer_len}")
    print(f"  VadDetector triggered after req1: {vad_op._triggered}")
    print(f"  VadDetector speech_buffer len: {len(vad_op._speech_buffer)}")
    print(f"  VadDetector current_sample: {vad_op._current_sample}")

    # Request 2 — reuse same graph
    result2 = await engine.run(inputs={})
    segments2 = result2.get("speech_audio", [])
    n2 = len(segments2) if isinstance(segments2, list) else (1 if segments2 is not None else 0)
    print(f"\nRequest 2: {n2} segments")

    print(f"  AudioProcessor buffer_len after req2: {audio_op._buffer_len}")
    print(f"  VadDetector current_sample: {vad_op._current_sample}")

    # Check
    if audio_op._buffer_len == 0 and vad_op._current_sample == 0:
        print("\n✓ State reset between requests")
    else:
        print(f"\n✗ STATE LEAK — buffer_len={audio_op._buffer_len}, current_sample={vad_op._current_sample}")
        print("  Request 2 inherited dirty state from Request 1!")


if __name__ == "__main__":
    asyncio.run(main())
