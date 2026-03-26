"""Test 3b: wav_source → AudioProcessor → MockVad → END.

MockVad: same interface as VadDetector but returns immediately (no ONNX).
Tests if deadlock is caused by VAD processing or by scheduler chained streaming.
"""

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
from speech.audio_processor import AudioProcessor


@op
async def wav_source(wav_path: str):
    sr, audio = wavfile.read(wav_path)
    silence = np.zeros(int(1.0 * sr), dtype=audio.dtype)
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


@op
def mock_vad(audio: np.ndarray, cmc_time: int, recv_time: int):
    """Mock VAD — pass through, no processing, no yield."""
    return {"speech_audio": audio, "speech_duration_ms": 32.0, "cmc_start_time": cmc_time}


@graph
def test_graph(wav_path):
    s = wav_source(wav_path=wav_path)
    a = AudioProcessor(
        name="audio",
        inputs={"raw_chunk": s["raw_chunk"], "cmc_time": s["cmc_time"]},
    )
    v = mock_vad(audio=a["audio"], cmc_time=a["cmc_time"], recv_time=a["recv_time"])
    START >> s >> a >> v >> END


async def main():
    wav = os.path.join(os.path.dirname(__file__), "../tests/speech/audio/03_confirm.wav")
    print(f"Test: wav_source → AudioProcessor → MockVad → END")
    wf = test_graph(wav_path=wav)
    engine = Hush(wf)
    t0 = time.time()
    result = await engine.run(inputs={})
    elapsed = (time.time() - t0) * 1000
    print(f"  Time: {elapsed:.0f}ms")
    keys = [k for k in result if k != "$state"]
    print(f"  Output keys: {keys}")
    for k in keys:
        v = result[k]
        if isinstance(v, list):
            print(f"    {k}: len={len(v)}")
        elif hasattr(v, "shape"):
            print(f"    {k}: shape={v.shape}")
        else:
            print(f"    {k}: {v}")
    print("  PASS" if keys else "  FAIL")


if __name__ == "__main__":
    asyncio.run(main())
