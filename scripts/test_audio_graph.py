"""Test @graph audio_processor in isolation."""

import asyncio
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ["LOG_LEVEL"] = "WARNING"

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "../.env"))

from hush.core import Hush, graph, START, END, PARENT, op
from speech.audio_processor import audio_processor


@op
async def source(n: int):
    for i in range(n):
        chunk = (np.random.randn(320) * 1000).astype(np.int16).tobytes()
        yield {"raw_chunk": chunk, "cmc_time": 1000 + i * 40}
        await asyncio.sleep(0)


@graph
def test(n):
    s = source(n=n)
    a = audio_processor(raw_chunk=s["raw_chunk"], cmc_time=s["cmc_time"])
    START >> s >> a >> END


async def main():
    wf = test(n=10)
    engine = Hush(wf)
    result = await engine.run(inputs={})
    audio = result.get("audio", [])
    print(f"Output chunks: {len(audio) if isinstance(audio, list) else 1}")
    if isinstance(audio, list) and audio:
        first = audio[0]
        print(f"First chunk: type={type(first).__name__}, shape={getattr(first, 'shape', 'N/A')}")


if __name__ == "__main__":
    asyncio.run(main())
