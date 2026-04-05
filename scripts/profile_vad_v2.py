"""Profile VAD v2 (@graph) vs v1 (BaseOp) to pinpoint where the 21ms overhead lives.

Uses cProfile + pstats for call-level breakdown, plus fine-grained perf_counter
timing around graph construction, engine init, and engine.run().
"""

import asyncio
import cProfile
import os
import pstats
import sys
import time
from io import StringIO

import numpy as np
from scipy.io import wavfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ["LOG_LEVEL"] = "WARNING"

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "../.env"))

from hush.core import Hush, graph, START, END, PARENT, op
from speech.audio_processor import AudioProcessor
from speech.vad_detector import VadDetector
from speech.vad_detector_v2 import vad_detector as vad_detector_v2

AUDIO_DIR = os.path.join(os.path.dirname(__file__), "../tests/speech/audio")
WAV_FILE = "06_read_phone.wav"


async def preprocess_wav(wav_path):
    """Run AudioProcessor v1 once, return list of (audio, cmc_time, recv_time) tuples."""
    sr, raw_audio = wavfile.read(wav_path)
    silence = np.zeros(int(sr * 1.0), dtype=raw_audio.dtype)
    raw_audio = np.concatenate([raw_audio, silence])
    proc = AudioProcessor(name="pre", use_preprocess=False)
    chunks = []
    for i in range(0, len(raw_audio), 320):
        chunk = raw_audio[i:i + 320]
        if len(chunk) < 320:
            chunk = np.pad(chunk, (0, 320 - len(chunk)))
        async for out in proc._process(raw_chunk=chunk.tobytes(), cmc_time=1000 + i):
            chunks.append((out["audio"].copy(), out["cmc_time"], out["recv_time"]))
    return chunks


# ── V1: direct BaseOp ──

async def run_v1(audio_chunks):
    vad = VadDetector(name="v1_vad")
    segments = []
    for audio, cmc_time, recv_time in audio_chunks:
        async for seg in vad._process(audio=audio, cmc_time=cmc_time, recv_time=recv_time):
            segments.append({
                "speech_audio": seg["speech_audio"].copy(),
                "speech_duration_ms": seg["speech_duration_ms"],
                "num_chunks": seg["num_chunks"],
            })
    return segments


# ── V2: @graph via engine ──

@op
async def chunk_source(audio_chunks: list):
    for audio, cmc_time, recv_time in audio_chunks:
        yield {"audio": audio, "cmc_time": cmc_time, "recv_time": recv_time}
        await asyncio.sleep(0)


@graph
def pipeline_v2(audio_chunks):
    source = chunk_source(audio_chunks=audio_chunks)
    vad = vad_detector_v2(
        audio=source["audio"], cmc_time=source["cmc_time"], recv_time=source["recv_time"],
    )
    START >> source >> vad >> END


def _collect_segments(result):
    segments = []
    speech_audios = result.get("speech_audio", [])
    durations = result.get("speech_duration_ms", [])
    num_chunks = result.get("num_chunks", [])
    if not isinstance(speech_audios, list):
        speech_audios = [speech_audios] if speech_audios is not None else []
    if not isinstance(durations, list):
        durations = [durations] if durations is not None else []
    if not isinstance(num_chunks, list):
        num_chunks = [num_chunks] if num_chunks is not None else []
    for i in range(len(speech_audios)):
        segments.append({
            "speech_audio": np.array(speech_audios[i], dtype=np.float32)
                if not isinstance(speech_audios[i], np.ndarray) else speech_audios[i],
            "speech_duration_ms": durations[i] if i < len(durations) else 0,
            "num_chunks": num_chunks[i] if i < len(num_chunks) else 0,
        })
    return segments


# ── Profiling helpers ──

def profile_async(coro_func, *args, **kwargs):
    """Profile an async function via cProfile, return (result, pstats_string)."""
    pr = cProfile.Profile()
    pr.enable()
    result = asyncio.get_event_loop().run_until_complete(coro_func(*args, **kwargs))
    pr.disable()
    s = StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
    ps.print_stats(20)
    return result, s.getvalue()


async def run_v2_phased(chunks):
    """Run v2 with fine-grained timing around each phase."""
    timings = {}

    t0 = time.perf_counter()
    wf = pipeline_v2(audio_chunks=chunks)
    timings["graph_construct"] = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    engine = Hush(wf)
    timings["engine_init"] = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    result = await engine.run(inputs={})
    timings["engine_run"] = (time.perf_counter() - t0) * 1000

    segs = _collect_segments(result)
    return segs, timings


async def main():
    wav_path = os.path.join(AUDIO_DIR, WAV_FILE)
    if not os.path.exists(wav_path):
        print(f"ERROR: {wav_path} not found")
        return

    print(f"Profiling VAD v1 vs v2 on: {WAV_FILE}")
    print("=" * 75)

    # Preprocess
    print("\nPreprocessing audio...")
    chunks = await preprocess_wav(wav_path)
    print(f"  {len(chunks)} chunks ready\n")

    # ── Warmup (1 run each to JIT / cache) ──
    print("Warmup run (v1)...")
    await run_v1(chunks)
    print("Warmup run (v2)...")
    wf = pipeline_v2(audio_chunks=chunks)
    engine = Hush(wf)
    await engine.run(inputs={})

    # ── Phase 1: Fine-grained perf_counter timing for v2 ──
    print("\n" + "=" * 75)
    print("PHASE 1: Fine-grained timing breakdown (v2)")
    print("=" * 75)

    N_RUNS = 5
    all_timings = []
    for i in range(N_RUNS):
        _, timings = await run_v2_phased(chunks)
        all_timings.append(timings)

    # Also time v1 for comparison
    v1_times = []
    for i in range(N_RUNS):
        t0 = time.perf_counter()
        await run_v1(chunks)
        v1_times.append((time.perf_counter() - t0) * 1000)

    avg = lambda key: sum(t[key] for t in all_timings) / N_RUNS
    avg_v1 = sum(v1_times) / N_RUNS
    avg_total_v2 = avg("graph_construct") + avg("engine_init") + avg("engine_run")

    print(f"\n  V1 total (BaseOp direct):     {avg_v1:7.2f}ms")
    print(f"  V2 total (@graph pipeline):   {avg_total_v2:7.2f}ms")
    print(f"  Overhead (V2 - V1):           {avg_total_v2 - avg_v1:+7.2f}ms")
    print()
    print(f"  V2 breakdown (avg of {N_RUNS} runs):")
    print(f"    graph construction:         {avg('graph_construct'):7.2f}ms")
    print(f"    Hush(wf) engine init:       {avg('engine_init'):7.2f}ms")
    print(f"    engine.run() execution:     {avg('engine_run'):7.2f}ms")
    print()
    print(f"  Per-run details:")
    for i, t in enumerate(all_timings):
        total = t["graph_construct"] + t["engine_init"] + t["engine_run"]
        print(f"    run {i}: construct={t['graph_construct']:5.2f}ms  "
              f"init={t['engine_init']:5.2f}ms  "
              f"run={t['engine_run']:5.2f}ms  "
              f"total={total:5.2f}ms")

    # ── Phase 2: cProfile for V1 ──
    print("\n" + "=" * 75)
    print("PHASE 2: cProfile — V1 (BaseOp direct)")
    print("=" * 75)

    async def _v1_profiled():
        return await run_v1(chunks)

    _, v1_stats = profile_async(_v1_profiled)
    print(v1_stats)

    # ── Phase 3: cProfile for V2 ──
    print("=" * 75)
    print("PHASE 3: cProfile — V2 (@graph pipeline)")
    print("=" * 75)

    async def _v2_profiled():
        wf = pipeline_v2(audio_chunks=chunks)
        engine = Hush(wf)
        result = await engine.run(inputs={})
        return _collect_segments(result)

    _, v2_stats = profile_async(_v2_profiled)
    print(v2_stats)

    # ── Phase 4: cProfile for V2 engine.run() ONLY ──
    print("=" * 75)
    print("PHASE 4: cProfile — V2 engine.run() only (graph+init outside profile)")
    print("=" * 75)

    wf = pipeline_v2(audio_chunks=chunks)
    engine = Hush(wf)

    async def _v2_run_only():
        return await engine.run(inputs={})

    _, v2_run_stats = profile_async(_v2_run_only)
    print(v2_run_stats)


if __name__ == "__main__":
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(main())
    finally:
        loop.close()
