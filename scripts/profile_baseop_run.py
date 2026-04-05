"""Profile BaseOp.run() internals — find where per-chunk overhead comes from.

Wraps key methods with timing to measure cost across 693 chunks.
"""

import asyncio
import os
import sys
import time
from collections import defaultdict
from functools import wraps

import numpy as np
from scipy.io import wavfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ["LOG_LEVEL"] = "WARNING"

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "../.env"))

from time import perf_counter
from datetime import datetime, timezone

from hush.core import Hush, graph, START, END, PARENT, op
from hush.core.ops.base import BaseOp
from hush.core.states.state import MemoryState
from speech.audio_processor import AudioProcessor
from speech.vad_detector import VadDetector
from speech.vad_detector_v2 import vad_detector as vad_detector_v2

AUDIO_DIR = os.path.join(os.path.dirname(__file__), "../tests/speech/audio")

# ── Timing accumulators ──
T = defaultdict(float)
C = defaultdict(int)


def wrap_method(cls, method_name, label):
    """Wrap a method to accumulate timing."""
    original = getattr(cls, method_name)

    @wraps(original)
    def timed(self, *args, **kwargs):
        t0 = perf_counter()
        result = original(self, *args, **kwargs)
        T[label] += perf_counter() - t0
        C[label] += 1
        return result

    setattr(cls, method_name, timed)


def wrap_state_access():
    """Wrap MemoryState __getitem__ and __setitem__."""
    orig_get = MemoryState.__getitem__
    orig_set = MemoryState.__setitem__

    @wraps(orig_get)
    def timed_get(self, key):
        t0 = perf_counter()
        result = orig_get(self, key)
        T["state_get"] += perf_counter() - t0
        C["state_get"] += 1
        return result

    @wraps(orig_set)
    def timed_set(self, key, value):
        t0 = perf_counter()
        orig_set(self, key, value)
        T["state_set"] += perf_counter() - t0
        C["state_set"] += 1

    MemoryState.__getitem__ = timed_get
    MemoryState.__setitem__ = timed_set


def install_patches():
    wrap_method(BaseOp, "get_inputs", "get_inputs")
    wrap_method(BaseOp, "store_result", "store_result")
    wrap_method(BaseOp, "_log", "_log")
    wrap_method(BaseOp, "_store_metrics", "_store_metrics")
    wrap_state_access()


# ── Audio preprocessing ──

async def preprocess_all():
    wav_files = sorted([
        f for f in os.listdir(AUDIO_DIR)
        if f.endswith(".wav") and f != "test_tts.wav"
    ])
    all_chunks = []
    for wav_file in wav_files:
        wav_path = os.path.join(AUDIO_DIR, wav_file)
        sr, raw_audio = wavfile.read(wav_path)
        silence = np.zeros(int(sr * 1.0), dtype=raw_audio.dtype)
        raw_audio = np.concatenate([raw_audio, silence])
        proc = AudioProcessor(name="pre", use_preprocess=False)
        for i in range(0, len(raw_audio), 320):
            chunk = raw_audio[i:i + 320]
            if len(chunk) < 320:
                chunk = np.pad(chunk, (0, 320 - len(chunk)))
            async for out in proc._process(raw_chunk=chunk.tobytes(), cmc_time=1000 + i):
                all_chunks.append((out["audio"].copy(), out["cmc_time"], out["recv_time"]))
    return all_chunks


# ── V1 baseline ──

async def run_v1(audio_chunks):
    vad = VadDetector(name="v1_vad")
    for audio, cmc_time, recv_time in audio_chunks:
        async for _ in vad._process(audio=audio, cmc_time=cmc_time, recv_time=recv_time):
            pass


# ── V2 pipeline ──

@op
def chunk_source(audio_chunks: list):
    for audio, cmc_time, recv_time in audio_chunks:
        yield {"audio": audio, "cmc_time": cmc_time, "recv_time": recv_time}


@graph
def pipeline_v2(audio_chunks):
    source = chunk_source(audio_chunks=audio_chunks)
    vad = vad_detector_v2(
        audio=source["audio"], cmc_time=source["cmc_time"], recv_time=source["recv_time"],
    )
    START >> source >> vad >> END


async def main():
    print("BaseOp.run() latency breakdown — 10 WAVs concatenated")
    print("=" * 70)

    chunks = await preprocess_all()
    print(f"  Preprocessed: {len(chunks)} chunks\n")

    # V1 baseline (no patches)
    t0 = perf_counter()
    await run_v1(chunks)
    t_v1 = (perf_counter() - t0) * 1000
    print(f"  V1 baseline (direct BaseOp): {t_v1:.0f}ms")

    # Install patches BEFORE building V2
    install_patches()

    # V2 — build
    t0 = perf_counter()
    wf = pipeline_v2(audio_chunks=chunks)
    t_build = (perf_counter() - t0) * 1000

    # V2 — engine init
    t0 = perf_counter()
    engine = Hush(wf)
    t_init = (perf_counter() - t0) * 1000

    # V2 — clear timers, run
    T.clear()
    C.clear()

    t0 = perf_counter()
    result = await engine.run(inputs={})
    t_run = (perf_counter() - t0) * 1000

    print(f"  V2 build: {t_build:.0f}ms | init: {t_init:.0f}ms | run: {t_run:.0f}ms | total: {t_build + t_init + t_run:.0f}ms")
    print(f"  V2 overhead (run - v1): {t_run - t_v1:+.0f}ms")

    # Breakdown
    print(f"\n{'=' * 70}")
    print(f"  engine.run() breakdown ({t_run:.0f}ms):")
    print(f"{'=' * 70}")

    # Compute scheduler/framework overhead
    measured = sum(T.values())
    scheduler = (t_run / 1000) - measured

    rows = [
        ("get_inputs()", T["get_inputs"], C["get_inputs"]),
        ("store_result()", T["store_result"], C["store_result"]),
        ("_log()", T["_log"], C["_log"]),
        ("_store_metrics()", T["_store_metrics"], C["_store_metrics"]),
        ("state[] get", T["state_get"], C["state_get"]),
        ("state[] set", T["state_set"], C["state_set"]),
        ("scheduler + exec + rest", scheduler, 0),
    ]

    print(f"\n    {'Component':<25} {'Time':>8} {'Calls':>8} {'Per-call':>10}  {'%':>5}")
    print(f"    {'─' * 60}")
    for label, secs, count in sorted(rows, key=lambda x: -x[1]):
        ms = secs * 1000
        pct = (secs / (t_run / 1000)) * 100 if t_run > 0 else 0
        per_call = f"{ms / count:.3f}ms" if count > 0 else "—"
        print(f"    {label:<25} {ms:7.1f}ms {count:>8} {per_call:>10}  {pct:4.1f}%")

    print(f"    {'─' * 60}")
    print(f"    {'TOTAL':<25} {t_run:7.1f}ms")

    # State access summary
    print(f"\n  State access: {C['state_get']} gets + {C['state_set']} sets = {C['state_get'] + C['state_set']} total")
    state_ms = (T["state_get"] + T["state_set"]) * 1000
    print(f"  State time: {state_ms:.1f}ms ({state_ms / t_run * 100:.1f}% of run)")


if __name__ == "__main__":
    asyncio.run(main())
