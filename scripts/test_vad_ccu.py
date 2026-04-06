"""Test VadDetector state isolation under concurrent calls (CCU).

Verifies that per-call VAD buffers keyed by state.request_id do not leak
between concurrent engine.run() calls sharing the same graph instance.

How it works:
  Spy on the ONNX session to record (input_vad_state, output_vad_state) per
  inference, attributed to the active request_id via ContextVar.
  For each call, verify: output_vad_state[chunk N] == input_vad_state[chunk N+1].
  If this chain breaks → another call overwrote the LSTM state between chunks.

Usage:
  cd /home/thang_ai/callbot-engine-hush
  uv run -m scripts.test_vad_ccu
"""

import asyncio
import sys
import time
from collections import defaultdict

import numpy as np
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, "/home/thang_ai/callbot-engine-hush")

from hush.core import Hush, GraphOp, START, END, PARENT
from hush.core.ops import op
from speech.vad_detector import VadDetector, _active_vad_buffers

RATE = 16000
CHUNK = 512
N_CALLS = 4


# ── Synthetic audio ──

def make_call_audio(freq: float = 440.0) -> np.ndarray:
    """[0.2s silence | 1.0s speech at freq | 1.2s silence]"""
    pre = np.zeros(int(RATE * 0.2), dtype=np.float32)
    t = np.linspace(0, 1.0, RATE)
    speech = (np.sin(2 * np.pi * freq * t) * 0.5).astype(np.float32)
    post = np.zeros(int(RATE * 1.2), dtype=np.float32)
    return np.concatenate([pre, speech, post])


# ── Mock source ──

@op
async def audio_source(audio: np.ndarray):
    for i in range(0, len(audio) - CHUNK + 1, CHUNK):
        yield {
            "audio": audio[i: i + CHUNK],
            "cmc_time": int(time.time() * 1000),
            "recv_time": int(time.time() * 1000),
        }
        await asyncio.sleep(0)  # force interleaving between concurrent calls


# ── ONNX state chain tracker ──

class StateChainTracker:
    """Wraps ONNX session. Records (input_vad_state, output_vad_state) per request_id.

    After N concurrent calls, verifies output[chunk N] == input[chunk N+1] for each call.
    A broken chain means another call overwrote the LSTM state between chunks.
    """

    def __init__(self, real_session):
        self._session = real_session
        self.chains: dict = defaultdict(list)

    def run(self, output_names, feed_dict):
        rid = _active_vad_buffers.get({}).get("_rid", "unknown")
        input_state = feed_dict["state"].copy()
        result = self._session.run(output_names, feed_dict)
        output_state = result[1].copy()
        self.chains[rid].append((input_state, output_state))
        return result

    def check_chains(self) -> dict:
        results = {}
        for rid, chain in self.chains.items():
            ok = True
            for n in range(len(chain) - 1):
                if not np.allclose(chain[n][1], chain[n + 1][0], atol=1e-7):
                    ok = False
                    break
            results[rid] = (ok, len(chain))
        return results


# ── Graph ──

def build_graph():
    with GraphOp(name="vad_ccu_test") as g:
        src = audio_source(audio=PARENT["audio"])
        vad = VadDetector(
            inputs={
                "audio": src["audio"],
                "cmc_time": src["cmc_time"],
                "recv_time": src["recv_time"],
            }
        )
        vad["speech_audio"] >> PARENT["speech_audio"]
        vad["speech_duration_ms"] >> PARENT["speech_duration_ms"]
        START >> src >> vad >> END
    return g, vad


# ── Main ──

async def main():
    print("VadDetector — Concurrent Call State Isolation Test")
    print(f"  {N_CALLS} concurrent calls, each ~2.4s audio ({int(RATE * 2.4) // CHUNK} chunks)")
    print()

    graph, vad_op = build_graph()
    real_session = vad_op._session
    tracker = StateChainTracker(real_session)

    # Inject _rid into each call's buffer so tracker can read it via ContextVar
    original_run = vad_op.run

    async def patched_run(state, context_id=None):
        rid = state.request_id
        if rid not in VadDetector._vad_buffers:
            VadDetector._vad_buffers[rid] = vad_op._init_vad_buffers()
        VadDetector._vad_buffers[rid]["_rid"] = rid
        async for item in original_run(state, context_id):
            yield item

    vad_op.run = patched_run
    vad_op._session = tracker

    engine = Hush(graph)
    call_audios = [make_call_audio(freq=300 + i * 200) for i in range(N_CALLS)]

    t0 = time.perf_counter()
    results = await asyncio.gather(*[engine.run({"audio": a}) for a in call_audios])
    elapsed = (time.perf_counter() - t0) * 1000

    print(f"  {N_CALLS} concurrent calls completed in {elapsed:.0f}ms")
    print()

    for i, r in enumerate(results):
        if isinstance(r, Exception):
            print(f"  Call {i}: ERROR — {r}")
        else:
            dur = r.get("speech_duration_ms") or 0
            detected = r.get("speech_audio") is not None
            print(f"  Call {i} (freq={300 + i * 200}Hz): detected={detected}, duration={dur:.0f}ms")

    # Buffer isolation check
    print()
    n_buffers = len(VadDetector._vad_buffers)
    buffer_ids = [id(b) for b in VadDetector._vad_buffers.values()]
    all_unique = len(set(buffer_ids)) == len(buffer_ids)
    print(f"  Buffer isolation: {n_buffers} separate buffer dicts, all unique={all_unique}")

    # ONNX state chain check
    chains = tracker.check_chains()
    print()
    print("  ONNX LSTM state chain per call (output[N] == input[N+1]):")
    n_corrupted = 0
    for rid, (ok, n) in chains.items():
        status = "OK     " if ok else "CORRUPT"
        print(f"    {status} — request={rid[:12]}... ({n} inferences)")
        if not ok:
            n_corrupted += 1

    print()
    if n_corrupted == 0 and all_unique:
        print("  PASS — all calls maintained isolated LSTM state chains")
    else:
        print(f"  FAIL — {n_corrupted} corrupted chains, buffer unique={all_unique}")

    # Cleanup
    for rid in list(VadDetector._vad_buffers.keys()):
        VadDetector.cleanup(rid)


if __name__ == "__main__":
    asyncio.run(main())
