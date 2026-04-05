"""Test: does PARENT.shared(bg={"vad_start_time": time.time()}) evaluate once at
graph-definition time, making vad_start_time stale on subsequent engine runs?

Hypothesis
----------
time.time() in PARENT.shared() is evaluated once when the @graph decorator
executes (module import), NOT per engine.run(). This means the 5-second
"always foreground" window in foreground_detect() drifts as wall-clock time
advances, because `elapsed = time.time() - bg["vad_start_time"]` uses a frozen
timestamp from graph construction.

In vad_detector.py (v1 / BaseOp), `_start_time = time.time()` is set in
__init__, which runs per-instance. In vad_detector_v2.py (@graph), the shared
dict is built once when the module is imported.

Usage:
    uv run python scripts/test_vad_v2_shared_time.py
"""

import asyncio
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ["LOG_LEVEL"] = "WARNING"

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "../.env"))

from hush.core import Hush, graph, START, END, PARENT, op
from hush.core.states.schema import StateSchema
from speech.vad_detector_v2 import vad_detector

# ── Config ──
RATE = 16000
CHUNK = 512
FREQ = 440  # Hz


def make_sine_chunks(n: int = 10) -> list:
    """Generate n 512-sample sine chunks at 440 Hz, 16 kHz SR."""
    total = n * CHUNK
    t = np.arange(total, dtype=np.float32) / RATE
    signal = (np.sin(2 * np.pi * FREQ * t) * 0.5).astype(np.float32)
    return [signal[i * CHUNK:(i + 1) * CHUNK] for i in range(n)]


@op
async def chunk_source(chunks: list):
    """Yield audio chunks one at a time."""
    for chunk in chunks:
        yield {
            "audio": chunk,
            "cmc_time": int(time.time() * 1000),
            "recv_time": int(time.time() * 1000),
        }
        await asyncio.sleep(0)


@graph
def test_pipeline(chunks):
    """Wrapper: source op -> vad_detector."""
    source = chunk_source(chunks=chunks)
    vad = vad_detector(
        audio=source["audio"],
        cmc_time=source["cmc_time"],
        recv_time=source["recv_time"],
    )
    START >> source >> vad >> END


async def main():
    print("=" * 70)
    print("TEST: PARENT.shared() vad_start_time staleness — vad_detector_v2")
    print("=" * 70)

    chunks = make_sine_chunks(n=10)

    # Build the pipeline. This calls test_pipeline() which internally calls
    # vad_detector(), building the nested GraphOp with PARENT.shared().
    pipeline_graph = test_pipeline(chunks=PARENT["chunks"])
    engine = Hush(pipeline_graph)
    schema = engine.schema

    # ── Step 1: Find the frozen vad_start_time in schema defaults ──
    print("\n--- Schema inspection ---")
    frozen_bg = None
    frozen_vad_start_time = None
    bg_var_key = None

    for (op_name, var_name), idx in schema._var_to_idx.items():
        if var_name == "bg":
            default_val = schema._defaults[idx]
            if isinstance(default_val, dict) and "vad_start_time" in default_val:
                frozen_bg = default_val
                frozen_vad_start_time = default_val["vad_start_time"]
                bg_var_key = (op_name, var_name)
                print(f"  Found bg dict at ({op_name}, {var_name}), schema index={idx}")
                break

    now = time.time()

    if frozen_vad_start_time is None:
        print("\n  Could not find bg.vad_start_time in schema defaults.")
        print("  Dumping all schema vars with dict defaults:")
        for (op_name, var_name), idx in sorted(schema._var_to_idx.items()):
            val = schema._defaults[idx]
            if isinstance(val, dict):
                print(f"    [{idx}] ({op_name}, {var_name}) = {repr(val)[:100]}")
        print("\n  Falling back to timing-based approach...")
    else:
        age_now = now - frozen_vad_start_time
        print(f"  frozen vad_start_time = {frozen_vad_start_time:.6f}")
        print(f"  current time          = {now:.6f}")
        print(f"  age right now         = {age_now:.3f}s")
        print(f"  (frozen at module import / @graph decoration time)")

    # ── Step 2: Run engine once, check state ──
    print("\n--- Run 1: immediate ---")
    t1 = time.time()
    if frozen_vad_start_time:
        print(f"  vad_start_time age at Run 1 start: {t1 - frozen_vad_start_time:.3f}s")

    try:
        result1 = await engine.run({"chunks": chunks})
        print(f"  Run 1 completed. Result: {_summarize(result1)}")
    except Exception as e:
        print(f"  Run 1 error: {type(e).__name__}: {e}")

    # Check if the frozen bg dict was mutated (state leak)
    if frozen_bg:
        print(f"  bg dict after Run 1: n={frozen_bg.get('n')}, mean={frozen_bg.get('mean', 0):.6f}")
        if frozen_bg.get("n", 0) > 0:
            print("  >> STATE LEAK: bg dict in schema._defaults was mutated by Run 1!")

    # ── Step 3: Sleep 6 seconds ──
    print(f"\n--- Sleeping 6 seconds... ---")
    time.sleep(6)

    # ── Step 4: Run engine again ──
    print("\n--- Run 2: after 6s sleep ---")
    t2 = time.time()
    if frozen_vad_start_time:
        age_at_run2 = t2 - frozen_vad_start_time
        print(f"  vad_start_time age at Run 2 start: {age_at_run2:.3f}s")

    try:
        result2 = await engine.run({"chunks": chunks})
        print(f"  Run 2 completed. Result: {_summarize(result2)}")
    except Exception as e:
        print(f"  Run 2 error: {type(e).__name__}: {e}")

    if frozen_bg:
        print(f"  bg dict after Run 2: n={frozen_bg.get('n')}, mean={frozen_bg.get('mean', 0):.6f}")

    # ── Analysis ──
    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)

    if frozen_vad_start_time:
        age_at_run2 = t2 - frozen_vad_start_time

        if age_at_run2 > 5.0:
            print(f"""
  BUG CONFIRMED: vad_start_time is {age_at_run2:.1f}s old at Run 2 start.

  The 5-second "always foreground" grace window has already expired before
  Run 2 processes its very first chunk. foreground_detect() will immediately
  use the background energy model instead of the 5s warm-up bypass.

  In vad_detector_v2.py line 85-86:
      elapsed = time.time() - bg["vad_start_time"]
      if elapsed < 5.0:
          is_fg = True   # <-- DEAD CODE for Run 2

  ROOT CAUSE: time.time() in PARENT.shared() is evaluated once when @graph
  decorator processes the function body (at module import). The float is
  baked into GraphOp._shared_vars -> StateSchema._defaults. Every engine.run()
  creates a new MemoryState whose Cell.default_value references the same bg
  dict — no re-evaluation of time.time().

  Compare with vad_detector.py (v1) where __init__ sets
  self._start_time = time.time() per instance.

  FIX OPTIONS:
    1. Add an init @op that sets bg["vad_start_time"] = time.time() and
       runs before vad_infer/foreground_detect
    2. Use a factory/lazy callback for shared state initial values
    3. Compute elapsed differently (e.g., count chunks instead of wall clock)
""")
        else:
            print(f"""
  NO BUG: vad_start_time age at Run 2 = {age_at_run2:.3f}s < 5.0s
  The timestamp appears to be fresh enough. (Test may need longer sleep.)
""")

        # State leak analysis
        if frozen_bg.get("n", 0) > 0:
            print("""  ADDITIONAL ISSUE — STATE LEAK:
  The bg dict in schema._defaults was mutated during execution.
  bg.n > 0 means foreground_detect's mutations persisted to the
  schema default. If Cell.default_value points to the same dict,
  Run 2 inherits Run 1's background model state.
""")
    else:
        print(f"""
  LIKELY BUG (could not inspect schema directly):
  time.time() in PARENT.shared(bg={{"vad_start_time": time.time()}}) is
  evaluated once at @graph decoration time. By Run 2 (~{t2 - t1:.0f}s later),
  the timestamp is stale and the 5-second grace window is expired.
""")


def _summarize(result):
    """Summarize engine result for display."""
    if not isinstance(result, dict):
        return str(type(result))
    keys = list(result.keys())
    out = {}
    for k in keys:
        v = result[k]
        if isinstance(v, np.ndarray):
            out[k] = f"ndarray({v.shape})"
        elif isinstance(v, (int, float, bool, str)):
            out[k] = v
        else:
            out[k] = type(v).__name__
    return str(out)


if __name__ == "__main__":
    asyncio.run(main())
