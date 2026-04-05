"""Profile PURE framework overhead — no-op functions, only Hush machinery.

Compares:
  V1: direct function call loop (baseline)
  V2: same functions inside @graph with scheduler
"""

import asyncio
import os
import sys
from collections import defaultdict
from functools import wraps
from time import perf_counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ["LOG_LEVEL"] = "WARNING"

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "../.env"))

from hush.core import Hush, graph, START, END, PARENT, op
from hush.core.ops.base import BaseOp
from hush.core.states.state import MemoryState

N_CHUNKS = 693  # same as 10 WAVs concat

# ── Timing ──
T = defaultdict(float)
C = defaultdict(int)


def wrap_method(cls, method_name, label):
    original = getattr(cls, method_name)
    @wraps(original)
    def timed(self, *args, **kwargs):
        t0 = perf_counter()
        result = original(self, *args, **kwargs)
        T[label] += perf_counter() - t0
        C[label] += 1
        return result
    setattr(cls, method_name, timed)


def wrap_state():
    orig_get = MemoryState.__getitem__
    orig_set = MemoryState.__setitem__
    @wraps(orig_get)
    def timed_get(self, key):
        t0 = perf_counter()
        r = orig_get(self, key)
        T["state_get"] += perf_counter() - t0
        C["state_get"] += 1
        return r
    @wraps(orig_set)
    def timed_set(self, key, value):
        t0 = perf_counter()
        orig_set(self, key, value)
        T["state_set"] += perf_counter() - t0
        C["state_set"] += 1
    MemoryState.__getitem__ = timed_get
    MemoryState.__setitem__ = timed_set


# ── No-op functions (simulate 3-op VAD pipeline) ──

def noop_a(x: int):
    return {"y": x}

def noop_b(x: int):
    return {"z": x}

def noop_c(y: int, z: int):
    return {"result": y + z}


# ── V1: direct call loop ──

def run_v1_direct():
    for _ in range(N_CHUNKS):
        r_a = noop_a(x=1)
        r_b = noop_b(x=1)
        r_c = noop_c(y=r_a["y"], z=r_b["z"])


# ── V2: @graph with 3 ops ──

@op
def source_op(n: int):
    for i in range(n):
        yield {"x": i}

@op
def op_a(x: int):
    return {"y": x}

@op
def op_b(x: int):
    return {"z": x}

@op
def op_c(y: int, z: int):
    return {"result": y + z}

@graph
def noop_pipeline(n):
    s = source_op(n=n)
    a = op_a(x=s["x"])
    b = op_b(x=s["x"])
    c = op_c(y=a["y"], z=b["z"])
    START >> s >> [a, b] >> c >> END


async def main():
    print(f"Pure framework overhead — {N_CHUNKS} chunks, 3 no-op ops each")
    print("=" * 70)

    # V1
    t0 = perf_counter()
    run_v1_direct()
    t_v1 = (perf_counter() - t0) * 1000
    print(f"  V1 (direct calls): {t_v1:.1f}ms")

    # Install patches
    wrap_method(BaseOp, "get_inputs", "get_inputs")
    wrap_method(BaseOp, "store_result", "store_result")
    wrap_method(BaseOp, "_log", "_log")
    wrap_method(BaseOp, "_store_metrics", "_store_metrics")
    wrap_state()

    T.clear()
    C.clear()

    # V2 build + init
    t0 = perf_counter()
    wf = noop_pipeline(n=N_CHUNKS)
    engine = Hush(wf)
    t_build = (perf_counter() - t0) * 1000

    # V2 run
    t0 = perf_counter()
    result = await engine.run(inputs={})
    t_run = (perf_counter() - t0) * 1000

    print(f"  V2 (build+init): {t_build:.1f}ms")
    print(f"  V2 (run):        {t_run:.1f}ms")
    print(f"  Pure overhead:   {t_run - t_v1:+.1f}ms")

    # Breakdown
    measured = sum(T.values())
    rest = (t_run / 1000) - measured

    rows = [
        ("get_inputs()", T["get_inputs"], C["get_inputs"]),
        ("store_result()", T["store_result"], C["store_result"]),
        ("_log()", T["_log"], C["_log"]),
        ("_store_metrics()", T["_store_metrics"], C["_store_metrics"]),
        ("state[] get", T["state_get"], C["state_get"]),
        ("state[] set", T["state_set"], C["state_set"]),
        ("rest (datetime+async+sched)", rest, 0),
    ]

    print(f"\n{'=' * 70}")
    print(f"  engine.run() breakdown ({t_run:.0f}ms):")
    print(f"{'=' * 70}")
    print(f"\n    {'Component':<30} {'Time':>8} {'Calls':>8} {'Per-call':>10}  {'%':>5}")
    print(f"    {'─' * 65}")
    for label, secs, count in sorted(rows, key=lambda x: -x[1]):
        ms = secs * 1000
        pct = (secs / (t_run / 1000)) * 100 if t_run > 0 else 0
        per_call = f"{ms / count:.4f}ms" if count > 0 else "—"
        print(f"    {label:<30} {ms:7.1f}ms {count:>8} {per_call:>10}  {pct:4.1f}%")
    print(f"    {'─' * 65}")
    print(f"    {'TOTAL':<30} {t_run:7.1f}ms")
    print(f"\n  State: {C['state_get']} gets + {C['state_set']} sets = {C['state_get'] + C['state_set']} total")


if __name__ == "__main__":
    asyncio.run(main())
