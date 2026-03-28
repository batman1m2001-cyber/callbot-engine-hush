"""Benchmark STT: compare all methods to find lowest latency at 20 CCU.

Methods:
1. Sync client, shared instance, run_in_executor
2. Sync client, per-call instance, run_in_executor
3. Async client, shared instance
4. Async client, per-call instance
5. Sync client, ThreadPoolExecutor(max_workers=20)
6. Sync client, ProcessPoolExecutor
"""

import asyncio
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import soxr
from scipy.io import wavfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

STT_URL = "192.168.1.212:8001"
MODEL = "fastconformer_asr"
CCU = 20

# Prepare audio
wav_path = os.path.join(os.path.dirname(__file__), "../tests/speech/audio/01_student_joining.wav")
sr, raw = wavfile.read(wav_path)
AUDIO = soxr.resample(raw.astype(np.float32) / 32768.0, sr, 16000, quality="VHQ").reshape(1, -1)


def _sync_infer(client):
    """Sync infer on given client."""
    import tritonclient.grpc as grpcclient
    inp = grpcclient.InferInput("AUDIO_SIGNAL", list(AUDIO.shape), "FP32")
    inp.set_data_from_numpy(AUDIO.astype(np.float32))
    out = grpcclient.InferRequestedOutput("TRANSCRIPT")
    client.infer(model_name=MODEL, inputs=[inp], outputs=[out], client_timeout=30)


def _sync_infer_new_client():
    """Sync infer with fresh client per call."""
    import tritonclient.grpc as grpcclient
    client = grpcclient.InferenceServerClient(url=STT_URL)
    inp = grpcclient.InferInput("AUDIO_SIGNAL", list(AUDIO.shape), "FP32")
    inp.set_data_from_numpy(AUDIO.astype(np.float32))
    out = grpcclient.InferRequestedOutput("TRANSCRIPT")
    client.infer(model_name=MODEL, inputs=[inp], outputs=[out], client_timeout=30)


async def bench_sync_shared():
    """Method 1: Sync client, shared, run_in_executor (default ThreadPool)."""
    import tritonclient.grpc as grpcclient
    client = grpcclient.InferenceServerClient(url=STT_URL)
    _sync_infer(client)  # warmup

    loop = asyncio.get_event_loop()

    async def one(i):
        t0 = time.perf_counter()
        await loop.run_in_executor(None, _sync_infer, client)
        return (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    times = await asyncio.gather(*[one(i) for i in range(CCU)])
    wall = (time.perf_counter() - t0) * 1000
    return "sync_shared_default_pool", times, wall


async def bench_sync_percall():
    """Method 2: Sync client, per-call instance, run_in_executor."""
    _sync_infer_new_client()  # warmup

    loop = asyncio.get_event_loop()

    async def one(i):
        t0 = time.perf_counter()
        await loop.run_in_executor(None, _sync_infer_new_client)
        return (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    times = await asyncio.gather(*[one(i) for i in range(CCU)])
    wall = (time.perf_counter() - t0) * 1000
    return "sync_percall_default_pool", times, wall


async def bench_async_shared():
    """Method 3: Async client, shared instance."""
    import tritonclient.grpc.aio as aio_grpc
    client = aio_grpc.InferenceServerClient(url=STT_URL)

    async def one(i):
        inp = aio_grpc.InferInput("AUDIO_SIGNAL", list(AUDIO.shape), "FP32")
        inp.set_data_from_numpy(AUDIO.astype(np.float32))
        out = aio_grpc.InferRequestedOutput("TRANSCRIPT")
        t0 = time.perf_counter()
        await client.infer(model_name=MODEL, inputs=[inp], outputs=[out], client_timeout=30)
        return (time.perf_counter() - t0) * 1000

    await one(-1)  # warmup

    t0 = time.perf_counter()
    times = await asyncio.gather(*[one(i) for i in range(CCU)])
    wall = (time.perf_counter() - t0) * 1000
    await client.close()
    return "async_shared", times, wall


async def bench_async_percall():
    """Method 4: Async client, per-call instance."""
    import tritonclient.grpc.aio as aio_grpc

    async def one(i):
        client = aio_grpc.InferenceServerClient(url=STT_URL)
        inp = aio_grpc.InferInput("AUDIO_SIGNAL", list(AUDIO.shape), "FP32")
        inp.set_data_from_numpy(AUDIO.astype(np.float32))
        out = aio_grpc.InferRequestedOutput("TRANSCRIPT")
        t0 = time.perf_counter()
        await client.infer(model_name=MODEL, inputs=[inp], outputs=[out], client_timeout=30)
        elapsed = (time.perf_counter() - t0) * 1000
        await client.close()
        return elapsed

    await one(-1)  # warmup

    t0 = time.perf_counter()
    times = await asyncio.gather(*[one(i) for i in range(CCU)])
    wall = (time.perf_counter() - t0) * 1000
    return "async_percall", times, wall


async def bench_sync_large_pool():
    """Method 5: Sync client, shared, ThreadPoolExecutor(max_workers=20)."""
    import tritonclient.grpc as grpcclient
    client = grpcclient.InferenceServerClient(url=STT_URL)
    _sync_infer(client)  # warmup

    pool = ThreadPoolExecutor(max_workers=CCU)
    loop = asyncio.get_event_loop()

    async def one(i):
        t0 = time.perf_counter()
        await loop.run_in_executor(pool, _sync_infer, client)
        return (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    times = await asyncio.gather(*[one(i) for i in range(CCU)])
    wall = (time.perf_counter() - t0) * 1000
    pool.shutdown(wait=False)
    return "sync_shared_pool20", times, wall


async def bench_sync_percall_large_pool():
    """Method 6: Sync client, per-call, ThreadPoolExecutor(max_workers=20)."""
    _sync_infer_new_client()  # warmup

    pool = ThreadPoolExecutor(max_workers=CCU)
    loop = asyncio.get_event_loop()

    async def one(i):
        t0 = time.perf_counter()
        await loop.run_in_executor(pool, _sync_infer_new_client)
        return (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    times = await asyncio.gather(*[one(i) for i in range(CCU)])
    wall = (time.perf_counter() - t0) * 1000
    pool.shutdown(wait=False)
    return "sync_percall_pool20", times, wall


async def main():
    print(f"STT Benchmark — CCU={CCU}")
    print(f"Audio: {AUDIO.shape}, Model: {MODEL}")
    print(f"{'='*70}")
    print(f"\n{'Method':<30} {'Avg':>8} {'P50':>8} {'P99':>8} {'Fast':>8} {'Slow':>8} {'Wall':>8}")
    print(f"{'─'*30} {'─'*8} {'─'*8} {'─'*8} {'─'*8} {'─'*8} {'─'*8}")

    for bench_fn in [
        bench_sync_shared,
        bench_sync_percall,
        bench_async_shared,
        bench_async_percall,
        bench_sync_large_pool,
        bench_sync_percall_large_pool,
    ]:
        name, times, wall = await bench_fn()
        times_sorted = sorted(times)
        avg = sum(times) / len(times)
        p50 = times_sorted[len(times) // 2]
        p99 = times_sorted[int(len(times) * 0.99)]
        print(f"{name:<30} {avg:>7.0f}ms {p50:>7.0f}ms {p99:>7.0f}ms {min(times):>7.0f}ms {max(times):>7.0f}ms {wall:>7.0f}ms")

    print(f"\n{'='*70}\nDone!")


if __name__ == "__main__":
    asyncio.run(main())
