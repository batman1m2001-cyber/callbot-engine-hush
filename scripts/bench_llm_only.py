"""Benchmark: LLM API only — 8 concurrent calls to isolate LLM bottleneck."""

import asyncio
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "../.env"))

import httpx

API_KEY = os.getenv("ANTHROPIC_API_KEY")
API_URL = "https://api.anthropic.com/v1/messages"
MODEL = "claude-3-haiku-20240307"

PROMPT = "Classify intent: 'bé đang vào rồi em'. Reply <result>student_joining</result>"


async def call_llm(call_id: int, client: httpx.AsyncClient) -> dict:
    t0 = time.perf_counter()
    resp = await client.post(
        API_URL,
        headers={
            "x-api-key": API_KEY,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        json={
            "model": MODEL,
            "max_tokens": 50,
            "messages": [{"role": "user", "content": PROMPT}],
        },
        timeout=30,
    )
    elapsed = (time.perf_counter() - t0) * 1000
    status = resp.status_code
    return {"call_id": call_id, "status": status, "time_ms": elapsed}


async def bench(ccu: int):
    print(f"\nCCU={ccu}: {ccu} concurrent LLM calls")
    async with httpx.AsyncClient() as client:
        # Warmup
        await call_llm(-1, client)

        t0 = time.perf_counter()
        tasks = [call_llm(i, client) for i in range(ccu)]
        results = await asyncio.gather(*tasks)
        total = (time.perf_counter() - t0) * 1000

    times = [r["time_ms"] for r in results]
    errors = [r for r in results if r["status"] != 200]
    avg = sum(times) / len(times)
    fastest = min(times)
    slowest = max(times)

    print(f"  Total: {total:.0f}ms  Avg: {avg:.0f}ms  Fast: {fastest:.0f}ms  Slow: {slowest:.0f}ms  Errors: {len(errors)}")
    for r in sorted(results, key=lambda x: x["call_id"]):
        print(f"    [{r['call_id']:2d}] {r['status']} {r['time_ms']:.0f}ms")


async def main():
    for ccu in [1, 2, 4, 8]:
        await bench(ccu)


if __name__ == "__main__":
    asyncio.run(main())
