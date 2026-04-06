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

MODELS = {
    "haiku-3":   "claude-3-haiku-20240307",
    "haiku-4.5": "claude-haiku-4-5-20251001",
}

# Realistic classify prompt (short version)
PROMPT = (
    "Phân loại ý định: 'anh bận lắm gọi lại sau đi'. "
    "Intents: busy, student_joining, fallback. "
    "Trả lời trong <result></result>."
)


async def call_llm(call_id: int, model: str, client: httpx.AsyncClient) -> dict:
    t0 = time.perf_counter()
    resp = await client.post(
        API_URL,
        headers={
            "x-api-key": API_KEY,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        json={
            "model": model,
            "max_tokens": 50,
            "messages": [{"role": "user", "content": PROMPT}],
        },
        timeout=30,
    )
    elapsed = (time.perf_counter() - t0) * 1000
    return {"call_id": call_id, "status": resp.status_code, "time_ms": elapsed}


async def bench(model_label: str, model: str, ccu: int):
    async with httpx.AsyncClient() as client:
        await call_llm(-1, model, client)  # warmup

        t0 = time.perf_counter()
        results = await asyncio.gather(*[call_llm(i, model, client) for i in range(ccu)])
        total = (time.perf_counter() - t0) * 1000

    times = [r["time_ms"] for r in results]
    errors = sum(1 for r in results if r["status"] != 200)
    print(f"  [{model_label}] CCU={ccu:2d}  total={total:5.0f}ms  avg={sum(times)/len(times):5.0f}ms  "
          f"min={min(times):5.0f}ms  max={max(times):5.0f}ms  errors={errors}")


async def main():
    print(f"Prompt: {PROMPT!r}\n")
    for label, model in MODELS.items():
        print(f"=== {label} ({model}) ===")
        for ccu in [1, 2, 4, 8]:
            await bench(label, model, ccu)
        print()


if __name__ == "__main__":
    asyncio.run(main())
