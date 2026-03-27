"""Benchmark: each pipeline component at 8 CCU to find bottleneck."""

import asyncio
import os
import sys
import time

import numpy as np
from scipy.io import wavfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "../.env"))

AUDIO_DIR = os.path.join(os.path.dirname(__file__), "../tests/speech/audio")
CCU = 8


# ── 1. VAD only ──
async def bench_vad():
    from speech.audio_processor import AudioProcessor
    from speech.vad_detector import VadDetector

    wav_path = os.path.join(AUDIO_DIR, "01_student_joining.wav")
    sr, raw_audio = wavfile.read(wav_path)
    silence = np.zeros(int(1.0 * sr), dtype=raw_audio.dtype)
    audio = np.concatenate([raw_audio, silence])

    async def run_one(call_id):
        proc = AudioProcessor(name=f"audio_{call_id}")
        vad = VadDetector(name=f"vad_{call_id}", model_path="models/silero_vad.onnx")
        t0 = time.perf_counter()
        segments = []
        for i in range(0, len(audio), 320):
            chunk = audio[i:i+320]
            if len(chunk) < 320:
                chunk = np.pad(chunk, (0, 320 - len(chunk)))
            async for proc_out in proc._process(raw_chunk=chunk.tobytes(), cmc_time=1000):
                async for seg in vad._process(audio=proc_out["audio"], cmc_time=proc_out["cmc_time"], recv_time=proc_out["recv_time"]):
                    segments.append(seg)
        return {"call_id": call_id, "time_ms": (time.perf_counter() - t0) * 1000, "segments": len(segments)}

    # Warmup
    await run_one(-1)

    t0 = time.perf_counter()
    results = await asyncio.gather(*[run_one(i) for i in range(CCU)])
    total = (time.perf_counter() - t0) * 1000

    times = [r["time_ms"] for r in results]
    print(f"VAD (AudioProcessor + VadDetector) x{CCU}:")
    print(f"  Total: {total:.0f}ms  Avg: {sum(times)/len(times):.0f}ms  Fast: {min(times):.0f}ms  Slow: {max(times):.0f}ms")


# ── 2. STT only ──
async def bench_stt():
    import tritonclient.grpc as grpcclient
    import soxr

    wav_path = os.path.join(AUDIO_DIR, "01_student_joining.wav")
    sr, raw_audio = wavfile.read(wav_path)
    audio_f32 = raw_audio.astype(np.float32) / 32768.0
    audio_16k = soxr.resample(audio_f32, sr, 16000, quality="VHQ")

    client = grpcclient.InferenceServerClient(url="192.168.1.212:8001")

    async def run_one(call_id):
        t0 = time.perf_counter()
        inp = grpcclient.InferInput("AUDIO_SIGNAL", list(audio_16k.shape), "FP32")
        inp.set_data_from_numpy(audio_16k.astype(np.float32))
        out = grpcclient.InferRequestedOutput("TRANSCRIPT")
        # Triton client is sync — wrap in thread
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, lambda: client.infer(
            model_name="fastconformer_asr", inputs=[inp], outputs=[out], client_timeout=30
        ))
        elapsed = (time.perf_counter() - t0) * 1000
        raw = result.as_numpy("TRANSCRIPT")
        text = raw.flat[0].decode("utf-8") if isinstance(raw.flat[0], bytes) else str(raw.flat[0])
        return {"call_id": call_id, "time_ms": elapsed, "text": text[:30]}

    # Warmup
    await run_one(-1)

    t0 = time.perf_counter()
    results = await asyncio.gather(*[run_one(i) for i in range(CCU)])
    total = (time.perf_counter() - t0) * 1000

    times = [r["time_ms"] for r in results]
    print(f"\nSTT (Triton gRPC) x{CCU}:")
    print(f"  Total: {total:.0f}ms  Avg: {sum(times)/len(times):.0f}ms  Fast: {min(times):.0f}ms  Slow: {max(times):.0f}ms")


# ── 3. TTS only ──
async def bench_tts():
    from hush.core import Hush
    from speech.tts_synthesizer import tts_pipeline

    async def run_one(call_id):
        t0 = time.perf_counter()
        wf = tts_pipeline(text="bé đang vào rồi em")
        engine = Hush(wf)
        result = await engine.run(inputs={})
        elapsed = (time.perf_counter() - t0) * 1000
        audio = result.get("audio")
        dur = result.get("audio_duration_ms", 0)
        return {"call_id": call_id, "time_ms": elapsed, "audio_ms": dur}

    # Warmup
    await run_one(-1)

    t0 = time.perf_counter()
    results = await asyncio.gather(*[run_one(i) for i in range(CCU)])
    total = (time.perf_counter() - t0) * 1000

    times = [r["time_ms"] for r in results]
    print(f"\nTTS (fastspeech2 + hifigan Triton) x{CCU}:")
    print(f"  Total: {total:.0f}ms  Avg: {sum(times)/len(times):.0f}ms  Fast: {min(times):.0f}ms  Slow: {max(times):.0f}ms")


# ── 4. Full pipeline via Hush ──
async def bench_full():
    from hush.core import Hush
    from pipeline.callbot import callbot_pipeline

    script_data = {"student_name": "Minh", "class_time": "19:00", "program_name": "AI CLASS",
                   "agent_name": "Linh", "hotline": "1900636464", "parent_name": "anh chị"}

    async def run_one(call_id):
        wav = os.path.join(AUDIO_DIR, "01_student_joining.wav")
        t0 = time.perf_counter()
        wf = callbot_pipeline(wav_path=wav, script_data=script_data)
        engine = Hush(wf, env=os.path.join(os.path.dirname(__file__), "../.env"),
                      resources=os.path.join(os.path.dirname(__file__), "../resources.yaml"))
        await engine.run(inputs={})
        return {"call_id": call_id, "time_ms": (time.perf_counter() - t0) * 1000}

    # Warmup
    await run_one(-1)

    t0 = time.perf_counter()
    results = await asyncio.gather(*[run_one(i) for i in range(CCU)])
    total = (time.perf_counter() - t0) * 1000

    times = [r["time_ms"] for r in results]
    print(f"\nFull Pipeline (Hush) x{CCU}:")
    print(f"  Total: {total:.0f}ms  Avg: {sum(times)/len(times):.0f}ms  Fast: {min(times):.0f}ms  Slow: {max(times):.0f}ms")


async def main():
    print(f"Component Benchmark — CCU={CCU}")
    print("=" * 60)
    await bench_vad()
    await bench_stt()
    await bench_tts()
    await bench_full()
    print("\n" + "=" * 60)
    print("Done!")


if __name__ == "__main__":
    asyncio.run(main())
