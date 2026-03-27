"""Benchmark: latency từ user nói xong → transcript ready.

Đo: silence tail bắt đầu (= user ngừng nói) → STT trả transcript.
"""

import asyncio
import os
import sys
import time

import numpy as np
from scipy.io import wavfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "../.env"))

from speech.audio_processor import AudioProcessor
from speech.vad_detector import VadDetector
import tritonclient.grpc as grpcclient

AUDIO_DIR = os.path.join(os.path.dirname(__file__), "../tests/speech/audio")
CHUNK_SIZE = 320  # 40ms at 8kHz
STT_URL = "192.168.1.212:8001"
STT_MODEL = "fastconformer_asr"


async def measure_latency(wav_path: str):
    sr, audio = wavfile.read(wav_path)
    speech_end_sample = len(audio)  # speech ends here
    silence_tail = np.zeros(int(1.0 * sr), dtype=audio.dtype)
    audio_with_tail = np.concatenate([audio, silence_tail])

    audio_proc = AudioProcessor(name="audio")
    vad = VadDetector(name="vad", model_path="models/silero_vad.onnx")

    segments = []
    speech_end_time = None  # timestamp when silence starts (= user stopped talking)
    vad_done_time = None
    stt_done_time = None

    for i in range(0, len(audio_with_tail), CHUNK_SIZE):
        chunk = audio_with_tail[i : i + CHUNK_SIZE]
        if len(chunk) < CHUNK_SIZE:
            chunk = np.pad(chunk, (0, CHUNK_SIZE - len(chunk)))

        # Mark when we start feeding silence (= speech ended)
        sample_pos = i
        if sample_pos >= speech_end_sample and speech_end_time is None:
            speech_end_time = time.perf_counter()

        raw_bytes = chunk.tobytes()
        cmc_time = int(time.time() * 1000)

        async for proc_out in audio_proc._process(raw_chunk=raw_bytes, cmc_time=cmc_time):
            async for vad_out in vad._process(
                audio=proc_out["audio"],
                cmc_time=proc_out["cmc_time"],
                recv_time=proc_out["recv_time"],
            ):
                vad_done_time = time.perf_counter()
                segments.append(vad_out)

    if not segments:
        return None

    # STT inference
    stt_start = time.perf_counter()
    client = grpcclient.InferenceServerClient(url=STT_URL)
    speech_audio = segments[0]["speech_audio"].astype(np.float32)
    inp = grpcclient.InferInput("AUDIO_SIGNAL", list(speech_audio.shape), "FP32")
    inp.set_data_from_numpy(speech_audio)
    out = grpcclient.InferRequestedOutput("TRANSCRIPT")
    result = client.infer(model_name=STT_MODEL, inputs=[inp], outputs=[out], client_timeout=30)
    raw = result.as_numpy("TRANSCRIPT")
    transcript = raw.flat[0].decode("utf-8") if isinstance(raw.flat[0], bytes) else str(raw.flat[0])
    stt_done_time = time.perf_counter()

    vad_latency = (vad_done_time - speech_end_time) * 1000 if speech_end_time else 0
    stt_latency = (stt_done_time - vad_done_time) * 1000
    total_latency = (stt_done_time - speech_end_time) * 1000 if speech_end_time else 0

    return {
        "transcript": transcript,
        "speech_duration_ms": segments[0]["speech_duration_ms"],
        "vad_latency_ms": vad_latency,
        "stt_latency_ms": stt_latency,
        "total_latency_ms": total_latency,
    }


async def main():
    files = [
        ("01_student_joining.wav", "bé đang vào rồi em"),
        ("03_confirm.wav", "vâng đúng rồi"),
        ("04_busy.wav", "anh bận lắm gọi lại sau đi"),
        ("06_read_phone.wav", "số điện thoại"),
    ]

    print(f"{'File':<25} {'Speech':>8} {'VAD':>8} {'STT':>8} {'Total':>8}  Transcript")
    print("-" * 100)

    for fname, _ in files:
        path = os.path.join(AUDIO_DIR, fname)
        result = await measure_latency(path)
        if result:
            print(
                f"{fname:<25} {result['speech_duration_ms']:>7.0f}ms "
                f"{result['vad_latency_ms']:>7.1f}ms "
                f"{result['stt_latency_ms']:>7.1f}ms "
                f"{result['total_latency_ms']:>7.1f}ms  "
                f"\"{result['transcript']}\""
            )
        else:
            print(f"{fname:<25}  NO SEGMENT")


if __name__ == "__main__":
    asyncio.run(main())