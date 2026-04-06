"""Compare: with vs without conformer preprocess → VAD + STT quality."""

import asyncio
import os
import sys

import numpy as np
from scipy.io import wavfile
import tritonclient.grpc as grpcclient

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ["LOG_LEVEL"] = "WARNING"

from speech.audio_processor import AudioProcessor
from speech.vad_detector import VadDetector

AUDIO_DIR = os.path.join(os.path.dirname(__file__), "../tests/speech/audio")
STT_URL = "192.168.1.212:8001"

WAVS = [
    ("01_student_joining.wav", "bé đang vào rồi em"),
    ("03_confirm.wav", "vâng đúng rồi"),
    ("04_busy.wav", "anh bận lắm gọi lại sau đi"),
    ("07_fallback.wav", "trời mưa quá"),
]


async def run_pipeline(wav_path, use_preprocess):
    sr, raw = wavfile.read(wav_path)
    silence = np.zeros(int(1.0 * sr), dtype=raw.dtype)
    audio = np.concatenate([raw, silence])

    proc = AudioProcessor(name="p", use_preprocess=use_preprocess)
    vad = VadDetector(name="v", model_path="models/silero_vad.onnx")

    segments = []
    for i in range(0, len(audio), 320):
        chunk = audio[i:i + 320]
        if len(chunk) < 320:
            chunk = np.pad(chunk, (0, 320 - len(chunk)))
        async for out in proc._process(raw_chunk=chunk.tobytes(), cmc_time=1000):
            async for seg in vad._process(audio=out["audio"], cmc_time=out["cmc_time"], recv_time=out["recv_time"]):
                segments.append(seg)

    if not segments:
        return 0, "(no speech)", 0.0

    speech = segments[0]["speech_audio"].astype(np.float32).reshape(1, -1)
    client = grpcclient.InferenceServerClient(url=STT_URL)
    inp = grpcclient.InferInput("AUDIO_SIGNAL", list(speech.shape), "FP32")
    inp.set_data_from_numpy(speech)
    out = grpcclient.InferRequestedOutput("TRANSCRIPT")
    result = client.infer(model_name="fastconformer_asr", inputs=[inp], outputs=[out], client_timeout=30)
    text = result.as_numpy("TRANSCRIPT").flat[0]
    text = text.decode("utf-8") if isinstance(text, bytes) else str(text)

    return len(segments), text, segments[0]["speech_duration_ms"]


async def main():
    print(f"{'File':<30} {'Preprocess':>10} {'Segs':>5} {'Duration':>10} {'Transcript'}")
    print("-" * 100)

    for fname, expected in WAVS:
        wav = os.path.join(AUDIO_DIR, fname)
        for pp in [True, False]:
            n, text, dur = await run_pipeline(wav, pp)
            label = "ON" if pp else "OFF"
            print(f"{fname:<30} {label:>10} {n:>5} {dur:>9.0f}ms  {text}")
        print()


if __name__ == "__main__":
    asyncio.run(main())
