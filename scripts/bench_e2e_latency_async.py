"""Benchmark: end-to-end latency from user speech end → agent first audio.

Measures at CCU=1, 8, 20.
Each call: WAV → pipeline → measure time from last speech chunk to TTS audio ready.
"""

import asyncio
import os
import sys
import time

import numpy as np
import soxr
from scipy.io import wavfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "../.env"))

from speech.audio_processor import AudioProcessor
from speech.vad_detector import VadDetector

import tritonclient.grpc.aio as grpcclient
import httpx

AUDIO_DIR = os.path.join(os.path.dirname(__file__), "../tests/speech/audio")
STT_URL = "192.168.1.212:8001"
TTS_URL = "192.168.1.212:3001"
LLM_URL = "https://api.anthropic.com/v1/messages"
LLM_KEY = os.getenv("ANTHROPIC_API_KEY")
LLM_MODEL = "claude-3-haiku-20240307"

SCRIPT_DATA = {"student_name": "Minh", "class_time": "19:00", "program_name": "AI CLASS"}


async def measure_one_call(call_id: int, wav_file: str, triton_client, llm_client):
    """Full pipeline, measure each stage."""
    wav_path = os.path.join(AUDIO_DIR, wav_file)
    sr, raw_audio = wavfile.read(wav_path)
    speech_end_sample = len(raw_audio)
    silence = np.zeros(int(1.0 * sr), dtype=raw_audio.dtype)
    audio = np.concatenate([raw_audio, silence])

    proc = AudioProcessor(name=f"a{call_id}")
    vad = VadDetector(name=f"v{call_id}", model_path="models/silero_vad.onnx")

    # ── Stage 1: Audio → VAD ──
    speech_end_time = None
    vad_done_time = None
    segment = None

    for i in range(0, len(audio), 320):
        chunk = audio[i:i+320]
        if len(chunk) < 320:
            chunk = np.pad(chunk, (0, 320 - len(chunk)))

        if i >= speech_end_sample and speech_end_time is None:
            speech_end_time = time.perf_counter()

        async for proc_out in proc._process(raw_chunk=chunk.tobytes(), cmc_time=1000):
            async for vad_out in vad._process(
                audio=proc_out["audio"], cmc_time=proc_out["cmc_time"], recv_time=proc_out["recv_time"]
            ):
                vad_done_time = time.perf_counter()
                segment = vad_out

    if not segment or not speech_end_time:
        return {"call_id": call_id, "error": "no segment"}

    t_vad = (vad_done_time - speech_end_time) * 1000

    # ── Stage 2: STT ──
    stt_start = time.perf_counter()
    speech_audio = segment["speech_audio"].astype(np.float32)
    if speech_audio.ndim == 1:
        speech_audio = speech_audio.reshape(1, -1)
    inp = grpcclient.InferInput("AUDIO_SIGNAL", list(speech_audio.shape), "FP32")
    inp.set_data_from_numpy(speech_audio)
    out = grpcclient.InferRequestedOutput("TRANSCRIPT")

    result = await triton_client.infer(
        model_name="fastconformer_asr", inputs=[inp], outputs=[out], client_timeout=30
    )
    raw = result.as_numpy("TRANSCRIPT")
    transcript = raw.flat[0].decode("utf-8") if isinstance(raw.flat[0], bytes) else str(raw.flat[0])
    t_stt = (time.perf_counter() - stt_start) * 1000

    # ── Stage 3: LLM ──
    llm_start = time.perf_counter()
    resp = await llm_client.post(
        LLM_URL,
        headers={"x-api-key": LLM_KEY, "anthropic-version": "2023-06-01", "content-type": "application/json"},
        json={
            "model": LLM_MODEL,
            "max_tokens": 200,
            "messages": [{"role": "user", "content": f"Classify intent and respond to: '{transcript}'. Reply short."}],
        },
        timeout=30,
    )
    t_llm = (time.perf_counter() - llm_start) * 1000

    # ── Stage 4: TTS ──
    tts_start = time.perf_counter()
    # Simple TTS call — fastspeech2 + hifigan
    response_text = "Dạ vâng em ghi nhận"
    from speech.tts.symbols import symbols
    from speech.tts import text_to_sequence, clean_vietnamese_text
    import speech.tts.vietnamese_phonemes as viphonemes
    import re

    text = re.sub(r"[,;.?\-!:]", " ", response_text)
    text = clean_vietnamese_text(text)

    from pipeline.mock_source import CHUNK_SIZE
    lexicon_path = os.path.join(os.path.dirname(__file__), "../speech/tts/vi-new-lexicon.txt")
    lexicon = {}
    with open(lexicon_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) == 2:
                lexicon[parts[0].lower()] = parts[1].split()

    phones = []
    for w in re.split(r"([,;.\-?\!\s+])", text):
        if not w:
            continue
        if w.lower() in lexicon:
            phones += lexicon[w.lower()]
        elif w.strip():
            phones += viphonemes.parse_word(w)

    phones_str = "{" + " ".join(phones) + "}"
    tokens = np.array(text_to_sequence(phones_str, ["vietnamese_cleaners"]), dtype=np.int64)
    T = tokens.size
    texts = tokens.reshape(1, T)

    # FastSpeech2
    ii_texts = grpcclient.InferInput("texts", [1, T], "INT64"); ii_texts.set_data_from_numpy(texts)
    ii_src = grpcclient.InferInput("src_lens.1", [1], "INT64"); ii_src.set_data_from_numpy(np.array([T], dtype=np.int64))
    ii_max = grpcclient.InferInput("max_src_len", [1], "INT64"); ii_max.set_data_from_numpy(np.array([T], dtype=np.int64))
    ii_p = grpcclient.InferInput("p_control", [1], "FP32"); ii_p.set_data_from_numpy(np.array([1.0], dtype=np.float32))
    ii_e = grpcclient.InferInput("e_control", [1], "FP32"); ii_e.set_data_from_numpy(np.array([1.0], dtype=np.float32))
    ii_d = grpcclient.InferInput("d_control", [1], "FP32"); ii_d.set_data_from_numpy(np.array([1.1], dtype=np.float32))

    tts_client = grpcclient.InferenceServerClient(url=TTS_URL)
    out_names = ["output", "postnet_output", "p_predictions", "e_predictions",
                 "log_d_predictions", "d_rounded", "src_masks", "mel_masks", "src_lens", "mel_lens"]
    outs = [grpcclient.InferRequestedOutput(n) for n in out_names]

    fs2_result = await tts_client.infer(
        model_name="fastspeech2", inputs=[ii_texts, ii_src, ii_max, ii_p, ii_e, ii_d], outputs=outs, client_timeout=30
    )
    postnet = fs2_result.as_numpy("postnet_output")
    mel = np.transpose(postnet, (0, 2, 1)).astype(np.float32)
    B, n_mels, Tm = mel.shape

    # HiFi-GAN
    ii_mels = grpcclient.InferInput("mels", [B, n_mels, Tm], "FP32"); ii_mels.set_data_from_numpy(mel)
    hifi_out = grpcclient.InferRequestedOutput("audio")
    hifi_result = await tts_client.infer(
        model_name="hifigan", inputs=[ii_mels], outputs=[hifi_out], client_timeout=30
    )

    t_tts = (time.perf_counter() - tts_start) * 1000

    total = t_vad + t_stt + t_llm + t_tts

    return {
        "call_id": call_id,
        "t_vad": t_vad,
        "t_stt": t_stt,
        "t_llm": t_llm,
        "t_tts": t_tts,
        "total": total,
        "transcript": transcript[:30],
    }


async def bench(ccu: int):
    wavs = ["01_student_joining.wav", "03_confirm.wav", "04_busy.wav", "06_read_phone.wav", "07_fallback.wav"]
    triton_client = grpcclient.InferenceServerClient(url=STT_URL)

    async with httpx.AsyncClient() as llm_client:
        # Warmup
        await measure_one_call(-1, wavs[0], triton_client, llm_client)

        print(f"\n{'='*70}")
        print(f"CCU={ccu}: End-to-end latency (speech end → agent audio ready)")
        print(f"{'='*70}")

        t0 = time.perf_counter()
        tasks = [measure_one_call(i, wavs[i % len(wavs)], triton_client, llm_client) for i in range(ccu)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        wall = (time.perf_counter() - t0) * 1000

    ok = [r for r in results if isinstance(r, dict) and "total" in r]
    errs = [r for r in results if not isinstance(r, dict) or "error" in r]

    if ok:
        print(f"\n  {'':>5} {'VAD':>8} {'STT':>8} {'LLM':>8} {'TTS':>8} {'TOTAL':>8}")
        print(f"  {'─'*5} {'─'*8} {'─'*8} {'─'*8} {'─'*8} {'─'*8}")
        for r in sorted(ok, key=lambda x: x["call_id"]):
            print(f"  #{r['call_id']:>3d} {r['t_vad']:>7.0f}ms {r['t_stt']:>7.0f}ms {r['t_llm']:>7.0f}ms {r['t_tts']:>7.0f}ms {r['total']:>7.0f}ms")

        totals = [r["total"] for r in ok]
        vads = [r["t_vad"] for r in ok]
        stts = [r["t_stt"] for r in ok]
        llms = [r["t_llm"] for r in ok]
        ttss = [r["t_tts"] for r in ok]

        print(f"\n  {'AVG':>5} {sum(vads)/len(vads):>7.0f}ms {sum(stts)/len(stts):>7.0f}ms {sum(llms)/len(llms):>7.0f}ms {sum(ttss)/len(ttss):>7.0f}ms {sum(totals)/len(totals):>7.0f}ms")
        print(f"  Wall time: {wall:.0f}ms  |  Errors: {len(errs)}")

    if errs:
        for e in errs:
            print(f"  ERROR: {e}")


async def main():
    for ccu in [1, 8, 16, 20]:
        await bench(ccu)
    print(f"\n{'='*70}\nDone!")


if __name__ == "__main__":
    asyncio.run(main())
