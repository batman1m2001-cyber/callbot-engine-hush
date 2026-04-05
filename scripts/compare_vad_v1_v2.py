"""Compare VadDetector: v1 (BaseOp) vs v2 (@graph 3 ops, reuse engine).

Uses the SAME AudioProcessor v1 for all — isolates only the VAD difference.
"""

import asyncio
import os
import sys
import time

import numpy as np
from scipy.io import wavfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ["LOG_LEVEL"] = "WARNING"

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "../.env"))

from hush.core import Hush, graph, START, END, PARENT, op
from speech.audio_processor import AudioProcessor
from speech.vad_detector import VadDetector
from speech.vad_detector_v2 import vad_detector as vad_detector_v2

AUDIO_DIR = os.path.join(os.path.dirname(__file__), "../tests/speech/audio")


async def preprocess_wav(wav_path):
    """Run AudioProcessor v1 once, return list of (audio, cmc_time, recv_time) tuples."""
    sr, raw_audio = wavfile.read(wav_path)
    silence = np.zeros(int(sr * 1.0), dtype=raw_audio.dtype)
    raw_audio = np.concatenate([raw_audio, silence])
    proc = AudioProcessor(name="pre", use_preprocess=False)
    chunks = []
    for i in range(0, len(raw_audio), 320):
        chunk = raw_audio[i:i + 320]
        if len(chunk) < 320:
            chunk = np.pad(chunk, (0, 320 - len(chunk)))
        async for out in proc._process(raw_chunk=chunk.tobytes(), cmc_time=1000 + i):
            chunks.append((out["audio"].copy(), out["cmc_time"], out["recv_time"]))
    return chunks


# ── V1: direct BaseOp ──

async def run_v1(audio_chunks):
    vad = VadDetector(name="v1_vad")
    segments = []
    for audio, cmc_time, recv_time in audio_chunks:
        async for seg in vad._process(audio=audio, cmc_time=cmc_time, recv_time=recv_time):
            segments.append({
                "speech_audio": seg["speech_audio"].copy(),
                "speech_duration_ms": seg["speech_duration_ms"],
                "num_chunks": seg["num_chunks"],
            })
    return segments


# ── V2: @graph via reusable engine ──

@op
def chunk_source(audio_chunks: list):
    for audio, cmc_time, recv_time in audio_chunks:
        yield {"audio": audio, "cmc_time": cmc_time, "recv_time": recv_time}


@graph
def pipeline_v2(audio_chunks):
    source = chunk_source(audio_chunks=audio_chunks)
    vad = vad_detector_v2(
        audio=source["audio"], cmc_time=source["cmc_time"], recv_time=source["recv_time"],
    )
    START >> source >> vad >> END


def _collect_segments(result):
    segments = []
    speech_audios = result.get("speech_audio", [])
    durations = result.get("speech_duration_ms", [])
    num_chunks = result.get("num_chunks", [])
    if not isinstance(speech_audios, list):
        speech_audios = [speech_audios] if speech_audios is not None else []
    if not isinstance(durations, list):
        durations = [durations] if durations is not None else []
    if not isinstance(num_chunks, list):
        num_chunks = [num_chunks] if num_chunks is not None else []
    for i in range(len(speech_audios)):
        segments.append({
            "speech_audio": np.array(speech_audios[i], dtype=np.float32)
                if not isinstance(speech_audios[i], np.ndarray) else speech_audios[i],
            "speech_duration_ms": durations[i] if i < len(durations) else 0,
            "num_chunks": num_chunks[i] if i < len(num_chunks) else 0,
        })
    return segments


# ── Comparison ──

def compare_segments(segs_a, segs_b):
    if len(segs_a) != len(segs_b):
        return False, f"count {len(segs_a)} vs {len(segs_b)}"
    if len(segs_a) == 0:
        return True, "both 0 segments"
    for i, (sa, sb) in enumerate(zip(segs_a, segs_b)):
        dur_diff = abs(sa["speech_duration_ms"] - sb["speech_duration_ms"])
        min_len = min(len(sa["speech_audio"]), len(sb["speech_audio"]))
        audio_diff = np.max(np.abs(sa["speech_audio"][:min_len] - sb["speech_audio"][:min_len])) if min_len > 0 else 0.0
        if dur_diff > 1.0 or audio_diff > 0.01:
            return False, f"seg{i}: dur={dur_diff:.0f}ms audio={audio_diff:.4f}"
    return True, "exact"


async def main():
    print("Compare VadDetector: v1 (BaseOp) vs v2 (@graph 3 ops, reuse engine)")
    print("Same AudioProcessor v1 input for all — isolating VAD only")
    print("=" * 75)

    wav_files = sorted([
        f for f in os.listdir(AUDIO_DIR)
        if f.endswith(".wav") and f != "test_tts.wav"
    ])

    # Preprocess all audio once
    all_chunks = {}
    for wav_file in wav_files:
        all_chunks[wav_file] = await preprocess_wav(os.path.join(AUDIO_DIR, wav_file))

    print(f"\n{'File':<28} {'V1':>5} {'V2':>5}  Match?")
    print("-" * 60)

    summary = []
    for wav_file in wav_files:
        chunks = all_chunks[wav_file]

        t0 = time.perf_counter()
        segs_v1 = await run_v1(chunks)
        t_v1 = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        wf = pipeline_v2(audio_chunks=chunks)
        engine = Hush(wf)
        result = await engine.run(inputs={})
        segs_v2 = _collect_segments(result)
        t_v2 = (time.perf_counter() - t0) * 1000

        match, detail = compare_segments(segs_v1, segs_v2)

        label = wav_file[:27]
        print(f"{label:<28} {t_v1:4.0f}ms {t_v2:4.0f}ms  {detail}")

        summary.append({"file": wav_file, "v1": t_v1, "v2": t_v2, "match": match})

    # Summary
    print(f"\n{'=' * 60}")
    avg = lambda k: sum(s[k] for s in summary) / len(summary)
    print(f"  Avg latency (VAD only, per file):")
    print(f"    V1 (BaseOp direct):       {avg('v1'):5.0f}ms")
    print(f"    V2 (3 ops, new engine):   {avg('v2'):5.0f}ms")
    print(f"    Overhead:                 {avg('v2') - avg('v1'):+5.0f}ms")

    matched = sum(1 for s in summary if s["match"])
    print(f"\n  V1 vs V2: {matched}/{len(summary)} exact match")

    # ── Concat all 10 WAVs into 1 big stream ──
    print(f"\n{'=' * 60}")
    print("  CONCAT TEST: all 10 WAVs concatenated (with silence gaps)")
    print(f"{'=' * 60}")

    # Concatenate all preprocessed chunks with 1s silence gaps between files
    all_concat = []
    for wav_file in wav_files:
        all_concat.extend(all_chunks[wav_file])
    total_chunks = len(all_concat)
    print(f"  Total chunks: {total_chunks}")

    # V1
    t0 = time.perf_counter()
    segs_v1 = await run_v1(all_concat)
    t_v1 = (time.perf_counter() - t0) * 1000

    # V2
    t0 = time.perf_counter()
    wf = pipeline_v2(audio_chunks=all_concat)
    engine = Hush(wf)
    result = await engine.run(inputs={})
    segs_v2 = _collect_segments(result)
    t_v2 = (time.perf_counter() - t0) * 1000

    match, detail = compare_segments(segs_v1, segs_v2)
    print(f"  V1: {len(segs_v1)} segments, {t_v1:.0f}ms")
    print(f"  V2: {len(segs_v2)} segments, {t_v2:.0f}ms")
    print(f"  Overhead: {t_v2 - t_v1:+.0f}ms")
    print(f"  Match: {detail}")

    matched = sum(1 for s in summary if s["match"])
    print(f"\n  V1 vs V2: {matched}/{len(summary)} exact match")


if __name__ == "__main__":
    asyncio.run(main())
