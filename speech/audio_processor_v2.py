"""AudioProcessor v2 — @graph + PARENT.shared() for per-request state.

Same logic as AudioProcessor BaseOp, but state in EngineState → no leak.
"""

import time
from collections import deque

import numpy as np
import soxr
from scipy import signal

from hush.core import graph, PARENT, START, END
from hush.core.ops import op

# ── Constants ──
TELCOM_RATE = 8000
TARGET_RATE = 16000
CHUNK_SIZE = 512
TARGET_RMS = 0.15
HIGH_PASS_CUTOFF = 60
LOW_PASS_CUTOFF = 4000
PRE_EMPHASIS_FACTOR = 0.97
GAIN_ATTACK = 0.2
GAIN_RELEASE = 0.05
GAIN_MIN = 0.1
GAIN_MAX = 5.0
VN_FREQ_WEIGHTS = {
    (60, 300): 1.5,
    (300, 800): 1.3,
    (1200, 2500): 1.4,
    (2500, 3500): 1.2,
}

# ── Shared (immutable) config — computed once at import ──
_nyquist = TARGET_RATE / 2
_B_HP, _A_HP = signal.butter(4, HIGH_PASS_CUTOFF / _nyquist, btype="high")
_B_LP, _A_LP = signal.butter(4, LOW_PASS_CUTOFF / _nyquist, btype="low")


@op
def decode_pcm(raw_chunk: bytes) -> dict:
    """Decode PCM int16 → float32 [-1, 1]. Stateless."""
    samples = np.frombuffer(raw_chunk, dtype=np.int16).astype(np.float32) / 32768.0
    if samples.ndim != 1:
        samples = samples.ravel()
    return {"samples": samples}


@op
async def resample_buffer_preprocess(samples: np.ndarray, cmc_time: int, state: dict):
    """Resample + buffer + preprocess. Stateful via state dict (PARENT.shared).

    state keys: resampler, buffer, noise_samples, energy_buffer,
                rms_buffer, current_gain, speech_frames, silence_frames
    """
    resampler = state.get("resampler")
    buffer = state["buffer"]

    # Resample
    if resampler is not None:
        samples = resampler.resample_chunk(samples)
        if len(samples) == 0:
            return  # PENDING

    # Buffer
    buffer.append(samples)
    total = sum(len(b) for b in buffer)

    recv_time = int(time.time() * 1000)

    while total >= CHUNK_SIZE:
        # Extract chunk
        collected = []
        remaining = CHUNK_SIZE
        while remaining > 0:
            front = buffer[0]
            if len(front) <= remaining:
                collected.append(buffer.popleft())
                remaining -= len(front)
            else:
                collected.append(front[:remaining])
                buffer[0] = front[remaining:]
                remaining = 0
        total -= CHUNK_SIZE
        chunk = np.concatenate(collected) if len(collected) > 1 else collected[0]

        # Preprocess
        chunk = _preprocess(chunk, state)

        yield {"audio": chunk, "cmc_time": cmc_time, "recv_time": recv_time}


def _preprocess(chunk: np.ndarray, state: dict) -> np.ndarray:
    """Bandpass → pre-emphasis → speech detect → spectral gating → gain."""
    chunk = signal.filtfilt(_B_HP, _A_HP, chunk).astype(np.float32)
    chunk = signal.filtfilt(_B_LP, _A_LP, chunk).astype(np.float32)
    chunk = signal.lfilter([1, -PRE_EMPHASIS_FACTOR], [1], chunk).astype(np.float32)

    is_speech = _detect_speech(chunk, state)

    if not is_speech:
        _update_noise_profile(chunk, state)

    if len(state["noise_samples"]) > 5:
        chunk = _apply_spectral_gating(chunk, state)

    chunk = _normalize_volume(chunk, is_speech, state)
    return chunk


def _detect_speech(chunk: np.ndarray, state: dict) -> bool:
    energy = float(np.mean(chunk ** 2))
    state["energy_buffer"].append(energy)

    if len(state["energy_buffer"]) > 5:
        sorted_e = sorted(state["energy_buffer"])
        bg_energy = sorted_e[max(0, len(sorted_e) // 10)]
    else:
        bg_energy = 0.001

    zcr = float(np.sum(np.abs(np.diff(np.signbit(chunk))))) / len(chunk)

    if len(chunk) >= 256:
        fft_mag = np.abs(np.fft.rfft(chunk[:256]))
        freqs = np.fft.rfftfreq(256, 1.0 / TARGET_RATE)
        total = np.sum(fft_mag)
        centroid = float(np.sum(freqs * fft_mag) / total) if total > 0 else 0
    else:
        centroid = 1000

    is_speech_frame = energy > bg_energy * 3 and 0.05 < zcr < 0.3 and 500 < centroid < 4000

    if is_speech_frame:
        state["speech_frames"] += 1
        state["silence_frames"] = 0
    else:
        state["silence_frames"] += 1
        if state["silence_frames"] > 10:
            state["speech_frames"] = 0

    return state["speech_frames"] > 3


def _update_noise_profile(chunk: np.ndarray, state: dict):
    fft_mag = np.abs(np.fft.rfft(chunk))
    state["noise_samples"].append(fft_mag)
    if len(state["noise_samples"]) > 30:
        state["noise_samples"].pop(0)


def _apply_spectral_gating(chunk: np.ndarray, state: dict) -> np.ndarray:
    noise_samples = state["noise_samples"]
    if not noise_samples:
        return chunk

    fft = np.fft.rfft(chunk)
    magnitude = np.abs(fft)
    phase = np.angle(fft)

    noise_mag = np.mean(noise_samples[-10:], axis=0)
    threshold = noise_mag * 1.5

    freqs = np.fft.rfftfreq(len(chunk), 1.0 / TARGET_RATE)
    for (f_low, f_high), weight in VN_FREQ_WEIGHTS.items():
        band = (freqs >= f_low) & (freqs < f_high)
        threshold[band] /= weight

    mask = (magnitude > threshold).astype(np.float32)
    kernel = np.ones(5) / 5
    mask = np.convolve(mask, kernel, mode="same")
    mask = np.maximum(mask, 0.1)

    cleaned = magnitude * mask * np.exp(1j * phase)
    return np.fft.irfft(cleaned, n=len(chunk)).astype(np.float32)


def _normalize_volume(chunk: np.ndarray, is_speech: bool, state: dict) -> np.ndarray:
    rms = float(np.sqrt(np.mean(np.square(chunk))))
    state["rms_buffer"].append(rms)

    target_gain = (TARGET_RMS if is_speech else TARGET_RMS * 0.7) / max(rms, 0.001)
    target_gain = max(GAIN_MIN, min(target_gain, GAIN_MAX))

    if target_gain > state["current_gain"]:
        state["current_gain"] += GAIN_ATTACK * (target_gain - state["current_gain"])
    else:
        state["current_gain"] += GAIN_RELEASE * (target_gain - state["current_gain"])

    chunk = chunk * state["current_gain"]
    if np.max(np.abs(chunk)) > 0.95:
        chunk = np.tanh(chunk).astype(np.float32) * 0.95

    return chunk.astype(np.float32)


@graph
def audio_processor(raw_chunk, cmc_time):
    """Streaming audio processor: decode → resample → buffer → preprocess → yield chunks.

    State via PARENT.shared() → per-request, no leak.
    """
    PARENT.shared(
        proc_state={
            "resampler": soxr.ResampleStream(TELCOM_RATE, TARGET_RATE, 1, dtype=np.float32, quality="VHQ"),
            "buffer": deque(),
            "noise_samples": [],
            "energy_buffer": deque(maxlen=50),
            "rms_buffer": deque(maxlen=50),
            "current_gain": 1.0,
            "speech_frames": 0,
            "silence_frames": 0,
        }
    )

    dec = decode_pcm(raw_chunk=raw_chunk)
    proc = resample_buffer_preprocess(
        samples=dec["samples"],
        cmc_time=cmc_time,
        state=PARENT["proc_state"],
    )

    START >> dec >> proc >> END
