"""AudioProcessor — decode PCM, resample 8k→16k, buffer 512-sample chunks, optional conformer preprocess.

Extends BaseOp. Per-call state isolated via request_id + ContextVar (same pattern as VadDetector).
Filter coefficients computed once at class level — not per instance.
"""

import contextvars
import time
from collections import deque
from typing import Any, Dict, Optional

import numpy as np
import soxr
from scipy import signal

from hush.core.ops.base import BaseOp
from hush.core.utils.common import Param


# ── Filter coefficients — computed once at import ──
_NYQUIST = 16000 / 2
_B_HP, _A_HP = signal.butter(4, 60 / _NYQUIST, btype="high")
_B_LP, _A_LP = signal.butter(4, 4000 / _NYQUIST, btype="low")

# ── ContextVar: holds the active per-call buffer dict during _process ──
_active_audio_buffers: contextvars.ContextVar[dict] = contextvars.ContextVar("active_audio_buffers")


class AudioProcessor(BaseOp):
    """Streaming audio processor: decode → resample → buffer → preprocess → yield chunks.

    Per-call state isolated via class-level _audio_buffers dict keyed by state.request_id.
    """

    type = "code"

    TELCOM_RATE = 8000
    TARGET_RATE = 16000
    CHUNK_SIZE = 512

    TARGET_RMS = 0.15
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

    # Class-level: request_id → per-call buffer dict
    _audio_buffers: Dict[str, dict] = {}

    def __init__(
        self,
        telcom_rate: int = 8000,
        target_rate: int = 16000,
        use_preprocess: bool = True,
        inputs: Optional[Dict[str, Any]] = None,
        outputs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        parsed_inputs = {
            "raw_chunk": Param(type=bytes, required=True),
            "cmc_time": Param(type=int, required=True),
        }
        parsed_outputs = {
            "audio": Param(),
            "cmc_time": Param(),
            "recv_time": Param(),
        }
        self.inputs = self._merge_params(parsed_inputs, self._normalize_params(inputs))
        self.outputs = self._merge_params(parsed_outputs, self._normalize_params(outputs))

        self.telcom_rate = telcom_rate
        self.target_rate = target_rate
        self.use_preprocess = use_preprocess

        self._set_core(self._process)

    # ── Per-call buffer init ──

    def _init_audio_buffers(self) -> dict:
        """Create a fresh set of buffers for one phone call."""
        b = {
            "resampler": soxr.ResampleStream(
                self.telcom_rate, self.target_rate, 1, dtype=np.float32, quality="VHQ"
            ) if self.telcom_rate != self.target_rate else None,
            "buffer": deque(),
            "buffer_len": 0,
        }
        if self.use_preprocess:
            b.update({
                "noise_profile": None,
                "noise_samples": [],
                "rms_buffer": deque(maxlen=50),
                "energy_buffer": deque(maxlen=50),
                "current_gain": 1.0,
                "is_speech": False,
                "speech_frames": 0,
                "silence_frames": 0,
            })
        return b

    # ── run() override: bind per-call buffers via ContextVar ──

    async def run(self, state, context_id=None):
        rid = state.request_id
        if rid not in AudioProcessor._audio_buffers:
            AudioProcessor._audio_buffers[rid] = self._init_audio_buffers()

        token = _active_audio_buffers.set(AudioProcessor._audio_buffers[rid])
        try:
            async for item in super().run(state, context_id):
                yield item
        finally:
            _active_audio_buffers.reset(token)

    @classmethod
    def cleanup(cls, request_id: str) -> None:
        """Release per-call buffers. Call this when the phone call ends."""
        cls._audio_buffers.pop(request_id, None)

    # ── Core processing ──

    async def _process(self, raw_chunk: bytes, cmc_time: int):
        """Decode PCM → resample → buffer → preprocess → yield 512-sample chunks."""
        b = _active_audio_buffers.get()

        # 1. Decode PCM int16 → float32 [-1, 1]
        samples = np.frombuffer(raw_chunk, dtype=np.int16).astype(np.float32) / 32768.0
        if samples.ndim != 1:
            samples = samples.ravel()

        # 2. Resample
        if b["resampler"] is not None:
            samples = b["resampler"].resample_chunk(samples)
            if len(samples) == 0:
                return

        # 3. Buffer
        b["buffer"].append(samples)
        b["buffer_len"] += len(samples)

        # 4. Extract + process chunks
        recv_time = int(time.time() * 1000)
        while b["buffer_len"] >= self.CHUNK_SIZE:
            chunk = self._extract_chunk(b)

            if self.use_preprocess:
                chunk = self._preprocess(chunk, b)

            yield {"audio": chunk, "cmc_time": cmc_time, "recv_time": recv_time}

    def _extract_chunk(self, b: dict) -> np.ndarray:
        collected = []
        remaining = self.CHUNK_SIZE
        while remaining > 0:
            front = b["buffer"][0]
            if len(front) <= remaining:
                collected.append(b["buffer"].popleft())
                remaining -= len(front)
                b["buffer_len"] -= len(front)
            else:
                collected.append(front[:remaining])
                b["buffer"][0] = front[remaining:]
                b["buffer_len"] -= remaining
                remaining = 0
        return np.concatenate(collected) if len(collected) > 1 else collected[0]

    # ── Conformer Preprocess ──

    def _preprocess(self, chunk: np.ndarray, b: dict) -> np.ndarray:
        """Bandpass → pre-emphasis → speech detect → spectral gating → adaptive gain."""
        chunk = signal.filtfilt(_B_HP, _A_HP, chunk).astype(np.float32)
        chunk = signal.filtfilt(_B_LP, _A_LP, chunk).astype(np.float32)
        chunk = signal.lfilter([1, -self.PRE_EMPHASIS_FACTOR], [1], chunk).astype(np.float32)

        is_speech = self._detect_speech(chunk, b)

        if not is_speech:
            self._update_noise_profile(chunk, b)

        if len(b["noise_samples"]) > 5:
            chunk = self._apply_spectral_gating(chunk, b)

        chunk = self._normalize_volume(chunk, is_speech, b)
        return chunk

    def _detect_speech(self, chunk: np.ndarray, b: dict) -> bool:
        energy = float(np.mean(chunk ** 2))
        b["energy_buffer"].append(energy)

        if len(b["energy_buffer"]) > 5:
            sorted_e = sorted(b["energy_buffer"])
            bg_energy = sorted_e[max(0, len(sorted_e) // 10)]
        else:
            bg_energy = 0.001

        zcr = float(np.sum(np.abs(np.diff(np.signbit(chunk))))) / len(chunk)

        if len(chunk) >= 256:
            fft_mag = np.abs(np.fft.rfft(chunk[:256]))
            freqs = np.fft.rfftfreq(256, 1.0 / self.target_rate)
            total = np.sum(fft_mag)
            centroid = float(np.sum(freqs * fft_mag) / total) if total > 0 else 0
        else:
            centroid = 1000

        is_speech_frame = (
            energy > bg_energy * 3
            and 0.05 < zcr < 0.3
            and 500 < centroid < 4000
        )

        if is_speech_frame:
            b["speech_frames"] += 1
            b["silence_frames"] = 0
        else:
            b["silence_frames"] += 1
            if b["silence_frames"] > 10:
                b["speech_frames"] = 0

        b["is_speech"] = b["speech_frames"] > 3
        return b["is_speech"]

    def _update_noise_profile(self, chunk: np.ndarray, b: dict):
        fft_mag = np.abs(np.fft.rfft(chunk))
        if b["noise_profile"] is None:
            b["noise_profile"] = fft_mag.copy()
        b["noise_samples"].append(fft_mag)
        if len(b["noise_samples"]) > 30:
            b["noise_samples"].pop(0)

    def _apply_spectral_gating(self, chunk: np.ndarray, b: dict) -> np.ndarray:
        if not b["noise_samples"]:
            return chunk

        fft = np.fft.rfft(chunk)
        magnitude = np.abs(fft)
        phase = np.angle(fft)

        noise_mag = np.mean(b["noise_samples"][-10:], axis=0)
        threshold = noise_mag * 1.5

        freqs = np.fft.rfftfreq(len(chunk), 1.0 / self.target_rate)
        for (f_low, f_high), weight in self.VN_FREQ_WEIGHTS.items():
            band = (freqs >= f_low) & (freqs < f_high)
            threshold[band] /= weight

        mask = (magnitude > threshold).astype(np.float32)
        mask = np.convolve(mask, np.ones(5) / 5, mode="same")
        mask = np.maximum(mask, 0.1)

        cleaned = magnitude * mask * np.exp(1j * phase)
        return np.fft.irfft(cleaned, n=len(chunk)).astype(np.float32)

    def _normalize_volume(self, chunk: np.ndarray, is_speech: bool, b: dict) -> np.ndarray:
        rms = float(np.sqrt(np.mean(np.square(chunk))))
        b["rms_buffer"].append(rms)

        target_gain = (self.TARGET_RMS if is_speech else self.TARGET_RMS * 0.7) / max(rms, 0.001)
        target_gain = max(self.GAIN_MIN, min(target_gain, self.GAIN_MAX))

        if target_gain > b["current_gain"]:
            b["current_gain"] += self.GAIN_ATTACK * (target_gain - b["current_gain"])
        else:
            b["current_gain"] += self.GAIN_RELEASE * (target_gain - b["current_gain"])

        chunk = chunk * b["current_gain"]
        if np.max(np.abs(chunk)) > 0.95:
            chunk = np.tanh(chunk) * 0.95

        return chunk.astype(np.float32)
