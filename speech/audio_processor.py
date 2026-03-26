"""AudioProcessor — decode PCM, resample 8k→16k, buffer 512-sample chunks, optional conformer preprocess.

Extends BaseOp for per-call state (resampler, buffer, conformer state).
ONNX sessions not needed here — pure audio DSP.
"""

import time
from collections import deque
from typing import Any, Dict, Optional

import numpy as np
import soxr
from scipy import signal

from hush.core.ops.base import BaseOp
from hush.core.utils.common import Param


class AudioProcessor(BaseOp):
    """Streaming audio processor: decode → resample → buffer → preprocess → yield chunks.

    State per instance (per call):
        resampler: soxr streaming resampler (maintains phase)
        buffer/buffer_len: accumulate samples until 512
        conformer state: noise_profile, rms_buffer, current_gain, etc.
    """

    type = "code"

    # ── Config ──
    TELCOM_RATE = 8000
    TARGET_RATE = 16000
    CHUNK_SIZE = 512  # 32ms at 16kHz

    # Conformer preprocess config
    TARGET_RMS = 0.15
    HIGH_PASS_CUTOFF = 60
    LOW_PASS_CUTOFF = 4000
    PRE_EMPHASIS_FACTOR = 0.97
    GAIN_ATTACK = 0.2
    GAIN_RELEASE = 0.05
    GAIN_MIN = 0.1
    GAIN_MAX = 5.0

    # Vietnamese frequency weights for spectral gating
    VN_FREQ_WEIGHTS = {
        (60, 300): 1.5,
        (300, 800): 1.3,
        (1200, 2500): 1.4,
        (2500, 3500): 1.2,
    }

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

        # Merge I/O
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

        # ── Per-call state (initialized here, reset per call via new instance) ──

        # Resampler
        if telcom_rate != target_rate:
            self.resampler = soxr.ResampleStream(
                telcom_rate, target_rate, 1, dtype=np.float32, quality="VHQ"
            )
        else:
            self.resampler = None

        # Buffer
        self._buffer = deque()
        self._buffer_len = 0

        # Conformer preprocess state
        if use_preprocess:
            nyquist = target_rate / 2
            self._b_hp, self._a_hp = signal.butter(4, self.HIGH_PASS_CUTOFF / nyquist, btype="high")
            self._b_lp, self._a_lp = signal.butter(4, self.LOW_PASS_CUTOFF / nyquist, btype="low")
            self._noise_profile = None
            self._noise_samples = []
            self._rms_buffer = deque(maxlen=50)
            self._energy_buffer = deque(maxlen=50)
            self._current_gain = 1.0
            self._is_speech = False
            self._speech_frames = 0
            self._silence_frames = 0

        self.core = self._process

    async def _process(self, raw_chunk: bytes, cmc_time: int):
        """Process raw PCM → yield 512-sample float32 chunks."""

        # 1. Decode PCM int16 → float32 [-1, 1]
        samples = np.frombuffer(raw_chunk, dtype=np.int16).astype(np.float32) / 32768.0

        # 2. Resample if needed
        if self.resampler is not None:
            samples = self.resampler.resample_chunk(samples)
            if len(samples) == 0:
                return  # PENDING — resampler buffering

        # 3. Ensure 1D
        if samples.ndim != 1:
            samples = samples.ravel()

        # 4. Buffer
        self._buffer.append(samples)
        self._buffer_len += len(samples)

        # 5. Extract + process chunks
        recv_time = int(time.time() * 1000)
        while self._buffer_len >= self.CHUNK_SIZE:
            chunk = self._extract_chunk()

            # 6. Conformer preprocess (optional)
            if self.use_preprocess:
                chunk = self._preprocess(chunk)

            yield {"audio": chunk, "cmc_time": cmc_time, "recv_time": recv_time}

    def _extract_chunk(self) -> np.ndarray:
        """Extract exactly CHUNK_SIZE samples from buffer."""
        collected = []
        remaining = self.CHUNK_SIZE
        while remaining > 0:
            front = self._buffer[0]
            if len(front) <= remaining:
                collected.append(self._buffer.popleft())
                remaining -= len(front)
                self._buffer_len -= len(front)
            else:
                collected.append(front[:remaining])
                self._buffer[0] = front[remaining:]
                self._buffer_len -= remaining
                remaining = 0
        return np.concatenate(collected) if len(collected) > 1 else collected[0]

    # ── Conformer Preprocess ──

    def _preprocess(self, chunk: np.ndarray) -> np.ndarray:
        """Bandpass → pre-emphasis → speech detect → spectral gating → adaptive gain."""

        # Bandpass filter (60-4000 Hz)
        chunk = signal.filtfilt(self._b_hp, self._a_hp, chunk).astype(np.float32)
        chunk = signal.filtfilt(self._b_lp, self._a_lp, chunk).astype(np.float32)

        # Pre-emphasis
        chunk = signal.lfilter([1, -self.PRE_EMPHASIS_FACTOR], [1], chunk).astype(np.float32)

        # Speech detection
        is_speech = self._detect_speech(chunk)

        # Noise profile update (during silence)
        if not is_speech:
            self._update_noise_profile(chunk)

        # Spectral gating (when noise profile ready)
        if len(self._noise_samples) > 5:
            chunk = self._apply_spectral_gating(chunk)

        # Adaptive gain control
        chunk = self._normalize_volume(chunk, is_speech)

        return chunk

    def _detect_speech(self, chunk: np.ndarray) -> bool:
        """Energy + ZCR + spectral centroid based speech detection with hysteresis."""
        energy = float(np.mean(chunk ** 2))
        self._energy_buffer.append(energy)

        # Background energy estimate
        if len(self._energy_buffer) > 5:
            sorted_e = sorted(self._energy_buffer)
            bg_energy = sorted_e[max(0, len(sorted_e) // 10)]
        else:
            bg_energy = 0.001

        # Zero crossing rate
        zcr = float(np.sum(np.abs(np.diff(np.signbit(chunk))))) / len(chunk)

        # Spectral centroid
        if len(chunk) >= 256:
            fft_mag = np.abs(np.fft.rfft(chunk[:256]))
            freqs = np.fft.rfftfreq(256, 1.0 / self.target_rate)
            total = np.sum(fft_mag)
            centroid = float(np.sum(freqs * fft_mag) / total) if total > 0 else 0
        else:
            centroid = 1000  # default mid-range

        # Decision
        is_speech_frame = (
            energy > bg_energy * 3
            and 0.05 < zcr < 0.3
            and 500 < centroid < 4000
        )

        # Hysteresis
        if is_speech_frame:
            self._speech_frames += 1
            self._silence_frames = 0
        else:
            self._silence_frames += 1
            if self._silence_frames > 10:
                self._speech_frames = 0

        self._is_speech = self._speech_frames > 3
        return self._is_speech

    def _update_noise_profile(self, chunk: np.ndarray):
        """Learn noise spectrum from silence frames."""
        fft_mag = np.abs(np.fft.rfft(chunk))
        if self._noise_profile is None:
            self._noise_profile = fft_mag.copy()
        self._noise_samples.append(fft_mag)
        if len(self._noise_samples) > 30:
            self._noise_samples.pop(0)

    def _apply_spectral_gating(self, chunk: np.ndarray) -> np.ndarray:
        """Suppress noise using learned spectral profile with Vietnamese frequency weighting."""
        if not self._noise_samples:
            return chunk

        fft = np.fft.rfft(chunk)
        magnitude = np.abs(fft)
        phase = np.angle(fft)

        # Noise threshold from recent samples
        noise_mag = np.mean(self._noise_samples[-10:], axis=0)
        threshold = noise_mag * 1.5

        # Vietnamese frequency weighting
        freqs = np.fft.rfftfreq(len(chunk), 1.0 / self.target_rate)
        for (f_low, f_high), weight in self.VN_FREQ_WEIGHTS.items():
            band = (freqs >= f_low) & (freqs < f_high)
            threshold[band] /= weight  # lower threshold = preserve more

        # Mask
        mask = (magnitude > threshold).astype(np.float32)

        # Smooth mask
        kernel = np.ones(5) / 5
        mask = np.convolve(mask, kernel, mode="same")

        # Spectral floor
        mask = np.maximum(mask, 0.1)

        # Apply
        cleaned = magnitude * mask * np.exp(1j * phase)
        result = np.fft.irfft(cleaned, n=len(chunk)).astype(np.float32)
        return result

    def _normalize_volume(self, chunk: np.ndarray, is_speech: bool) -> np.ndarray:
        """Adaptive gain with attack/release smoothing and soft clipping."""
        rms = float(np.sqrt(np.mean(np.square(chunk))))
        self._rms_buffer.append(rms)

        # Target gain
        if is_speech:
            target_gain = self.TARGET_RMS / max(rms, 0.001)
        else:
            target_gain = (self.TARGET_RMS * 0.7) / max(rms, 0.001)

        target_gain = max(self.GAIN_MIN, min(target_gain, self.GAIN_MAX))

        # Smooth
        if target_gain > self._current_gain:
            self._current_gain += self.GAIN_ATTACK * (target_gain - self._current_gain)
        else:
            self._current_gain += self.GAIN_RELEASE * (target_gain - self._current_gain)

        chunk = chunk * self._current_gain

        # Soft clipping
        if np.max(np.abs(chunk)) > 0.95:
            chunk = np.tanh(chunk) * 0.95

        return chunk.astype(np.float32)
