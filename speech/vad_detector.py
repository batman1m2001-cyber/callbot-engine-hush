"""VadDetector — Silero VAD ONNX inference + foreground detection + speech segmentation.

Extends BaseOp. Streaming: yields speech segments when speech end detected, PENDING otherwise.
ONNX session shared globally, per-call state (LSTM, buffers, state machine).
"""

import time
import math
from collections import deque
from typing import Any, Dict, Optional

import numpy as np
import onnxruntime as ort

from hush.core.ops.base import BaseOp
from hush.core.utils.common import Param


# ── Global ONNX session (shared across all calls) ──
_vad_session: Optional[ort.InferenceSession] = None


def _get_vad_session(model_path: str = "models/silero_vad.onnx") -> ort.InferenceSession:
    global _vad_session
    if _vad_session is None:
        _vad_session = ort.InferenceSession(
            model_path,
            providers=["CPUExecutionProvider"],
            sess_options=_make_session_options(),
        )
    return _vad_session


def _make_session_options() -> ort.SessionOptions:
    opts = ort.SessionOptions()
    opts.inter_op_num_threads = 1
    opts.intra_op_num_threads = 1
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return opts


class OnlineGaussianModel:
    """Online background energy model — EWMA mean/variance tracker.

    Used for foreground detection: is this chunk louder than background?
    """

    def __init__(self, chunk_size: int = 512, rate: int = 16000, window_sec: float = 5.0):
        self.alpha = chunk_size / (rate * window_sec)  # ~0.0064
        self.mean = 0.0
        self.var = 1e-6
        self.n = 0

    def update(self, energy: float):
        self.n += 1
        if self.n == 1:
            self.mean = energy
            self.var = energy * energy * 0.1
            return
        delta = energy - self.mean
        self.mean += self.alpha * delta
        self.var = (1 - self.alpha) * self.var + self.alpha * delta * delta

    def p_background(self, energy: float) -> float:
        """Probability that energy belongs to background distribution."""
        if self.var < 1e-10:
            return 0.5
        std = math.sqrt(self.var)
        z = (energy - self.mean) / max(std, 1e-8)
        # Sigmoid approximation: high z → low p_background
        return 1.0 / (1.0 + math.exp(min(z, 20)))


class VadDetector(BaseOp):
    """Streaming VAD: ONNX Silero inference + foreground detection + speech segmentation.

    Yields speech segments when speech end confirmed. PENDING between.

    Config:
        model_path: path to silero_vad.onnx
        rate: 16000
        chunk_size: 512 (32ms)
        trigger_threshold: 0.5
        neg_threshold: 0.35
        min_speech_duration_ms: 60
        max_speech_duration_ms: 45000
        min_silence_duration_ms: 500
        speech_pad_start_ms: 500
        speech_pad_end_ms: 300
    """

    type = "code"

    def __init__(
        self,
        model_path: str = "models/silero_vad.onnx",
        rate: int = 16000,
        chunk_size: int = 512,
        trigger_threshold: float = 0.5,
        neg_threshold: float = 0.35,
        min_speech_duration_ms: float = 60,
        max_speech_duration_ms: float = 45000,
        min_silence_duration_ms: float = 500,
        speech_pad_start_ms: float = 500,
        speech_pad_end_ms: float = 300,
        inputs: Optional[Dict[str, Any]] = None,
        outputs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # I/O
        parsed_inputs = {
            "audio": Param(required=True),
            "cmc_time": Param(type=int, required=True),
            "recv_time": Param(type=int, required=True),
        }
        parsed_outputs = {
            "speech_audio": Param(),
            "speech_duration_ms": Param(),
            "num_chunks": Param(),
            "cmc_start_time": Param(),
            "vad_time": Param(),
        }
        self.inputs = self._merge_params(parsed_inputs, self._normalize_params(inputs))
        self.outputs = self._merge_params(parsed_outputs, self._normalize_params(outputs))

        # Config
        self.rate = rate
        self.chunk_size = chunk_size
        self.trigger_threshold = trigger_threshold
        self.neg_threshold = neg_threshold
        self.min_speech_samples = int(rate * min_speech_duration_ms / 1000)
        self.max_speech_samples = int(rate * max_speech_duration_ms / 1000)
        self.min_silence_samples = int(rate * min_silence_duration_ms / 1000)
        self.pad_start_chunks = int(speech_pad_start_ms / (chunk_size / rate * 1000))
        self.pad_end_chunks = int(speech_pad_end_ms / (chunk_size / rate * 1000))

        # ONNX session (shared)
        self._session = _get_vad_session(model_path)

        # ── Per-call state ──
        self._vad_state = np.zeros((2, 1, 128), dtype=np.float32)
        self._vad_context = np.zeros((1, 64), dtype=np.float32)
        self._sr = np.array(rate, dtype=np.int64)

        # State machine
        self._triggered = False
        self._temp_end = 0
        self._current_sample = 0
        self._possible_ends = []

        # Buffers
        self._speech_buffer = deque()   # [(chunk, prob, cmc_time), ...]
        self._temp_buffer = deque(maxlen=self.pad_start_chunks)

        # Background model
        self._bg_model = OnlineGaussianModel(chunk_size, rate)
        self._start_time = time.time()

        self.core = self._process

    async def _process(self, audio: np.ndarray, cmc_time: int, recv_time: int):
        """Process one 512-sample chunk. Yield speech segment when end detected."""

        # ── ONNX Inference ──
        input_with_ctx = np.concatenate(
            [self._vad_context, audio.reshape(1, -1)], axis=1
        )  # (1, 576)

        ort_outs = self._session.run(
            ["output", "stateN"],
            {
                "input": input_with_ctx.astype(np.float32),
                "state": self._vad_state,
                "sr": self._sr,
            },
        )
        speech_prob = float(ort_outs[0][0][0])
        self._vad_state = ort_outs[1]
        self._vad_context = input_with_ctx[:, -64:]

        # ── Foreground detection ──
        energy = float(np.sqrt(np.mean(audio ** 2)))
        self._bg_model.update(energy)
        elapsed = time.time() - self._start_time
        if elapsed < 5.0:
            is_foreground = True
        else:
            p_bg = self._bg_model.p_background(energy)
            is_foreground = p_bg < 0.2

        # ── State machine ──
        self._current_sample += self.chunk_size

        if not self._triggered:
            # IDLE state
            self._temp_buffer.append((audio, speech_prob, cmc_time))

            if speech_prob >= self.trigger_threshold and is_foreground:
                # IDLE → SPEECH
                self._triggered = True
                self._temp_end = 0
                self._possible_ends = []

                # Prepend context
                for item in self._temp_buffer:
                    self._speech_buffer.append(item)
                self._temp_buffer.clear()

        else:
            # SPEECH state — accumulate
            self._speech_buffer.append((audio, speech_prob, cmc_time))

            # Check for speech end
            if speech_prob < self.neg_threshold:
                if self._temp_end == 0:
                    self._temp_end = self._current_sample

                silence_samples = self._current_sample - self._temp_end

                if silence_samples >= self.min_silence_samples:
                    # Confirmed end
                    segment = self._finalize_segment()
                    if segment is not None:
                        yield segment
                    return
            else:
                if self._temp_end > 0:
                    # Speech resumed — record silence point
                    silence_ms = (self._current_sample - self._temp_end) / self.rate * 1000
                    self._possible_ends.append((self._temp_end, silence_ms))
                    self._temp_end = 0

            # Max duration check
            speech_samples = len(self._speech_buffer) * self.chunk_size
            if speech_samples >= self.max_speech_samples:
                segment = self._force_end_segment()
                if segment is not None:
                    yield segment

    def _finalize_segment(self) -> Optional[Dict[str, Any]]:
        """Finalize speech segment: trim, validate, reset state."""
        if not self._speech_buffer:
            self._reset_state()
            return None

        # Trim leading/trailing non-speech
        trimmed = self._trim_buffer(self._speech_buffer)

        # Validate min duration
        total_samples = sum(len(item[0]) for item in trimmed)
        if total_samples < self.min_speech_samples:
            self._reset_state()
            return None

        # Add end padding
        # (padding chunks already in speech_buffer from accumulation)

        # Concatenate
        audio = np.concatenate([item[0] for item in trimmed])
        cmc_start = trimmed[0][2]
        duration_ms = len(audio) / self.rate * 1000

        result = {
            "speech_audio": audio,
            "speech_duration_ms": duration_ms,
            "num_chunks": len(trimmed),
            "cmc_start_time": cmc_start,
            "vad_time": int(time.time() * 1000),
        }

        self._reset_state()
        return result

    def _force_end_segment(self) -> Optional[Dict[str, Any]]:
        """Force end at best silence point when max duration exceeded."""
        best_end = None
        best_silence = 0

        for sample_idx, silence_ms in self._possible_ends:
            if silence_ms >= 98 and silence_ms > best_silence:
                best_silence = silence_ms
                best_end = sample_idx

        if best_end is not None:
            # Cut at best silence point
            cut_chunks = (best_end - (self._current_sample - len(self._speech_buffer) * self.chunk_size)) // self.chunk_size
            cut_chunks = max(1, min(cut_chunks, len(self._speech_buffer)))
            cut_buffer = deque(list(self._speech_buffer)[:cut_chunks])
            remaining = deque(list(self._speech_buffer)[cut_chunks:])

            # Finalize cut portion
            trimmed = self._trim_buffer(cut_buffer)
            if not trimmed:
                self._reset_state()
                return None

            audio = np.concatenate([item[0] for item in trimmed])
            cmc_start = trimmed[0][2]

            # Keep remaining for next segment
            self._speech_buffer = remaining
            self._possible_ends = []
            self._temp_end = 0

            return {
                "speech_audio": audio,
                "speech_duration_ms": len(audio) / self.rate * 1000,
                "num_chunks": len(trimmed),
                "cmc_start_time": cmc_start,
                "vad_time": int(time.time() * 1000),
            }
        else:
            # No good silence point — cut at max
            return self._finalize_segment()

    def _trim_buffer(self, buffer: deque) -> list:
        """Trim leading/trailing non-speech chunks (prob < 0.25)."""
        items = list(buffer)
        if not items:
            return items

        # Find first speech chunk
        start = 0
        for i, (_, prob, _) in enumerate(items):
            if prob >= 0.25:
                start = max(0, i - 3)  # Keep 2-3 frames before
                break

        # Find last speech chunk
        end = len(items)
        for i in range(len(items) - 1, -1, -1):
            if items[i][1] >= 0.25:
                end = min(len(items), i + 5)  # Keep 3-5 frames after
                break

        return items[start:end]

    def _reset_state(self):
        """Reset state machine for next segment. Keep ONNX/background state."""
        self._triggered = False
        self._temp_end = 0
        self._possible_ends = []
        self._speech_buffer.clear()
        self._temp_buffer.clear()
