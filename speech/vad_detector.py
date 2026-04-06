"""VadDetector v3 — v1 logic, per-call isolation via request_id + ContextVar.

Fixes v1's shared instance state bug under concurrent calls:
  v1: self._vad_state, self._speech_buffer etc. are instance vars →
      two concurrent engine.run() calls share the same VadDetector instance →
      call B's ONNX state overwrites call A's between chunks.

  v3: per-call VAD buffers stored in class-level dict keyed by state.request_id.
      ContextVar propagates the correct buffer dict into _process without changing
      the method signature or the BaseOp input resolution pipeline.

Cleanup: call VadDetector.cleanup(request_id) when the call ends.
"""

import contextvars
import math
import time
from collections import deque
from typing import Any, Dict, Optional

import numpy as np
import onnxruntime as ort

from hush.core.ops.base import BaseOp
from hush.core.utils.common import Param


# ── Global ONNX session (shared, stateless) ──
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
    """Online background energy model — EWMA mean/variance tracker."""

    def __init__(self, chunk_size: int = 512, rate: int = 16000, window_sec: float = 5.0):
        self.alpha = chunk_size / (rate * window_sec)
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
        if self.var < 1e-10:
            return 0.5
        std = math.sqrt(self.var)
        z = (energy - self.mean) / max(std, 1e-8)
        return 1.0 / (1.0 + math.exp(min(z, 20)))


# ContextVar: holds the active per-call buffer dict during _process execution
_active_vad_buffers: contextvars.ContextVar[dict] = contextvars.ContextVar("active_vad_buffers")


class VadDetector(BaseOp):
    """Streaming VAD: ONNX Silero + foreground detection + speech segmentation.

    Per-call state isolated via class-level _vad_buffers dict keyed by request_id.
    """

    type = "code"

    # Class-level: request_id → per-call VAD buffer dict
    _vad_buffers: Dict[str, dict] = {}

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
        silence_timeout: float = 0.0,
        inputs: Optional[Dict[str, Any]] = None,
        outputs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.silence_timeout = silence_timeout  # 0 = disabled

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

        # Config — no per-call state here
        self.rate = rate
        self.chunk_size = chunk_size
        self.trigger_threshold = trigger_threshold
        self.neg_threshold = neg_threshold
        self.min_speech_samples = int(rate * min_speech_duration_ms / 1000)
        self.max_speech_samples = int(rate * max_speech_duration_ms / 1000)
        self.min_silence_samples = int(rate * min_silence_duration_ms / 1000)
        self.pad_start_chunks = int(speech_pad_start_ms / (chunk_size / rate * 1000))
        self.pad_end_chunks = int(speech_pad_end_ms / (chunk_size / rate * 1000))
        self._sr = np.array(rate, dtype=np.int64)

        self._session = _get_vad_session(model_path)
        self._set_core(self._process)

    # ── Per-call buffer init ──

    def _init_vad_buffers(self) -> dict:
        """Create a fresh set of VAD buffers for one phone call."""
        return {
            "vad_state": np.zeros((2, 1, 128), dtype=np.float32),
            "vad_context": np.zeros((1, 64), dtype=np.float32),
            "triggered": False,
            "temp_end": 0,
            "current_sample": 0,
            "speech_buffer": deque(),
            "temp_buffer": deque(maxlen=self.pad_start_chunks),
            "possible_ends": [],
            "bg_model": OnlineGaussianModel(self.chunk_size, self.rate),
            "start_time": time.time(),
            "last_speech_time": time.time(),
        }

    # ── run() override: bind per-call buffers via ContextVar ──

    async def run(self, state, context_id=None):
        rid = state.request_id
        if rid not in VadDetector._vad_buffers:
            VadDetector._vad_buffers[rid] = self._init_vad_buffers()

        token = _active_vad_buffers.set(VadDetector._vad_buffers[rid])
        try:
            async for item in super().run(state, context_id):
                yield item
        finally:
            _active_vad_buffers.reset(token)

    @classmethod
    def cleanup(cls, request_id: str) -> None:
        """Release per-call buffers. Call this when the phone call ends."""
        cls._vad_buffers.pop(request_id, None)

    # ── Core processing — identical to v1, reads from ContextVar ──

    async def _process(self, audio: np.ndarray, cmc_time: int, recv_time: int):
        b = _active_vad_buffers.get()

        # ── ONNX Inference ──
        input_with_ctx = np.concatenate(
            [b["vad_context"], audio.reshape(1, -1)], axis=1
        )
        ort_outs = self._session.run(
            ["output", "stateN"],
            {
                "input": input_with_ctx.astype(np.float32),
                "state": b["vad_state"],
                "sr": self._sr,
            },
        )
        speech_prob = float(ort_outs[0][0][0])
        b["vad_state"] = ort_outs[1]
        b["vad_context"] = input_with_ctx[:, -64:]

        # ── Foreground detection ──
        energy = float(np.sqrt(np.mean(audio ** 2)))
        b["bg_model"].update(energy)
        elapsed = time.time() - b["start_time"]
        if elapsed < 5.0:
            is_foreground = True
        else:
            p_bg = b["bg_model"].p_background(energy)
            is_foreground = p_bg < 0.2

        # ── State machine ──
        b["current_sample"] += self.chunk_size

        if not b["triggered"]:
            b["temp_buffer"].append((audio, speech_prob, cmc_time))

            if speech_prob >= self.trigger_threshold and is_foreground:
                b["triggered"] = True
                b["temp_end"] = 0
                b["possible_ends"] = []
                for item in b["temp_buffer"]:
                    b["speech_buffer"].append(item)
                b["temp_buffer"].clear()
                b["last_speech_time"] = time.time()
            elif self.silence_timeout > 0:
                # No speech — check if silence_timeout exceeded
                elapsed = time.time() - b["last_speech_time"]
                if elapsed >= self.silence_timeout:
                    b["last_speech_time"] = time.time()  # reset timer
                    yield {
                        "speech_audio": np.zeros(160, dtype=np.float32),
                        "speech_duration_ms": 0.0,
                        "num_chunks": 0,
                        "cmc_start_time": cmc_time,
                        "vad_time": int(time.time() * 1000),
                    }
        else:
            b["speech_buffer"].append((audio, speech_prob, cmc_time))

            if speech_prob < self.neg_threshold:
                if b["temp_end"] == 0:
                    b["temp_end"] = b["current_sample"]

                silence_samples = b["current_sample"] - b["temp_end"]
                if silence_samples >= self.min_silence_samples:
                    segment = self._finalize_segment(b)
                    if segment is not None:
                        b["last_speech_time"] = time.time()
                        yield segment
                    return
            else:
                if b["temp_end"] > 0:
                    silence_ms = (b["current_sample"] - b["temp_end"]) / self.rate * 1000
                    b["possible_ends"].append((b["temp_end"], silence_ms))
                    b["temp_end"] = 0

            speech_samples = len(b["speech_buffer"]) * self.chunk_size
            if speech_samples >= self.max_speech_samples:
                segment = self._force_end_segment(b)
                if segment is not None:
                    yield segment

    def _finalize_segment(self, b: dict) -> Optional[Dict[str, Any]]:
        if not b["speech_buffer"]:
            self._reset_buffers(b)
            return None

        trimmed = self._trim_buffer(b["speech_buffer"])
        total_samples = sum(len(item[0]) for item in trimmed)
        if total_samples < self.min_speech_samples:
            self._reset_buffers(b)
            return None

        audio = np.concatenate([item[0] for item in trimmed])
        cmc_start = trimmed[0][2]

        result = {
            "speech_audio": audio,
            "speech_duration_ms": len(audio) / self.rate * 1000,
            "num_chunks": len(trimmed),
            "cmc_start_time": cmc_start,
            "vad_time": int(time.time() * 1000),
        }
        self._reset_buffers(b)
        return result

    def _force_end_segment(self, b: dict) -> Optional[Dict[str, Any]]:
        best_end = None
        best_silence = 0

        for sample_idx, silence_ms in b["possible_ends"]:
            if silence_ms >= 98 and silence_ms > best_silence:
                best_silence = silence_ms
                best_end = sample_idx

        if best_end is not None:
            cut_chunks = (best_end - (b["current_sample"] - len(b["speech_buffer"]) * self.chunk_size)) // self.chunk_size
            cut_chunks = max(1, min(cut_chunks, len(b["speech_buffer"])))
            cut_buffer = deque(list(b["speech_buffer"])[:cut_chunks])
            remaining = deque(list(b["speech_buffer"])[cut_chunks:])

            trimmed = self._trim_buffer(cut_buffer)
            if not trimmed:
                self._reset_buffers(b)
                return None

            audio = np.concatenate([item[0] for item in trimmed])
            cmc_start = trimmed[0][2]

            b["speech_buffer"] = remaining
            b["possible_ends"] = []
            b["temp_end"] = 0

            return {
                "speech_audio": audio,
                "speech_duration_ms": len(audio) / self.rate * 1000,
                "num_chunks": len(trimmed),
                "cmc_start_time": cmc_start,
                "vad_time": int(time.time() * 1000),
            }
        else:
            return self._finalize_segment(b)

    def _trim_buffer(self, buffer: deque) -> list:
        items = list(buffer)
        if not items:
            return items

        start = 0
        for i, (_, prob, _) in enumerate(items):
            if prob >= 0.25:
                start = max(0, i - 3)
                break

        end = len(items)
        for i in range(len(items) - 1, -1, -1):
            if items[i][1] >= 0.25:
                end = min(len(items), i + 5)
                break

        return items[start:end]

    def _reset_buffers(self, b: dict) -> None:
        """Reset state machine fields. Keeps ONNX state + background model."""
        b["triggered"] = False
        b["temp_end"] = 0
        b["possible_ends"] = []
        b["speech_buffer"].clear()
        b["temp_buffer"].clear()
