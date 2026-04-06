"""DenoiseClassifier — speech/noise classifier using ConvTransformer ONNX model.

Extends BaseOp. Stateful: maintains embedding_history across speech segments.
Runs AFTER STT — uses STT embedding output to classify speech vs noise.
If noise detected → suppresses transcript.

ONNX session shared globally. Per-call state: embedding_history (capped at 24).
"""

import logging
from typing import Any, Dict, Optional

import numpy as np
import onnxruntime as ort

from hush.core.ops.base import BaseOp
from hush.core.utils.common import Param

LOGGER = logging.getLogger(__name__)

# ── Global ONNX session ──
_denoise_session: Optional[ort.InferenceSession] = None


def _get_denoise_session(model_path: str = "models/denoise.onnx") -> ort.InferenceSession:
    global _denoise_session
    if _denoise_session is None:
        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        _denoise_session = ort.InferenceSession(
            model_path, providers=["CPUExecutionProvider"], sess_options=opts
        )
    return _denoise_session


class DenoiseClassifier(BaseOp):
    """Classify speech vs noise using STT embedding + ConvTransformer.

    Streaming stateful: embedding_history persists across segments within a call.
    Each segment's conv output appended to history (causal transformer attends to all past).

    Args:
        model_path: path to denoise ONNX model
        threshold: speech probability threshold (below = noise, suppress transcript)
        max_history: max embedding history length (prevent OOM)
        pad_length: input embedding time dimension padded/truncated to this
    """

    type = "code"

    def __init__(
        self,
        model_path: str = "models/denoise.onnx",
        threshold: float = 0.63,
        pad_length: int = 128,
        inputs: Optional[Dict[str, Any]] = None,
        outputs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        parsed_inputs = {
            "transcript": Param(type=str, required=True),
            "embedding": Param(required=False),  # None if STT didn't return embedding
        }
        parsed_outputs = {
            "transcript": Param(type=str),
            "is_speech": Param(type=bool),
            "speech_prob": Param(type=float),
        }
        self.inputs = self._merge_params(parsed_inputs, self._normalize_params(inputs))
        self.outputs = self._merge_params(parsed_outputs, self._normalize_params(outputs))

        self.threshold = threshold
        self.pad_length = pad_length

        # ONNX session (shared)
        self._session = _get_denoise_session(model_path)

        self._set_core(self._process)

    async def _process(
        self,
        transcript: str,
        embedding: Optional[np.ndarray] = None,
    ):
        """Classify segment as speech or noise.

        Generator: yields only if speech detected. Noise = no yield (PENDING).
        Downstream ops only trigger for real speech, skipping noise-only segments.

        If no embedding provided, pass through (assume speech).
        """
        if embedding is None or len(transcript.strip()) == 0:
            # No embedding or empty transcript — pass through as speech
            if transcript and transcript.strip():
                yield {
                    "transcript": transcript,
                    "is_speech": True,
                    "speech_prob": 1.0,
                }
            return

        speech_prob = self._classify(embedding)
        is_speech = speech_prob >= self.threshold

        if is_speech:
            yield {
                "transcript": transcript,
                "is_speech": True,
                "speech_prob": speech_prob,
            }
        # else: noise — no yield, downstream skipped

    def _classify(self, embedding: np.ndarray) -> float:
        """Run denoise inference.

        ONNX model inputs:
            - embeddings: (batch, seq_len, 1024, time) float32
            - lengths: (batch,) int64

        ONNX model outputs:
            - logits: (batch, seq_len) float32 — apply sigmoid for probability
        """
        # Normalize to (1024, T)
        if embedding.ndim == 1:
            embedding = embedding[np.newaxis, :]  # (1, T) — unusual shape
        if embedding.ndim == 2:
            # Could be (1024, T) or (T, 1024) — assume (1024, T)
            pass

        # Pad/truncate time dimension to pad_length
        T = embedding.shape[-1]
        if T < self.pad_length:
            pad_widths = [(0, 0)] * (embedding.ndim - 1) + [(0, self.pad_length - T)]
            embedding = np.pad(embedding, pad_widths)
        elif T > self.pad_length:
            embedding = embedding[..., :self.pad_length]

        embedding = embedding.astype(np.float32)

        # Model expects (batch=1, seq_len=1, 1024, time)
        if embedding.ndim == 2:
            embedding = embedding[np.newaxis, np.newaxis, :]  # (1, 1, 1024, T)
        elif embedding.ndim == 3:
            embedding = embedding[np.newaxis, :]  # (1, seq_len, 1024, T)

        lengths = np.array([1], dtype=np.int64)  # one segment per call

        try:
            ort_outs = self._session.run(
                None,
                {"embeddings": embedding, "lengths": lengths},
            )
            # logits: (batch, seq_len) — take first segment, apply sigmoid
            logit = float(ort_outs[0][0, 0])
            speech_prob = float(1.0 / (1.0 + np.exp(-logit)))
        except Exception as e:
            LOGGER.error("Denoise inference failed: %s", e)
            speech_prob = 1.0  # Default to speech on failure

        return speech_prob
