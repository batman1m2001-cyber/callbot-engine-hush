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
        max_history: int = 24,
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
        self.max_history = max_history
        self.pad_length = pad_length

        # ONNX session (shared)
        self._session = _get_denoise_session(model_path)

        # Per-call state
        self._embedding_history = []  # list of conv outputs (1, 256) each

        self.core = self._process

    async def _process(
        self,
        transcript: str,
        embedding: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Classify segment as speech or noise.

        If no embedding provided, pass through transcript unchanged.
        If embedding provided, run denoise model:
            - prob >= threshold → speech → keep transcript
            - prob < threshold → noise → suppress transcript (empty string)
        """
        if embedding is None or len(transcript.strip()) == 0:
            return {
                "transcript": transcript,
                "is_speech": True,
                "speech_prob": 1.0,
            }

        speech_prob = self._classify(embedding)
        is_speech = speech_prob >= self.threshold

        return {
            "transcript": transcript if is_speech else "",
            "is_speech": is_speech,
            "speech_prob": speech_prob,
        }

    def _classify(self, embedding: np.ndarray) -> float:
        """Run ConvTransformer denoise inference with streaming state.

        ONNX model expected inputs:
            - embedding: (1, 1024, T) float32
            - history_len: (1,) int64 — number of past embeddings
            - past_embeddings: (1, max_history, 256) float32 — padded history

        ONNX model expected outputs:
            - speech_prob: (1,) float32
            - conv_output: (1, 256) float32 — to append to history
        """
        # Pad/truncate time dimension to pad_length
        if embedding.ndim == 2:
            embedding = embedding[np.newaxis, ...]  # (1, 1024, T)

        T = embedding.shape[2]
        if T < self.pad_length:
            padded = np.zeros((1, embedding.shape[1], self.pad_length), dtype=np.float32)
            padded[:, :, :T] = embedding
            embedding = padded
        elif T > self.pad_length:
            embedding = embedding[:, :, :self.pad_length]

        embedding = embedding.astype(np.float32)

        # Build history tensor
        history_len = len(self._embedding_history)
        past_emb = np.zeros((1, self.max_history, 256), dtype=np.float32)
        for i, h in enumerate(self._embedding_history[-self.max_history:]):
            past_emb[0, i, :] = h.flatten()[:256]

        try:
            # Try full model with history
            ort_outs = self._session.run(
                None,
                {
                    "embedding": embedding,
                    "history_len": np.array([history_len], dtype=np.int64),
                    "past_embeddings": past_emb,
                },
            )
            speech_prob = float(ort_outs[0][0])

            # Update history with conv output
            if len(ort_outs) > 1:
                conv_out = ort_outs[1]
                self._embedding_history.append(conv_out)
                if len(self._embedding_history) > self.max_history:
                    self._embedding_history = self._embedding_history[-self.max_history:]

        except Exception as e:
            # Fallback: simple model without history (just embedding → prob)
            LOGGER.warning("Denoise full model failed, trying simple: %s", e)
            try:
                ort_outs = self._session.run(None, {"embedding": embedding})
                speech_prob = float(ort_outs[0][0])
            except Exception as e2:
                LOGGER.error("Denoise inference failed: %s", e2)
                speech_prob = 1.0  # Default to speech on failure

        return speech_prob
