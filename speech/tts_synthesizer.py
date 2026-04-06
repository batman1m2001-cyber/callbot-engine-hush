"""TTS Pipeline — text → phonemes → fastspeech2 (Triton) → hifigan (Triton) → audio.

@graph with @op preprocess + 2x TritonOp. Stateless.
"""

import os
import re
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from hush.core import graph, START, END, PARENT
from hush.core.ops import op
from hush.providers.ops import TritonOp
from speech.tts import vietnamese_phonemes as viphonemes
from speech.tts import text_to_sequence, clean_vietnamese_text

LOGGER = logging.getLogger(__name__)

# ── Lexicon (loaded once) ──
_LEXICON: Optional[Dict[str, List[str]]] = None
_LEXICON_PATH = str(Path(__file__).parent / "tts" / "vi-new-lexicon.txt")


def _load_lexicon(path: str = _LEXICON_PATH) -> Dict[str, List[str]]:
    global _LEXICON
    if _LEXICON is not None:
        return _LEXICON
    _LEXICON = {}
    if not os.path.exists(path):
        LOGGER.warning("Lexicon not found: %s", path)
        return _LEXICON
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) == 2:
                word, phones = parts
                _LEXICON[word.lower()] = phones.split()
    LOGGER.info("Loaded lexicon: %d entries", len(_LEXICON))
    return _LEXICON


@op
def text_to_phonemes(text: str) -> dict:
    """Vietnamese text → phoneme token IDs + fastspeech2 inputs.

    Returns all 6 inputs needed for fastspeech2 Triton model.
    """
    from speech.tts import vietnamese_phonemes as viphonemes
    from speech.tts import text_to_sequence, clean_vietnamese_text

    if not text or not text.strip():
        return {
            "texts": np.zeros((1, 1), dtype=np.int64),
            "src_lens": np.array([1], dtype=np.int64),
            "max_src_len": np.array([1], dtype=np.int64),
            "p_control": np.array([1.0], dtype=np.float32),
            "e_control": np.array([1.0], dtype=np.float32),
            "d_control": np.array([1.1], dtype=np.float32),
            "is_empty": True,
        }

    # Clean punctuation
    text = re.sub(r"[,;.?\-!:]", " ", text)
    text = clean_vietnamese_text(text)

    # Phoneme conversion
    lexicon = _load_lexicon()
    phones: List[str] = []
    words = re.split(r"([,;.\-?\!\s+])", text)
    for w in words:
        if not w:
            continue
        if w.lower() in lexicon:
            phones += lexicon[w.lower()]
        elif w.strip():
            phones += viphonemes.parse_word(w)

    phones_str = "{" + " ".join(phones) + "}"
    sequence = np.array(text_to_sequence(phones_str, ["vietnamese_cleaners"]), dtype=np.int64)

    T = int(sequence.size)

    return {
        "texts": sequence.reshape(1, T),
        "src_lens": np.array([T], dtype=np.int64),
        "max_src_len": np.array([T], dtype=np.int64),
        "p_control": np.array([1.0], dtype=np.float32),
        "e_control": np.array([1.0], dtype=np.float32),
        "d_control": np.array([1.1], dtype=np.float32),
        "is_empty": False,
    }


@op
def postprocess_mel(postnet_output: np.ndarray, mel_lens: np.ndarray) -> dict:
    """Transpose fastspeech2 output mel and compute audio lengths.

    fastspeech2 output: [B, Tm, n_mels] → hifigan needs [B, n_mels, Tm]
    """
    mel_postnet = np.transpose(postnet_output, (0, 2, 1)).astype(np.float32)
    hop_length = 256
    lengths_samples = (mel_lens * hop_length).astype(np.int64)

    return {
        "mels": mel_postnet,
        "lengths_samples": lengths_samples,
    }


@op
def postprocess_audio(audio_raw: np.ndarray, lengths_samples: np.ndarray) -> dict:
    """Trim and convert hifigan output to int16."""
    # Handle shape: [B, 1, L] or [B, L]
    if audio_raw.ndim == 3 and audio_raw.shape[1] == 1:
        audio_raw = audio_raw[:, 0, :]

    # Convert to int16
    if audio_raw.dtype != np.int16:
        audio = (audio_raw * 32768).clip(-32768, 32767).astype(np.int16)
    else:
        audio = audio_raw

    # Trim to actual length
    if lengths_samples is not None and len(lengths_samples) > 0:
        audio = audio[0][:int(lengths_samples[0])]
    else:
        audio = audio[0]

    audio_duration_ms = len(audio) / 22050 * 1000

    return {
        "audio": audio,
        "audio_duration_ms": audio_duration_ms,
    }


@graph
def tts_pipeline(text):
    """Full TTS: text → phonemes → fastspeech2 → hifigan → audio.

    Input:
        text: str — Vietnamese text to synthesize

    Output:
        audio: np.ndarray int16 — waveform at 22050Hz
        audio_duration_ms: float — duration in ms
    """
    # Op 1: Text → phoneme tokens + fastspeech2 params
    phonemes = text_to_phonemes(text=text)

    # Op 2: FastSpeech2 — tokens → mel spectrogram
    fs2 = TritonOp(
        resource="tts-fastspeech2",
        inputs={
            "texts": phonemes["texts"],
            "src_lens": phonemes["src_lens"],
            "max_src_len": phonemes["max_src_len"],
            "p_control": phonemes["p_control"],
            "e_control": phonemes["e_control"],
            "d_control": phonemes["d_control"],
        },
    )

    # Op 3: Transpose mel + compute lengths
    mel = postprocess_mel(
        postnet_output=fs2["postnet_output"],
        mel_lens=fs2["mel_lens"],
    )

    # Op 4: HiFi-GAN — mel → waveform
    hifigan = TritonOp(
        resource="tts-hifigan",
        inputs={"mels": mel["mels"]},
    )

    # Op 5: Trim + convert to int16
    final = postprocess_audio(
        audio_raw=hifigan["audio_raw"],
        lengths_samples=mel["lengths_samples"],
    )

    START >> phonemes >> fs2 >> mel >> hifigan >> final >> END
