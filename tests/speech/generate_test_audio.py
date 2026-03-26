"""Generate test WAV files using Hush TTS pipeline.

Generates 8kHz int16 mono WAV files simulating telco input for speech pipeline testing.
Output: tests/speech/audio/*.wav
"""

import asyncio
import os
import sys

import numpy as np
import soxr
from scipy.io import wavfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "../../.env"))

from hush.core import Hush
from speech.tts_synthesizer import tts_pipeline

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "audio")
TTS_RATE = 22050
TELCO_RATE = 8000

# ── Test cases matching educa workflow ──
TEST_CASES = [
    ("01_student_joining.wav", "bé đang vào rồi em", "REMINDER: student joining"),
    ("02_silence.wav", None, "REMINDER: silence"),
    ("03_confirm.wav", "vâng đúng rồi", "CONFIRM_CUSTOMER: confirm"),
    ("04_busy.wav", "anh bận lắm gọi lại sau đi", "REMINDER: busy"),
    ("05_technical_issue.wav", "con vào mãi không được em ạ", "REMINDER: technical issue"),
    ("06_read_phone.wav", "số không chín tám bảy sáu năm bốn ba hai một em nhé", "ASK_PHONE: read phone"),
    ("07_fallback.wav", "trời mưa quá", "REMINDER: fallback"),
    ("08_deny.wav", "không nhà tôi không có ai học", "CONFIRM_CUSTOMER: deny"),
    ("09_wrong_number.wav", "bạn gọi nhầm số rồi", "CONFIRM_CUSTOMER: wrong number"),
    ("10_unclear.wav", "ừm ờ để xem nào", "CONFIRM_CUSTOMER: unclear"),
]


async def tts_generate(text: str) -> np.ndarray:
    """Generate audio via Hush TTS pipeline."""
    wf = tts_pipeline(text=text)
    engine = Hush(wf)
    result = await engine.run(inputs={})
    audio = result.get("audio")
    if audio is None or len(audio) == 0:
        return None
    # Convert int16 → float32 for resampling
    if audio.dtype == np.int16:
        return audio.astype(np.float32) / 32768.0
    return audio.astype(np.float32)


def downsample_to_telco(audio: np.ndarray) -> np.ndarray:
    """Downsample TTS output (22050Hz) → telco rate (8000Hz)."""
    return soxr.resample(audio, TTS_RATE, TELCO_RATE, quality="VHQ")


def save_wav(filepath: str, audio: np.ndarray, rate: int = TELCO_RATE):
    """Save float32 audio as int16 WAV."""
    if np.max(np.abs(audio)) > 0:
        audio = audio / np.max(np.abs(audio)) * 0.9
    wavfile.write(filepath, rate, (audio * 32768).astype(np.int16))


async def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Generating test audio → {OUTPUT_DIR}\n")

    for filename, text, desc in TEST_CASES:
        filepath = os.path.join(OUTPUT_DIR, filename)
        print(f"  [{filename}] {desc}")

        if text is None:
            silence = np.zeros(int(3.0 * TELCO_RATE), dtype=np.float32)
            save_wav(filepath, silence)
            print(f"    → silence, 3.0s")
            continue

        print(f"    Text: \"{text}\"")
        audio = await tts_generate(text)

        if audio is None:
            print(f"    → SKIPPED (TTS failed)")
            continue

        audio_8k = downsample_to_telco(audio)
        save_wav(filepath, audio_8k)
        duration_ms = len(audio_8k) / TELCO_RATE * 1000
        print(f"    → {duration_ms:.0f}ms")

    print(f"\nDone!")


if __name__ == "__main__":
    asyncio.run(main())
