"""Mock audio source — reads WAV file and yields chunks like telco."""

import asyncio
import time

import numpy as np
from scipy.io import wavfile

from hush.core.ops import op

CHUNK_SIZE = 320  # 40ms at 8kHz


@op
async def wav_source(wav_path: str):
    """Read WAV file, yield 320-sample chunks + silence tail like telco stream."""
    sr, audio = wavfile.read(wav_path)
    # Append 1s silence to trigger VAD end detection
    silence = np.zeros(int(1.0 * sr), dtype=audio.dtype)
    audio = np.concatenate([audio, silence])

    for i in range(0, len(audio), CHUNK_SIZE):
        chunk = audio[i : i + CHUNK_SIZE]
        if len(chunk) < CHUNK_SIZE:
            chunk = np.pad(chunk, (0, CHUNK_SIZE - len(chunk)))
        yield {"raw_chunk": chunk.tobytes(), "cmc_time": int(time.time() * 1000)}
        await asyncio.sleep(0)


@op
async def multi_wav_source(wav_paths: list):
    """Read multiple WAV files sequentially, simulating multi-turn conversation.

    Each WAV = 1 user utterance. Silence between WAVs simulates pause.
    """
    for wav_path in wav_paths:
        sr, audio = wavfile.read(wav_path)
        # Append 1s silence after each utterance
        silence = np.zeros(int(1.0 * sr), dtype=audio.dtype)
        audio = np.concatenate([audio, silence])

        for i in range(0, len(audio), CHUNK_SIZE):
            chunk = audio[i : i + CHUNK_SIZE]
            if len(chunk) < CHUNK_SIZE:
                chunk = np.pad(chunk, (0, CHUNK_SIZE - len(chunk)))
            yield {"raw_chunk": chunk.tobytes(), "cmc_time": int(time.time() * 1000)}
            await asyncio.sleep(0)
