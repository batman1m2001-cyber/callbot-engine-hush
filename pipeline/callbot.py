"""Full callbot pipeline — single Hush graph.

Speech In → STT → LLM Workflow → TTS → Speech Out.
Shared state persists across turns via PARENT.shared().
"""

import asyncio
import time

import numpy as np
from scipy.io import wavfile

from hush.core import graph, START, END, PARENT
from hush.core.ops import op
from hush.providers.ops import TritonOp

from speech.audio_processor import AudioProcessor
from speech.vad_detector import VadDetector
from speech.denoise_classifier import DenoiseClassifier
from speech.tts_synthesizer import tts_pipeline, resample_for_telco

from agents.educa_reminder.workflow import educa_workflow

TELCO_CHUNK = 320  # 40ms at 8kHz


@op
def prepare_stt_input(speech_audio: np.ndarray) -> dict:
    """Reshape (N,) → (1, N) for FastConformer batch dimension."""
    return {"speech_audio": speech_audio.reshape(1, -1).astype(np.float32)}


@op
async def wav_source(wav_path: str):
    """Read WAV file, yield 320-sample chunks + silence tail like telco stream."""
    sr, audio = wavfile.read(wav_path)
    silence = np.zeros(int(1.0 * sr), dtype=audio.dtype)
    audio = np.concatenate([audio, silence])

    for i in range(0, len(audio), TELCO_CHUNK):
        chunk = audio[i: i + TELCO_CHUNK]
        if len(chunk) < TELCO_CHUNK:
            chunk = np.pad(chunk, (0, TELCO_CHUNK - len(chunk)))
        yield {"raw_chunk": chunk.tobytes(), "cmc_time": int(time.time() * 1000)}
        await asyncio.sleep(0)


@op
async def ws_source(audio_queue: asyncio.Queue):
    """Yield audio chunks from WebSocket queue. None sentinel = call ended."""
    while True:
        item = await audio_queue.get()
        if item is None:
            return
        yield {"raw_chunk": item["audio_bytes"], "cmc_time": item["cmc_time"]}


@op
def update_conversation(
    transcript: str,
    response: str,
    intent: str,
    new_state: str,
    conversation_history: list,
    intent_retry_counts: dict,
) -> dict:
    """Merge turn results back into shared state."""
    new_history = list(conversation_history)
    if transcript:
        new_history.append({"role": "user", "content": transcript})
    if response:
        new_history.append({"role": "assistant", "content": response})

    new_retry_counts = dict(intent_retry_counts)
    if intent:
        new_retry_counts[intent] = new_retry_counts.get(intent, 0) + 1

    return {
        "updated_state": new_state,
        "updated_history": new_history,
        "updated_retry_counts": new_retry_counts,
        "updated_response": response or "",
    }


@graph
def callbot_pipeline(wav_path, script_data):
    """Full callbot: audio → VAD → STT → LLM → TTS.

    Single graph, streaming. PARENT.shared() for multi-turn state.

    Input:
        wav_path: str — path to WAV file (mock telco audio)
        script_data: dict — student/class info for educa agent

    Output (per turn):
        response: str — agent response text
        audio: ndarray — TTS audio
        intent: str — detected intent
        new_state: str — state after this turn
    """
    PARENT.shared(
        current_state="REMINDER",
        conversation_history=[],
        intent_retry_counts={},
        last_agent_response="",
    )

    source = wav_source(wav_path=wav_path)

    audio = AudioProcessor(
        inputs={"raw_chunk": source["raw_chunk"], "cmc_time": source["cmc_time"]},
    )

    vad = VadDetector(
        inputs={
            "audio": audio["audio"],
            "cmc_time": audio["cmc_time"],
            "recv_time": audio["recv_time"],
        },
    )

    stt_input = prepare_stt_input(speech_audio=vad["speech_audio"])

    stt = TritonOp(
        resource="stt",
        inputs={"speech_audio": stt_input["speech_audio"]},
    )

    denoise = DenoiseClassifier(
        inputs={
            "transcript": stt["transcript"],
            "embedding": stt["embedding"],
        },
    )

    workflow = educa_workflow(
        customer_speech=denoise["transcript"],
        agent_speech=PARENT["last_agent_response"],
        current_state=PARENT["current_state"],
        script_data=script_data,
        intent_retry_counts=PARENT["intent_retry_counts"],
        conversation_history=PARENT["conversation_history"],
    )

    update = update_conversation(
        transcript=denoise["transcript"],
        response=workflow["response"],
        intent=workflow["intent"],
        new_state=workflow["new_state"],
        conversation_history=PARENT["conversation_history"],
        intent_retry_counts=PARENT["intent_retry_counts"],
    )
    update["updated_state"] >> PARENT["current_state"]
    update["updated_history"] >> PARENT["conversation_history"]
    update["updated_retry_counts"] >> PARENT["intent_retry_counts"]
    update["updated_response"] >> PARENT["last_agent_response"]

    tts = tts_pipeline(text=workflow["response"])

    workflow["response"] >> PARENT["response"]
    workflow["intent"] >> PARENT["intent"]
    workflow["new_state"] >> PARENT["new_state"]

    START >> source >> audio >> vad >> stt_input >> stt >> denoise >> workflow >> update >> tts >> END


@graph
def ws_callbot_pipeline(audio_queue, script_data):
    """WebSocket callbot: live audio → VAD → STT → LLM → TTS → 8kHz.

    Same as callbot_pipeline but reads from asyncio.Queue (WebSocket)
    instead of WAV file, and resamples TTS output to 8kHz for telco.
    """
    PARENT.shared(
        current_state="REMINDER",
        conversation_history=[],
        intent_retry_counts={},
        last_agent_response="",
    )

    source = ws_source(audio_queue=audio_queue)

    audio = AudioProcessor(
        inputs={"raw_chunk": source["raw_chunk"], "cmc_time": source["cmc_time"]},
    )

    vad = VadDetector(
        inputs={
            "audio": audio["audio"],
            "cmc_time": audio["cmc_time"],
            "recv_time": audio["recv_time"],
        },
    )

    stt_input = prepare_stt_input(speech_audio=vad["speech_audio"])

    stt = TritonOp(
        resource="stt",
        inputs={"speech_audio": stt_input["speech_audio"]},
    )

    denoise = DenoiseClassifier(
        inputs={
            "transcript": stt["transcript"],
            "embedding": stt["embedding"],
        },
    )

    educa_agent = educa_workflow(
        customer_speech=denoise["transcript"],
        agent_speech=PARENT["last_agent_response"],
        current_state=PARENT["current_state"],
        script_data=script_data,
        intent_retry_counts=PARENT["intent_retry_counts"],
        conversation_history=PARENT["conversation_history"],
    )

    update = update_conversation(
        transcript=denoise["transcript"],
        response=educa_agent["response"],
        intent=educa_agent["intent"],
        new_state=educa_agent["new_state"],
        conversation_history=PARENT["conversation_history"],
        intent_retry_counts=PARENT["intent_retry_counts"],
    )
    update["updated_state"] >> PARENT["current_state"]
    update["updated_history"] >> PARENT["conversation_history"]
    update["updated_retry_counts"] >> PARENT["intent_retry_counts"]
    update["updated_response"] >> PARENT["last_agent_response"]

    tts = tts_pipeline(text=educa_agent["response"])
    tts_8khz = resample_for_telco(audio=tts["audio"])

    START >> source >> audio >> vad >> stt_input >> stt >> denoise >> educa_agent >> update >> tts >> tts_8khz >> END
