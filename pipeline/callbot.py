"""Full callbot pipeline — single Hush graph.

Speech In → STT → LLM Workflow → TTS → Speech Out.
Shared state persists across turns via PARENT.shared().
"""

from hush.core import graph, START, END, PARENT
from hush.providers.ops import TritonOp

from speech.audio_processor import AudioProcessor
from speech.vad_detector import VadDetector
from speech.tts_synthesizer import tts_pipeline

from agents.educa_reminder.workflow import educa_workflow
from pipeline.mock_source import wav_source, multi_wav_source
from pipeline.conversation_manager import update_conversation


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
    # ── Shared state (persist across turns) ──
    PARENT.shared(
        current_state="REMINDER",
        conversation_history=[],
        intent_retry_counts={},
        last_agent_response="",
    )

    # ── Speech In (streaming) ──
    source = wav_source(wav_path=wav_path)

    audio = AudioProcessor(
        name="audio",
        inputs={"raw_chunk": source["raw_chunk"], "cmc_time": source["cmc_time"]},
    )

    vad = VadDetector(
        name="vad",
        inputs={
            "audio": audio["audio"],
            "cmc_time": audio["cmc_time"],
            "recv_time": audio["recv_time"],
        },
    )

    stt = TritonOp(
        name="stt",
        url="192.168.1.212:8001",
        model_name="fastconformer_asr",
        inputs_map={"AUDIO_SIGNAL": "speech_audio"},
        outputs_map={"TRANSCRIPT": "transcript"},
        inputs={"speech_audio": vad["speech_audio"]},
    )

    # ── Brain (per-turn, reads shared state) ──
    workflow = educa_workflow(
        customer_speech=stt["transcript"],
        agent_speech=PARENT["last_agent_response"],
        current_state=PARENT["current_state"],
        script_data=script_data,
        intent_retry_counts=PARENT["intent_retry_counts"],
        conversation_history=PARENT["conversation_history"],
    )

    # ── Update shared state ──
    update = update_conversation(
        transcript=stt["transcript"],
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

    # ── Speech Out (per-turn) ──
    tts = tts_pipeline(text=workflow["response"])

    # ── Forward outputs ──
    workflow["response"] >> PARENT["response"]
    workflow["intent"] >> PARENT["intent"]
    workflow["new_state"] >> PARENT["new_state"]

    # ── Wiring ──
    START >> source >> audio >> vad >> stt >> workflow >> update >> tts >> END


@graph
def callbot_pipeline_multi(wav_paths, script_data):
    """Multi-turn callbot with multiple WAV files.

    Same as callbot_pipeline but uses multi_wav_source.
    """
    PARENT.shared(
        current_state="REMINDER",
        conversation_history=[],
        intent_retry_counts={},
        last_agent_response="",
    )

    source = multi_wav_source(wav_paths=wav_paths)

    audio = AudioProcessor(
        name="audio",
        inputs={"raw_chunk": source["raw_chunk"], "cmc_time": source["cmc_time"]},
    )

    vad = VadDetector(
        name="vad",
        inputs={
            "audio": audio["audio"],
            "cmc_time": audio["cmc_time"],
            "recv_time": audio["recv_time"],
        },
    )

    stt = TritonOp(
        name="stt",
        url="192.168.1.212:8001",
        model_name="fastconformer_asr",
        inputs_map={"AUDIO_SIGNAL": "speech_audio"},
        outputs_map={"TRANSCRIPT": "transcript"},
        inputs={"speech_audio": vad["speech_audio"]},
    )

    workflow = educa_workflow(
        customer_speech=stt["transcript"],
        agent_speech=PARENT["last_agent_response"],
        current_state=PARENT["current_state"],
        script_data=script_data,
        intent_retry_counts=PARENT["intent_retry_counts"],
        conversation_history=PARENT["conversation_history"],
    )

    update = update_conversation(
        transcript=stt["transcript"],
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

    START >> source >> audio >> vad >> stt >> workflow >> update >> tts >> END
