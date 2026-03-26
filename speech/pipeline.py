"""Speech pipeline graph: audio → VAD → STT → denoise → transcript.

Usage:
    from speech.pipeline import create_speech_pipeline

    pipeline = create_speech_pipeline()
    engine = Hush(pipeline, resources="resources.yaml")
    # Feed audio chunks via engine
"""

from hush.core import graph, START, END, PARENT
from hush.providers.ops import TritonOp

from speech.audio_processor import AudioProcessor
from speech.vad_detector import VadDetector
from speech.denoise_classifier import DenoiseClassifier


@graph
def speech_pipeline(raw_chunk, cmc_time):
    """Full speech pipeline: audio → preprocess → VAD → STT → denoise.

    Input:
        raw_chunk: bytes — PCM int16 audio from telco (8kHz)
        cmc_time: int — timestamp ms from telco

    Output:
        transcript: str — recognized text (empty if noise)
        is_speech: bool — True if speech detected
        speech_prob: float — denoise confidence
        speech_duration_ms: float — duration of speech segment
        cmc_start_time: int — when user started speaking
    """

    # Op 1: Decode PCM + resample 8k→16k + buffer 512 + conformer preprocess
    audio = AudioProcessor(
        name="audio",
        inputs={
            "raw_chunk": PARENT["raw_chunk"],
            "cmc_time": PARENT["cmc_time"],
        },
    )

    # Op 2: Silero VAD + foreground detection + speech segmentation
    vad = VadDetector(
        name="vad",
        inputs={
            "audio": audio["audio"],
            "cmc_time": audio["cmc_time"],
            "recv_time": audio["recv_time"],
        },
    )

    # Op 3: STT via Triton gRPC
    stt = TritonOp(
        name="stt",
        resource="stt",
        inputs_map={"AUDIO_SIGNAL": "speech_audio"},
        outputs_map={"TRANSCRIPT": "transcript", "EMBEDDING": "embedding"},
        inputs={
            "speech_audio": vad["speech_audio"],
        },
    )

    # Op 4: Denoise classifier (suppress noise-only segments)
    denoise = DenoiseClassifier(
        name="denoise",
        inputs={
            "transcript": stt["transcript"],
            "embedding": stt["embedding"],
        },
    )

    # Forward VAD timing info to output
    vad["speech_duration_ms"] >> PARENT["speech_duration_ms"]
    vad["cmc_start_time"] >> PARENT["cmc_start_time"]

    START >> audio >> vad >> stt >> denoise >> END
