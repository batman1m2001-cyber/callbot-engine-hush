"""VadDetector v2 — @graph with separate ops, PARENT.shared() flat vars.

Ops:
  vad_infer: ONNX Silero inference → speech_prob
  foreground_detect: energy-based background model → is_foreground
  speech_segmenter: state machine + buffer → yield speech segments
"""

import math
import time
from collections import deque

import numpy as np
import onnxruntime as ort

from hush.core import graph, PARENT, START, END
from hush.core.ops import op

# ── Global ONNX session (shared, stateless) ──
_SESSION = None


def _get_session(path="models/silero_vad.onnx"):
    global _SESSION
    if _SESSION is None:
        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        _SESSION = ort.InferenceSession(path, providers=["CPUExecutionProvider"], sess_options=opts)
    return _SESSION


# Pre-load
_get_session()

# Config
RATE = 16000
CHUNK = 512
TRIGGER = 0.5
NEG = 0.35
MIN_SPEECH = int(RATE * 60 / 1000)
MAX_SPEECH = int(RATE * 45000 / 1000)
MIN_SILENCE = int(RATE * 500 / 1000)
PAD_START = int(500 / (CHUNK / RATE * 1000))


@op
def vad_infer(audio, onnx_state: dict) -> dict:
    """ONNX Silero VAD inference. Mutates onnx_state dict in place."""
    if not isinstance(audio, np.ndarray):
        audio = np.array(audio, dtype=np.float32)

    vad_state = onnx_state["vad_state"]
    vad_context = onnx_state["vad_context"]

    input_ctx = np.concatenate([vad_context, audio.reshape(1, -1)], axis=1)
    outs = _SESSION.run(
        ["output", "stateN"],
        {"input": input_ctx.astype(np.float32), "state": vad_state, "sr": np.array(RATE, dtype=np.int64)},
    )
    # Mutate in place
    onnx_state["vad_state"] = outs[1]
    onnx_state["vad_context"] = input_ctx[:, -64:]

    return {"speech_prob": float(outs[0][0][0])}


@op
def foreground_detect(audio, bg: dict) -> dict:
    """Energy-based foreground detection. Mutates bg dict in place."""
    if not isinstance(audio, np.ndarray):
        audio = np.array(audio, dtype=np.float32)

    energy = float(np.sqrt(np.mean(audio ** 2)))
    bg["n"] += 1

    if bg["n"] == 1:
        bg["mean"] = energy
        bg["var"] = energy * energy * 0.1
    else:
        delta = energy - bg["mean"]
        bg["mean"] += bg["alpha"] * delta
        bg["var"] = (1 - bg["alpha"]) * bg["var"] + bg["alpha"] * delta * delta

    elapsed = time.time() - bg["vad_start_time"]
    if elapsed < 5.0:
        is_fg = True
    else:
        std = math.sqrt(max(bg["var"], 1e-10))
        z = (energy - bg["mean"]) / max(std, 1e-8)
        p_bg = 1.0 / (1.0 + math.exp(min(z, 20)))
        is_fg = p_bg < 0.2

    return {"is_foreground": is_fg}


@op
def speech_segmenter(
    audio, cmc_time: int,
    speech_prob: float, is_foreground: bool,
    sm: dict, possible_ends: list, speech_buffer: deque, temp_buffer: deque,
):
    """State machine: IDLE → SPEECH → yield segment. N-to-M.

    All state mutated in place via mutable containers (sm dict, deque, list).
    No push refs needed — mutations persist through shared var references.
    """
    # Schema hint: declare output keys for AST extraction
    if False:
        yield {"speech_audio": None, "speech_duration_ms": None, "num_chunks": None, "cmc_start_time": None, "vad_time": None}

    if not isinstance(audio, np.ndarray):
        audio = np.array(audio, dtype=np.float32)

    sm["current_sample"] += CHUNK

    if not sm["triggered"]:
        temp_buffer.append((audio, speech_prob, cmc_time))
        while len(temp_buffer) > PAD_START:
            temp_buffer.popleft()

        if speech_prob >= TRIGGER and is_foreground:
            sm["triggered"] = True
            sm["temp_end"] = 0
            possible_ends.clear()
            for item in temp_buffer:
                speech_buffer.append(item)
            temp_buffer.clear()
    else:
        speech_buffer.append((audio, speech_prob, cmc_time))

        if speech_prob < NEG:
            if sm["temp_end"] == 0:
                sm["temp_end"] = sm["current_sample"]
            silence = sm["current_sample"] - sm["temp_end"]
            if silence >= MIN_SILENCE:
                seg = _finalize(speech_buffer)
                _reset(sm, possible_ends, speech_buffer, temp_buffer)
                if seg:
                    yield seg
                return
        else:
            if sm["temp_end"] > 0:
                sil_ms = (sm["current_sample"] - sm["temp_end"]) / RATE * 1000
                possible_ends.append((sm["temp_end"], sil_ms))
                sm["temp_end"] = 0

        if len(speech_buffer) * CHUNK >= MAX_SPEECH:
            seg = _force_end(speech_buffer, possible_ends, sm["current_sample"])
            if seg:
                possible_ends.clear()
                sm["temp_end"] = 0
                yield seg
                return


def _finalize(buf):
    if not buf:
        return None
    trimmed = _trim(list(buf))
    total = sum(len(item[0]) for item in trimmed)
    if total < MIN_SPEECH:
        return None
    audio = np.concatenate([item[0] for item in trimmed])
    return {
        "speech_audio": audio,
        "speech_duration_ms": len(audio) / RATE * 1000,
        "num_chunks": len(trimmed),
        "cmc_start_time": trimmed[0][2],
        "vad_time": int(time.time() * 1000),
    }


def _force_end(buf, possible_ends, current_sample):
    best_end, best_sil = None, 0
    for idx, sil_ms in possible_ends:
        if sil_ms >= 98 and sil_ms > best_sil:
            best_sil = sil_ms
            best_end = idx
    items = list(buf)
    if best_end:
        cut = (best_end - (current_sample - len(buf) * CHUNK)) // CHUNK
        cut = max(1, min(cut, len(items)))
        trimmed = _trim(items[:cut])
        if trimmed:
            audio = np.concatenate([item[0] for item in trimmed])
            # Keep remaining in buffer
            remaining = items[cut:]
            buf.clear()
            buf.extend(remaining)
            return {
                "speech_audio": audio,
                "speech_duration_ms": len(audio) / RATE * 1000,
                "num_chunks": len(trimmed),
                "cmc_start_time": trimmed[0][2],
                "vad_time": int(time.time() * 1000),
            }
    return _finalize(buf)


def _reset(sm, possible_ends, speech_buffer, temp_buffer):
    sm["triggered"] = False
    sm["temp_end"] = 0
    possible_ends.clear()
    speech_buffer.clear()
    temp_buffer.clear()


def _trim(items):
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


@graph
def vad_detector(audio, cmc_time, recv_time):
    """Streaming VAD: 3 ops, flat shared vars, per-request state."""
    # Mutable containers: mutations persist directly via shared ref.
    # - onnx_state dict: vad_state, vad_context (ndarray mutated by vad_infer)
    # - bg dict: bg_mean, bg_var, bg_n (floats/ints mutated by foreground_detect)
    # - sm dict: triggered, temp_end, current_sample (primitives mutated by segmenter)
    # - speech_buffer, temp_buffer, possible_ends: deque/list mutated in place
    PARENT.shared(
        onnx_state={"vad_state": np.zeros((2, 1, 128), dtype=np.float32),
                     "vad_context": np.zeros((1, 64), dtype=np.float32)},
        bg={"mean": 0.0, "var": 1e-6, "n": 0, "alpha": CHUNK / (RATE * 5.0),
            "vad_start_time": time.time()},
        sm={"triggered": False, "temp_end": 0, "current_sample": 0},
        possible_ends=[],
        speech_buffer=deque(),
        temp_buffer=deque(maxlen=PAD_START),
    )

    # Op 1: ONNX inference (mutates onnx_state dict in place)
    infer = vad_infer(
        audio=audio,
        onnx_state=PARENT["onnx_state"],
    )

    # Op 2: Foreground detection (mutates bg dict in place)
    fg = foreground_detect(
        audio=audio,
        bg=PARENT["bg"],
    )

    # Op 3: State machine (mutates sm dict + buffers in place)
    seg = speech_segmenter(
        audio=audio,
        cmc_time=cmc_time,
        speech_prob=infer["speech_prob"],
        is_foreground=fg["is_foreground"],
        sm=PARENT["sm"],
        possible_ends=PARENT["possible_ends"],
        speech_buffer=PARENT["speech_buffer"],
        temp_buffer=PARENT["temp_buffer"],
    )

    START >> [infer, fg]
    [infer, fg] >> seg >> END
