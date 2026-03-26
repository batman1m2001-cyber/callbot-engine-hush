# Audio → VAD → STT Pipeline Spec

## Mục tiêu

Cô lập toàn bộ speech pipeline (audio → denoise → VAD → STT) thành Hush graph standalone.
Test được với mock audio (WAV file) mà không cần WebSocket, Redis, agent.

---

## Pipeline Overview

```
mock_audio_source (async iterator)
  │ yields {"raw_chunk": bytes, "cmc_time": int}
  │
  ▼
audio_processor (@op, streaming/yield)
  │ decode PCM → float32, resample 8k→16k, buffer → 512-sample chunks
  │ yields {"audio": ndarray(512,), "cmc_time": int, "recv_time": int}
  │
  ▼
conformer_preprocess (@op, streaming/yield)
  │ bandpass 60-4000Hz, pre-emphasis, spectral gating, adaptive gain
  │ yields {"audio": ndarray(512,), "cmc_time": int, "recv_time": int}
  │ (pass-through nếu disable)
  │
  ▼
vad_detector (@op, streaming/yield, stateful)
  │ ONNX Silero inference → speech_prob
  │ foreground detection (energy-based background model)
  │ state machine: IDLE → SPEECH → END
  │ buffer + trim + validate
  │ yields {"speech_audio": ndarray, "duration_ms": float, "cmc_start_time": int, "vad_time": int}
  │ (PENDING khi chưa detect speech end)
  │
  ▼
denoise_classifier (@op, stateful)
  │ Input: speech embedding từ STT (1024, T)
  │ ConvTransformerClassifier (ONNX, 9MB)
  │ Streaming state: embedding history capped 24
  │ Output: speech_prob [0-1], nếu < 0.63 → suppress transcript
  │
  ▼
stt_client (@op)
  │ gRPC tới Triton: audio float32 → text + embedding
  │ Optional denoise check: embedding → denoise_classifier → suppress nếu noise
  │ Output: {"transcript": str, "embedding": ndarray, "stt_time": int}
  │
  ▼
output (transcript ready for LLM workflow)
```

---

## Op Specs

### Op 1: `audio_processor` (streaming)

**Purpose:** Decode raw PCM, resample, buffer thành fixed-size chunks.

**Input per call:**
| Name | Type | Description |
|------|------|-------------|
| `raw_chunk` | `bytes` | Raw PCM từ tổng đài (8kHz, int16, mono) |
| `cmc_time` | `int` | Timestamp ms từ tổng đài |

**Internal State:**
| State | Type | Init | Description |
|-------|------|------|-------------|
| `resampler` | `soxr.ResampleStream` | 8k→16k VHQ | Streaming resampler, maintain phase |
| `buffer` | `collections.deque` | empty | Accumulate samples chờ đủ 512 |
| `buffer_len` | `int` | 0 | Total samples in buffer |

**Processing:**
1. `np.frombuffer(raw_chunk, dtype=np.int16)` → int16 array
2. `.astype(np.float32) / 32768.0` → normalize [-1, 1]
3. `resampler.resample_chunk(data)` → 16kHz float32
4. Append to buffer
5. While `buffer_len >= 512`:
   - Extract 512 samples
   - **yield** `{"audio": chunk, "cmc_time": cmc_time, "recv_time": now_ms()}`

**Output per yield:**
| Name | Type | Shape | Description |
|------|------|-------|-------------|
| `audio` | `np.ndarray` | (512,) float32 | 32ms chunk at 16kHz |
| `cmc_time` | `int` | — | Preserved timestamp |
| `recv_time` | `int` | — | Backend processing timestamp ms |

**Edge cases:**
- Chunk < 512 samples → buffer, PENDING (no yield)
- Resampler output rỗng → no yield
- Input already 16kHz → skip resampler

---

### Op 2: `conformer_preprocess` (streaming, optional)

**Purpose:** Audio enhancement — filter noise, boost speech, normalize volume.
Có thể disable (pass-through) nếu không cần.

**Input per call:**
| Name | Type | Description |
|------|------|-------------|
| `audio` | `np.ndarray` (512,) | Float32 chunk |
| `cmc_time` | `int` | Timestamp |
| `recv_time` | `int` | Timestamp |

**Internal State:**
| State | Type | Description |
|-------|------|-------------|
| `noise_samples` | `list` | Accumulated noise profile (FFT) |
| `speech_count` | `int` | Consecutive speech frames for hysteresis |
| `rms_buffer` | `deque(50)` | Recent RMS values for adaptive gain |
| `current_gain` | `float` | Smooth gain factor |

**Processing chain:**
1. **Bandpass filter** — `scipy.signal.filtfilt`, Butterworth order 4
   - High-pass: 60 Hz (remove DC drift, subsonic noise)
   - Low-pass: 4000 Hz (remove high-freq hiss)

2. **Pre-emphasis** — `y[n] = x[n] - 0.97 * x[n-1]`
   - Boost consonants/fricatives (quan trọng cho tiếng Việt)

3. **Speech detection** (internal heuristics)
   - Energy threshold: `bg_energy * 3`
   - Zero-crossing rate: 0.05 - 0.3
   - Spectral centroid: 500 - 4000 Hz
   - Hysteresis: maintain state ≥ 3 frames (~60ms)

4. **Noise profile update** (during detected silence)
   - Accumulate non-speech frames
   - Estimate spectrum via FFT

5. **Spectral gating** (khi `noise_samples > 5`)
   - Mask frequencies dưới noise threshold
   - Vietnamese frequency band weights:
     - 60-300 Hz: 1.5x (nguyên âm)
     - 300-800 Hz: 1.3x (nguyên âm)
     - 1200-2500 Hz: 1.4x (phụ âm)
     - 2500-3500 Hz: 1.2x (thanh điệu)

6. **Adaptive gain control**
   - Track RMS over 50-frame window
   - Attack: 0.2, Release: 0.05
   - Normalize toward target_rms = 0.15

**Output:** Same as input format — `{"audio": processed, "cmc_time": ..., "recv_time": ...}`

---

### Op 3: `vad_detector` (streaming, stateful)

**Purpose:** Detect speech segments, emit concatenated audio khi speech kết thúc.

**Input per chunk:**
| Name | Type | Description |
|------|------|-------------|
| `audio` | `np.ndarray` (512,) float32 | 16kHz audio chunk |
| `cmc_time` | `int` | Timestamp ms |
| `recv_time` | `int` | Backend timestamp ms |

**Config:**
```python
rate = 16000
chunk = 512  # 32ms
speech_threshold = 0.5
trigger_threshold = 0.5
neg_threshold = 0.35  # max(trigger - 0.15, 0.01)
min_speech_duration_ms = 60
max_speech_duration_ms = 45000
min_silence_duration_ms = 500
speech_pad_start_ms = 500  # ~16 chunks context before trigger
speech_pad_end_ms = 300    # ~10 chunks after release
```

**Internal State:**
| State | Type | Init | Description |
|-------|------|------|-------------|
| `vad_state` | `np.ndarray` | zeros (2,1,128) | ONNX LSTM hidden/cell state |
| `vad_context` | `np.ndarray` | zeros (1,64) | Last 64 samples for context |
| `triggered` | `bool` | False | Currently in speech segment |
| `temp_end` | `int` | 0 | Sample index of potential end |
| `current_sample` | `int` | 0 | Running sample counter |
| `speech_buffer` | `deque` | empty | [(chunk, prob, cmc_time), ...] |
| `temp_buffer` | `deque` | empty | Context buffer before speech (max ~16 chunks) |
| `possible_ends` | `list` | [] | [(sample_idx, silence_duration_ms)] |
| `bg_model` | `OnlineGaussianModel` | new | Energy-based background model |

**ONNX Inference:**
```python
input_with_context = np.concatenate([vad_context, audio.reshape(1, -1)])  # (1, 576)
speech_prob, new_state = ort_session.run(
    ["output", "stateN"],
    {"input": input_with_context, "state": vad_state, "sr": np.int64(16000)}
)
vad_state = new_state
vad_context = input_with_context[:, -64:]
```

**Foreground detection:**
```python
energy = np.sqrt(np.mean(audio ** 2))
bg_model.update(energy)
p_bg = bg_model.p_background(energy)
# First 5 seconds: always foreground
# After: foreground if p_bg < 0.2
```

**State Machine:**
```
                    prob ≥ 0.5 + foreground
         ┌────────────────────────────────────┐
         │                                    ▼
      [IDLE]                              [SPEECH]
         ▲                                    │
         │         silence ≥ 500ms            │
         │    OR   duration > 45s             │
         └────────────────────────────────────┘
                   → yield speech segment
```

**Transitions:**

1. **IDLE → SPEECH**: `prob >= 0.5` AND `is_foreground` AND NOT `triggered`
   - Prepend `temp_buffer` → `speech_buffer` (pre-speech context 500ms)
   - Set `triggered = True`

2. **SPEECH accumulate**: append `(chunk, prob, cmc_time)` to `speech_buffer`

3. **SPEECH → potential end**: `prob < 0.35` AND `triggered`
   - Set `temp_end = current_sample`
   - Track silence duration

4. **Speech resumes**: `prob >= 0.5` while `temp_end > 0`
   - Record silence point in `possible_ends`
   - Clear `temp_end`, continue accumulating

5. **SPEECH → confirmed end**: `silence_duration >= 500ms` (= 8000 samples at 16kHz)
   - **Trim**: find first/last chunk with `prob >= 0.25`, keep 2-3 frames before, 3-5 after
   - **Validate**: `total_samples >= 960` (60ms min) — discard if shorter
   - **End padding**: append ~10 chunks (300ms) tail
   - **Concatenate**: `np.concatenate([item[0] for item in trimmed_buffer])`
   - **yield** speech segment
   - Reset state machine

6. **SPEECH → force end**: `duration > 45s`
   - Find best silence point from `possible_ends` (≥ 98ms silence)
   - Cut at that point, yield segment

**Output per yield (only when speech ends):**
| Name | Type | Description |
|------|------|-------------|
| `speech_audio` | `np.ndarray` float32 | Concatenated speech (variable length) |
| `duration_ms` | `float` | Speech duration in ms |
| `num_chunks` | `int` | Number of 512-sample chunks |
| `cmc_start_time` | `int` | Timestamp of first speech chunk |
| `vad_time` | `int` | Timestamp when VAD processed |

**PENDING:** Chưa detect speech end → không yield, downstream chờ.

---

### Op 4: `stt_client`

**Purpose:** gRPC call tới Triton, transcribe speech audio → text.

**Input:**
| Name | Type | Description |
|------|------|-------------|
| `speech_audio` | `np.ndarray` float32 | Speech segment (N,) at 16kHz |
| `duration_ms` | `float` | Duration info |
| `cmc_start_time` | `int` | Timing |

**Processing:**
1. **Normalize**: if `max(abs(audio)) > 1.0` → `audio / max(abs(audio))`
2. **Rescale** (optional): `(audio * 32767).astype(np.int16)` nếu model cần int16
3. **gRPC call** tới Triton:
   ```
   Input: AUDIO_SIGNAL (FP32, shape=(N,))
   Output: TRANSCRIPT (string), EMBEDDING (FP32, shape=(1024, T)) [optional]
   ```
4. **Filter noisy utterances** (heuristic post-filter)

**Output:**
| Name | Type | Description |
|------|------|-------------|
| `transcript` | `str` | Recognized text |
| `embedding` | `np.ndarray` or None | (1024, T) speech embedding for denoise |
| `stt_time` | `int` | Timestamp ms when STT finished |
| `cmc_start_time` | `int` | Preserved from input |
| `duration_ms` | `float` | Preserved from input |

**Config (via resources.yaml):**
```yaml
triton:
  stt:
    url: ${TRITON_URL:192.168.1.212:8001}
    model: fastconformer_asr
    rescale_factor: 0  # or 32767 for int16
```
```python
stt = TritonOp(resource="stt", ...)  # resolve từ ResourceHub
```

---

### Op 5: `denoise_classifier` (stateful, optional)

**Purpose:** Classify speech vs noise using STT embedding. Suppress noise-only segments.

**Input:**
| Name | Type | Description |
|------|------|-------------|
| `transcript` | `str` | STT output |
| `embedding` | `np.ndarray` (1024, T) | Speech embedding |

**Internal State:**
| State | Type | Init | Description |
|-------|------|------|-------------|
| `embedding_history` | `list` | [start_token] | Past conv outputs, capped at 24 |
| `start_token` | `np.ndarray` (1, 256) | learnable | Initial token for transformer |

**Processing:**
1. **Pad/truncate** embedding to (1024, 128)
2. **ConvEncoder**: (1024, 128) → (256,) via 3x Conv1D + GlobalAvgPool + FC
3. **Append** to embedding_history (cap at 24)
4. **Causal Transformer**: stack history → (1, seq_len, 256) → attend → (256,)
5. **Classifier**: FC(256→128) + ReLU + Dropout + FC(128→1) → sigmoid → prob

**Output:**
| Name | Type | Description |
|------|------|-------------|
| `transcript` | `str` | Empty string if noise (prob < 0.63), else original |
| `is_speech` | `bool` | True if speech, False if noise |
| `speech_prob` | `float` | [0, 1] confidence |

**Config (via resources.yaml):**
```yaml
onnx:
  denoise:
    model_path: models/denoise.onnx
    denoise_threshold: 0.63
    max_history: 24
```

**Note:** Denoise runs **after** STT, not before — it uses the embedding from STT model output. Nếu denoise detect noise → transcript bị suppress thành empty string.

---

## Graph Wiring

```python
@graph
def speech_pipeline(raw_chunk, cmc_time):
    # Op 1: AudioProcessor (extends BaseOp, streaming, stateful)
    # decode PCM + resample 8k→16k + buffer 512 + conformer preprocess (optional)
    proc = AudioProcessor(
        name="audio",
        inputs={"raw_chunk": PARENT["raw_chunk"], "cmc_time": PARENT["cmc_time"]},
    )

    # Op 2: VadDetector (extends BaseOp, streaming, stateful)
    # ONNX Silero inference + foreground detection + state machine + segmenter
    # Shared ONNX session (global), per-call LSTM state + buffers
    speech = VadDetector(
        name="vad",
        inputs={
            "audio": proc["audio"],
            "cmc_time": proc["cmc_time"],
            "recv_time": proc["recv_time"],
        },
    )

    # Op 3: STT (TritonOp — stateless, 1 segment → 1 transcript)
    stt = TritonOp(
        name="stt",
        resource="stt",
        inputs_map={"AUDIO_SIGNAL": "speech_audio"},
        outputs_map={"TRANSCRIPT": "transcript", "EMBEDDING": "embedding"},
        inputs={
            "speech_audio": speech["speech_audio"],
        },
    )

    # Op 4: DenoiseClassifier (extends BaseOp, stateful)
    # ONNX ConvTransformer + streaming embedding_history (capped 24)
    # Shared ONNX session (global), per-call embedding history
    final = DenoiseClassifier(
        name="denoise",
        inputs={
            "transcript": stt["transcript"],
            "embedding": stt["embedding"],
        },
    )

    START >> proc >> speech >> stt >> final >> END
```

**Op implementation:**
- `AudioProcessor(BaseOp)` — stateful: resampler, buffer, conformer state. Streaming yield per 512-sample chunk.
- `VadDetector(BaseOp)` — stateful: ONNX LSTM state, vad_context, speech_buffer, background model, state machine. ONNX session shared global. Streaming yield per speech segment (PENDING between).
- `TritonOp(BaseOp)` — generic Triton gRPC client in hush-providers. Stateless per call.
- `DenoiseClassifier(BaseOp)` — stateful: embedding_history. ONNX session shared global. Runs once per speech segment.

**Streaming behavior:**
- `audio_processor` yields per 512-sample chunk (~32ms)
- `conformer_preprocess` yields per chunk (pass-through timing)
- `vad_detector` PENDING until speech end detected, then yields
- `stt_client` runs once per speech segment (~300-500ms inference)
- `denoise_classifier` runs once per speech segment (~25ms)

---

## Data Format Summary

| Point | Format | Sample Rate | Dtype | Shape |
|-------|--------|-------------|-------|-------|
| Raw input | PCM bytes | 8kHz | int16 | variable |
| After decode | ndarray | 8kHz | float32 | variable |
| After resample | ndarray | 16kHz | float32 | variable |
| After buffer | ndarray | 16kHz | float32 | (512,) fixed |
| After preprocess | ndarray | 16kHz | float32 | (512,) fixed |
| VAD output | ndarray | 16kHz | float32 | (N,) variable |
| STT input | ndarray | 16kHz | float32 or int16 | (N,) |
| STT output | string | — | — | — |
| Embedding | ndarray | — | float32 | (1024, T) |

---

## Test Plan

### Test 1: Silence only
- Input: 5s zeros
- Expected: vad_detector PENDING (no yield), no STT call

### Test 2: Clear speech
- Input: WAV with 3s clear speech
- Expected: 1 speech segment → 1 transcript

### Test 3: Speech → pause → speech
- Input: 2s speech + 1s silence + 2s speech
- Expected: 2 speech segments → 2 transcripts

### Test 4: Very short speech (< 60ms)
- Input: 30ms click/pop
- Expected: Discarded by VAD, no STT

### Test 5: Max duration (> 45s)
- Input: 50s continuous speech
- Expected: Force-cut at ~45s, 1 transcript

### Test 6: Noise suppression
- Input: background noise only (fan, traffic)
- Expected: denoise prob < 0.63 → transcript suppressed

### Test 7: End-to-end timing
- Input: 2s speech WAV
- Expected: total latency < 500ms (VAD + STT + denoise)

### Mock audio source
```python
async def mock_audio_source(wav_path: str, chunk_ms: int = 40):
    """Read WAV, yield 320-sample chunks (40ms at 8kHz) like tổng đài."""
    import soundfile as sf
    audio, sr = sf.read(wav_path, dtype='int16')
    if sr != 8000:
        import soxr
        audio = soxr.resample(audio.astype(np.float32), sr, 8000)
        audio = (audio * 32768).astype(np.int16)
    chunk_size = int(8000 * chunk_ms / 1000)  # 320 samples
    for i in range(0, len(audio), chunk_size):
        chunk = audio[i:i+chunk_size]
        yield {
            "raw_chunk": chunk.tobytes(),
            "cmc_time": int(time.time() * 1000),
        }
        await asyncio.sleep(chunk_ms / 1000)
```

---

## Resources Config (`resources.yaml`)

```yaml
triton:
  stt:
    url: ${TRITON_URL:192.168.1.212:8001}
    model: fastconformer_asr
    rescale_factor: 0
  tts:
    url: ${TTS_TRITON_URL:192.168.1.212:3001}
    model: tts_tritonv2

onnx:
  vad:
    model_path: models/silero_vad.onnx
  denoise:
    model_path: models/denoise.onnx
    denoise_threshold: 0.63
    max_history: 24
```

Tất cả model inference config tập trung 1 chỗ. Ops đọc từ ResourceHub:
- `TritonOp(resource="stt", ...)` — generic Triton client (hush-providers)
- `OnnxOp(resource="vad", ...)` — ONNX inference (hush-providers, đã có)

## Hush-providers Enhancement: TritonOp

Generic op cho Triton Inference Server, tương tự LLMOp/OnnxOp:

```python
class TritonOp(BaseOp):
    """Generic Triton gRPC inference op."""
    def __init__(self, resource, inputs_map, outputs_map, **kwargs):
        # resource: key in resources.yaml["triton"]
        # inputs_map: {"AUDIO_SIGNAL": "audio"}  (triton_name → op_input_name)
        # outputs_map: {"TRANSCRIPT": "transcript"}  (triton_name → op_output_name)

    async def _process(self, **inputs):
        # 1. Format inputs → InferInput[]
        # 2. grpcclient.infer(model_name, inputs)
        # 3. Parse outputs → dict
```

Dùng cho STT, TTS, và bất kỳ Triton model nào sau này.

## Dependencies

- `soxr` — streaming resampler
- `onnxruntime` — Silero VAD + denoise inference
- `numpy` — audio array processing
- `scipy` — bandpass filter, pre-emphasis (conformer_preprocess)
- `tritonclient[grpc]` — Triton gRPC client (STT, TTS)

## Model files

| Model | File | Size | Format | Config key |
|-------|------|------|--------|------------|
| Silero VAD | `models/silero_vad.onnx` | 2.3MB | ONNX | `onnx.vad` |
| Denoise classifier | `models/denoise.onnx` | 9MB | ONNX | `onnx.denoise` |
| STT | External Triton server | — | FastConformer | `triton.stt` |
| TTS | External Triton server | — | — | `triton.tts` |

---

## Không include (scope ngoài)

- TTS pipeline (agent → audio output) — separate graph
- WebSocket server — thay bằng mock iterator
- Agent LLM workflow — separate graph, receives transcript
- Conversation logging — downstream consumer
- Metrics/Prometheus — add later
- ChunkerSTTService variant — add later if needed
- LightGBM noise classifier — legacy, replaced by ConvTransformer denoise
