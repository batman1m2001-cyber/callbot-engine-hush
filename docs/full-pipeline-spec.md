# Full Callbot Pipeline — Single Hush Graph

## Mục tiêu

Merge speech pipeline + LLM workflow + TTS thành 1 graph end-to-end.
Chạy được với WAV file input, không cần WebSocket.

## Pipeline

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         callbot_pipeline                                │
│                                                                         │
│  ┌─── Speech In ───┐   ┌─── Brain ───┐   ┌─── Speech Out ───┐         │
│  │                  │   │             │   │                   │         │
│  │  wav_source      │   │  educa      │   │  tts_pipeline     │         │
│  │  ↓               │   │  _workflow  │   │  (text → audio)   │         │
│  │  AudioProcessor  │   │             │   │                   │         │
│  │  ↓               │   │             │   │                   │         │
│  │  VadDetector     │   │             │   │                   │         │
│  │  ↓               │   │             │   │                   │         │
│  │  STT (Triton)    │   │             │   │                   │         │
│  │  ↓               │   │             │   │                   │         │
│  │  [Denoise]       │   │             │   │                   │         │
│  └──────┬───────────┘   └──────┬──────┘   └──────┬────────────┘         │
│         │ transcript           │ response        │ audio                │
│         └──────────►───────────┘─────────►───────┘                      │
└─────────────────────────────────────────────────────────────────────────┘
```

## Data Flow (per turn)

```
1. wav_source yield audio chunks (8kHz PCM, 40ms each)
     ↓
2. AudioProcessor: decode → resample 8k→16k → buffer 512 → preprocess
     ↓ yield 512-sample chunks
3. VadDetector: ONNX inference → state machine → yield speech segment
     ↓ yield (speech_audio, duration_ms, cmc_start_time)
4. STT (TritonOp): speech_audio → Triton gRPC → transcript
     ↓ (transcript, embedding)
5. [Denoise]: embedding → speech/noise classify → suppress if noise
     ↓ transcript (or empty)
6. educa_workflow: transcript + state + context → intent → response text
     ↓ (response, intent, new_state, ...)
7. tts_pipeline: response text → phonemes → fastspeech2 → hifigan → audio
     ↓ (audio int16, duration_ms)
```

## Vấn đề cần giải quyết

### 1. Stateful context giữa các turns

Một cuộc gọi = nhiều turns (user nói → agent trả lời → user nói lại...).
Cần maintain giữa các turns:
- `current_state` (REMINDER → CONFIRM_CUSTOMER → ...)
- `conversation_history` (list of messages)
- `intent_retry_counts` ({intent: count})
- `last_agent_response` (câu agent vừa nói)

### Giải pháp: PARENT.shared() — Hush feature mới

```python
@graph
def callbot(wav_path, script_data):
    # Khai báo shared vars — persist xuyên tất cả stream contexts (turns)
    PARENT.shared(
        current_state="REMINDER",
        conversation_history=[],
        intent_retry_counts={},
        last_agent_response="",
    )

    # Tất cả ops đọc/ghi shared vars qua PARENT["key"] như bình thường
    # Khác biệt: shared vars KHÔNG copy per stream context
    # Turn 1 ghi current_state = "CONFIRM_CUSTOMER"
    # Turn 2 đọc current_state → "CONFIRM_CUSTOMER" (không phải "REMINDER")

    workflow = educa_workflow(
        current_state=PARENT["current_state"],  # đọc shared var
        conversation_history=PARENT["conversation_history"],
        ...
    )
    workflow["new_state"] >> PARENT["current_state"]  # ghi ngược lên shared var
```

**Cách hoạt động:**
- `PARENT.shared(key=initial_value)` — khai báo + set giá trị ban đầu
- Scheduler biết vars nào là shared → không copy per stream context
- Ops đọc/ghi qua `PARENT["key"]` syntax quen thuộc — không API mới
- Race condition: scheduler đảm bảo sequential access (turn trước xong mới turn sau)

**So sánh PARENT vars:**
```
Normal PARENT var:  mỗi stream context copy riêng
                    Turn 1: PARENT["x"] = "a"
                    Turn 2: PARENT["x"] = initial (không thấy "a")

Shared PARENT var:  tất cả stream contexts share 1 bản
                    Turn 1: PARENT["x"] = "a"
                    Turn 2: PARENT["x"] = "a" (thấy giá trị từ turn 1)
```

**Cần implement trong Hush:**
- `PARENT.shared()` method → register vars as shared
- StateSchema: shared vars lưu ở graph-level cell, không per-context
- Scheduler: resolve shared vars từ graph-level cell thay vì context cell

### 2. Speech pipeline streaming + LLM workflow batch

Speech pipeline là streaming (yield per chunk/segment).
LLM workflow là batch (1 input → 1 output).
TTS cũng batch.

Hush graph tự handle: VAD yield segment → downstream dispatch STT → workflow → TTS.
Mỗi VAD segment = 1 stream context → trigger full batch pipeline.

### 3. Multi-turn tự động qua streaming

```
wav_source yield chunks liên tục
  → AudioProcessor yield 512-sample chunks
    → VAD yield segment khi user nói xong (N-to-M)
      → STT transcript
        → educa_workflow (đọc PARENT.shared state)
          → update shared state (ghi PARENT.shared)
            → TTS audio response
```

Không cần loop hay outer manager. Streaming graph + shared vars = multi-turn tự động.
Mỗi VAD segment trigger 1 turn. Shared vars carry state giữa turns.

## Graph Design

```python
@graph
def callbot_pipeline(wav_path, script_data):
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
        inputs={"audio": audio["audio"], "cmc_time": audio["cmc_time"], "recv_time": audio["recv_time"]},
    )
    stt = TritonOp(
        name="stt",
        url="192.168.1.212:8001",
        model_name="fastconformer_asr",
        inputs_map={"AUDIO_SIGNAL": "speech_audio"},
        outputs_map={"TRANSCRIPT": "transcript"},
        inputs={"speech_audio": vad["speech_audio"]},
    )

    # ── Brain (batch, per-turn) ──
    # Reads shared vars via PARENT — gets latest state from previous turn
    workflow = educa_workflow(
        customer_speech=stt["transcript"],
        agent_speech=PARENT["last_agent_response"],
        current_state=PARENT["current_state"],
        script_data=script_data,
        intent_retry_counts=PARENT["intent_retry_counts"],
        conversation_history=PARENT["conversation_history"],
    )

    # ── Update shared state for next turn ──
    update = update_conversation(
        transcript=stt["transcript"],
        response=workflow["response"],
        intent=workflow["intent"],
        new_state=workflow["new_state"],
    )
    # Write back to shared vars
    update["new_state"] >> PARENT["current_state"]
    update["history"] >> PARENT["conversation_history"]
    update["retry_counts"] >> PARENT["intent_retry_counts"]
    update["response"] >> PARENT["last_agent_response"]

    # ── Speech Out (batch, per-turn) ──
    tts = tts_pipeline(text=workflow["response"])

    # ── Wiring ──
    START >> source >> audio >> vad >> stt >> workflow >> update >> tts >> END
```

## Folder Structure

```
callbot-engine-hush/
├── agents/
│   └── educa_reminder/
│       ├── ops/                    ← agent-specific ops
│       ├── data/                   ← prompts, intents, responses
│       └── workflow.py             ← @graph educa_workflow
│
├── speech/
│   ├── audio_processor.py          ← AudioProcessor (BaseOp)
│   ├── vad_detector.py             ← VadDetector (BaseOp)
│   ├── denoise_classifier.py       ← DenoiseClassifier (BaseOp)
│   ├── tts_synthesizer.py          ← tts_pipeline (@graph)
│   ├── tts/                        ← phoneme processing + lexicon
│   └── pipeline.py                 ← speech_pipeline (@graph, speech-in only)
│
├── pipeline/                       ← NEW: full callbot pipeline
│   ├── callbot.py                  ← callbot_pipeline (@graph)
│   ├── conversation_manager.py     ← stateful turn context + history
│   └── mock_source.py              ← wav_source for testing
│
├── models/
│   ├── silero_vad.onnx
│   └── denoise.onnx
│
├── scripts/
│   ├── bench_latency.py
│   └── run_callbot.py              ← NEW: chạy full pipeline với WAV
│
├── tests/
│   ├── agents/
│   │   └── test_educa_reminder.py
│   ├── speech/
│   │   ├── audio/                  ← test WAV files
│   │   ├── test_speech_pipeline.py
│   │   └── generate_test_audio.py
│   └── pipeline/                   ← NEW
│       └── test_callbot.py         ← full pipeline test
│
├── docs/
│   ├── workflow-spec.md
│   ├── audio-vad-pipeline-spec.md
│   └── full-pipeline-spec.md       ← this file
│
├── resources.yaml
├── pyproject.toml
└── .env
```

## Test Plan

### Test 1: Single turn — user confirms
```
Input:  "03_confirm.wav" (vâng đúng rồi)
State:  CONFIRM_CUSTOMER
Expect: intent=confirm, response contains greeting, TTS audio output
```

### Test 2: Single turn — user silent
```
Input:  "02_silence.wav"
State:  REMINDER
Expect: intent=silent, response asks to confirm, TTS audio
```

### Test 3: Multi-turn conversation
```
Turn 1: silence → agent: "nhà mình có bạn tham gia AI CLASS?"
Turn 2: "vâng đúng rồi" → agent: "bé Minh có lịch học lúc 19:00..."
Turn 3: "bé đang vào rồi" → agent: "cảm ơn anh chị..."
```

### Test 4: Latency benchmark
```
Measure per-turn: VAD end → transcript → LLM → TTS → total
Target: < 2s per turn (VAD 13ms + STT 50ms + LLM 1000ms + TTS 300ms)
```

## Implementation Steps

1. Tạo `pipeline/conversation_manager.py` — stateful turn context
2. Tạo `pipeline/callbot.py` — full pipeline @graph
3. Tạo `pipeline/mock_source.py` — wav_source op
4. Tạo `tests/pipeline/test_callbot.py` — single + multi-turn tests
5. Tạo `scripts/run_callbot.py` — interactive demo
6. Test end-to-end