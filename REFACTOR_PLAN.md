# Callbot Engine Hush — WebSocket Server Refactor Plan (v3)

> Goal: make `callbot-engine-hush` a drop-in replacement for `educa-reminder-agent`,
> accepting the same WebSocket + HTTP API contract.
>
> Principle: Hush-ai is a general-purpose workflow framework — voice/telco specifics
> live in callbot-engine-hush. Hush-ai stays unchanged.

---

## Architecture

```
CMC (Telco Bridge)
    │
    │  ws://host:9922/ws/call?call_id=...&customer_id=...
    ▼
┌──────────────────────────────────────────────────────────┐
│  server/ws_server.py                                      │
│                                                           │
│  recv_loop ──► audio_queue ──► Hush engine.start()       │
│                                      │                    │
│                              ExecutionHandle              │
│                              ┌───────┴────────┐          │
│  send_loop ◄── tts_queue ◄──│ frame monitor  │          │
│  (audio out)                 │ (VAD events → │          │
│                              │  interrupt)   │          │
│                              └───────────────┘          │
└──────────────────────────────────────────────────────────┘
```

### Key insight: interrupt via ExecutionHandle

`engine.start()` returns an `ExecutionHandle` that streams every root-level op output
as frames. We monitor it to bridge pipeline events → WebSocket control signals.

**Important**: Only root-level op frames appear in the ExecutionHandle stream.
Ops inside nested graphs (e.g. `state_transition` inside `educa_workflow`,
`postprocess_audio` inside `tts_pipeline`) do NOT produce separate frames.
The nested graph's aggregated outputs appear under its root-level variable name.

Root ops for `ws_callbot_pipeline`:
```
source, audio, vad, stt_input, stt, denoise, educa_agent, update, tts, tts_8khz
```

```python
handle = engine.start(inputs={})

async def monitor_handle():
    async for op_name, ctx, data in handle:
        if op_name == "vad" and data.get("speech_audio") is not None:
            if send_active:
                interrupt_event.set()
        if op_name == "educa_agent" and data.get("response"):
            last_response = data["response"]
        if op_name == "tts_8khz" and data.get("audio") is not None:
            await tts_queue.put({"audio": data["audio"], "text": last_response, ...})
```

---

## Changes Overview

```
  MODIFIED FILES (4):
    agents/educa_reminder/workflow.py  — remove crm from graph, simplify wiring
    pipeline/callbot.py                — add ws_source + ws_callbot_pipeline
    speech/vad_detector.py             — add silence_timeout param
    speech/tts_synthesizer.py          — add resample_for_telco op

  DEMOTED (1):
    agents/educa_reminder/ops/call_result.py  — remove @op, keep as plain function

  NEW FILES (7):
    server/config.py          — env-configurable values
    server/ws_server.py       — WebSocket handler + monitor_handle + interrupt
    server/http_server.py     — REST: health, customer-info, call summary
    server/customer_store.py  — customer_info CRUD (JSON files)
    server/call_logger.py     — JSONL log writer + builds call_result post-call
    main.py                   — entry point: start WS + HTTP servers
    scripts/test_ws_client.py — mock CMC client for integration test
```

---

## File Structure (after refactor)

```
callbot-engine-hush/
├── agents/
│   └── educa_reminder/
│       ├── workflow.py                  ← MODIFY: remove crm, m_response >> END
│       ├── data/
│       │   ├── prompts.yaml
│       │   ├── response_templates.py
│       │   └── state_config.py
│       └── ops/
│           ├── build_intent_context.py
│           ├── call_result.py           ← DEMOTE: remove @op, plain function
│           ├── generate_rule.py
│           ├── merge.py
│           ├── normalize.py
│           ├── quick_detect.py
│           ├── skip.py
│           └── state_transition.py
├── pipeline/
│   └── callbot.py                       ← MODIFY: add ws_source + ws_callbot_pipeline
├── speech/
│   ├── audio_processor.py
│   ├── denoise_classifier.py
│   ├── vad_detector.py                  ← MODIFY: add silence_timeout param
│   ├── tts_synthesizer.py              ← MODIFY: add resample_for_telco op
│   └── tts/                             (unchanged)
├── server/
│   ├── __init__.py                      (exists)
│   ├── config.py                        ★ NEW
│   ├── ws_server.py                     ★ NEW
│   ├── http_server.py                   ★ NEW
│   ├── customer_store.py                ★ NEW
│   └── call_logger.py                   ★ NEW
├── main.py                              ★ NEW
├── models/                              (unchanged)
├── tests/                               (unchanged)
├── scripts/
│   ├── (existing bench scripts)
│   └── test_ws_client.py                ★ NEW
├── docs/                                (unchanged)
├── resources.yaml
├── .env
└── pyproject.toml
```

---

## TIER 1 — Blocking

### T1-1: `ws_source` op + `ws_callbot_pipeline`  _(pipeline/callbot.py)_

```python
@op
async def ws_source(audio_queue: asyncio.Queue):
    """Yield 320-sample chunks from WS audio queue. None sentinel = call ended."""
    while True:
        item = await audio_queue.get()
        if item is None:
            return
        yield {"raw_chunk": item["audio_bytes"], "cmc_time": item["cmc_time"]}
```

Add `ws_callbot_pipeline` alongside existing `callbot_pipeline` (WAV stays for tests):

```python
@graph
def ws_callbot_pipeline(audio_queue, script_data):
    PARENT.shared(
        current_state="REMINDER",
        conversation_history=[],
        intent_retry_counts={},
        last_agent_response="",
    )
    source  = ws_source(audio_queue=audio_queue)
    audio   = AudioProcessor(inputs={"raw_chunk": source["raw_chunk"], "cmc_time": source["cmc_time"]})
    vad     = VadDetector(inputs={"audio": audio["audio"], "cmc_time": audio["cmc_time"], "recv_time": audio["recv_time"]})
    stt_in  = prepare_stt_input(speech_audio=vad["speech_audio"])
    stt     = TritonOp(resource="stt", inputs={"speech_audio": stt_in["speech_audio"]})
    denoise = DenoiseClassifier(inputs={"transcript": stt["transcript"], "embedding": stt["embedding"]})
    educa_agent = educa_workflow(
        customer_speech=denoise["transcript"],
        agent_speech=PARENT["last_agent_response"],
        current_state=PARENT["current_state"],
        script_data=script_data,
        intent_retry_counts=PARENT["intent_retry_counts"],
        conversation_history=PARENT["conversation_history"],
    )
    update  = update_conversation(
        transcript=denoise["transcript"], response=educa_agent["response"],
        intent=educa_agent["intent"], new_state=educa_agent["new_state"],
        conversation_history=PARENT["conversation_history"],
        intent_retry_counts=PARENT["intent_retry_counts"],
    )
    update["updated_state"]        >> PARENT["current_state"]
    update["updated_history"]      >> PARENT["conversation_history"]
    update["updated_retry_counts"] >> PARENT["intent_retry_counts"]
    update["updated_response"]     >> PARENT["last_agent_response"]

    tts      = tts_pipeline(text=educa_agent["response"])
    tts_8khz = resample_for_telco(audio=tts["audio"])

    START >> source >> audio >> vad >> stt_in >> stt >> denoise >> educa_agent >> update >> tts >> tts_8khz >> END
```

**Note**: `educa_agent` outputs (`response`, `intent`, `new_state`) are available to
`monitor_handle` via the ExecutionHandle frame for `op_name == "educa_agent"`.
No explicit `>> PARENT` forwarding needed — they're the graph's natural outputs.

---

### T1-2: `server/ws_server.py`

**Endpoint**: `ws://host:9922/ws/call`

**Query params**:
| Param | Required | Notes |
|-------|----------|-------|
| `call_id` | ✓ | unique call id |
| `phone_number` | ✓ | |
| `customer_id` | ✓ | int, used to load script_data |
| `agent_type` | ✓ | |
| `customer_info` | ✓ | URL-encoded JSON |
| `lead_id` | — | |
| `campaign_id` | — | |

**Inbound** (Client → Server):
```json
{"event": "media", "timestamp": 1700000000000, "media": {"payload": "<base64>", "tracking_id": "t1"}}
{"event": "ack",   "data": {"tracking_id": "t1"}}
```

**Outbound** (Server → Client):
| event | trigger | priority |
|-------|---------|----------|
| `media` | TTS audio ready | 8 |
| `interrupt` | VAD detects speech during agent speaking | 9 |
| `heartbeat` | every 3s | 1 |
| `transfer_hotline` | workflow new_state == TRANSFER_HOTLINE | 9 |
| `hangup` | call end, last message | 0 |

**Media message**:
```json
{
  "event": "media",
  "media_name": "audio_<id>.wav",
  "timestamp": 1700000000000,
  "text": "<agent response text>",
  "audio_dur": 2.5,
  "media": {"payload": "<base64_wav_8khz>", "is_sync": true, "tracking_id": "<uuid>"}
}
```

**Handler structure**:
```python
@app.websocket("/ws/call")
async def ws_call(websocket, call_id, customer_id, ...):
    script_data   = customer_store.get(customer_id)
    audio_queue   = asyncio.Queue()
    tts_queue     = asyncio.Queue()
    interrupt_event = asyncio.Event()
    send_active   = False

    engine = Hush(ws_callbot_pipeline(audio_queue=audio_queue, script_data=script_data),
                  resources="resources.yaml")
    handle = engine.start(inputs={})

    last_response = ""
    is_final_turn = False

    async def monitor_handle():
        """Bridge: pipeline frames → tts_queue + interrupt signals.

        IMPORTANT: only ROOT-LEVEL ops appear in the handle stream.
        Nested graph ops (e.g. state_transition inside educa_workflow,
        postprocess_audio inside tts_pipeline) do NOT produce separate
        frames. The nested graph's aggregated outputs appear under
        its root-level variable name.

        Root ops: source, audio, vad, stt_input, stt, denoise,
                  educa_agent, update, tts, tts_8khz
        """
        nonlocal send_active, last_response, is_final_turn
        async for op_name, ctx, data in handle:
            # ── VAD: speech detected → interrupt signal ──
            if op_name == "vad" and data.get("speech_audio") is not None:
                if send_active:
                    interrupt_event.set()

            # ── educa_agent completed: capture response + detect terminal state ──
            # Outputs: response, intent, new_state (aggregated from nested graph)
            if op_name == "educa_agent":
                if data.get("response") is not None:
                    last_response = data["response"]
                if data.get("new_state") in ("FINISH", "TRANSFER_HOTLINE"):
                    is_final_turn = True
                    if data["new_state"] == "TRANSFER_HOTLINE":
                        await send_json(websocket, {"event": "transfer_hotline", "call_id": call_id})

            # ── TTS completed: queue audio for sending ──
            # tts_8khz is root-level (after resample). Contains: audio
            if op_name == "tts_8khz" and data.get("audio") is not None:
                await tts_queue.put({"audio": data["audio"], "text": last_response,
                                     "duration": len(data["audio"]) / 8000})
                # Only stop AFTER the final turn's TTS audio is queued
                if is_final_turn:
                    await tts_queue.put(None)    # stop send_loop after this audio
                    await audio_queue.put(None)  # stop ws_source

    async def recv_loop():
        """Receive audio from WebSocket → audio_queue."""
        async for msg in websocket.iter_json():
            if msg.get("event") == "media":
                audio_b64 = msg["media"]["payload"]
                audio_bytes = base64.b64decode(audio_b64)
                cmc_time = msg.get("timestamp", int(time.time()*1000))
                await audio_queue.put({"audio_bytes": audio_bytes, "cmc_time": cmc_time})
            elif msg.get("type") == "websocket.disconnect":
                break
        await audio_queue.put(None)   # sentinel

    async def send_loop():
        """Send TTS audio + interrupt messages to WebSocket."""
        nonlocal send_active
        while True:
            item = await tts_queue.get()
            if item is None:
                break
            send_active = True
            interrupt_event.clear()
            audio_id = str(uuid.uuid4())
            wav_b64 = base64.b64encode(to_wav_8khz(item["audio"])).decode()
            await send_json(websocket, {
                "event": "media", "media_name": f"audio_{audio_id}.wav",
                "timestamp": now_ms(), "text": item["text"],
                "audio_dur": item["duration"],
                "media": {"payload": wav_b64, "is_sync": True, "tracking_id": audio_id},
            })
            send_active = False
            # If interrupted during send, send interrupt event
            if interrupt_event.is_set():
                await send_json(websocket, {"event": "interrupt", "call_id": call_id,
                                            "audio_id": audio_id, "timestamp": now_ms()})

    async def heartbeat_loop():
        while True:
            await asyncio.sleep(3.0)
            await send_json(websocket, {"event": "heartbeat", "call_id": call_id, "timestamp": now_ms()})

    tasks = [
        asyncio.create_task(recv_loop()),
        asyncio.create_task(send_loop()),
        asyncio.create_task(heartbeat_loop()),
        asyncio.create_task(monitor_handle()),
    ]
    try:
        done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_EXCEPTION)
        for t in done:
            t.result()   # re-raise any exception
    except Exception:
        pass
    finally:
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await audio_queue.put(None)   # ensure ws_source stops if not already
        await tts_queue.put(None)     # ensure send_loop stops if not already
        log_summary = call_logger.build_log_summary(handle.state, script_data, call_id)
        await send_json(websocket, {"event": "hangup", "call_id": call_id, "log_summary": log_summary})
        call_logger.write(call_id, log_summary)
```

**Audio format**:
- Input from telco: int16 PCM, 8kHz, 320 samples/chunk, base64
- Output to telco: WAV, 8kHz, mono, base64

---

### T1-3: TTS resample to 8kHz  _(speech/tts_synthesizer.py)_

TTS currently outputs 22050Hz. Add a `resample_for_telco` op at the end:

```python
# speech/tts_synthesizer.py — add op, keep existing tts_pipeline unchanged
@op
def resample_for_telco(audio: np.ndarray) -> dict:
    import soxr
    return {"audio": soxr.resample(audio.astype(np.float32), 22050, 8000)}
```

In `ws_callbot_pipeline`, chain it after `tts_pipeline`:
```python
tts      = tts_pipeline(text=educa_agent["response"])
tts_8khz = resample_for_telco(audio=tts["audio"])
START >> ... >> tts >> tts_8khz >> END
```

`callbot_pipeline` (WAV test) keeps using `tts["audio"]` at 22050Hz — no change.

---

### T1-4: `server/http_server.py`

**Port**: 9923

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/health` | GET | `{"status": "healthy", "redis_status": "n/a", "version": "..."}` |
| `/version` | GET | version info |
| `/api/v1/customer-info` | POST | Store script_data for customer_id |
| `/api/v1/customer-info/upload-xlsx` | POST | Bulk upload from Excel |
| `/api/v1/calls/summary/batch` | POST | Retrieve call logs |

**`POST /api/v1/customer-info`**:
```json
{"Customer_id": "c001", "student_name": "Minh", "class_time": "19:00",
 "program_name": "AI CLASS", "hotline": "1900636464", "agent_name": "Linh"}
```
Stored in `data/customers/{customer_id}.json`.

**`POST /api/v1/calls/summary/batch`**:
```json
{"call_ids": ["call_001", "call_002"], "date": "2025-04-06"}
```
Response: reads from `logs/educa_reminder/{YYYYMMDD}/calls.jsonl`.

---

### T1-5: `server/call_logger.py`

Builds call_result **post-call** from `handle.state` (not inside the graph).
`build_call_result` from `agents/educa_reminder/ops/call_result.py` is called
as a **plain function** (no longer an `@op`).

```python
from agents.educa_reminder.ops.call_result import build_call_result

def build_log_summary(handle_state, script_data, call_id) -> dict:
    """Build log_summary for hangup message + JSONL storage.

    Reads final state/intent from handle_state, calls build_call_result()
    to get ARId/Comment/report_result, then assembles the summary.
    """
    # Extract final values from pipeline state
    final_state = ...   # from handle_state
    final_intent = ...  # from handle_state
    conversation_history = ...  # from handle_state

    # Build CRM data (same logic as original, just called post-call)
    cr = build_call_result(
        current_state=final_state,
        intent=final_intent,
        previous_state=...,
        customer_speech=...,
        customer_confirmed=...,
        new_phone_number=...,
        script_data=script_data,
    )

    return {
        "call_id": call_id,
        "action_code": cr["call_result"]["ARId"],
        "end_reason": final_state,
        "duration_seconds": ...,
        "transcript": [
            {"speaker": "agent",    "text": "...", "intent": None},
            {"speaker": "customer", "text": "...", "intent": "busy"},
        ],
        "call_result": cr["call_result"],
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }

def write(call_id: str, log_summary: dict):
    """Append to logs/educa_reminder/{YYYYMMDD}/calls.jsonl"""
    date_dir = datetime.utcnow().strftime("%Y%m%d")
    path = Path(config.LOG_BASE_DIR) / "educa_reminder" / date_dir / "calls.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(log_summary, ensure_ascii=False) + "\n")
```

Original flow comparison:
| Original educa-reminder | Hush refactor |
|------------------------|---------------|
| `state_manager.update_call_result()` per-turn in graph | `build_call_result()` once, post-call in `call_logger` |
| Result stored in Redis `call_result:{call_id}` | No Redis — built from `handle.state` |
| `conversation_logger.export_to_s3()` | Future: S3 upload in call_logger |
| `log_summary` in hangup WS message | Same: `log_summary` in hangup message |

---

### T1-6: `agents/educa_reminder/workflow.py` — remove `crm` from graph

**Current wiring** (broken — `crm` fans out from `trans`, creates two exit paths):
```python
m_intent >> trans >> [gen_rule, crm]
crm >> END                              # ← isolated exit path
gen_rule >> router2
router2 >> gen_llm >> ~m_response >> END
router2 >> skip_gen >> ~m_response
```

**New wiring** (clean — `crm` removed, `m_response >> END` auto-forwards `response`):
```python
# ── Wiring ──
START >> norm >> detect >> router1
router1 >> ctx >> classify >> ~m_intent
router1 >> skip_cls >> ~m_intent
m_intent >> trans >> gen_rule
gen_rule >> router2
router2 >> gen_llm >> ~m_response >> END
router2 >> skip_gen >> ~m_response
```

**Changes**:
1. Delete `crm = build_call_result(...)` instantiation (lines 116-124)
2. Delete `crm >> END` wiring
3. Delete `from agents.educa_reminder.ops.call_result import build_call_result` import
4. Remove `[gen_rule, crm]` fan-out → just `trans >> gen_rule`
5. `m_response >> END` auto-forwards `response` — no explicit PARENT mapping needed

**In `call_result.py`**: remove `@op` decorator, keep as plain function for `call_logger`.

---

## TIER 2 — Important

### T2-1: Interrupt — already handled by T1-2 design

The `monitor_handle()` loop in ws_server watches the ExecutionHandle stream.
When `vad` yields (new speech detected) while `send_active=True`, it sets
`interrupt_event` — send_loop detects this after finishing current audio chunk
and sends `{"event": "interrupt"}` to client.

Config: `USE_INTERRUPT=false` by default. When false, send_loop ignores the event
and plays audio to completion.

**No Hush changes needed.** ExecutionHandle is the exact analogue of Pipecat's
frame transport — frames flow out of the pipeline and the WS layer reacts to them.

---

### T2-2: Silence detection  _(speech/vad_detector.py)_

Add `silence_timeout` param to VadDetector. If no speech detected for N seconds,
yield a synthetic silent segment so the pipeline can handle the inactivity:

```python
class VadDetector(BaseOp):
    def __init__(self, ..., silence_timeout: float = 0.0):
        self.silence_timeout = silence_timeout  # 0 = disabled

    def _init_vad_buffers(self):
        b = { ..., "last_speech_time": time.time() }
        return b

    async def _process(self, audio, cmc_time, recv_time):
        b = _active_vad_buffers.get()
        # ... existing VAD logic ...
        if speech_detected:
            b["last_speech_time"] = time.time()
            yield {"speech_audio": segment, "is_silence": False}
        elif self.silence_timeout > 0:
            elapsed = time.time() - b["last_speech_time"]
            if elapsed >= self.silence_timeout:
                b["last_speech_time"] = time.time()  # reset timer
                yield {"speech_audio": np.zeros(160, dtype=np.float32), "is_silence": True}
```

In `ws_callbot_pipeline`:
```python
import random
PARENT.shared(..., silence_timeout=random.uniform(6.5, 8.5))
vad = VadDetector(..., silence_timeout=PARENT["silence_timeout"])
```

`quick_detect` already handles empty transcript → `silent` intent → state machine loops.
`silence_timeout=0` (default) keeps existing behavior — no change to test pipeline.

---

### T2-3: `server/config.py`

```python
WS_PORT                    = int(os.getenv("WS_API_PORT", "9922"))
HTTP_PORT                  = int(os.getenv("HTTP_API_PORT", "9923"))
SERVER_DECODE_BASE64        = os.getenv("SERVER_DECODE_BASE64", "true") == "true"
USE_INTERRUPT               = os.getenv("USE_INTERRUPT", "false") == "true"
INTERRUPT_STREAMING_TIMEOUT = float(os.getenv("INTERRUPT_STREAMING_TIMEOUT", "2.0"))
SILENCE_THRESHOLD_MIN       = float(os.getenv("SILENCE_THRESHOLD_MIN", "6.5"))
SILENCE_THRESHOLD_MAX       = float(os.getenv("SILENCE_THRESHOLD_MAX", "8.5"))
HEARTBEAT_INTERVAL_S        = 3.0
HANGUP_FLUSH_TIMEOUT_S      = 5.0
AGENT_SAMPLE_RATE           = int(os.getenv("AGENT_SAMPLE_RATE", "8000"))
TTS_SAMPLE_RATE             = 22050
LOG_BASE_DIR                = os.getenv("LOG_BASE_DIR", "logs")
CUSTOMER_INFO_DIR           = os.getenv("CUSTOMER_INFO_DIR", "data/customers")
MAX_CALL_DURATION_S         = int(os.getenv("MAX_CALL_DURATION", "600"))
```

---

## Implementation Status

```
Phase 1 — Core pipeline                              ✅ DONE
  1. T1-6  Remove crm from workflow.py               ✅
  2. T1-1  ws_source + ws_callbot_pipeline            ✅
  3. T1-3  Add resample_for_telco to tts_pipeline     ✅
  4. T2-3  server/config.py                           ✅

Phase 2 — WebSocket server                           ✅ DONE
  5. T1-5  server/call_logger.py                      ✅
  6. T1-4a server/customer_store.py                   ✅
  7. T1-2  server/ws_server.py                        ✅
          (recv_loop, monitor_handle, send_loop, heartbeat, hangup)

Phase 3 — HTTP + silence + entry point               ✅ DONE
  8. T1-4b server/http_server.py                      ✅
  9. T2-2  silence_timeout in VadDetector              ✅
  10. T2-1 Interrupt wiring in send_loop              ✅ (USE_INTERRUPT=false default)
  11. main.py entry point (WS:9922 + HTTP:9923)       ✅

TODO — Integration testing
  12. scripts/test_ws_client.py (mock CMC client)
  13. End-to-end test with live Triton + LLM
```

---

## Integration Test Script

After Phase 2, test with a mock CMC client:

```bash
# scripts/test_ws_client.py — sends WAV file as if CMC
uv run python scripts/test_ws_client.py \
  --wav tests/speech/audio/04_busy.wav \
  --call-id test_001 \
  --customer-id 42 \
  --phone 0912345678

# Expect:
#   → heartbeat messages every 3s
#   → media message with agent audio after first speech segment
#   → hangup with log_summary containing action_code at end
```

---

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **No Redis** | Original uses Redis as cross-process bus. Hush is single-process async — `asyncio.Queue` replaces all Redis queues. Simpler, faster, no infra. |
| **One `engine.start()` per call** | The pipeline runs for the full call lifetime. `ws_source` keeps yielding until `None` sentinel. All `PARENT.shared()` state persists across turns automatically. |
| **`ExecutionHandle` as interrupt bus** | Streaming handle frames from root-level ops replaces Pipecat's frame transport. VAD frame detected in `monitor_handle()` → interrupt signal → `send_loop`. No Hush changes needed. |
| **WAV pipeline stays** | `callbot_pipeline(wav_path=...)` keeps working for unit tests. Production uses `ws_callbot_pipeline(audio_queue=...)`. |
| **`crm` removed from graph** | `build_call_result()` is a post-call concern — original only uses it for CRM mapping (ARId/Comment) at call end. Now called as plain function in `call_logger.py` from `handle.state`. Simplifies workflow graph (single `m_response >> END` path). |
| **silence_timeout + resample in callbot** | Both are voice/telco specifics — they live in `callbot-engine-hush/speech/`, not Hush-ai. |
| **Root-level frame monitoring** | Only root ops produce ExecutionHandle frames. `"educa_agent"` frame carries all nested outputs (response, intent, new_state). `"tts_8khz"` frame carries resampled audio. Terminal state detection waits for TTS to finish before sending sentinels. |
