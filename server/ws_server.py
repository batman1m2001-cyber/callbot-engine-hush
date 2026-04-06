"""WebSocket server — bridges CMC telco calls to Hush pipeline.

Endpoint: ws://host:9922/ws/call?call_id=...&customer_id=...
Protocol matches original educa-reminder-agent for drop-in replacement.
"""

import asyncio
import base64
import json
import io
import logging
import time
import uuid
import struct

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query

from hush.core import Hush

from pipeline.callbot import ws_callbot_pipeline
from server import config, call_logger, customer_store

LOGGER = logging.getLogger(__name__)

app = FastAPI(title="Callbot Engine Hush — WS Server")


# ── Helpers ──────────────────────────────────────────────────────────────────

def now_ms() -> int:
    return int(time.time() * 1000)


async def send_json(ws: WebSocket, payload: dict):
    """Send JSON message to WebSocket, ignoring closed connection errors."""
    try:
        await ws.send_json(payload)
    except Exception:
        pass


def to_wav_8khz(audio: np.ndarray) -> bytes:
    """Convert float32/int16 audio array to WAV bytes (8kHz, mono, int16)."""
    if audio.dtype == np.float32:
        audio = np.clip(audio, -1.0, 1.0)
        audio = (audio * 32767).astype(np.int16)
    elif audio.dtype != np.int16:
        audio = audio.astype(np.int16)

    buf = io.BytesIO()
    n_samples = len(audio)
    data_size = n_samples * 2  # int16 = 2 bytes
    # WAV header
    buf.write(b"RIFF")
    buf.write(struct.pack("<I", 36 + data_size))
    buf.write(b"WAVE")
    buf.write(b"fmt ")
    buf.write(struct.pack("<IHHIIHH", 16, 1, 1, 8000, 16000, 2, 16))
    buf.write(b"data")
    buf.write(struct.pack("<I", data_size))
    buf.write(audio.tobytes())
    return buf.getvalue()


# ── WebSocket Handler ────────────────────────────────────────────────────────

@app.websocket("/ws/call")
async def ws_call(
    websocket: WebSocket,
    call_id: str = Query(...),
    phone_number: str = Query(...),
    customer_id: str = Query(...),
    agent_type: str = Query("educa_reminder"),
    customer_info: str = Query("{}"),
    lead_id: str = Query(""),
    campaign_id: str = Query(""),
):
    await websocket.accept()
    start_time = time.time()
    LOGGER.info(f"[{call_id}] WS connected: customer_id={customer_id} phone={phone_number}")

    # Load customer script_data
    script_data = customer_store.get(customer_id)
    if script_data is None:
        # Try parsing from query param
        try:
            script_data = json.loads(customer_info)
        except json.JSONDecodeError:
            script_data = {}
    script_data.setdefault("phone_number", phone_number)

    # Pipeline queues
    audio_queue: asyncio.Queue = asyncio.Queue()
    tts_queue: asyncio.Queue = asyncio.Queue()
    interrupt_event = asyncio.Event()
    send_active = False
    last_response = ""
    is_final_turn = False

    # Build and start pipeline
    engine = Hush(
        ws_callbot_pipeline(audio_queue=audio_queue, script_data=script_data),
        resources="resources.yaml",
    )
    handle = engine.start(inputs={})

    # ── Coroutines ───────────────────────────────────────────────────────

    async def monitor_handle():
        """Bridge: pipeline frames → tts_queue + interrupt signals.

        Only ROOT-LEVEL ops appear in the handle stream:
        source, audio, vad, stt_input, stt, denoise, educa_agent, update, tts, tts_8khz
        """
        nonlocal send_active, last_response, is_final_turn
        async for op_name, ctx, data in handle:
            # VAD: speech detected → interrupt signal
            if op_name == "vad" and data.get("speech_audio") is not None:
                if send_active:
                    interrupt_event.set()

            # educa_agent completed: capture response + detect terminal state
            if op_name == "educa_agent":
                if data.get("response") is not None:
                    last_response = data["response"]
                if data.get("new_state") in ("FINISH", "TRANSFER_HOTLINE"):
                    is_final_turn = True
                    if data["new_state"] == "TRANSFER_HOTLINE":
                        await send_json(websocket, {
                            "event": "transfer_hotline",
                            "call_id": call_id,
                            "timestamp": now_ms(),
                        })

            # TTS completed (after resample): queue audio for sending
            if op_name == "tts_8khz" and data.get("audio") is not None:
                await tts_queue.put({
                    "audio": data["audio"],
                    "text": last_response,
                    "duration": len(data["audio"]) / config.AGENT_SAMPLE_RATE,
                })
                # Only stop AFTER the final turn's TTS audio is queued
                if is_final_turn:
                    await tts_queue.put(None)
                    await audio_queue.put(None)

    async def recv_loop():
        """Receive audio from WebSocket → audio_queue."""
        try:
            async for msg_text in websocket.iter_text():
                try:
                    msg = json.loads(msg_text)
                except json.JSONDecodeError:
                    continue
                if msg.get("event") == "media":
                    media = msg.get("media", {})
                    audio_b64 = media.get("payload", "")
                    if not audio_b64:
                        continue
                    audio_bytes = base64.b64decode(audio_b64)
                    cmc_time = msg.get("timestamp", now_ms())
                    await audio_queue.put({
                        "audio_bytes": audio_bytes,
                        "cmc_time": cmc_time,
                    })
        except WebSocketDisconnect:
            LOGGER.info(f"[{call_id}] WS disconnected")
        except Exception as e:
            LOGGER.error(f"[{call_id}] recv_loop error: {e}")
        finally:
            await audio_queue.put(None)  # sentinel — stop ws_source

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
                "event": "media",
                "media_name": f"audio_{audio_id}.wav",
                "timestamp": now_ms(),
                "text": item["text"],
                "audio_dur": item["duration"],
                "media": {
                    "payload": wav_b64,
                    "is_sync": True,
                    "tracking_id": audio_id,
                },
            })
            send_active = False
            # If interrupted during send, notify client
            if config.USE_INTERRUPT and interrupt_event.is_set():
                await send_json(websocket, {
                    "event": "interrupt",
                    "call_id": call_id,
                    "audio_id": audio_id,
                    "timestamp": now_ms(),
                })

    async def heartbeat_loop():
        """Send heartbeat every N seconds."""
        try:
            while True:
                await asyncio.sleep(config.HEARTBEAT_INTERVAL_S)
                await send_json(websocket, {
                    "event": "heartbeat",
                    "call_id": call_id,
                    "timestamp": now_ms(),
                })
        except asyncio.CancelledError:
            pass

    # ── Run all loops ────────────────────────────────────────────────────

    tasks = [
        asyncio.create_task(recv_loop(), name=f"{call_id}-recv"),
        asyncio.create_task(send_loop(), name=f"{call_id}-send"),
        asyncio.create_task(heartbeat_loop(), name=f"{call_id}-heartbeat"),
        asyncio.create_task(monitor_handle(), name=f"{call_id}-monitor"),
    ]
    try:
        done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_EXCEPTION)
        for t in done:
            if t.exception():
                LOGGER.error(f"[{call_id}] Task {t.get_name()} failed: {t.exception()}")
    except Exception as e:
        LOGGER.error(f"[{call_id}] ws_call error: {e}")
    finally:
        # Cancel all tasks
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

        # Ensure pipeline stops
        await audio_queue.put(None)
        await tts_queue.put(None)

        # Build and send hangup
        end_time = time.time()
        try:
            log_summary = call_logger.build_log_summary(
                handle_state=handle.state,
                script_data=script_data,
                call_id=call_id,
                phone_number=phone_number,
                customer_id=customer_id,
                start_time=start_time,
                end_time=end_time,
            )
            await send_json(websocket, {
                "event": "hangup",
                "call_id": call_id,
                "log_summary": log_summary,
                "timestamp": now_ms(),
            })
            call_logger.write(call_id, log_summary)
        except Exception as e:
            LOGGER.error(f"[{call_id}] Failed to build log_summary: {e}")
            await send_json(websocket, {
                "event": "hangup",
                "call_id": call_id,
                "timestamp": now_ms(),
            })

        LOGGER.info(f"[{call_id}] Call ended, duration={end_time - start_time:.1f}s")
