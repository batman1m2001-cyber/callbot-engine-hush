"""Mock CMC WebSocket client — sends WAV audio to callbot WS server.

Usage:
    # Terminal 1: start server
    uv run python main.py

    # Terminal 2: run this client
    uv run python scripts/test_ws_client.py --wav tests/speech/audio/04_busy.wav
    uv run python scripts/test_ws_client.py --wav tests/speech/audio/01_student_joining.wav
"""

import argparse
import asyncio
import base64
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from urllib.parse import quote

import numpy as np
from scipy.io import wavfile

try:
    import websockets
except ImportError:
    print("Install websockets: uv add websockets")
    sys.exit(1)


CHUNK_SIZE = 320  # 40ms at 8kHz — matches telco
CHUNK_INTERVAL = 0.04  # 40ms between chunks


async def send_audio(ws, wav_path: str):
    """Read WAV, chunk it, send as base64 media events like CMC."""
    sr, audio = wavfile.read(wav_path)
    if sr != 8000:
        print(f"  WARNING: WAV is {sr}Hz, expected 8kHz")

    # Add 1s silence tail (like real telco stream)
    silence = np.zeros(int(1.0 * sr), dtype=audio.dtype)
    audio = np.concatenate([audio, silence])

    total_chunks = (len(audio) + CHUNK_SIZE - 1) // CHUNK_SIZE
    print(f"  Sending {total_chunks} chunks ({len(audio)/sr:.2f}s audio)")

    for i in range(0, len(audio), CHUNK_SIZE):
        chunk = audio[i: i + CHUNK_SIZE]
        if len(chunk) < CHUNK_SIZE:
            chunk = np.pad(chunk, (0, CHUNK_SIZE - len(chunk)))

        payload = base64.b64encode(chunk.tobytes()).decode()
        msg = {
            "event": "media",
            "timestamp": int(time.time() * 1000),
            "media": {
                "payload": payload,
                "tracking_id": f"t{i // CHUNK_SIZE}",
            },
        }
        await ws.send(json.dumps(msg))
        await asyncio.sleep(CHUNK_INTERVAL)

    print("  All audio sent, waiting for server responses...")


async def recv_messages(ws, results: dict):
    """Receive and print all server messages until hangup or disconnect."""
    try:
        async for msg_text in ws:
            msg = json.loads(msg_text)
            event = msg.get("event", "unknown")
            ts = msg.get("timestamp", "")

            if event == "heartbeat":
                results["heartbeats"] += 1
                print(f"  <- heartbeat #{results['heartbeats']}")

            elif event == "media":
                text = msg.get("text", "")[:80]
                dur = msg.get("audio_dur", 0)
                media = msg.get("media", {})
                payload_len = len(media.get("payload", ""))
                results["media_count"] += 1
                print(f"  <- media #{results['media_count']}: \"{text}\" dur={dur:.1f}s payload={payload_len} bytes")

            elif event == "interrupt":
                audio_id = msg.get("audio_id", "")
                results["interrupts"] += 1
                print(f"  <- interrupt audio_id={audio_id}")

            elif event == "transfer_hotline":
                results["transfer"] = True
                print(f"  <- transfer_hotline")

            elif event == "hangup":
                log_summary = msg.get("log_summary", {})
                action_code = log_summary.get("action_code", "N/A")
                end_reason = log_summary.get("end_reason", "N/A")
                duration = log_summary.get("duration_seconds", 0)
                transcript = log_summary.get("transcript", [])
                results["hangup"] = True
                print(f"  <- hangup: action_code={action_code} end_reason={end_reason} duration={duration:.1f}s")
                if transcript:
                    print(f"     transcript ({len(transcript)} turns):")
                    for t in transcript[:6]:
                        print(f"       [{t['speaker']}] {t['text'][:60]}")
                break

            else:
                print(f"  <- {event}: {json.dumps(msg)[:120]}")

    except websockets.exceptions.ConnectionClosed as e:
        print(f"  Connection closed: {e}")


async def run_test(
    wav_path: str,
    call_id: str = "test_001",
    customer_id: str = "42",
    phone: str = "0912345678",
    host: str = "localhost",
    port: int = 9922,
):
    """Run a single test call."""
    # Build customer_info as URL-encoded JSON
    customer_info = json.dumps({
        "Customer_id": customer_id,
        "student_name": "Minh",
        "class_time": "19:00",
        "program_name": "AI CLASS",
        "agent_name": "Linh",
        "company": "Edupia",
        "hotline": "1900636464",
        "phone_number": phone,
    })

    uri = (
        f"ws://{host}:{port}/ws/call"
        f"?call_id={call_id}"
        f"&phone_number={phone}"
        f"&customer_id={customer_id}"
        f"&agent_type=educa_reminder"
        f"&customer_info={quote(customer_info)}"
    )

    print(f"\n{'='*60}")
    print(f"TEST: {os.path.basename(wav_path)}")
    print(f"  URI: ws://{host}:{port}/ws/call?call_id={call_id}")
    print(f"  WAV: {wav_path}")

    results = {
        "heartbeats": 0,
        "media_count": 0,
        "interrupts": 0,
        "transfer": False,
        "hangup": False,
    }

    t0 = time.time()
    try:
        async with websockets.connect(uri, close_timeout=5) as ws:
            print("  Connected!")

            # Run sender and receiver concurrently
            send_task = asyncio.create_task(send_audio(ws, wav_path))
            recv_task = asyncio.create_task(recv_messages(ws, results))

            # Wait for sender to finish, then wait for hangup or timeout
            await send_task
            try:
                await asyncio.wait_for(recv_task, timeout=30)
            except asyncio.TimeoutError:
                print("  TIMEOUT: no hangup received within 30s")
                recv_task.cancel()

    except ConnectionRefusedError:
        print(f"  ERROR: cannot connect to ws://{host}:{port} — is the server running?")
        print(f"  Start it with: uv run python main.py")
        return

    elapsed = time.time() - t0

    print(f"\n  SUMMARY:")
    print(f"    Duration:    {elapsed:.1f}s")
    print(f"    Heartbeats:  {results['heartbeats']}")
    print(f"    Media msgs:  {results['media_count']}")
    print(f"    Interrupts:  {results['interrupts']}")
    print(f"    Transfer:    {results['transfer']}")
    print(f"    Hangup:      {results['hangup']}")
    print(f"{'='*60}")


async def main():
    parser = argparse.ArgumentParser(description="Mock CMC WebSocket client")
    parser.add_argument("--wav", required=True, help="Path to WAV file (8kHz int16)")
    parser.add_argument("--call-id", default="test_001", help="Call ID")
    parser.add_argument("--customer-id", default="42", help="Customer ID")
    parser.add_argument("--phone", default="0912345678", help="Phone number")
    parser.add_argument("--host", default="localhost", help="Server host")
    parser.add_argument("--port", type=int, default=9922, help="Server WS port")
    args = parser.parse_args()

    if not os.path.exists(args.wav):
        print(f"ERROR: WAV file not found: {args.wav}")
        sys.exit(1)

    await run_test(
        wav_path=args.wav,
        call_id=args.call_id,
        customer_id=args.customer_id,
        phone=args.phone,
        host=args.host,
        port=args.port,
    )


if __name__ == "__main__":
    asyncio.run(main())
