"""Server configuration — env-configurable values."""

import os

WS_PORT = int(os.getenv("WS_API_PORT", "9922"))
HTTP_PORT = int(os.getenv("HTTP_API_PORT", "9923"))
SERVER_DECODE_BASE64 = os.getenv("SERVER_DECODE_BASE64", "true") == "true"
USE_INTERRUPT = os.getenv("USE_INTERRUPT", "false") == "true"
INTERRUPT_STREAMING_TIMEOUT = float(os.getenv("INTERRUPT_STREAMING_TIMEOUT", "2.0"))
SILENCE_THRESHOLD_MIN = float(os.getenv("SILENCE_THRESHOLD_MIN", "6.5"))
SILENCE_THRESHOLD_MAX = float(os.getenv("SILENCE_THRESHOLD_MAX", "8.5"))
HEARTBEAT_INTERVAL_S = 3.0
HANGUP_FLUSH_TIMEOUT_S = 5.0
AGENT_SAMPLE_RATE = int(os.getenv("AGENT_SAMPLE_RATE", "8000"))
TTS_SAMPLE_RATE = 22050
LOG_BASE_DIR = os.getenv("LOG_BASE_DIR", "logs")
CUSTOMER_INFO_DIR = os.getenv("CUSTOMER_INFO_DIR", "data/customers")
MAX_CALL_DURATION_S = int(os.getenv("MAX_CALL_DURATION", "600"))
