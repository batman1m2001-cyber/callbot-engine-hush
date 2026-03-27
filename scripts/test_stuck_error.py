"""Reproduce: TritonOp build error → should crash, not stuck."""

import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ["LOG_LEVEL"] = "DEBUG"

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "../.env"))

from hush.core import Hush
from pipeline.callbot import callbot_pipeline

print("Building graph...")
try:
    wf = callbot_pipeline(
        wav_path="tests/speech/audio/03_confirm.wav",
        script_data={
            "student_name": "Minh", "class_time": "19:00",
            "program_name": "AI CLASS", "agent_name": "Linh",
            "hotline": "1900636464", "parent_name": "anh chị",
        },
    )
    print("Graph built OK")
except Exception as e:
    print(f"Graph build FAILED: {type(e).__name__}: {e}")
    sys.exit(1)

print("Running engine...")
try:
    engine = Hush(wf, env=os.path.join(os.path.dirname(__file__), "../.env"),
                  resources=os.path.join(os.path.dirname(__file__), "../resources.yaml"))
    result = asyncio.run(engine.run(inputs={}))
    print("DONE")
except Exception as e:
    print(f"Engine FAILED: {type(e).__name__}: {e}")
