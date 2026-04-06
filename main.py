"""Entry point — start WebSocket + HTTP servers."""

import asyncio
import logging
import os

import uvicorn
from dotenv import load_dotenv

load_dotenv()

from server import config

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)-7s %(name)s — %(message)s",
)
LOGGER = logging.getLogger(__name__)


async def main():
    ws_config = uvicorn.Config(
        "server.ws_server:app",
        host="0.0.0.0",
        port=config.WS_PORT,
        log_level="info",
    )
    http_config = uvicorn.Config(
        "server.http_server:app",
        host="0.0.0.0",
        port=config.HTTP_PORT,
        log_level="info",
    )

    ws_server = uvicorn.Server(ws_config)
    http_server = uvicorn.Server(http_config)

    LOGGER.info(f"Starting WS server on :{config.WS_PORT}")
    LOGGER.info(f"Starting HTTP server on :{config.HTTP_PORT}")

    await asyncio.gather(
        ws_server.serve(),
        http_server.serve(),
    )


if __name__ == "__main__":
    asyncio.run(main())
