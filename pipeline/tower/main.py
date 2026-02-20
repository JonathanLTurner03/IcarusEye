"""
IcarusEye v2 — Tower Service
pipeline/tower/main.py

Dashboard and stream server.
Connects to the pipeline service via Unix sockets to receive MJPEG frames.
Serves the UI, MJPEG feeds, and Redis-backed stats.
"""

import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from loguru import logger

from pipeline.tower.api.stream import router as stream_router
from pipeline.tower.mjpeg.receiver import FrameReceiver
from pipeline.shared.redis_client import Channels, Subscriber
from pipeline.frame_server import RAW_PORT, ANNOTATED_PORT


@asynccontextmanager
async def lifespan(app: FastAPI):
    redis_host    = os.getenv("REDIS_HOST", "redis")
    redis_port    = int(os.getenv("REDIS_PORT", 6379))
    pipeline_host = os.getenv("PIPELINE_HOST", "pipeline")

    # ── Frame receivers — connect to pipeline TCP frame servers ───────────────
    raw_receiver = FrameReceiver(pipeline_host, RAW_PORT, "raw")
    ann_receiver = FrameReceiver(pipeline_host, ANNOTATED_PORT, "annotated")
    raw_receiver.start()
    ann_receiver.start()
    app.state.raw_receiver       = raw_receiver
    app.state.annotated_receiver = ann_receiver

    # ── Redis stats subscriber ────────────────────────────────────────────────
    app.state.pipeline_stats: dict[str, dict] = {}

    def _on_stats(channel: str, data: dict) -> None:
        source_name = data.get("source", "unknown")
        data["_received_at"] = time.time()
        app.state.pipeline_stats[source_name] = data
        logger.debug("Tower stats from {}: fps={}", source_name, data.get("fps"))

    subscriber = Subscriber(host=redis_host, port=redis_port)
    subscriber.on(Channels.STATS, _on_stats)
    subscriber.start()
    app.state.subscriber = subscriber

    logger.info("Tower started — awaiting pipeline connection")

    yield

    # ── Shutdown ──────────────────────────────────────────────────────────────
    raw_receiver.stop()
    ann_receiver.stop()
    subscriber.stop()
    logger.info("Tower shutdown")


app = FastAPI(title="IcarusEye Tower", version="2.0.0", lifespan=lifespan)
app.include_router(stream_router, prefix="/stream")

static_dir = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/")
async def index():
    return FileResponse(os.path.join(static_dir, "index.html"))


@app.get("/health")
async def health():
    return {"status": "ok"}