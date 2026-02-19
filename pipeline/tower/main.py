"""
Tower is IcarusEye Dashboard & Control
FastAPI application serving the UI, HLS stream, and Redis-backed stats.
"""
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from loguru import logger

from pipeline.tower.api.stream import router as stream_router
from pipeline.tower.hls.transcoder import Transcoder

# ── Transcoder singleton ──
_transcoder: Transcoder = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _transcoder

    source      = os.getenv("HLS_SOURCE", "sample/sample.mp4")
    segment_dir = os.getenv("HLS_SEGMENT_DIR", "/tmp/icaruseye_hls")

    _transcoder         = Transcoder(source=source, segment_dir=segment_dir)
    app.state.transcoder = _transcoder

    await _transcoder.start()
    logger.info(f"Tower started — HLS source: {source}")

    yield

    await _transcoder.stop()
    logger.info("Tower shutdown")


# ── App ──
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