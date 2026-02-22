"""
IcarusEye v2 — Stream API
pipeline/tower/api/stream.py

MJPEG streaming endpoints and pipeline stats.
Replaces the HLS/ffmpeg approach entirely.

Endpoints:
  GET /stream/feed       — annotated feed (falls back to raw if no annotations yet)
  GET /stream/raw        — raw feed from lens (no detections drawn)
  GET /stream/annotated  — annotated feed from mark
  GET /stream/status     — pipeline connection status
  GET /stream/stats      — per-service Redis stats
"""

import asyncio
import time
from typing import AsyncGenerator

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from pipeline.shared.redis_client import ControlMessage, publish_control

router = APIRouter()

# MJPEG boundary string
BOUNDARY = "icaruseye_frame"


async def _mjpeg_generator(receiver, request: Request) -> AsyncGenerator[bytes, None]:
    """
    Async generator that yields MJPEG frames as fast as the pipeline produces them.
    Tracks frame count instead of identity comparison to detect new frames.
    """
    last_frame_count = -1

    while not await request.is_disconnected():
        current_count = receiver.frame_count

        if current_count == last_frame_count:
            # No new frame yet — yield briefly and check again
            await asyncio.sleep(0.01)
            continue

        jpeg = receiver.latest_jpeg
        if jpeg is None:
            await asyncio.sleep(0.05)
            continue

        last_frame_count = current_count

        yield (
            f"--{BOUNDARY}\r\n"
            f"Content-Type: image/jpeg\r\n"
            f"Content-Length: {len(jpeg)}\r\n"
            f"\r\n"
        ).encode() + jpeg + b"\r\n"


@router.get("/feed")
async def feed(request: Request):
    """Annotated feed — falls back to raw if annotated not connected."""
    ann = getattr(request.app.state, "annotated_receiver", None)
    raw = getattr(request.app.state, "raw_receiver", None)

    # Use annotated if connected and has frames, else raw
    receiver = (ann if (ann and ann.is_connected and ann.latest_jpeg is not None)
                else raw)

    if receiver is None:
        return JSONResponse({"error": "No pipeline connected"}, status_code=503)

    return StreamingResponse(
        _mjpeg_generator(receiver, request),
        media_type=f"multipart/x-mixed-replace; boundary={BOUNDARY}",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/raw")
async def raw_feed(request: Request):
    """Raw feed from lens — no detections drawn."""
    receiver = getattr(request.app.state, "raw_receiver", None)
    if receiver is None or not receiver.is_connected:
        return JSONResponse({"error": "Raw feed not connected"}, status_code=503)
    return StreamingResponse(
        _mjpeg_generator(receiver, request),
        media_type=f"multipart/x-mixed-replace; boundary={BOUNDARY}",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/annotated")
async def annotated_feed(request: Request):
    """Annotated feed from mark."""
    receiver = getattr(request.app.state, "annotated_receiver", None)
    if receiver is None or not receiver.is_connected:
        return JSONResponse({"error": "Annotated feed not connected"}, status_code=503)
    return StreamingResponse(
        _mjpeg_generator(receiver, request),
        media_type=f"multipart/x-mixed-replace; boundary={BOUNDARY}",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/status")
async def status(request: Request):
    raw = getattr(request.app.state, "raw_receiver", None)
    ann = getattr(request.app.state, "annotated_receiver", None)
    return JSONResponse({
        "raw_connected":        raw.is_connected if raw else False,
        "annotated_connected":  ann.is_connected if ann else False,
        "raw_frames":           raw.frame_count if raw else 0,
        "annotated_frames":     ann.frame_count if ann else 0,
    })


@router.get("/stats")
async def pipeline_stats(request: Request):
    """Return latest stats published by each pipeline service via Redis."""
    raw_stats: dict = getattr(request.app.state, "pipeline_stats", {})
    now = time.time()
    stale_after = 15.0

    result = {}
    for service, data in raw_stats.items():
        received_at = data.get("_received_at", 0)
        result[service] = {
            "fps":          data.get("fps", 0.0),
            "drop_rate":    data.get("drop_rate", 0.0),
            "inference_ms": data.get("inference_ms", 0.0),
            "extra":        data.get("extra", {}),
            "stale":        (now - received_at) > stale_after,
        }

    return JSONResponse(result)


_VALID_TYPES = {"file", "v4l2", "rtsp", "gstreamer_uri"}


@router.post("/control/source")
async def change_source(request: Request):
    """Send a swap_source control command to the pipeline via Redis."""
    body = await request.json()
    source_type = body.get("type", "")
    if source_type not in _VALID_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid source type '{source_type}'. Must be one of: {sorted(_VALID_TYPES)}",
        )

    msg = ControlMessage(
        command="swap_source",
        payload={
            "type":      source_type,
            "uri":       body.get("uri", ""),
            "device":    body.get("device", ""),
            "width":     int(body.get("width", 0)),
            "height":    int(body.get("height", 0)),
            "framerate": int(body.get("framerate", 30)),
            "loop":      bool(body.get("loop", True)),
        },
    )
    r = request.app.state.redis_client
    publish_control(msg, r)
    return JSONResponse({"status": "ok", "source_type": source_type})