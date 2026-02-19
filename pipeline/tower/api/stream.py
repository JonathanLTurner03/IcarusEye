"""
Stream API serves HLS playlist and segments, plus source control.
"""
import os
from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

router = APIRouter()


def _segment_dir(request: Request) -> str:
    return os.getenv("HLS_SEGMENT_DIR", "/tmp/icaruseye_hls")


@router.get("/index.m3u8")
async def playlist(request: Request):
    path = os.path.join(_segment_dir(request), "index.m3u8")
    if not os.path.exists(path):
        raise HTTPException(status_code=503, detail="Stream not ready yet")
    return FileResponse(
        path,
        media_type="application/vnd.apple.mpegurl",
        headers={"Cache-Control": "no-cache"},
    )


@router.get("/{segment}")
async def segment(segment: str, request: Request):
    if not segment.endswith(".ts"):
        raise HTTPException(status_code=400, detail="Invalid segment")
    path = os.path.join(_segment_dir(request), segment)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Segment not found")
    return FileResponse(path, media_type="video/mp2t")


@router.get("/status")
async def status(request: Request):
    transcoder = request.app.state.transcoder
    return JSONResponse({
        "running": transcoder.is_running if transcoder else False,
        "source":  transcoder.source if transcoder else None,
    })


class SourceUpdate(BaseModel):
    source: str


@router.post("/source")
async def change_source(body: SourceUpdate, request: Request):
    """Hot-swap the video source without restarting Tower."""
    transcoder = request.app.state.transcoder
    if not transcoder:
        raise HTTPException(status_code=503, detail="Transcoder not initialized")
    await transcoder.change_source(body.source)
    return {"status": "ok", "source": body.source}