"""
IcarusEye v2 — Capture Factory
pipeline/lens/factory.py

Creates the correct CaptureSource based on pipeline config.
Swapping capture sources is a config/env change only — no code changes.

    DEV_MODE=true   → FileSource  (reads sample.mp4)
    CAPTURE_TYPE=v4l2  → V4L2Source (USB capture card, default for Jetson)
    CAPTURE_TYPE=rtsp  → RTSPSource (IP camera / network stream)
"""

from __future__ import annotations

from loguru import logger

from pipeline.lens.base import CaptureSource
from pipeline.lens.file_source import FileSource
from pipeline.lens.rtsp_source import RTSPSource
from pipeline.lens.v4l2_source import V4L2Source
from pipeline.shared.config import CaptureConfig

# Map config type strings → source classes
_REGISTRY: dict[str, type[CaptureSource]] = {
    "file":          FileSource,
    "v4l2":          V4L2Source,
    "rtsp":          RTSPSource,
    "gstreamer_uri": RTSPSource,  # RTSPSource handles generic GStreamer URIs too
}


def create(cfg: CaptureConfig) -> CaptureSource:
    """
    Instantiate and return the correct CaptureSource for the given config.

    Raises ValueError for unknown capture types.
    Does NOT call .open() — the caller is responsible for lifecycle.
    """
    source_type = cfg.type.lower()
    cls = _REGISTRY.get(source_type)

    if cls is None:
        valid = ", ".join(_REGISTRY)
        raise ValueError(
            f"Unknown capture type '{source_type}'. Valid types: {valid}"
        )

    logger.info("CaptureFactory: creating {} source", source_type)
    return cls(cfg)