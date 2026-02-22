"""
IcarusEye v2 — RTSP / Generic URI Capture Source
pipeline/lens/rtsp_source.py

Reads frames from an RTSP stream or any GStreamer URI via OpenCV.
Also handles generic GStreamer pipeline strings when type=gstreamer_uri.

Examples:
    rtsp://192.168.1.x:554/stream     — IP camera
    rtp://127.0.0.1:5000              — RTP stream
"""

from __future__ import annotations

import os
from typing import Optional

import cv2
from loguru import logger

from pipeline.lens.base import CaptureSource, Frame
from pipeline.shared.config import CaptureConfig


class RTSPSource(CaptureSource):
    """
    Captures from an RTSP stream or generic URI via OpenCV.

    Config fields used:
        capture.uri       — full URI string
        capture.width     — resize output width  (0 = native)
        capture.height    — resize output height (0 = native)
    """

    def __init__(self, cfg: CaptureConfig):
        super().__init__(name="rtsp")
        self._uri    = cfg.uri
        self._width  = cfg.width
        self._height = cfg.height
        self._cap: Optional[cv2.VideoCapture] = None

    # ── CaptureSource interface ───────────────────────────────────────────────

    def open(self) -> None:
        if not self._uri:
            raise ValueError("RTSPSource: capture.uri must be set in config")

        # Force TCP transport so FFmpeg doesn't use UDP (avoids packet-loss artifacts).
        # Must be set before VideoCapture() is called; restored afterward so other
        # sources are unaffected.
        _prev = os.environ.get("OPENCV_FFMPEG_CAPTURE_OPTIONS")
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

        self._cap = cv2.VideoCapture(self._uri, cv2.CAP_FFMPEG)

        if _prev is not None:
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = _prev
        else:
            os.environ.pop("OPENCV_FFMPEG_CAPTURE_OPTIONS", None)

        if not self._cap.isOpened():
            raise RuntimeError(f"RTSPSource: could not open {self._uri}")

        # Buffer 3 frames so the H.264 decoder has room to reorder B-frames.
        # Buffer=1 caused frame corruption with many RTSP sources.
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)

        actual_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self._opened = True
        logger.info("RTSPSource opened (TCP): {} — {}x{}", self._uri, actual_w, actual_h)

    def read(self) -> Optional[Frame]:
        if not self._cap or not self._opened:
            return None

        ok, frame = self._cap.read()
        if not ok:
            logger.warning("RTSPSource: read failed — stream may have dropped")
            return None

        # Copy immediately — OpenCV's FFmpeg backend may reuse the internal decode
        # buffer on the next read(), which would corrupt any queued frame that still
        # holds a reference to that memory.
        frame = frame.copy()

        if self._width and self._height:
            frame = cv2.resize(frame, (self._width, self._height))

        return self._make_frame(frame, source_id=f"rtsp:{self._uri}")

    def close(self) -> None:
        if self._cap:
            self._cap.release()
            self._cap = None
        self._opened = False
        logger.info("RTSPSource closed: {}", self._uri)