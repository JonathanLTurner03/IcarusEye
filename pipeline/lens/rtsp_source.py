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


def _gstreamer_available() -> bool:
    return "GStreamer" in cv2.getBuildInformation()


def _is_jetson() -> bool:
    """Return True only when the Jetson HW decoder device is present and mapped."""
    return os.path.exists("/dev/nvhost-nvdec")


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

        # Try backends in order: Jetson hardware → GStreamer software → FFmpeg
        self._cap = (
            self._open_gstreamer_hw() or
            self._open_gstreamer_sw() or
            self._open_ffmpeg()
        )

        if not self._cap or not self._cap.isOpened():
            raise RuntimeError(f"RTSPSource: could not open {self._uri}")

        actual_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._opened = True
        logger.info("RTSPSource opened: {} — {}x{}", self._uri, actual_w, actual_h)

    def _open_gstreamer_hw(self) -> Optional[cv2.VideoCapture]:
        """
        GStreamer pipeline using Jetson hardware decoder (nvv4l2decoder).
        Hardcodes H.264; swap rtph264depay/h264parse for rtph265depay/h265parse
        if your stream uses HEVC.
        """
        if not _gstreamer_available() or not _is_jetson():
            return None
        pipeline = (
            f"rtspsrc location={self._uri} latency=50 protocols=tcp ! "
            "rtph264depay ! h264parse ! "
            "nvv4l2decoder ! "
            "nvvidconv ! video/x-raw,format=BGRx ! "
            "videoconvert ! video/x-raw,format=BGR ! "
            "appsink drop=true max-buffers=2 sync=false"
        )
        cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        if cap.isOpened():
            logger.info("RTSPSource: GStreamer hardware pipeline (nvv4l2decoder)")
            return cap
        cap.release()
        return None

    def _open_gstreamer_sw(self) -> Optional[cv2.VideoCapture]:
        """GStreamer pipeline with software H.264 decoder (non-Jetson GStreamer)."""
        if not _gstreamer_available():
            return None
        pipeline = (
            f"rtspsrc location={self._uri} latency=100 protocols=tcp ! "
            "rtph264depay ! h264parse ! avdec_h264 ! "
            "videoconvert ! video/x-raw,format=BGR ! "
            "appsink drop=true max-buffers=2 sync=false"
        )
        cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        if cap.isOpened():
            logger.info("RTSPSource: GStreamer software pipeline (avdec_h264)")
            return cap
        cap.release()
        return None

    def _open_ffmpeg(self) -> Optional[cv2.VideoCapture]:
        """FFmpeg fallback with forced TCP transport."""
        logger.info("RTSPSource: GStreamer unavailable, using FFmpeg for {}", self._uri)
        prev = os.environ.get("OPENCV_FFMPEG_CAPTURE_OPTIONS")
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
        cap = cv2.VideoCapture(self._uri)
        if prev is not None:
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = prev
        else:
            os.environ.pop("OPENCV_FFMPEG_CAPTURE_OPTIONS", None)
        # Buffer 3 frames for B-frame reordering
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)
        if cap.isOpened():
            return cap
        cap.release()
        return None

    def read(self) -> Optional[Frame]:
        if not self._cap or not self._opened:
            return None

        ok, frame = self._cap.read()
        if not ok:
            logger.warning("RTSPSource: read failed — stream may have dropped")
            return None

        # Copy immediately — the backend may reuse the internal decode buffer on
        # the next read(), which would corrupt any frame still sitting in a queue.
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
