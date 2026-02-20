"""
IcarusEye v2 — V4L2 Capture Source
pipeline/lens/v4l2_source.py

Reads frames from a V4L2 device (USB HDMI capture card) via OpenCV.
Used in production on the Jetson with /dev/capture0 (udev symlink).

For full hardware-accelerated GStreamer pipelines on Jetson, this can be
swapped for GStreamerURISource using a v4l2src pipeline string — but OpenCV
V4L2 is sufficient for Phase 1 and works without GStreamer Python bindings.
"""

from __future__ import annotations

from typing import Optional

import cv2
from loguru import logger

from pipeline.lens.base import CaptureSource, Frame
from pipeline.shared.config import CaptureConfig


class V4L2Source(CaptureSource):
    """
    Captures from a V4L2 device via OpenCV (e.g. USB HDMI capture card).

    Config fields used:
        capture.device    — device path, e.g. /dev/capture0
        capture.width     — capture width
        capture.height    — capture height
        capture.framerate — target framerate (hint to driver, not guaranteed)
        capture.format    — V4L2 fourcc format string, e.g. MJPG
    """

    def __init__(self, cfg: CaptureConfig):
        super().__init__(name="v4l2")
        self._device    = cfg.device
        self._width     = cfg.width
        self._height    = cfg.height
        self._framerate = cfg.framerate
        self._format    = cfg.format
        self._cap: Optional[cv2.VideoCapture] = None

    # ── CaptureSource interface ───────────────────────────────────────────────

    def open(self) -> None:
        self._cap = cv2.VideoCapture(self._device, cv2.CAP_V4L2)
        if not self._cap.isOpened():
            raise RuntimeError(
                f"V4L2Source: could not open {self._device} — "
                "check udev symlink and that the capture card is connected"
            )

        # Set capture properties — driver may clamp to nearest supported value
        if self._format:
            fourcc = cv2.VideoWriter_fourcc(*self._format)
            self._cap.set(cv2.CAP_PROP_FOURCC, fourcc)
        if self._width:
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
        if self._height:
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
        if self._framerate:
            self._cap.set(cv2.CAP_PROP_FPS, self._framerate)

        # Read back actual values the driver accepted
        actual_w   = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h   = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self._cap.get(cv2.CAP_PROP_FPS)

        self._opened = True
        logger.info(
            "V4L2Source opened: {} — {}x{} @ {:.1f}fps (requested {}x{} @ {}fps {})",
            self._device, actual_w, actual_h, actual_fps,
            self._width, self._height, self._framerate, self._format,
        )

        if actual_w != self._width or actual_h != self._height:
            logger.warning(
                "V4L2Source: driver returned {}x{}, requested {}x{} — "
                "check your capture card's supported resolutions",
                actual_w, actual_h, self._width, self._height,
            )

    def read(self) -> Optional[Frame]:
        if not self._cap or not self._opened:
            return None

        ok, frame = self._cap.read()
        if not ok:
            logger.warning("V4L2Source: read failed — capture card may have disconnected")
            return None

        return self._make_frame(frame, source_id=f"v4l2:{self._device}")

    def close(self) -> None:
        if self._cap:
            self._cap.release()
            self._cap = None
        self._opened = False
        logger.info("V4L2Source closed: {}", self._device)