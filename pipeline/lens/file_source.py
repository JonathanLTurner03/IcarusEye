"""
IcarusEye v2 — File Capture Source
pipeline/lens/file_source.py

Reads frames from a local video file via OpenCV.
Used in DEV_MODE — no hardware required.
Loops the file indefinitely when cfg.capture.loop is True.
Always delivers at native FPS — frame skipping for inference is handled
upstream by the pipeline service, not here.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import cv2
from loguru import logger

from pipeline.lens.base import CaptureSource, Frame
from pipeline.shared.config import CaptureConfig


class FileSource(CaptureSource):
    """
    Reads frames from a local video file at native FPS.

    Config fields used:
        capture.uri       — path to video file
        capture.width     — resize output width  (0 = native)
        capture.height    — resize output height (0 = native)
        capture.framerate — recorded for reference but NOT used to throttle.
                            Frame skipping is handled by the pipeline service.
        capture.loop      — restart from beginning when EOF is reached
    """

    def __init__(self, cfg: CaptureConfig):
        super().__init__(name="file")
        self._path       = cfg.uri or cfg.device
        self._width      = cfg.width
        self._height     = cfg.height
        self._loop       = cfg.loop
        self._cfg_fps    = cfg.framerate
        self._native_fps = 0.0
        self._cap: Optional[cv2.VideoCapture] = None
        self._next_frame_time: float = 0.0  # absolute deadline for next frame

    @property
    def native_fps(self) -> float:
        return self._native_fps

    @property
    def cfg_fps(self) -> float:
        return float(self._cfg_fps) if self._cfg_fps else self._native_fps

    # ── CaptureSource interface ───────────────────────────────────────────────

    def open(self) -> None:
        path = Path(self._path)
        if not path.exists():
            raise FileNotFoundError(f"FileSource: video not found at {self._path}")

        self._cap = cv2.VideoCapture(str(path))
        if not self._cap.isOpened():
            raise RuntimeError(f"FileSource: OpenCV could not open {self._path}")

        self._native_fps = self._cap.get(cv2.CAP_PROP_FPS) or 30.0
        native_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        native_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frames   = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))

        self._opened = True
        self._next_frame_time = time.monotonic()  # deadline for first frame
        logger.info(
            "FileSource opened: {} — {}x{} @ {:.1f}fps, {} frames, loop={}",
            self._path, native_w, native_h, self._native_fps, frames, self._loop,
        )

    def read(self) -> Optional[Frame]:
        if not self._cap or not self._opened:
            return None

        # Deadline-based timing — sleep until the next frame is due
        # This accumulates correctly and doesn't drift like fixed-interval sleep
        interval = 1.0 / self._native_fps
        now = time.monotonic()
        if self._next_frame_time > now:
            time.sleep(self._next_frame_time - now)
        self._next_frame_time += interval

        ok, frame = self._cap.read()

        if not ok:
            if self._loop:
                logger.debug("FileSource: EOF — looping")
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self._next_frame_time = time.monotonic()  # reset deadline on loop
                ok, frame = self._cap.read()
                if not ok:
                    return None
            else:
                logger.info("FileSource: EOF — stopping")
                self._opened = False
                return None

        if self._width and self._height:
            frame = cv2.resize(frame, (self._width, self._height))

        return self._make_frame(frame, source_id=f"file:{self._path}")

    def close(self) -> None:
        if self._cap:
            self._cap.release()
            self._cap = None
        self._opened = False
        logger.info("FileSource closed: {}", self._path)