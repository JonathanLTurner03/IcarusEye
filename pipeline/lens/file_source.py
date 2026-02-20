"""
IcarusEye v2 — File Capture Source
pipeline/lens/file_source.py

Reads frames from a local video file via OpenCV.
Used in DEV_MODE — no hardware required.
Loops the file indefinitely when cfg.capture.loop is True.
Throttles to cfg.capture.framerate to simulate real capture hardware.
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
    Reads frames from a local video file.

    Config fields used:
        capture.uri       — path to video file
        capture.width     — resize output width  (0 = native)
        capture.height    — resize output height (0 = native)
        capture.framerate — target FPS throttle (0 = use file's native FPS)
        capture.loop      — restart from beginning when EOF is reached
    """

    def __init__(self, cfg: CaptureConfig):
        super().__init__(name="file")
        self._path      = cfg.uri or cfg.device
        self._width     = cfg.width
        self._height    = cfg.height
        self._loop      = cfg.loop
        self._cfg_fps   = cfg.framerate   # requested FPS from config (may be 0)
        self._target_fps = 0.0            # resolved after open()
        self._cap: Optional[cv2.VideoCapture] = None
        self._last_read: float = 0.0

    # ── CaptureSource interface ───────────────────────────────────────────────

    def open(self) -> None:
        path = Path(self._path)
        if not path.exists():
            raise FileNotFoundError(f"FileSource: video not found at {self._path}")

        self._cap = cv2.VideoCapture(str(path))
        if not self._cap.isOpened():
            raise RuntimeError(f"FileSource: OpenCV could not open {self._path}")

        native_fps = self._cap.get(cv2.CAP_PROP_FPS) or 30.0
        native_w   = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        native_h   = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frames     = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Throttle to config FPS if set, but never faster than native —
        # if native is lower than config, just use native.
        if self._cfg_fps:
            self._target_fps = min(float(self._cfg_fps), native_fps)
        else:
            self._target_fps = native_fps

        self._opened = True
        logger.info(
            "FileSource opened: {} — {}x{} @ {:.1f}fps native, {} frames, "
            "throttle={:.1f}fps, loop={}",
            self._path, native_w, native_h, native_fps, frames,
            self._target_fps, self._loop,
        )

    def read(self) -> Optional[Frame]:
        if not self._cap or not self._opened:
            return None

        # Throttle to target FPS
        if self._target_fps:
            target_interval = 1.0 / self._target_fps
            elapsed = time.monotonic() - self._last_read
            if elapsed < target_interval:
                time.sleep(target_interval - elapsed)

        ok, frame = self._cap.read()

        if not ok:
            if self._loop:
                logger.debug("FileSource: EOF — looping")
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, frame = self._cap.read()
                if not ok:
                    return None
            else:
                logger.info("FileSource: EOF — stopping")
                self._opened = False
                return None

        self._last_read = time.monotonic()

        if self._width and self._height:
            frame = cv2.resize(frame, (self._width, self._height))

        return self._make_frame(frame, source_id=f"file:{self._path}")

    def close(self) -> None:
        if self._cap:
            self._cap.release()
            self._cap = None
        self._opened = False
        logger.info("FileSource closed: {}", self._path)