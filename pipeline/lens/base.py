"""
IcarusEye v2 — Capture Source Base Class
pipeline/lens/base.py

All capture sources implement this interface.
The pipeline only talks to CaptureSource — never to concrete implementations directly.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class Frame:
    """
    A single captured video frame passed through the pipeline.

    data        — BGR image as numpy array (H, W, 3), uint8
    frame_id    — monotonically increasing counter from the capture source
    timestamp   — time.monotonic() at capture
    width       — frame width in pixels
    height      — frame height in pixels
    source      — capture source identifier e.g. "v4l2:/dev/capture0"
    """
    data:      np.ndarray
    frame_id:  int
    timestamp: float
    width:     int
    height:    int
    source:    str
    meta:      dict = field(default_factory=dict)  # optional extras


class CaptureSource(ABC):
    """
    Abstract base class for all IcarusEye capture sources.

    Subclasses implement:
        open()    — initialise the source (open device, connect, etc.)
        read()    — return the next Frame, or None if not ready
        close()   — release all resources

    The pipeline uses the context manager protocol:

        with factory.create(cfg) as src:
            while running:
                frame = src.read()
                if frame:
                    process(frame)
    """

    def __init__(self, name: str):
        self._name = name
        self._frame_id = 0
        self._opened = False

    # ── Interface ────────────────────────────────────────────────────────────

    @abstractmethod
    def open(self) -> None:
        """Open/initialise the capture source. Raises on failure."""
        ...

    @abstractmethod
    def read(self) -> Optional[Frame]:
        """
        Return the next frame, or None if no frame is available yet.
        Should not block indefinitely — return None and let the caller retry.
        """
        ...

    @abstractmethod
    def close(self) -> None:
        """Release all resources. Safe to call multiple times."""
        ...

    # ── Properties ───────────────────────────────────────────────────────────

    @property
    def name(self) -> str:
        return self._name

    @property
    def is_open(self) -> bool:
        return self._opened

    # ── Context manager ───────────────────────────────────────────────────────

    def __enter__(self) -> "CaptureSource":
        self.open()
        return self

    def __exit__(self, *_) -> None:
        self.close()

    # ── Helpers for subclasses ────────────────────────────────────────────────

    def _next_id(self) -> int:
        self._frame_id += 1
        return self._frame_id

    def _make_frame(self, data: np.ndarray, source_id: str) -> Frame:
        h, w = data.shape[:2]
        return Frame(
            data=data,
            frame_id=self._next_id(),
            timestamp=time.monotonic(),
            width=w,
            height=h,
            source=source_id,
        )