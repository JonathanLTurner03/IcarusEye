"""
IcarusEye v2 — Inference Engine Base Class
pipeline/hawk/base.py

All inference engines implement this interface.
The pipeline only talks to InferenceEngine — never to concrete implementations directly.

DetectionResult carries one detected object.
InferenceResult carries the full output for one frame, including timing.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class DetectionResult:
    """
    Single detected object from inference.

    bbox        — [x1, y1, x2, y2] normalized 0–1 relative to frame size
    label       — class name string (e.g. "person", "car")
    confidence  — detection confidence 0–1
    class_id    — integer class index
    """
    bbox:       list[float]   # [x1, y1, x2, y2], normalized 0–1
    label:      str
    confidence: float
    class_id:   int = 0


@dataclass
class InferenceResult:
    """
    Full inference output for one frame.

    frame_id     — matches Frame.frame_id from lens
    timestamp    — time.monotonic() at inference completion
    detections   — list of DetectionResult
    inference_ms — wall-clock inference time in milliseconds
    engine_name  — identifies the engine (e.g. "mock", "tensorrt")
    """
    frame_id:     int
    timestamp:    float
    detections:   list[DetectionResult] = field(default_factory=list)
    inference_ms: float = 0.0
    engine_name:  str = "unknown"


class InferenceEngine(ABC):
    """
    Abstract base class for all IcarusEye inference engines.

    Subclasses implement:
        load()     — load model weights / engine file, warm up if needed
        infer()    — run inference on a single BGR frame, return InferenceResult
        unload()   — release all GPU/CPU resources

    The pipeline uses the context manager protocol:

        with factory.create(cfg) as engine:
            result = engine.infer(frame)
    """

    def __init__(self, name: str):
        self._name = name
        self._loaded = False

    # ── Interface ─────────────────────────────────────────────────────────────

    @abstractmethod
    def load(self) -> None:
        """Load/initialise the model. Raises on failure."""
        ...

    @abstractmethod
    def infer(self, frame) -> InferenceResult:
        """
        Run inference on a Frame object.
        Returns InferenceResult with detections and timing.
        Must not raise — return empty detections on error.
        """
        ...

    @abstractmethod
    def unload(self) -> None:
        """Release all resources. Safe to call multiple times."""
        ...

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def name(self) -> str:
        return self._name

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    # ── Context manager ───────────────────────────────────────────────────────

    def __enter__(self) -> "InferenceEngine":
        self.load()
        return self

    def __exit__(self, *_) -> None:
        self.unload()