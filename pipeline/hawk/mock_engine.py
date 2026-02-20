"""
IcarusEye v2 — Mock Inference Engine
pipeline/hawk/mock_engine.py

Produces synthetic detections without any real model.
Used in DEV_MODE so the full pipeline (hawk → mark → annotated feed) can be
validated without a TensorRT engine or GPU.

Behaviour:
  - Generates 1–4 random bounding boxes per frame
  - Boxes drift slowly across the frame over time (makes the UI feel alive)
  - Simulates configurable inference latency (default: 5ms)
  - Labels come from a small subset of VisDrone classes
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field

from loguru import logger

from pipeline.hawk.base import DetectionResult, InferenceEngine, InferenceResult
from pipeline.shared.config import ModelConfig


# VisDrone class subset — mirrors the real dataset labels hawk will eventually use
_VISDRONE_LABELS = [
    "pedestrian", "people", "bicycle", "car", "van",
    "truck", "tricycle", "awning-tricycle", "bus", "motor",
]

# How many ms to sleep to simulate inference time
_MOCK_LATENCY_MS = 5.0


@dataclass
class _DriftingBox:
    """A bounding box that drifts slowly across the frame."""
    cx: float         # centre x, 0–1
    cy: float         # centre y, 0–1
    w:  float         # width, 0–1
    h:  float         # height, 0–1
    vx: float         # velocity x per frame
    vy: float         # velocity y per frame
    label: str
    class_id: int
    confidence: float

    def step(self) -> None:
        self.cx = (self.cx + self.vx) % 1.0
        self.cy = (self.cy + self.vy) % 1.0

    def to_detection(self) -> DetectionResult:
        x1 = max(0.0, self.cx - self.w / 2)
        y1 = max(0.0, self.cy - self.h / 2)
        x2 = min(1.0, self.cx + self.w / 2)
        y2 = min(1.0, self.cy + self.h / 2)
        return DetectionResult(
            bbox=[round(x1, 4), round(y1, 4), round(x2, 4), round(y2, 4)],
            label=self.label,
            confidence=round(self.confidence, 3),
            class_id=self.class_id,
        )


def _random_box() -> _DriftingBox:
    w = random.uniform(0.05, 0.18)
    h = random.uniform(0.04, 0.12)
    idx = random.randrange(len(_VISDRONE_LABELS))
    speed = random.uniform(0.001, 0.004)
    angle = random.uniform(0, 2 * math.pi)
    return _DriftingBox(
        cx=random.uniform(w / 2, 1 - w / 2),
        cy=random.uniform(h / 2, 1 - h / 2),
        w=w,
        h=h,
        vx=speed * math.cos(angle),
        vy=speed * math.sin(angle),
        label=_VISDRONE_LABELS[idx],
        class_id=idx,
        confidence=random.uniform(0.55, 0.97),
    )


class MockEngine(InferenceEngine):
    """
    Mock inference engine for DEV_MODE.

    Returns synthetic drifting bounding boxes so the full pipeline
    (hawk → mark → annotated feed) can be tested without hardware.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__(name="mock")
        self._cfg       = cfg
        self._boxes:    list[_DriftingBox] = []
        self._n_boxes   = random.randint(2, 5)
        self._latency_s = _MOCK_LATENCY_MS / 1000.0

    # ── InferenceEngine interface ─────────────────────────────────────────────

    def load(self) -> None:
        self._boxes = [_random_box() for _ in range(self._n_boxes)]
        self._loaded = True
        logger.info(
            "MockEngine loaded — {} drifting boxes, {:.0f}ms simulated latency",
            self._n_boxes, _MOCK_LATENCY_MS,
        )

    def infer(self, frame) -> InferenceResult:
        t0 = time.monotonic()

        # Simulate inference time
        time.sleep(self._latency_s)

        # Advance all boxes
        for box in self._boxes:
            box.step()

        # Apply confidence threshold filter (mirrors real engine behaviour)
        detections = [
            b.to_detection()
            for b in self._boxes
            if b.confidence >= self._cfg.confidence
        ]

        inference_ms = (time.monotonic() - t0) * 1000

        return InferenceResult(
            frame_id=frame.frame_id,
            timestamp=time.monotonic(),
            detections=detections,
            inference_ms=round(inference_ms, 2),
            engine_name="mock",
        )

    def unload(self) -> None:
        self._boxes = []
        self._loaded = False
        logger.info("MockEngine unloaded")