"""
IcarusEye v2 — TensorRT Inference Engine
pipeline/hawk/tensorrt_engine.py

Runs YOLO26 inference via Ultralytics on a TensorRT .engine file.
Used on Jetson (Phase 2+). Activated by setting model.mock=false in config.

The engine file is exported from a trained .pt checkpoint:
    yolo export model=best.pt format=engine device=0 int8=True imgsz=640

Key design decisions:
  - Uses Ultralytics YOLO() wrapper — handles preprocessing, postprocessing,
    NMS, and TensorRT warmup transparently
  - Normalises bboxes to 0–1 so mark never needs to know frame dimensions
  - Inference runs on the inference thread (not the main thread) — GPU context
    must be established in the same thread that calls load()
  - Returns empty InferenceResult on any error — never raises from infer()

Swapping mock → real engine is a config change only:
    models:
      - engine: models/yolo26s_visdrone.engine
        mock: false   # was true in DEV_MODE
"""

from __future__ import annotations

import time
from pathlib import Path

from loguru import logger

from pipeline.hawk.base import DetectionResult, InferenceEngine, InferenceResult
from pipeline.shared.config import ModelConfig


class TensorRTEngine(InferenceEngine):
    """
    YOLO26 TensorRT inference engine for NVIDIA Jetson.

    Requires:
      - ultralytics >= 8.3
      - TensorRT 8.x (included in JetPack 6.x)
      - A .engine file exported from a YOLO26 .pt checkpoint
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__(name="tensorrt")
        self._cfg        = cfg
        self._model      = None
        self._engine_path = Path(cfg.engine)
        self._class_names: list[str] = []

    # ── InferenceEngine interface ─────────────────────────────────────────────

    def load(self) -> None:
        """
        Load the TensorRT engine via Ultralytics YOLO.

        This must be called from the inference thread — CUDA contexts are
        thread-local and TensorRT will error if load() and infer() run in
        different threads.
        """
        if not self._engine_path.exists():
            raise FileNotFoundError(
                f"TensorRTEngine: engine file not found at {self._engine_path}\n"
                f"Export first: yolo export model=best.pt format=engine "
                f"device=0 int8=True imgsz={self._cfg.imgsz}"
            )

        try:
            from ultralytics import YOLO
        except ImportError as e:
            raise ImportError(
                "ultralytics not installed. Run: pip install ultralytics"
            ) from e

        logger.info("TensorRTEngine: loading {}", self._engine_path)
        t0 = time.monotonic()

        self._model = YOLO(str(self._engine_path), task="detect")

        # Warm-up run — TensorRT JIT-compiles kernels on first inference.
        # Run a dummy forward pass so the first real frame isn't slow.
        import numpy as np
        h = w = self._cfg.imgsz
        dummy = np.zeros((h, w, 3), dtype=np.uint8)
        self._model(dummy, verbose=False)

        self._class_names = self._model.names  # dict {int: str}
        load_ms = (time.monotonic() - t0) * 1000

        self._loaded = True
        logger.info(
            "TensorRTEngine loaded in {:.0f}ms — {} classes, engine={}",
            load_ms, len(self._class_names), self._engine_path.name,
        )

    def infer(self, frame) -> InferenceResult:
        """
        Run inference on a Frame. Returns InferenceResult.
        Never raises — returns empty detections on error.
        """
        if not self._loaded or self._model is None:
            return InferenceResult(
                frame_id=frame.frame_id,
                timestamp=time.monotonic(),
                engine_name="tensorrt",
            )

        t0 = time.monotonic()

        try:
            results = self._model(
                frame.data,
                conf=self._cfg.confidence,
                classes=self._cfg.classes,  # None = all classes
                verbose=False,
                imgsz=self._cfg.imgsz,
            )
        except Exception:
            logger.exception("TensorRTEngine: inference error on frame {}", frame.frame_id)
            return InferenceResult(
                frame_id=frame.frame_id,
                timestamp=time.monotonic(),
                engine_name="tensorrt",
            )

        inference_ms = (time.monotonic() - t0) * 1000
        detections = self._parse_results(results, frame.width, frame.height)

        return InferenceResult(
            frame_id=frame.frame_id,
            timestamp=time.monotonic(),
            detections=detections,
            inference_ms=round(inference_ms, 2),
            engine_name="tensorrt",
        )

    def unload(self) -> None:
        self._model = None
        self._loaded = False
        logger.info("TensorRTEngine unloaded")

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _parse_results(self, results, frame_w: int, frame_h: int) -> list[DetectionResult]:
        """
        Convert Ultralytics Results to normalized DetectionResult list.
        Bboxes are normalized to 0–1 so mark is resolution-agnostic.
        """
        detections = []

        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue

            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf    = float(box.conf[0])
                cls_id  = int(box.cls[0])
                label   = self._class_names.get(cls_id, str(cls_id))

                detections.append(DetectionResult(
                    bbox=[
                        round(x1 / frame_w, 4),
                        round(y1 / frame_h, 4),
                        round(x2 / frame_w, 4),
                        round(y2 / frame_h, 4),
                    ],
                    label=label,
                    confidence=round(conf, 3),
                    class_id=cls_id,
                ))

        return detections