"""
IcarusEye v2 — PyTorch Inference Engine
pipeline/hawk/pytorch_engine.py

Runs YOLO inference via Ultralytics on a .pt (or .onnx) model file.
Works on any platform — CPU, CUDA (x86 GPU), or MPS (Apple Silicon).
No TensorRT required.

Use this engine when:
  - Running on Mac / Apple Silicon (MPS acceleration)
  - Running on CPU-only hardware (dev, CI)
  - Running on x86 NVIDIA GPU without a pre-built .engine file

Activated automatically by factory.py when cfg.engine ends in .pt or .onnx.
Point to a model via MODEL_PATH env var or configs/pipeline.yaml:
    models:
      - engine: /app/models/yolo11n.pt
        mock: false
"""

from __future__ import annotations

import time
from pathlib import Path

from loguru import logger

from pipeline.hawk.base import DetectionResult, InferenceEngine, InferenceResult
from pipeline.shared.config import ModelConfig


def _best_device() -> str:
    """Pick the fastest available device: mps > cuda > cpu."""
    try:
        import torch
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


class PyTorchEngine(InferenceEngine):
    """
    YOLO inference engine using Ultralytics + PyTorch.

    Supports .pt and .onnx model files on any device.
    Auto-selects MPS on Apple Silicon, CUDA on NVIDIA, CPU otherwise.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__(name="pytorch")
        self._cfg         = cfg
        self._model       = None
        self._model_path  = Path(cfg.engine)
        self._device      = _best_device()
        self._class_names: dict[int, str] = {}

    def load(self) -> None:
        if not self._model_path.exists():
            raise FileNotFoundError(
                f"PyTorchEngine: model file not found at {self._model_path}\n"
                f"Download a YOLO model: yolo export model=yolo11n.pt  "
                f"or place a .pt file at that path."
            )

        try:
            from ultralytics import YOLO
        except ImportError as e:
            raise ImportError(
                "ultralytics not installed. Run: pip install ultralytics"
            ) from e

        logger.info(
            "PyTorchEngine: loading {} on device={}",
            self._model_path, self._device,
        )
        t0 = time.monotonic()

        self._model = YOLO(str(self._model_path), task="detect")

        # Warm-up pass so the first real frame isn't slow
        import numpy as np
        h = w = self._cfg.imgsz
        dummy = np.zeros((h, w, 3), dtype=np.uint8)
        self._model(dummy, device=self._device, verbose=False)

        self._class_names = self._model.names or {}
        load_ms = (time.monotonic() - t0) * 1000

        self._loaded = True
        logger.info(
            "PyTorchEngine loaded in {:.0f}ms — {} classes, device={}, model={}",
            load_ms, len(self._class_names), self._device, self._model_path.name,
        )

    def infer(self, frame) -> InferenceResult:
        if not self._loaded or self._model is None:
            return InferenceResult(
                frame_id=frame.frame_id,
                timestamp=time.monotonic(),
                engine_name="pytorch",
            )

        t0 = time.monotonic()

        try:
            results = self._model(
                frame.data,
                conf=self._cfg.confidence,
                classes=self._cfg.classes,
                device=self._device,
                verbose=False,
                imgsz=self._cfg.imgsz,
            )
        except Exception:
            logger.exception("PyTorchEngine: inference error on frame {}", frame.frame_id)
            return InferenceResult(
                frame_id=frame.frame_id,
                timestamp=time.monotonic(),
                engine_name="pytorch",
            )

        inference_ms = (time.monotonic() - t0) * 1000
        detections = self._parse_results(results, frame.width, frame.height)

        return InferenceResult(
            frame_id=frame.frame_id,
            timestamp=time.monotonic(),
            detections=detections,
            inference_ms=round(inference_ms, 2),
            engine_name=f"pytorch/{self._device}",
        )

    def unload(self) -> None:
        self._model = None
        self._loaded = False
        logger.info("PyTorchEngine unloaded")

    def _parse_results(self, results, frame_w: int, frame_h: int) -> list[DetectionResult]:
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf   = float(box.conf[0])
                cls_id = int(box.cls[0])
                label  = self._class_names.get(cls_id, str(cls_id))
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
