"""
IcarusEye v2 — Inference Engine Factory
pipeline/hawk/factory.py

Selects the correct InferenceEngine based on model config and file extension.

    model.mock = true          → MockEngine      (no hardware needed)
    model.engine ends in .pt   → PyTorchEngine   (CPU / MPS / CUDA, any platform)
    model.engine ends in .onnx → PyTorchEngine   (CPU / CUDA)
    model.engine ends in .engine/.trt → TensorRTEngine  (Jetson / x86 GPU)
"""

from __future__ import annotations

from pathlib import Path

from loguru import logger

from pipeline.hawk.base import InferenceEngine
from pipeline.hawk.mock_engine import MockEngine
from pipeline.hawk.pytorch_engine import PyTorchEngine
from pipeline.hawk.tensorrt_engine import TensorRTEngine
from pipeline.shared.config import ModelConfig

_PYTORCH_EXTS  = {".pt", ".onnx"}
_TENSORRT_EXTS = {".engine", ".trt"}


def create(cfg: ModelConfig) -> InferenceEngine:
    """
    Instantiate and return the correct InferenceEngine for the given model config.

    Does NOT call .load() — caller is responsible for lifecycle management.
    Use as context manager or call .load() / .unload() explicitly.
    """
    if cfg.mock:
        logger.info("HawkFactory: MockEngine for '{}' (mock=true)", cfg.name)
        return MockEngine(cfg)

    ext = Path(cfg.engine).suffix.lower()

    if ext in _PYTORCH_EXTS:
        logger.info(
            "HawkFactory: PyTorchEngine for '{}' — model={}",
            cfg.name, cfg.engine,
        )
        return PyTorchEngine(cfg)

    if ext in _TENSORRT_EXTS:
        logger.info(
            "HawkFactory: TensorRTEngine for '{}' — engine={}",
            cfg.name, cfg.engine,
        )
        return TensorRTEngine(cfg)

    # Unknown extension — default to PyTorch and let it fail with a clear message
    logger.warning(
        "HawkFactory: unknown engine extension '{}' for '{}' — defaulting to PyTorchEngine",
        ext, cfg.name,
    )
    return PyTorchEngine(cfg)