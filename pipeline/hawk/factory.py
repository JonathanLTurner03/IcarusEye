"""
IcarusEye v2 — Inference Engine Factory
pipeline/hawk/factory.py

Creates the correct InferenceEngine based on model config.
Swapping mock ↔ TensorRT is config-only — no code changes.

    model.mock = true   → MockEngine   (DEV_MODE, no GPU required)
    model.mock = false  → TensorRTEngine (Jetson, requires .engine file)
"""

from __future__ import annotations

from loguru import logger

from pipeline.hawk.base import InferenceEngine
from pipeline.hawk.mock_engine import MockEngine
from pipeline.hawk.tensorrt_engine import TensorRTEngine
from pipeline.shared.config import ModelConfig


def create(cfg: ModelConfig) -> InferenceEngine:
    """
    Instantiate and return the correct InferenceEngine for the given model config.

    Does NOT call .load() — caller is responsible for lifecycle management.
    Use as context manager or call .load() / .unload() explicitly.

    Args:
        cfg: ModelConfig from pipeline.yaml / load_config()

    Returns:
        InferenceEngine instance (MockEngine or TensorRTEngine)
    """
    if cfg.mock:
        logger.info(
            "HawkFactory: creating MockEngine for model '{}' (mock=true)",
            cfg.name,
        )
        return MockEngine(cfg)

    logger.info(
        "HawkFactory: creating TensorRTEngine for model '{}' — engine={}",
        cfg.name, cfg.engine,
    )
    return TensorRTEngine(cfg)