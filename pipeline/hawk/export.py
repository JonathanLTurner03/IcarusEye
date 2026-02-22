"""
IcarusEye v2 — TensorRT engine export helper
pipeline/hawk/export.py

Finds a source .pt checkpoint and re-exports a TensorRT .engine file.
Must be called from the inference GPU thread (CUDA context is thread-local).

The .pt file is auto-derived from the engine path by replacing the extension:
    models/yolo26m.engine  →  models/yolo26m.pt

The source .pt file is never deleted or moved — only the .engine file is
created/replaced.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

from loguru import logger

from pipeline.shared.config import ModelConfig


def find_pt(cfg: ModelConfig) -> Optional[Path]:
    """Return the .pt checkpoint path for auto-rebuild, or None if not found."""
    pt = Path(cfg.engine).with_suffix(".pt")
    return pt if pt.exists() else None


def build_engine(
    cfg: ModelConfig,
    progress: Callable[[str], None] | None = None,
) -> Path:
    """
    Export a TensorRT .engine from the matching .pt checkpoint.

    MUST be called from the inference GPU thread — CUDA contexts are
    thread-local and TensorRT will error if load() and this function run
    in different threads.

    Returns the path of the newly created engine file.
    Raises FileNotFoundError if no .pt checkpoint is found.
    """
    from ultralytics import YOLO

    pt_path = find_pt(cfg)
    if pt_path is None:
        raise FileNotFoundError(
            f"No .pt checkpoint found for {cfg.engine}. "
            f"Expected: {Path(cfg.engine).with_suffix('.pt')}"
        )

    if progress:
        progress(
            f"Exporting TRT engine from {pt_path.name} "
            f"(this takes several minutes on first run)…"
        )

    logger.info("Engine export starting — source={} imgsz={}", pt_path, cfg.imgsz)
    model = YOLO(str(pt_path))
    exported = model.export(format="engine", device=0, imgsz=cfg.imgsz)
    engine_path = Path(exported)
    # NOTE: .pt file is never deleted or moved — only the .engine file is created/replaced.

    if progress:
        progress(f"Export complete → {engine_path.name}")

    logger.info("Engine exported to {}", engine_path)
    return engine_path
