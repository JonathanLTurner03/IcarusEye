"""
IcarusEye v2 — Pipeline Configuration
pipeline/shared/config.py

Loads configs/pipeline.yaml and returns a fully typed PipelineConfig.
Environment variables override YAML values — set them in .env or docker-compose.yml.

Usage:
    from pipeline.shared.config import load_config

    cfg = load_config()                        # auto-finds configs/pipeline.yaml
    cfg = load_config("configs/pipeline.yaml") # explicit path

    cfg.capture.device      # "/dev/capture0"
    cfg.redis.host          # "redis"
    cfg.models[0].engine    # "models/yolo26s_visdrone.engine"
    cfg.dev.enabled         # True when DEV_MODE=true
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml
from loguru import logger


# ── Sub-configs ──────────────────────────────────────────────────────────────

@dataclass
class CaptureConfig:
    type: str = "v4l2"               # v4l2 | rtsp | file | gstreamer_uri
    device: str = "/dev/capture0"
    uri: str = ""                    # RTSP / file path / GStreamer URI
    width: int = 1280
    height: int = 720
    framerate: int = 30
    format: str = "MJPG"
    loop: bool = True                # loop file source in dev mode


@dataclass
class ModelConfig:
    name: str = "aerial_detection"
    engine: str = "models/yolo26s_visdrone.engine"
    type: str = "detection"          # detection | segmentation | pose
    confidence: float = 0.5
    imgsz: int = 640
    classes: Optional[list[int]] = None   # None = all classes
    mock: bool = False               # replaced by MockEngine when True


@dataclass
class StreamConfig:
    enabled: bool = True
    host: str = "127.0.0.1"
    port: int = 5000
    bitrate: int = 2000              # kbps
    codec: str = "h264"


@dataclass
class RedisConfig:
    host: str = "redis"
    port: int = 6379
    db: int = 0
    channel_detections: str = "icaruseye:detections"
    channel_stats: str = "icaruseye:stats"
    channel_control: str = "icaruseye:control"


@dataclass
class DevConfig:
    enabled: bool = False
    sample_video: str = "/app/sample/sample.mp4"
    mock_inference: bool = True
    mock_fps: int = 30


# ── Root config ───────────────────────────────────────────────────────────────

@dataclass
class PipelineConfig:
    capture: CaptureConfig = field(default_factory=CaptureConfig)
    models: list[ModelConfig] = field(default_factory=lambda: [ModelConfig()])
    stream: StreamConfig = field(default_factory=StreamConfig)
    redis: RedisConfig = field(default_factory=RedisConfig)
    dev: DevConfig = field(default_factory=DevConfig)


# ── Loader ────────────────────────────────────────────────────────────────────

_DEFAULT_PATH = Path("configs/pipeline.yaml")


def load_config(path: str | Path | None = None) -> PipelineConfig:
    """
    Load pipeline.yaml and return a typed PipelineConfig.

    Environment variable overrides (take precedence over YAML):
        DEV_MODE        → dev.enabled          ("true" / "1" / "yes")
        REDIS_HOST      → redis.host
        REDIS_PORT      → redis.port
        STREAM_HOST     → stream.host
        STREAM_PORT     → stream.port
        CAPTURE_DEVICE  → capture.device
        CAPTURE_TYPE    → capture.type
        CAPTURE_URI     → capture.uri
        SAMPLE_VIDEO    → dev.sample_video + capture.uri
    """
    config_path = Path(path) if path else _DEFAULT_PATH

    raw: dict = {}
    if config_path.exists():
        with config_path.open() as f:
            raw = yaml.safe_load(f) or {}
        logger.debug("Config loaded from {}", config_path)
    else:
        logger.warning("Config not found at {} — using defaults", config_path)

    cfg = PipelineConfig(
        capture=_build_capture(raw.get("capture", {})),
        models=_build_models(raw.get("models", [])),
        stream=_build_stream(raw.get("stream", {})),
        redis=_build_redis(raw.get("redis", {})),
        dev=_build_dev(raw.get("dev", {})),
    )

    _apply_env(cfg)
    return cfg


# ── Section builders ──────────────────────────────────────────────────────────

def _build_capture(d: dict) -> CaptureConfig:
    return CaptureConfig(
        type=d.get("type", "v4l2"),
        device=d.get("device", "/dev/capture0"),
        uri=d.get("uri", ""),
        width=d.get("width", 1280),
        height=d.get("height", 720),
        framerate=d.get("framerate", 30),
        format=d.get("format", "MJPG"),
        loop=d.get("loop", True),
    )


def _build_models(raw: list | dict) -> list[ModelConfig]:
    if not raw:
        return [ModelConfig()]
    if isinstance(raw, dict):
        raw = [raw]
    return [
        ModelConfig(
            name=m.get("name", "model"),
            engine=m.get("engine", "models/yolo26s_visdrone.engine"),
            type=m.get("type", "detection"),
            confidence=m.get("confidence", 0.5),
            imgsz=m.get("imgsz", 640),
            classes=m.get("classes"),
            mock=m.get("mock", False),
        )
        for m in raw
    ]


def _build_stream(d: dict) -> StreamConfig:
    return StreamConfig(
        enabled=d.get("enabled", True),
        host=d.get("host", "127.0.0.1"),
        port=int(d.get("port", 5000)),
        bitrate=int(d.get("bitrate", 2000)),
        codec=d.get("codec", "h264"),
    )


def _build_redis(d: dict) -> RedisConfig:
    ch = d.get("channels", {})
    return RedisConfig(
        host=d.get("host", "redis"),
        port=int(d.get("port", 6379)),
        db=int(d.get("db", 0)),
        channel_detections=ch.get("detections", "icaruseye:detections"),
        channel_stats=ch.get("stats", "icaruseye:stats"),
        channel_control=ch.get("control", "icaruseye:control"),
    )


def _build_dev(d: dict) -> DevConfig:
    return DevConfig(
        enabled=_truthy(d.get("enabled", False)),
        sample_video=d.get("sample_video", "/app/sample/sample.mp4"),
        mock_inference=d.get("mock_inference", True),
        mock_fps=int(d.get("mock_fps", 30)),
    )


# ── Environment overrides ─────────────────────────────────────────────────────

def _apply_env(cfg: PipelineConfig) -> None:
    env = os.environ

    if v := env.get("REDIS_HOST"):
        cfg.redis.host = v
    if v := env.get("REDIS_PORT"):
        cfg.redis.port = int(v)

    if v := env.get("STREAM_HOST"):
        cfg.stream.host = v
    if v := env.get("STREAM_PORT"):
        cfg.stream.port = int(v)

    if v := env.get("CAPTURE_DEVICE"):
        cfg.capture.device = v
    if v := env.get("CAPTURE_TYPE"):
        cfg.capture.type = v
    if v := env.get("CAPTURE_URI"):
        cfg.capture.uri = v

    if v := env.get("SAMPLE_VIDEO"):
        cfg.dev.sample_video = v
        cfg.capture.uri = v

    if v := env.get("DEV_MODE"):
        cfg.dev.enabled = _truthy(v)

    # Dev mode side-effects — apply after all other env vars
    if cfg.dev.enabled:
        if not env.get("CAPTURE_TYPE"):
            cfg.capture.type = "file"
        if not cfg.capture.uri:
            cfg.capture.uri = cfg.dev.sample_video
        for m in cfg.models:
            m.mock = True
        logger.info("DEV_MODE enabled — capture→file, models→mock")


def _truthy(val: str | bool | int) -> bool:
    if isinstance(val, bool):
        return val
    return str(val).strip().lower() in ("true", "1", "yes", "on")