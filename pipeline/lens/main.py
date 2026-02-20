"""
IcarusEye v2 — Lens Service
pipeline/lens/main.py

Capture service. Reads frames from the configured source and publishes
stats to Redis every few seconds so tower can display capture health.

Run:
    python -m pipeline.lens.main

Environment:
    DEV_MODE=true       → FileSource (sample.mp4), no hardware needed
    CAPTURE_TYPE=v4l2   → V4L2Source (USB capture card)
    CAPTURE_DEVICE=...  → override device path
    REDIS_HOST=...      → Redis host (default: redis)
"""

import signal
import time

from loguru import logger

from pipeline.lens import factory
from pipeline.shared.config import load_config
from pipeline.shared.redis_client import StatsMessage, publish_stats, wait_for_redis


def run() -> None:
    cfg = load_config()
    logger.info("Lens starting — capture type: {}", cfg.capture.type)

    r = wait_for_redis(host=cfg.redis.host, port=cfg.redis.port)

    running = True

    def _stop(sig, _frame):
        nonlocal running
        logger.info("Lens received signal {} — shutting down", sig)
        running = False

    signal.signal(signal.SIGINT,  _stop)
    signal.signal(signal.SIGTERM, _stop)

    frame_count = 0
    drop_count  = 0
    stats_every = 3.0
    last_stats  = time.monotonic()
    fps_window: list[float] = []

    src = factory.create(cfg.capture)

    with src:
        logger.info("Lens capture loop started — source: {}", src.name)

        while running:
            frame = src.read()

            if frame is None:
                if not src.is_open:
                    logger.warning("Lens: source closed unexpectedly")
                    break
                time.sleep(0.001)
                drop_count += 1
                continue

            frame_count += 1
            now = time.monotonic()
            fps_window.append(now)
            fps_window = [t for t in fps_window if now - t <= 2.0]

            if now - last_stats >= stats_every:
                fps       = len(fps_window) / min(now - last_stats, 2.0) if fps_window else 0.0
                total     = frame_count + drop_count
                drop_rate = drop_count / total if total > 0 else 0.0

                msg = StatsMessage(
                    source="lens",
                    timestamp=now,
                    fps=round(fps, 1),
                    drop_rate=round(drop_rate, 4),
                    extra={
                        "frame_count": frame_count,
                        "source_type": cfg.capture.type,
                        "resolution":  f"{frame.width}x{frame.height}",
                    },
                )
                publish_stats(msg, r)
                logger.debug(
                    "Lens stats — fps={:.1f} frames={} drops={} res={}x{}",
                    fps, frame_count, drop_count, frame.width, frame.height,
                )

                last_stats = now
                drop_count = 0

    logger.info("Lens stopped — {} frames captured total", frame_count)


if __name__ == "__main__":
    run()