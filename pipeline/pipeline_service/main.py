"""
IcarusEye v2 — Pipeline Service
pipeline/pipeline_service/main.py

Single process running all pipeline stages as threads:
  lens   → raw_queue → [frame_server:raw]
  raw_queue → hawk   → mark → annotated_queue → [frame_server:annotated]

Tower connects to the frame servers via Unix sockets to get MJPEG feeds.
Redis carries only metadata (stats, detections, control).

Run:
    python -m pipeline.pipeline_service.main

Environment:
    DEV_MODE=true       → FileSource + MockEngine
    REDIS_HOST=...
"""

import signal
import threading
import time

from loguru import logger

from pipeline.lens import factory as lens_factory
from pipeline.shared.config import load_config
from pipeline.shared.redis_client import StatsMessage, publish_stats, wait_for_redis
from pipeline.frame_bus import display_queue, inference_queue, annotated_queue, put_nowait_drop
from pipeline.frame_server import FrameServer, RAW_PORT, ANNOTATED_PORT


def lens_thread(cfg, r, stop_event: threading.Event) -> None:
    """
    Capture frames and fan out:
      display_queue   — every frame at native fps (tower always gets full rate)
      inference_queue — every Nth frame at cfg_fps (hawk runs at lower rate)
    """
    from pipeline.lens.file_source import FileSource

    src = lens_factory.create(cfg.capture)
    frame_count  = 0
    drop_count   = 0
    fps_window   = []
    last_stats   = time.monotonic()
    last_deliver = 0.0
    stats_every  = 3.0

    # Determine inference skip ratio after source is created
    # (native_fps only available after open(), computed below)
    inference_skip = 1
    inference_frame = 0

    with src:
        # Compute skip ratio now that source is open and native_fps is known
        if isinstance(src, FileSource) and src.cfg_fps < src.native_fps:
            inference_skip = max(1, round(src.native_fps / src.cfg_fps))
        logger.info(
            "Lens thread started — source: {} native_fps={:.1f} inference_skip={}",
            src.name,
            src.native_fps if isinstance(src, FileSource) else cfg.capture.framerate,
            inference_skip,
        )

        while not stop_event.is_set():
            frame = src.read()
            if frame is None:
                if not src.is_open:
                    logger.warning("Lens: source closed")
                    break
                continue

            now = time.monotonic()
            gap_ms = (now - last_deliver) * 1000 if last_deliver else 0
            last_deliver = now

            frame_count += 1
            inference_frame += 1

            # Tower always gets every frame at native fps
            put_nowait_drop(display_queue, frame)

            # Hawk gets every Nth frame
            if inference_skip == 1 or (inference_frame % inference_skip) == 0:
                put_nowait_drop(inference_queue, frame)

            fps_window.append(now)
            fps_window = [t for t in fps_window if now - t <= 2.0]

            if now - last_stats >= stats_every:
                fps       = len(fps_window) / min(now - last_stats, 2.0) if fps_window else 0.0
                total     = frame_count + drop_count
                drop_rate = drop_count / total if total > 0 else 0.0
                logger.info(
                    "Lens: fps={:.1f} delivered={} inf_skip={} last_gap={:.1f}ms",
                    fps, frame_count, inference_skip, gap_ms,
                )
                publish_stats(StatsMessage(
                    source="lens",
                    timestamp=now,
                    fps=round(fps, 1),
                    drop_rate=round(drop_rate, 4),
                    extra={
                        "frame_count": frame_count,
                        "source_type": cfg.capture.type,
                        "resolution":  f"{frame.width}x{frame.height}",
                    },
                ), r)
                last_stats = now
                drop_count = 0

    logger.info("Lens thread stopped — {} frames delivered", frame_count)


def hawk_mark_thread(cfg, r, stop_event: threading.Event) -> None:
    """
    Inference + annotation thread.
    Pulls from inference_queue (separate from display so tower gets every frame),
    runs hawk (stub), mark draws boxes, pushes to annotated_queue.
    """
    frame_count  = 0
    last_detections = []

    logger.info("Hawk/mark thread started (stub — passthrough)")

    while not stop_event.is_set():
        try:
            frame = inference_queue.get(timeout=0.5)
        except Exception:
            continue

        frame_count += 1

        # Passthrough — annotated feed mirrors raw until hawk is built
        put_nowait_drop(annotated_queue, frame)

    logger.info("Hawk/mark thread stopped")


def run() -> None:
    cfg = load_config()
    logger.info("Pipeline service starting")

    r = wait_for_redis(host=cfg.redis.host, port=cfg.redis.port)

    # ── Frame servers ─────────────────────────────────────────────────────────
    raw_server = FrameServer(RAW_PORT, display_queue, "raw")
    ann_server = FrameServer(ANNOTATED_PORT, annotated_queue, "annotated")
    raw_server.start()
    ann_server.start()

    # ── Stop event ────────────────────────────────────────────────────────────
    stop_event = threading.Event()

    def _stop(sig, _frame):
        logger.info("Pipeline received signal {} — stopping", sig)
        stop_event.set()

    signal.signal(signal.SIGINT,  _stop)
    signal.signal(signal.SIGTERM, _stop)

    # ── Pipeline threads ──────────────────────────────────────────────────────
    threads = [
        threading.Thread(target=lens_thread,      args=(cfg, r, stop_event),
                         name="lens",      daemon=True),
        threading.Thread(target=hawk_mark_thread, args=(cfg, r, stop_event),
                         name="hawk-mark", daemon=True),
    ]

    for t in threads:
        t.start()

    logger.info("Pipeline running — {} threads active", len(threads))

    # Wait for stop signal
    stop_event.wait()

    logger.info("Pipeline shutting down")
    for t in threads:
        t.join(timeout=5.0)

    raw_server.stop()
    ann_server.stop()
    logger.info("Pipeline stopped")


if __name__ == "__main__":
    run()