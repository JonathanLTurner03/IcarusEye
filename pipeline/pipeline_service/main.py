"""
IcarusEye v2 — Pipeline Service
pipeline/pipeline_service/main.py

Single process running all pipeline stages as threads:
  lens   → raw_queue → [frame_server:raw]
  raw_queue → hawk → mark → annotated_queue → [frame_server:annotated]

Tower connects to the frame servers via TCP sockets to get MJPEG feeds.
Redis carries only metadata (stats, detections, control).

Run:
    python -m pipeline.pipeline_service.main

Environment:
    DEV_MODE=true       → FileSource + MockEngine
    REDIS_HOST=...
"""

import queue
import signal
import threading
import time

from loguru import logger

from pipeline.lens import factory as lens_factory
from pipeline.hawk import factory as hawk_factory
from pipeline.mark.annotator import draw as mark_draw
from pipeline.shared.config import load_config, CaptureConfig
from pipeline.shared.redis_client import (
    DetectionMessage,
    StatsMessage,
    Subscriber,
    publish_detection,
    publish_stats,
    wait_for_redis,
)
from pipeline.frame_bus import (
    display_queue,
    inference_queue,
    annotated_queue,
    put_nowait_drop,
)
from pipeline.frame_server import FrameServer, RAW_PORT, ANNOTATED_PORT


def lens_thread(cfg, r, stop_event: threading.Event, swap_queue: queue.Queue) -> None:
    """
    Capture frames and fan out:
      display_queue   — every frame at native fps (tower always gets full rate)
      inference_queue — every Nth frame at cfg_fps (hawk runs at lower rate)

    Supports hot-swapping the capture source at runtime via swap_queue.
    If the source fails to open, the thread enters standby and waits for a
    swap command rather than crashing the pipeline.
    """
    from pipeline.lens.file_source import FileSource

    capture_cfg  = cfg.capture
    total_frames = 0

    while not stop_event.is_set():
        # ── Create and attempt to open source ────────────────────────────────
        src = lens_factory.create(capture_cfg)
        try:
            src.open()
        except Exception as exc:
            logger.error(
                "Lens: failed to open source type={} — {} — entering standby",
                capture_cfg.type, exc,
            )
            # Standby: wait for a swap command or stop signal
            while not stop_event.is_set():
                try:
                    capture_cfg = swap_queue.get(timeout=1.0)
                    logger.info(
                        "Lens: swap command received while in standby — type={} uri={}",
                        capture_cfg.type, capture_cfg.uri or capture_cfg.device,
                    )
                    break
                except queue.Empty:
                    continue
            continue  # retry outer loop with new capture_cfg

        # ── Source opened successfully ─────────────────────────────────────
        frame_count     = 0
        drop_count      = 0
        fps_window      = []
        last_stats      = time.monotonic()
        last_deliver    = 0.0
        stats_every     = 3.0
        inference_skip  = 1
        inference_frame = 0

        if isinstance(src, FileSource) and src.cfg_fps < src.native_fps:
            inference_skip = max(1, round(src.native_fps / src.cfg_fps))

        logger.info(
            "Lens: opened {} native_fps={:.1f} inference_skip={}",
            src.name,
            src.native_fps if isinstance(src, FileSource) else capture_cfg.framerate,
            inference_skip,
        )

        try:
            while not stop_event.is_set():
                # Check for a pending swap (non-blocking)
                try:
                    capture_cfg = swap_queue.get_nowait()
                    logger.info(
                        "Lens: swap command received — type={} uri={}",
                        capture_cfg.type, capture_cfg.uri or capture_cfg.device,
                    )
                    break  # close current source, restart outer loop
                except queue.Empty:
                    pass

                frame = src.read()
                if frame is None:
                    if not src.is_open:
                        logger.warning("Lens: source closed unexpectedly")
                        break
                    continue

                now = time.monotonic()
                gap_ms = (now - last_deliver) * 1000 if last_deliver else 0
                last_deliver = now

                frame_count  += 1
                inference_frame += 1
                total_frames += 1

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
                            "source_type": capture_cfg.type,
                            "resolution":  f"{frame.width}x{frame.height}",
                        },
                    ), r)
                    last_stats = now
                    drop_count = 0
        finally:
            src.close()
            logger.info("Lens: source closed — {} frames this session", frame_count)

    logger.info("Lens thread stopped — {} frames total", total_frames)


def hawk_mark_thread(cfg, r, stop_event: threading.Event) -> None:
    """
    Inference + annotation thread.

    Pulls frames from inference_queue (throttled by lens to cfg_fps).
    Runs hawk (mock or TensorRT) → publishes DetectionMessage to Redis.
    Passes InferenceResult to mark → draws bboxes → pushes to annotated_queue.

    Tower's annotated feed shows the live annotated stream.
    Tower's raw feed shows the unmodified stream at full native fps.
    """
    model_cfg = cfg.models[0]  # primary model — multi-model cascade is Phase 4

    engine      = hawk_factory.create(model_cfg)
    frame_count = 0
    fps_window  = []
    last_stats  = time.monotonic()
    stats_every = 3.0

    logger.info(
        "Hawk/mark thread starting — engine={} mock={}",
        model_cfg.engine, model_cfg.mock,
    )

    with engine:
        logger.info("Hawk/mark thread ready — engine loaded")

        while not stop_event.is_set():
            try:
                frame = inference_queue.get(timeout=0.5)
            except Exception:
                continue

            # ── Hawk: run inference ───────────────────────────────────────────
            result = engine.infer(frame)
            frame_count += 1

            # ── Publish detection metadata to Redis ───────────────────────────
            # mark subscribes to this if it runs as a separate service.
            # In this single-process design we pass result directly to mark,
            # but we still publish for tower's stats panel + future decoupling.
            det_msg = DetectionMessage(
                frame_id=frame.frame_id,
                timestamp=result.timestamp,
                source="hawk",
                detections=[
                    {
                        "label":      d.label,
                        "confidence": d.confidence,
                        "bbox":       d.bbox,
                        "class_id":   d.class_id,
                    }
                    for d in result.detections
                ],
                inference_ms=result.inference_ms,
            )
            try:
                publish_detection(det_msg, r)
            except Exception:
                pass  # Redis publish failures are non-fatal

            # ── Mark: annotate frame ──────────────────────────────────────────
            annotated = mark_draw(frame, result)
            put_nowait_drop(annotated_queue, annotated)

            # ── Stats ─────────────────────────────────────────────────────────
            now = time.monotonic()
            fps_window.append(now)
            fps_window = [t for t in fps_window if now - t <= 2.0]

            if now - last_stats >= stats_every:
                fps = len(fps_window) / min(now - last_stats, 2.0) if fps_window else 0.0
                n_det = len(result.detections)
                logger.info(
                    "Hawk: fps={:.1f} det={} inf={:.1f}ms engine={}",
                    fps, n_det, result.inference_ms, result.engine_name,
                )
                publish_stats(StatsMessage(
                    source="hawk",
                    timestamp=now,
                    fps=round(fps, 1),
                    inference_ms=result.inference_ms,
                    extra={
                        "frame_count":     frame_count,
                        "detections":      n_det,
                        "engine":          result.engine_name,
                        "model":           model_cfg.name,
                    },
                ), r)
                last_stats = now

    logger.info("Hawk/mark thread stopped — {} frames processed", frame_count)


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

    # ── Swap queue (lens source hot-swap) ─────────────────────────────────────
    swap_queue: queue.Queue = queue.Queue(maxsize=1)

    def _handle_control(_channel: str, data: dict) -> None:
        cmd = data.get("command")
        if cmd != "swap_source":
            return
        payload = data.get("payload", {})
        new_cfg = CaptureConfig(
            type=payload.get("type", cfg.capture.type),
            uri=payload.get("uri", ""),
            device=payload.get("device", cfg.capture.device),
            width=int(payload.get("width", cfg.capture.width)),
            height=int(payload.get("height", cfg.capture.height)),
            framerate=int(payload.get("framerate", cfg.capture.framerate)),
            format=payload.get("format", cfg.capture.format),
            loop=bool(payload.get("loop", cfg.capture.loop)),
        )
        # Drop any pending swap — only the latest matters
        try:
            swap_queue.get_nowait()
        except queue.Empty:
            pass
        swap_queue.put_nowait(new_cfg)
        logger.info(
            "Control: swap_source queued — type={} uri={}",
            new_cfg.type, new_cfg.uri or new_cfg.device,
        )

    control_sub = Subscriber(host=cfg.redis.host, port=cfg.redis.port)
    control_sub.on(cfg.redis.channel_control, _handle_control)
    control_sub.start()

    # ── Pipeline threads ──────────────────────────────────────────────────────
    threads = [
        threading.Thread(target=lens_thread,      args=(cfg, r, stop_event, swap_queue),
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

    control_sub.stop()
    raw_server.stop()
    ann_server.stop()
    logger.info("Pipeline stopped")


if __name__ == "__main__":
    run()
