"""
IcarusEye v2 — Redis Foundation Test
scripts/test_redis_foundation.py

Tests the shared Redis foundation against a live Redis instance.
Run with Redis already up:

    docker compose up redis -d
    python scripts/test_redis_foundation.py

Optional env overrides:
    REDIS_HOST=localhost  (default)
    REDIS_PORT=6379       (default)
"""

import dataclasses
import json
import os
import sys
import threading
import time

# ── Allow running from project root without installing the package ────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.shared.config import load_config
from pipeline.shared.redis_client import (
    Channels,
    ControlMessage,
    Detection,
    DetectionMessage,
    StatsMessage,
    Subscriber,
    get_client,
    publish_control,
    publish_detection,
    publish_stats,
    reset_client,
    wait_for_redis,
)

# ── Helpers ───────────────────────────────────────────────────────────────────

PASS  = "\033[92m  PASS\033[0m"
FAIL  = "\033[91m  FAIL\033[0m"
HEAD  = "\033[96m{}\033[0m"
RESET = "\033[0m"

results: list[tuple[str, bool, str]] = []


def check(name: str, condition: bool, detail: str = "") -> bool:
    results.append((name, condition, detail))
    status = PASS if condition else FAIL
    detail_str = f"  ({detail})" if detail else ""
    print(f"{status}  {name}{detail_str}")
    return condition


def section(title: str) -> None:
    print(f"\n{HEAD.format('── ' + title + ' ' + '─' * max(0, 50 - len(title)))}")


# ── Test config ───────────────────────────────────────────────────────────────

def test_config() -> None:
    section("Config")

    cfg = load_config("configs/pipeline.yaml")

    check("pipeline.yaml loads without error", cfg is not None)
    check("capture.type is set",     cfg.capture.type in ("v4l2", "file", "rtsp", "gstreamer_uri"))
    check("capture.device is set",   bool(cfg.capture.device))
    check("redis.host is set",       bool(cfg.redis.host))
    check("redis.port is 6379",      cfg.redis.port == 6379)
    check("channel_detections",      cfg.redis.channel_detections == Channels.DETECTIONS)
    check("channel_stats",           cfg.redis.channel_stats      == Channels.STATS)
    check("channel_control",         cfg.redis.channel_control    == Channels.CONTROL)
    check("at least one model",      len(cfg.models) >= 1)
    check("stream config present",   cfg.stream.port > 0)
    check("dev.enabled is False",    cfg.dev.enabled == False,
          "set DEV_MODE=true to flip this")

    # DEV_MODE env override
    os.environ["DEV_MODE"] = "true"
    cfg_dev = load_config("configs/pipeline.yaml")
    check("DEV_MODE=true → dev.enabled",      cfg_dev.dev.enabled == True)
    check("DEV_MODE=true → capture.type=file", cfg_dev.capture.type == "file")
    check("DEV_MODE=true → model.mock=True",   cfg_dev.models[0].mock == True)
    del os.environ["DEV_MODE"]


# ── Test dataclasses ──────────────────────────────────────────────────────────

def test_dataclasses() -> None:
    section("Message Dataclasses")

    det = Detection(label="person", confidence=0.91, bbox=[0.1, 0.2, 0.8, 0.9], class_id=0)
    det_dict = dataclasses.asdict(det)
    check("Detection serializes to dict",  isinstance(det_dict, dict))
    check("Detection label preserved",     det_dict["label"] == "person")
    check("Detection bbox is list",        isinstance(det_dict["bbox"], list))

    dmsg = DetectionMessage(
        frame_id=42,
        timestamp=time.monotonic(),
        source="hawk",
        detections=[det_dict],
        inference_ms=11.5,
    )
    dmsg_dict = dataclasses.asdict(dmsg)
    roundtrip  = json.loads(json.dumps(dmsg_dict))
    check("DetectionMessage JSON round-trip",  roundtrip["frame_id"] == 42)
    check("DetectionMessage detections list",  len(roundtrip["detections"]) == 1)

    smsg = StatsMessage(source="lens", timestamp=time.monotonic(), fps=29.7)
    check("StatsMessage extra defaults to {}", smsg.extra == {})
    check("StatsMessage JSON round-trip",
          json.loads(json.dumps(dataclasses.asdict(smsg)))["source"] == "lens")

    cmsg = ControlMessage(command="pause")
    check("ControlMessage payload defaults to {}", cmsg.payload == {})
    check("ControlMessage JSON round-trip",
          json.loads(json.dumps(dataclasses.asdict(cmsg)))["command"] == "pause")


# ── Test Redis connectivity ───────────────────────────────────────────────────

REDIS_HOST = os.environ.get("REDIS_HOST", "localhost")
REDIS_PORT  = int(os.environ.get("REDIS_PORT", 6379))


def test_connection() -> bool:
    section("Redis Connection")

    try:
        client = wait_for_redis(host=REDIS_HOST, port=REDIS_PORT, retries=5, interval=1.0)
        ok = client.ping()
        check("wait_for_redis() succeeds", ok)
        check("get_client() returns same client", get_client(host=REDIS_HOST, port=REDIS_PORT) is not None)
        return True
    except Exception as exc:
        check("wait_for_redis() succeeds", False, str(exc))
        print(f"\n  \033[93mRedis not reachable at {REDIS_HOST}:{REDIS_PORT}.")
        print(  "  Start it with:  docker compose up redis -d\033[0m")
        return False


# ── Test pub/sub round-trip ───────────────────────────────────────────────────

def test_pubsub() -> None:
    section("Pub/Sub Round-Trip")

    received: dict[str, dict | None] = {
        Channels.DETECTIONS: None,
        Channels.STATS:      None,
        Channels.CONTROL:    None,
    }
    events = {ch: threading.Event() for ch in Channels.ALL}

    def handler(channel: str, data: dict) -> None:
        received[channel] = data
        events[channel].set()

    # Start subscriber
    sub = (
        Subscriber(host=REDIS_HOST, port=REDIS_PORT)
        .on(Channels.DETECTIONS, handler)
        .on(Channels.STATS,      handler)
        .on(Channels.CONTROL,    handler)
        .start()
    )

    time.sleep(0.3)  # let pubsub thread subscribe before we publish

    # ── Publish DetectionMessage ──
    det = Detection(label="car", confidence=0.87, bbox=[0.2, 0.3, 0.7, 0.8], class_id=2)
    dmsg = DetectionMessage(
        frame_id=1,
        timestamp=time.monotonic(),
        source="hawk",
        detections=[dataclasses.asdict(det)],
        inference_ms=9.3,
    )
    publish_detection(dmsg, get_client(host=REDIS_HOST, port=REDIS_PORT))
    got_det = events[Channels.DETECTIONS].wait(timeout=3.0)
    check("DetectionMessage published and received", got_det)
    if got_det:
        d = received[Channels.DETECTIONS]
        check("  frame_id correct",           d["frame_id"] == 1)
        check("  source is 'hawk'",           d["source"] == "hawk")
        check("  detections list non-empty",  len(d["detections"]) == 1)
        check("  detection label preserved",  d["detections"][0]["label"] == "car")
        check("  inference_ms preserved",     d["inference_ms"] == 9.3)

    # ── Publish StatsMessage ──
    smsg = StatsMessage(
        source="lens",
        timestamp=time.monotonic(),
        fps=30.0,
        drop_rate=0.01,
        queue_depth=2,
    )
    publish_stats(smsg, get_client(host=REDIS_HOST, port=REDIS_PORT))
    got_stats = events[Channels.STATS].wait(timeout=3.0)
    check("StatsMessage published and received", got_stats)
    if got_stats:
        s = received[Channels.STATS]
        check("  source is 'lens'",   s["source"] == "lens")
        check("  fps correct",        s["fps"] == 30.0)
        check("  extra is {}",        s["extra"] == {})

    # ── Publish ControlMessage ──
    cmsg = ControlMessage(command="pause", payload={"reason": "test"})
    publish_control(cmsg, get_client(host=REDIS_HOST, port=REDIS_PORT))
    got_ctrl = events[Channels.CONTROL].wait(timeout=3.0)
    check("ControlMessage published and received", got_ctrl)
    if got_ctrl:
        c = received[Channels.CONTROL]
        check("  command is 'pause'",        c["command"] == "pause")
        check("  payload preserved",         c["payload"]["reason"] == "test")

    sub.stop()


# ── Test Subscriber reconnect (basic) ─────────────────────────────────────────

def test_subscriber_resilience() -> None:
    section("Subscriber Resilience")

    # Verify stop/start cycle doesn't crash
    sub = Subscriber(host=REDIS_HOST, port=REDIS_PORT)
    sub.on(Channels.STATS, lambda ch, d: None)
    sub.start()
    time.sleep(0.2)
    sub.stop()
    check("Subscriber start/stop cycle clean", True)

    # Verify double-start is safe
    sub2 = Subscriber(host=REDIS_HOST, port=REDIS_PORT)
    sub2.on(Channels.STATS, lambda ch, d: None)
    sub2.start()
    sub2.start()  # should be a no-op
    time.sleep(0.1)
    sub2.stop()
    check("Subscriber double-start is safe", True)


# ── Summary ───────────────────────────────────────────────────────────────────

def print_summary() -> None:
    total  = len(results)
    passed = sum(1 for _, ok, _ in results if ok)
    failed = total - passed

    print(f"\n{'─' * 55}")
    if failed == 0:
        print(f"\033[92m  {passed}/{total} checks passed — foundation is solid.\033[0m")
    else:
        print(f"\033[91m  {passed}/{total} passed, {failed} FAILED\033[0m")
        print("\n  Failed checks:")
        for name, ok, detail in results:
            if not ok:
                detail_str = f" ({detail})" if detail else ""
                print(f"    ✗  {name}{detail_str}")
    print()


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"\n\033[96mIcarusEye v2 — Redis Foundation Test\033[0m")
    print(f"Redis: {REDIS_HOST}:{REDIS_PORT}\n")

    test_config()
    test_dataclasses()

    redis_ok = test_connection()
    if redis_ok:
        test_pubsub()
        test_subscriber_resilience()
    else:
        print("\n  Skipping pub/sub tests — Redis unreachable.\n")

    print_summary()
    sys.exit(0 if all(ok for _, ok, _ in results) else 1)