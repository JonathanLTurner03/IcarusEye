"""
IcarusEye v2 — Lens Service Test
scripts/test_lens.py

Tests the lens capture pipeline against a real video file.
No hardware or Docker required — uses DEV_MODE FileSource.

Run from project root:
    python3 scripts/test_lens.py

Optional:
    SAMPLE_VIDEO=sample/sample.mp4  (default)
    REDIS_HOST=localhost             (default)
    REDIS_PORT=6379                 (default)
"""

import os
import sys
import time
import threading

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Force DEV_MODE for all tests
os.environ.setdefault("DEV_MODE",     "true")
os.environ.setdefault("REDIS_HOST",   "localhost")
os.environ.setdefault("SAMPLE_VIDEO", "sample/sample.mp4")

from pipeline.shared.config import load_config
from pipeline.lens import factory
from pipeline.lens.base import CaptureSource, Frame
from pipeline.lens.file_source import FileSource
from pipeline.lens.v4l2_source import V4L2Source
from pipeline.lens.rtsp_source import RTSPSource
from pipeline.lens.factory import create

# ── Helpers ───────────────────────────────────────────────────────────────────

PASS  = "\033[92m  PASS\033[0m"
FAIL  = "\033[91m  FAIL\033[0m"
HEAD  = "\033[96m{}\033[0m"
SKIP  = "\033[93m  SKIP\033[0m"

results: list[tuple[str, bool, str]] = []


def check(name: str, condition: bool, detail: str = "") -> bool:
    results.append((name, condition, detail))
    status = PASS if condition else FAIL
    suffix = f"  ({detail})" if detail else ""
    print(f"{status}  {name}{suffix}")
    return condition


def skip(name: str, reason: str) -> None:
    print(f"{SKIP}  {name}  ({reason})")


def section(title: str) -> None:
    print(f"\n{HEAD.format('── ' + title + ' ' + '─' * max(0, 50 - len(title)))}")


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_config_dev_mode() -> None:
    section("Config — DEV_MODE")

    cfg = load_config()
    check("DEV_MODE sets dev.enabled",      cfg.dev.enabled == True)
    check("DEV_MODE sets capture.type=file", cfg.capture.type == "file")
    check("DEV_MODE sets model.mock=True",   cfg.models[0].mock == True)
    check("capture.uri points to sample",   "sample" in cfg.capture.uri)


def test_factory() -> None:
    section("CaptureFactory")

    cfg = load_config()

    # DEV_MODE → FileSource
    src = create(cfg.capture)
    check("factory returns FileSource in DEV_MODE", isinstance(src, FileSource))
    check("source is not open before open()",       not src.is_open)

    # Explicit v4l2
    cfg.capture.type = "v4l2"
    src_v4l2 = create(cfg.capture)
    check("factory returns V4L2Source for v4l2",  isinstance(src_v4l2, V4L2Source))

    # Explicit rtsp
    cfg.capture.type = "rtsp"
    src_rtsp = create(cfg.capture)
    check("factory returns RTSPSource for rtsp",  isinstance(src_rtsp, RTSPSource))

    # Invalid type
    cfg.capture.type = "invalid_type"
    try:
        create(cfg.capture)
        check("factory raises on unknown type", False)
    except ValueError:
        check("factory raises on unknown type", True)


def test_file_source() -> None:
    section("FileSource")

    sample = os.environ.get("SAMPLE_VIDEO", "sample/sample.mp4")

    if not os.path.exists(sample):
        skip("FileSource tests", f"{sample} not found — add a video file to sample/")
        return

    cfg = load_config()
    cfg.capture.uri  = sample
    cfg.capture.loop = False   # don't loop for this test

    src = FileSource(cfg.capture)

    # Test context manager + open
    check("FileSource not open before context", not src.is_open)

    frames_read = []
    with src:
        check("FileSource opens successfully",  src.is_open)
        check("source name is 'file'",          src.name == "file")

        # Read up to 10 frames
        for _ in range(10):
            frame = src.read()
            if frame is None:
                break
            frames_read.append(frame)

    check("FileSource closed after context",    not src.is_open)
    check("Read at least 1 frame",              len(frames_read) >= 1)

    if frames_read:
        f = frames_read[0]
        check("Frame has numpy data",           hasattr(f, 'data') and f.data is not None)
        check("Frame data is 3-channel",        f.data.ndim == 3 and f.data.shape[2] == 3)
        check("Frame width > 0",                f.width > 0)
        check("Frame height > 0",               f.height > 0)
        check("Frame id starts at 1",           f.frame_id == 1)
        check("Frame timestamp > 0",            f.timestamp > 0)
        check("Frame source contains 'file'",   "file" in f.source)

    if len(frames_read) >= 2:
        check("Frame IDs increment",            frames_read[1].frame_id == frames_read[0].frame_id + 1)
        check("Timestamps increase",            frames_read[1].timestamp >= frames_read[0].timestamp)


def test_file_source_loop() -> None:
    section("FileSource — Loop")

    sample = os.environ.get("SAMPLE_VIDEO", "sample/sample.mp4")
    if not os.path.exists(sample):
        skip("FileSource loop test", f"{sample} not found")
        return

    cfg = load_config()
    cfg.capture.uri  = sample
    cfg.capture.loop = True

    src = FileSource(cfg.capture)
    frame_count = 0
    target = 200   # read well past one loop of any short sample

    with src:
        start = time.monotonic()
        while frame_count < target and (time.monotonic() - start) < 10.0:
            frame = src.read()
            if frame:
                frame_count += 1
            else:
                time.sleep(0.001)

    check(
        f"FileSource loops — read {frame_count}/{target} frames",
        frame_count >= target,
        f"got {frame_count}"
    )


def test_file_source_missing() -> None:
    section("FileSource — Error Handling")

    cfg = load_config()
    cfg.capture.uri = "/nonexistent/path/video.mp4"
    src = FileSource(cfg.capture)

    try:
        src.open()
        check("FileSource raises on missing file", False)
    except FileNotFoundError:
        check("FileSource raises FileNotFoundError on missing file", True)
    finally:
        src.close()


def test_throughput() -> None:
    section("FileSource — Throughput")

    sample = os.environ.get("SAMPLE_VIDEO", "sample/sample.mp4")
    if not os.path.exists(sample):
        skip("Throughput test", f"{sample} not found")
        return

    cfg = load_config()
    cfg.capture.uri  = sample
    cfg.capture.loop = True

    src = FileSource(cfg.capture)
    frame_count = 0
    duration = 2.0

    with src:
        start = time.monotonic()
        while time.monotonic() - start < duration:
            frame = src.read()
            if frame:
                frame_count += 1

    fps = frame_count / duration
    check(
        f"FileSource throughput ≥ 20 fps  (got {fps:.1f})",
        fps >= 20.0,
        f"{fps:.1f} fps over {duration}s"
    )


# ── Summary ───────────────────────────────────────────────────────────────────

def print_summary() -> None:
    total  = len(results)
    passed = sum(1 for _, ok, _ in results if ok)
    failed = total - passed

    print(f"\n{'─' * 55}")
    if failed == 0:
        print(f"\033[92m  {passed}/{total} checks passed — lens is solid.\033[0m")
    else:
        print(f"\033[91m  {passed}/{total} passed, {failed} FAILED\033[0m")
        print("\n  Failed checks:")
        for name, ok, detail in results:
            if not ok:
                suffix = f" ({detail})" if detail else ""
                print(f"    ✗  {name}{suffix}")
    print()


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"\n\033[96mIcarusEye v2 — Lens Test\033[0m")
    print(f"Sample: {os.environ.get('SAMPLE_VIDEO', 'sample/sample.mp4')}\n")

    test_config_dev_mode()
    test_factory()
    test_file_source_missing()
    test_file_source()
    test_file_source_loop()
    test_throughput()

    print_summary()
    sys.exit(0 if all(ok for _, ok, _ in results) else 1)