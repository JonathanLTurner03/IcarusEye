"""
IcarusEye v2 — Frame Bus
pipeline/frame_bus.py

Shared in-process frame queues connecting pipeline stages.

Queue design:
  display_queue    lens → tower raw feed (display only, maxsize=2)
  inference_queue  lens → hawk (inference, maxsize=2)
  annotated_queue  mark → tower annotated feed (maxsize=2)

Lens fans out to both display_queue and inference_queue independently.
This means tower always gets every frame regardless of hawk's speed.
"""

from __future__ import annotations
import queue

# Raw frames for display — consumed only by tower's raw feed
display_queue: queue.Queue = queue.Queue(maxsize=2)

# Raw frames for inference — consumed only by hawk
inference_queue: queue.Queue = queue.Queue(maxsize=2)

# Annotated frames from mark — consumed by tower's annotated feed
annotated_queue: queue.Queue = queue.Queue(maxsize=2)


def put_nowait_drop(q: queue.Queue, item) -> None:
    """Put item in queue, dropping oldest if full (keeps latency low)."""
    try:
        q.put_nowait(item)
    except queue.Full:
        try:
            q.get_nowait()
        except queue.Empty:
            pass
        try:
            q.put_nowait(item)
        except queue.Full:
            pass