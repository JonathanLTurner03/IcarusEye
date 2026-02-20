"""
IcarusEye v2 — Redis Client Foundation
pipeline/shared/redis_client.py

Provides:
  - Channels       — channel name constants (single source of truth)
  - Message types  — DetectionMessage, StatsMessage, ControlMessage dataclasses
  - get_client()   — thread-safe singleton Redis client
  - publish_*()    — typed publish helpers for each channel
  - Subscriber     — background pub/sub listener with auto-reconnect
  - wait_for_redis() — startup utility, blocks until Redis is reachable
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import asdict, dataclass
from typing import Callable, Optional

import redis
from loguru import logger


# ── Channel constants ──

class Channels:
    """Redis pub/sub channel names. Import this everywhere — never hardcode."""

    DETECTIONS = "icaruseye:detections"   # hawk → mark
    STATS      = "icaruseye:stats"        # all services → tower
    CONTROL    = "icaruseye:control"      # tower → all services

    ALL = (DETECTIONS, STATS, CONTROL)


# ── Message schemas ──

@dataclass
class Detection:
    """Single object detection result from hawk."""
    label: str
    confidence: float
    bbox: list[float]     # [x1, y1, x2, y2] normalized 0–1
    class_id: int = 0


@dataclass
class DetectionMessage:
    """
    Published by hawk on DETECTIONS channel.
    Consumed by mark to draw bounding boxes.
    """
    frame_id: int
    timestamp: float          # time.monotonic() at inference
    source: str               # "hawk"
    detections: list[dict]    # list of Detection.asdict()
    inference_ms: float = 0.0


@dataclass
class StatsMessage:
    """
    Published by any service on STATS channel.
    Consumed by tower for the dashboard stats panel.
    """
    source: str               # "lens" | "hawk" | "mark" | "relay"
    timestamp: float
    fps: float = 0.0
    drop_rate: float = 0.0    # dropped frames / total frames
    queue_depth: int = 0
    inference_ms: float = 0.0 # hawk only
    extra: dict = None        # service-specific extras

    def __post_init__(self):
        if self.extra is None:
            self.extra = {}


@dataclass
class ControlMessage:
    """
    Published by tower on CONTROL channel.
    Consumed by all services. Commands: pause | resume | swap_source | shutdown
    """
    command: str
    payload: dict = None

    def __post_init__(self):
        if self.payload is None:
            self.payload = {}


# ── Singleton client ──

_lock: threading.Lock = threading.Lock()
_client: Optional[redis.Redis] = None


def get_client(
    host: str = "redis",
    port: int = 6379,
    db: int = 0,
    decode_responses: bool = True,
    socket_connect_timeout: float = 5.0,
    socket_timeout: float = 5.0,
    retry_on_timeout: bool = True,
) -> redis.Redis:
    """
    Return a thread-safe singleton Redis client.

    First call connects and caches the client.
    Subsequent calls return the cached instance.
    Raises redis.ConnectionError if Redis is unreachable.
    """
    global _client
    if _client is not None:
        return _client

    with _lock:
        if _client is None:
            _client = redis.Redis(
                host=host,
                port=port,
                db=db,
                decode_responses=decode_responses,
                socket_connect_timeout=socket_connect_timeout,
                socket_timeout=socket_timeout,
                retry_on_timeout=retry_on_timeout,
            )
            _client.ping()
            logger.info("Redis client connected → {}:{}", host, port)

    return _client


def reset_client() -> None:
    """Close and discard the singleton (useful in tests)."""
    global _client
    with _lock:
        if _client is not None:
            try:
                _client.close()
            except Exception:
                pass
            _client = None


# ── Publish helpers ──

def _publish(channel: str, payload: dict, client: Optional[redis.Redis] = None) -> int:
    r = client or get_client()
    return r.publish(channel, json.dumps(payload))


def publish_detection(msg: DetectionMessage, client: Optional[redis.Redis] = None) -> int:
    """hawk → mark: publish inference results."""
    return _publish(Channels.DETECTIONS, asdict(msg), client)


def publish_stats(msg: StatsMessage, client: Optional[redis.Redis] = None) -> int:
    """Any service → tower: publish telemetry."""
    return _publish(Channels.STATS, asdict(msg), client)


def publish_control(msg: ControlMessage, client: Optional[redis.Redis] = None) -> int:
    """tower → all: publish control commands."""
    return _publish(Channels.CONTROL, asdict(msg), client)


# ── Subscriber ──

HandlerFn = Callable[[str, dict], None]


class Subscriber:
    """
    Manages a Redis pub/sub subscription in a daemon background thread.

    Features:
      - Register handlers per channel with .on()
      - Auto-reconnects on connection loss with exponential back-off
      - JSON parse errors are logged and skipped, not fatal
      - Handler exceptions are caught and logged, not fatal

    Usage:
        sub = Subscriber()
        sub.on(Channels.DETECTIONS, handle_detections)
        sub.on(Channels.CONTROL, handle_control)
        sub.start()
        ...
        sub.stop()
    """

    def __init__(
        self,
        host: str = "redis",
        port: int = 6379,
        db: int = 0,
        reconnect_delay: float = 1.0,
        reconnect_max: float = 30.0,
    ):
        self._host = host
        self._port = port
        self._db = db
        self._reconnect_delay = reconnect_delay
        self._reconnect_max = reconnect_max

        self._handlers: dict[str, list[HandlerFn]] = {}
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()

    def on(self, channel: str, handler: HandlerFn) -> "Subscriber":
        """Register a handler for a channel. Chainable."""
        self._handlers.setdefault(channel, []).append(handler)
        return self

    def start(self) -> "Subscriber":
        """Start the listener in a daemon thread. Chainable."""
        if self._thread and self._thread.is_alive():
            return self
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run,
            name="icaruseye-subscriber",
            daemon=True,
        )
        self._thread.start()
        logger.info("Subscriber started → channels: {}", list(self._handlers))
        return self

    def stop(self, timeout: float = 5.0) -> None:
        """Signal the listener to stop and wait for thread exit."""
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=timeout)
            self._thread = None
        logger.info("Subscriber stopped")

    def _run(self) -> None:
        delay = self._reconnect_delay
        while not self._stop.is_set():
            try:
                r = redis.Redis(
                    host=self._host,
                    port=self._port,
                    db=self._db,
                    decode_responses=True,
                    socket_connect_timeout=5.0,
                )
                pubsub = r.pubsub(ignore_subscribe_messages=True)
                pubsub.subscribe(*self._handlers)
                logger.info("Subscriber listening on {}", list(self._handlers))
                delay = self._reconnect_delay  # reset on successful connect

                for message in pubsub.listen():
                    if self._stop.is_set():
                        break
                    if message["type"] != "message":
                        continue
                    channel: str = message["channel"]
                    try:
                        data = json.loads(message["data"])
                    except (json.JSONDecodeError, TypeError) as exc:
                        logger.warning("Bad message on {}: {}", channel, exc)
                        continue
                    for handler in self._handlers.get(channel, []):
                        try:
                            handler(channel, data)
                        except Exception:
                            logger.exception(
                                "Handler {} raised on channel {}", handler, channel
                            )

                pubsub.close()

            except redis.RedisError as exc:
                if self._stop.is_set():
                    break
                logger.warning(
                    "Subscriber lost connection ({}) — reconnecting in {:.1f}s", exc, delay
                )
                time.sleep(delay)
                delay = min(delay * 2, self._reconnect_max)


# ── Startup utility ──

def wait_for_redis(
    host: str = "redis",
    port: int = 6379,
    retries: int = 10,
    interval: float = 2.0,
) -> redis.Redis:
    """
    Block until Redis is reachable, then return a connected client.
    Every pipeline service calls this at startup before doing anything else.
    Raises redis.ConnectionError if all retries are exhausted.
    """
    last_exc: Optional[Exception] = None
    for attempt in range(1, retries + 1):
        try:
            client = redis.Redis(
                host=host,
                port=port,
                decode_responses=True,
                socket_connect_timeout=2.0,
            )
            client.ping()
            logger.info("Redis ready (attempt {}/{})", attempt, retries)
            return client
        except redis.RedisError as exc:
            last_exc = exc
            logger.warning("Redis not ready ({}/{}) — {}", attempt, retries, exc)
            if attempt < retries:
                time.sleep(interval)

    raise redis.ConnectionError(
        f"Redis at {host}:{port} unreachable after {retries} attempts"
    ) from last_exc