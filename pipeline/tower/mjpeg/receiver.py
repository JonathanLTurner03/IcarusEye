"""
IcarusEye v2 — MJPEG Frame Receiver
pipeline/tower/mjpeg/receiver.py

Connects to the pipeline's TCP frame server and maintains
the latest JPEG frame for tower to serve via MJPEG.
Auto-reconnects if the pipeline service restarts.
"""

from __future__ import annotations

import socket
import struct
import threading
import time
from typing import Optional

from loguru import logger

from pipeline.frame_server import RAW_PORT, ANNOTATED_PORT  # noqa — port constants


class FrameReceiver:
    """Connects to a FrameServer TCP port and keeps the latest JPEG in memory."""

    def __init__(self, host: str, port: int, name: str):
        self._host      = host
        self._port      = port
        self._name      = name
        self._latest:   Optional[bytes] = None
        self._lock      = threading.Lock()
        self._running   = False
        self._connected = False
        self._frame_count = 0

    @property
    def latest_jpeg(self) -> Optional[bytes]:
        with self._lock:
            return self._latest

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def frame_count(self) -> int:
        return self._frame_count

    def start(self) -> None:
        self._running = True
        threading.Thread(target=self._run, daemon=True,
                         name=f"frame-receiver-{self._name}").start()
        logger.info("FrameReceiver [{}] started — connecting to {}:{}",
                    self._name, self._host, self._port)

    def stop(self) -> None:
        self._running = False

    def _run(self) -> None:
        while self._running:
            try:
                self._connect_and_read()
            except Exception as e:
                if self._running:
                    logger.debug("FrameReceiver [{}]: disconnected ({}), retrying in 2s",
                                 self._name, e)
            self._connected = False
            if self._running:
                time.sleep(2.0)

    def _connect_and_read(self) -> None:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5.0)

        try:
            sock.connect((self._host, self._port))
        except (ConnectionRefusedError, OSError):
            raise ConnectionRefusedError(
                f"Pipeline not ready at {self._host}:{self._port}"
            )

        self._connected = True
        logger.info("FrameReceiver [{}]: connected to pipeline at {}:{}",
                    self._name, self._host, self._port)

        try:
            while self._running:
                raw_len = self._recvall(sock, 4)
                if not raw_len:
                    break
                length = struct.unpack(">I", raw_len)[0]

                jpeg = self._recvall(sock, length)
                if not jpeg:
                    break

                with self._lock:
                    self._latest = jpeg
                self._frame_count += 1

        finally:
            sock.close()

    @staticmethod
    def _recvall(sock: socket.socket, n: int) -> Optional[bytes]:
        data = bytearray()
        while len(data) < n:
            try:
                chunk = sock.recv(n - len(data))
            except socket.timeout:
                continue
            if not chunk:
                return None
            data.extend(chunk)
        return bytes(data)