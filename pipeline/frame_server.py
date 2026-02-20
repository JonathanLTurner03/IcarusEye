"""
IcarusEye v2 — Frame Server
pipeline/frame_server.py

TCP server that streams JPEG frames to tower.
Uses TCP instead of Unix sockets for WSL/cross-platform compatibility.

Protocol:
  Server sends: [4-byte big-endian length][JPEG bytes]
  Repeated for each frame. Client reads length, reads bytes, decodes JPEG.

Two servers run on different ports:
  RAW_PORT       — raw frames from lens
  ANNOTATED_PORT — annotated frames from mark
"""

from __future__ import annotations

import queue
import socket
import struct
import threading
import time
from typing import Optional

import cv2
from loguru import logger

RAW_PORT       = 5100
ANNOTATED_PORT = 5101
BIND_HOST      = "0.0.0.0"

JPEG_QUALITY = 80


class FrameServer:
    """
    Serves frames from a queue over a TCP socket.
    Handles multiple simultaneous clients.
    """

    def __init__(self, port: int, source_queue: queue.Queue, name: str):
        self._port    = port
        self._queue   = source_queue
        self._name    = name
        self._clients: list[socket.socket] = []
        self._lock    = threading.Lock()
        self._running = False

    def start(self) -> None:
        self._running = True
        threading.Thread(target=self._accept_loop, daemon=True,
                         name=f"frame-server-accept-{self._name}").start()
        threading.Thread(target=self._broadcast_loop, daemon=True,
                         name=f"frame-server-broadcast-{self._name}").start()
        logger.info("FrameServer [{}] listening on port {}", self._name, self._port)

    def stop(self) -> None:
        self._running = False

    def _accept_loop(self) -> None:
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind((BIND_HOST, self._port))
        srv.listen(5)
        srv.settimeout(1.0)

        while self._running:
            try:
                conn, addr = srv.accept()
                conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                with self._lock:
                    self._clients.append(conn)
                logger.info("FrameServer [{}]: client connected from {} ({} total)",
                            self._name, addr, len(self._clients))
            except socket.timeout:
                continue
            except Exception as e:
                if self._running:
                    logger.warning("FrameServer [{}] accept error: {}", self._name, e)

        srv.close()

    def _broadcast_loop(self) -> None:
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]

        while self._running:
            try:
                frame = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue

            ok, buf = cv2.imencode(".jpg", frame.data, encode_params)
            if not ok:
                continue
            jpeg = buf.tobytes()
            packet = struct.pack(">I", len(jpeg)) + jpeg

            dead = []
            with self._lock:
                clients = list(self._clients)

            for conn in clients:
                try:
                    conn.sendall(packet)
                except (BrokenPipeError, ConnectionResetError, OSError):
                    dead.append(conn)

            if dead:
                with self._lock:
                    for c in dead:
                        try:
                            c.close()
                        except Exception:
                            pass
                        if c in self._clients:
                            self._clients.remove(c)
                logger.info("FrameServer [{}]: {} client(s) disconnected",
                            self._name, len(dead))

        logger.info("FrameServer [{}] stopped", self._name)