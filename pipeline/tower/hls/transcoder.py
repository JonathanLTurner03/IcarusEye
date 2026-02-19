"""
Transcoder manages the ffmpeg subprocess that converts any video source
into HLS segments Tower can serve to the browser.

Supports any source ffmpeg understands:
  - Local file:   sample/sample.mp4
  - RTSP stream:  rtsp://192.168.1.x:554/stream
  - RTP:          rtp://127.0.0.1:5000
  - V4L2 device:  /dev/video0
"""
import asyncio
import os
import shutil
from loguru import logger


class Transcoder:
    def __init__(
        self,
        source: str,
        segment_dir: str = "/tmp/icaruseye_hls",
        segment_duration: int = 2,
        playlist_size: int = 5,
    ):
        self.source          = source
        self.segment_dir     = segment_dir
        self.segment_duration = segment_duration
        self.playlist_size   = playlist_size
        self._process: asyncio.subprocess.Process = None

    # ── Lifecycle ──

    async def start(self):
        os.makedirs(self.segment_dir, exist_ok=True)
        cmd = self._build_ffmpeg_cmd()
        logger.info(f"Transcoder starting: {' '.join(cmd)}")

        self._process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
        )
        # Give ffmpeg a moment to write the first segment
        await asyncio.sleep(2)
        logger.info(f"Transcoder running — segments at {self.segment_dir}")

    async def stop(self):
        if self._process and self._process.returncode is None:
            self._process.terminate()
            try:
                await asyncio.wait_for(self._process.wait(), timeout=5)
            except asyncio.TimeoutError:
                self._process.kill()
            logger.info("Transcoder stopped")

        # Clean up segments
        if os.path.exists(self.segment_dir):
            shutil.rmtree(self.segment_dir, ignore_errors=True)

    @property
    def is_running(self) -> bool:
        return self._process is not None and self._process.returncode is None

    # ── Source swap (TODO: called from control API later) ──

    async def change_source(self, new_source: str):
        logger.info(f"Transcoder changing source: {self.source} → {new_source}")
        self.source = new_source
        await self.stop()
        await self.start()

    # ── Internal ──

    def _build_ffmpeg_cmd(self) -> list[str]:
        playlist_path = os.path.join(self.segment_dir, "index.m3u8")
        segment_path  = os.path.join(self.segment_dir, "seg%03d.ts")

        # Detect if source is a loopable file (not a live stream)
        is_file = os.path.isfile(self.source)

        cmd = ["ffmpeg", "-y"]

        if is_file:
            cmd += ["-stream_loop", "-1"]           # loop the file indefinitely

        cmd += [
            "-i", self.source,
            "-c:v", "libx264",
            "-preset", "ultrafast",                 # low latency encode
            "-tune", "zerolatency",
            "-profile:v", "baseline",               # broadest browser compatibility
            "-level", "3.0",
            "-crf", "23",
            "-g", str(self.segment_duration * 30),  # keyframe every segment
            "-sc_threshold", "0",
            "-an",                                  # no audio for now
            "-f", "hls",
            "-hls_time", str(self.segment_duration),
            "-hls_list_size", str(self.playlist_size),
            "-hls_flags", "delete_segments+append_list",
            "-hls_segment_filename", segment_path,
            playlist_path,
        ]
        return cmd