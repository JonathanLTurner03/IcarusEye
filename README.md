# IcarusEye v2

Real-time aerial object detection and tracking pipeline. Captures video, runs YOLO inference, annotates frames, and serves a live MJPEG feed through a web dashboard.

```
Lens (capture) → Hawk (inference) → Mark (annotate) → TCP frame servers → Tower (web UI)
```

Inter-service communication is via Redis pub/sub. The pipeline and tower run as separate containers sharing a bridge network.

---

## Quick Start

```bash
# 1. Detect your platform and generate config
./setup.sh

# 2. Edit video source (optional — defaults to sample file in dev mode)
nano .env

# 3. Start everything
docker-compose up --build
```

Open **http://localhost:8080** for the live dashboard.

---

## Platform Support

`setup.sh` auto-detects your hardware and writes a `docker-compose.override.yml` that selects the right pipeline image and enables GPU access where available.

| Platform | Dockerfile | Inference | Notes |
|---|---|---|---|
| NVIDIA Jetson (JetPack 6.x) | `Dockerfile.jetson` | TensorRT | L4T base image, HW encode/decode |
| x86\_64 + NVIDIA GPU | `Dockerfile.gpu` | TensorRT (pip) | Requires `nvidia-container-toolkit` |
| macOS / CPU / any arch | `Dockerfile.cpu` | Mock (dev mode) | No GPU needed, file source only |

### Manual platform selection

If you want to bypass auto-detection:

```bash
# Jetson
docker-compose -f docker-compose.yml -f docker-compose.jetson.yml up --build

# x86_64 NVIDIA GPU
docker-compose -f docker-compose.yml -f docker-compose.gpu.yml up --build

# CPU / dev (default)
docker-compose up --build
```

---

## Configuration

Copy `.env.example` to `.env` and adjust:

```bash
cp .env.example .env
```

Key variables:

| Variable | Default | Description |
|---|---|---|
| `DEV_MODE` | `true` (CPU), `false` (GPU/Jetson) | Enables mock inference + file source |
| `HLS_SOURCE` | `/app/sample/sample.mp4` | Video source path or RTSP URL |
| `CAPTURE_TYPE` | `file` | `file`, `v4l2`, `rtsp`, `gstreamer_uri` |
| `CAPTURE_DEVICE` | `/dev/capture0` | V4L2 device path |
| `CAPTURE_URI` | — | RTSP/RTMP/GStreamer URI |
| `REDIS_HOST` | `redis` | Redis hostname |

All pipeline settings (resolution, framerate, model path, confidence threshold, etc.) live in `configs/pipeline.yaml`.

### Dev mode (no hardware)

`DEV_MODE=true` automatically:
- Switches capture to `FileSource` (`HLS_SOURCE` path)
- Replaces `TensorRTEngine` with `MockEngine` (synthetic drifting bounding boxes)
- Requires no GPU, V4L2 device, or `.engine` model file

Place sample videos in `sample/` — they mount to `/app/sample/` inside the container.

---

## TensorRT Engine Export

Required for GPU and Jetson inference. Must be run on the target hardware (engines are arch-specific):

```bash
python models/export_tensorrt.py
```

Exports a `.pt` YOLO model to a `.engine` binary in `models/`. The pipeline auto-detects version mismatches and can rebuild from the `.pt` source.

---

## Architecture

### Services

- **pipeline** — Runs lens + hawk + mark as daemon threads in a single process. Exposes raw frames on TCP port 5100, annotated frames on TCP port 5101.
- **tower** — FastAPI app. Receives frames from pipeline via TCP, serves MJPEG streams and a web dashboard on port 8080.
- **redis** — Message broker for detection metadata, per-component stats, and live control commands.

### Hot-swap source

The capture source can be changed at runtime via the dashboard without restarting the pipeline. Tower publishes a `swap_source` command to the `icaruseye:control` Redis channel; the lens thread picks it up and swaps the active `CaptureSource`.

### Ports

| Port | Service | Description |
|---|---|---|
| 8080 | tower | Web dashboard + MJPEG streams |
| 5100 | pipeline | Raw TCP frame server |
| 5101 | pipeline | Annotated TCP frame server |

---

## File Layout

```
docker/
  pipeline/
    Dockerfile.jetson   # Jetson / L4T
    Dockerfile.gpu      # x86_64 NVIDIA
    Dockerfile.cpu      # CPU / dev / any arch
  tower/
    Dockerfile
docker-compose.yml          # Base (CPU default)
docker-compose.jetson.yml   # Jetson overlay
docker-compose.gpu.yml      # x86_64 GPU overlay
setup.sh                    # Auto-detects platform, writes override
configs/
  pipeline.yaml             # All pipeline settings + defaults
requirements/
  base.txt                  # redis, pyyaml, loguru
  lens.txt                  # ultralytics, numpy (GPU/Jetson)
  pipeline-cpu.txt          # numpy, opencv-headless (CPU dev)
  tower.txt                 # fastapi, uvicorn, opencv-headless
pipeline/
  lens/                     # Capture sources (file, v4l2, rtsp)
  hawk/                     # Inference engines (mock, tensorrt)
  mark/                     # Frame annotation
  tower/                    # FastAPI dashboard
  shared/                   # Config loader, Redis client
  pipeline_service/         # Single-process orchestrator
```
