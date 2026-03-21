#!/usr/bin/env bash
# IcarusEye v2 — Cross-platform setup script
# Detects hardware, reads models/registry.json, prompts for a model,
# downloads it, and writes docker-compose.override.yml + .env.
#
# Environment:
#   HF_TOKEN=hf_xxx   HuggingFace token for private model repos (optional)
set -euo pipefail

BOLD='\033[1m'
CYAN='\033[0;36m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31d'
DIM='\033[2m'
NC='\033[0m'

info()    { echo -e "${CYAN}[setup]${NC} $*"; }
success() { echo -e "${GREEN}[setup]${NC} $*"; }
warn()    { echo -e "${YELLOW}[setup]${NC} $*"; }
error()   { echo -e "${RED}[setup]${NC} $*" >&2; }

REGISTRY="models/registry.json"

# ── Require Python ────────────────────────────────────────────────────────────

PYTHON_BIN=""
for py in python3 python; do
    if command -v "$py" >/dev/null 2>&1; then
        PYTHON_BIN="$py"
        break
    fi
done

if [ -z "$PYTHON_BIN" ]; then
    error "Python 3 is required to run setup. Install it and try again."
    exit 1
fi

# ── Platform detection ────────────────────────────────────────────────────────

OS=$(uname -s)
ARCH=$(uname -m)
PLATFORM="cpu"

is_jetson() {
    [ -f /proc/device-tree/model ] && grep -qi "nvidia jetson" /proc/device-tree/model 2>/dev/null && return 0
    command -v tegrastats >/dev/null 2>&1 && return 0
    return 1
}

has_nvidia_gpu() {
    command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1
}

has_nvidia_container_toolkit() {
    docker info 2>/dev/null | grep -q "Runtimes.*nvidia"
}

# ── Header ────────────────────────────────────────────────────────────────────

echo ""
echo -e "${BOLD}IcarusEye v2 — Setup${NC}"
echo "─────────────────────────────────────────"
info "OS:   $OS"
info "Arch: $ARCH"
echo ""

# ── Detect platform ───────────────────────────────────────────────────────────

if [ "$OS" = "Darwin" ]; then
    warn "macOS — Docker Desktop does not support NVIDIA runtime."
    warn "PyTorchEngine will run on MPS (Apple Silicon) or CPU."
    PLATFORM="cpu"
elif is_jetson; then
    success "NVIDIA Jetson detected."
    PLATFORM="jetson"
elif has_nvidia_gpu; then
    if [ "$ARCH" = "x86_64" ] || [ "$ARCH" = "amd64" ]; then
        success "NVIDIA GPU detected on x86_64."
        if ! has_nvidia_container_toolkit; then
            warn "nvidia-container-toolkit not found — falling back to CPU mode."
            warn "Install it: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html"
            PLATFORM="cpu"
        else
            PLATFORM="gpu"
        fi
    else
        warn "NVIDIA GPU on $ARCH (non-Jetson) — falling back to CPU mode."
        PLATFORM="cpu"
    fi
else
    info "No NVIDIA GPU found — using PyTorchEngine (CPU/MPS)."
    PLATFORM="cpu"
fi

echo ""

# ── Check registry ────────────────────────────────────────────────────────────

if [ ! -f "$REGISTRY" ]; then
    error "Model registry not found at ${REGISTRY}"
    error "Expected: $(pwd)/${REGISTRY}"
    exit 1
fi

# ── Read registry and build menu via Python ───────────────────────────────────
# Outputs one line per model: INDEX|NAME|DISPLAY|DESCRIPTION|FILENAME|SOURCE|URL

REGISTRY_LINES=$("$PYTHON_BIN" - <<'PYEOF'
import json, sys

with open("models/registry.json") as f:
    registry = json.load(f)

models = registry.get("models", [])
if not models:
    print("ERROR: No models found in registry.json", file=sys.stderr)
    sys.exit(1)

for i, m in enumerate(models, 1):
    name     = m.get("name", "")
    display  = m.get("display_name", name)
    desc     = m.get("description", "")
    filename = m.get("filename", name + ".pt")
    source   = m.get("source", "ultralytics")
    url      = m.get("url", "")
    print(f"{i}|{name}|{display}|{desc}|{filename}|{source}|{url}")
PYEOF
)

# ── Print model menu ──────────────────────────────────────────────────────────

echo -e "${BOLD}Select a model:${NC}"
echo ""

while IFS='|' read -r idx name display desc filename source url; do
    # Flag models that need a URL but don't have one yet
    if [ "$source" = "url" ] && [ -z "$url" ]; then
        echo -e "  ${BOLD}${idx})${NC} ${display}  ${DIM}${desc}${NC}  ${YELLOW}[URL not set in registry.json]${NC}"
    else
        echo -e "  ${BOLD}${idx})${NC} ${display}  ${DIM}${desc}${NC}"
    fi
done <<< "$REGISTRY_LINES"

REGISTRY_COUNT=$(echo "$REGISTRY_LINES" | wc -l | tr -d ' ')
CUSTOM_IDX=$((REGISTRY_COUNT + 1))
echo -e "  ${BOLD}${CUSTOM_IDX})${NC} Custom  ${DIM}enter a name + URL manually${NC}"
echo ""

# ── Prompt for selection ──────────────────────────────────────────────────────

while true; do
    read -rp "  Choice [1-${CUSTOM_IDX}] (default: 1): " MODEL_CHOICE
    MODEL_CHOICE="${MODEL_CHOICE:-1}"

    if [ "$MODEL_CHOICE" = "$CUSTOM_IDX" ]; then
        read -rp "  Model name (without extension, e.g. yolo26m): " MODEL_NAME
        read -rp "  Download URL (leave blank to use ultralytics API): " MODEL_URL
        MODEL_FILENAME="${MODEL_NAME}.pt"
        MODEL_SOURCE="url"
        [ -z "$MODEL_URL" ] && MODEL_SOURCE="ultralytics"
        break
    fi

    # Validate numeric range
    if ! echo "$MODEL_CHOICE" | grep -qE '^[0-9]+$'; then
        warn "Enter a number between 1 and ${CUSTOM_IDX}."; continue
    fi
    if [ "$MODEL_CHOICE" -lt 1 ] || [ "$MODEL_CHOICE" -gt "$REGISTRY_COUNT" ]; then
        warn "Enter a number between 1 and ${CUSTOM_IDX}."; continue
    fi

    # Extract selected row
    SELECTED=$(echo "$REGISTRY_LINES" | sed -n "${MODEL_CHOICE}p")
    IFS='|' read -r _ MODEL_NAME _ _ MODEL_FILENAME MODEL_SOURCE MODEL_URL <<< "$SELECTED"

    # Guard: url source with no URL
    if [ "$MODEL_SOURCE" = "url" ] && [ -z "$MODEL_URL" ]; then
        warn "This model has no URL set in ${REGISTRY}."
        warn "Add a \"url\" for \"${MODEL_NAME}\" and re-run, or choose a different model."
        continue
    fi

    break
done

echo ""
info "Selected: ${BOLD}${MODEL_NAME}${NC} (${MODEL_FILENAME})"
echo ""

# ── Resolve container model path ──────────────────────────────────────────────

mkdir -p models

case "$PLATFORM" in
    jetson|gpu)
        ENGINE_FILENAME="${MODEL_FILENAME%.pt}.engine"
        MODEL_PATH="/app/models/${ENGINE_FILENAME}"
        MODEL_PATH_NOTE="TensorRTEngine — auto-builds from ${MODEL_FILENAME} on first run if .engine absent"
        ;;
    cpu)
        MODEL_PATH="/app/models/${MODEL_FILENAME}"
        MODEL_PATH_NOTE="PyTorchEngine — loads .pt directly (MPS on Apple Silicon, else CPU)"
        ;;
esac

# ── Download model ────────────────────────────────────────────────────────────

PT_DEST="models/${MODEL_FILENAME}"

if [ -f "$PT_DEST" ]; then
    success "${PT_DEST} already exists — skipping download."
else
    info "Downloading ${MODEL_FILENAME}..."

    if [ "$MODEL_SOURCE" = "ultralytics" ]; then
        # Use the Ultralytics Python API — handles mirrors and caching automatically
        if ! "$PYTHON_BIN" -c "import ultralytics" 2>/dev/null; then
            error "ultralytics not installed on this machine."
            error "Run: pip install ultralytics"
            error "Then re-run setup.sh."
            exit 1
        fi
        (
            cd models
            "$PYTHON_BIN" - <<PYEOF
from ultralytics import YOLO
import shutil, pathlib, os

filename = "${MODEL_FILENAME}"
model = YOLO(filename)

# ultralytics may save to a cache dir — ensure it lands in models/
dest = pathlib.Path(filename)
if not dest.exists():
    cache_candidates = [
        pathlib.Path.home() / ".config" / "Ultralytics" / filename,
        pathlib.Path.home() / ".cache"  / "ultralytics" / filename,
    ]
    for src in cache_candidates:
        if src.exists():
            shutil.copy2(src, dest)
            print(f"Copied from cache: {src} -> {dest}")
            break
PYEOF
        )

    else
        # Direct URL download — works for HuggingFace, GitHub releases, S3, etc.
        HF_TOKEN="${HF_TOKEN:-}"
        CURL_ARGS=("-L" "--progress-bar" "-o" "$PT_DEST")
        [ -n "$HF_TOKEN" ] && CURL_ARGS+=("-H" "Authorization: Bearer ${HF_TOKEN}")

        if command -v curl >/dev/null 2>&1; then
            curl "${CURL_ARGS[@]}" "$MODEL_URL"
        elif command -v wget >/dev/null 2>&1; then
            WGET_ARGS=("-O" "$PT_DEST" "--show-progress")
            [ -n "$HF_TOKEN" ] && WGET_ARGS+=("--header" "Authorization: Bearer ${HF_TOKEN}")
            wget "${WGET_ARGS[@]}" "$MODEL_URL"
        else
            # Fall back to Python urllib
            "$PYTHON_BIN" - <<PYEOF
import urllib.request, os, sys

url   = "${MODEL_URL}"
dest  = "${PT_DEST}"
token = os.environ.get("HF_TOKEN", "")

req = urllib.request.Request(url)
if token:
    req.add_header("Authorization", f"Bearer {token}")

print(f"Downloading {url} ...")
urllib.request.urlretrieve(url, dest)
print(f"Saved to {dest}")
PYEOF
        fi
    fi

    if [ -f "$PT_DEST" ]; then
        success "Downloaded → ${PT_DEST}"
    else
        error "Download failed — ${PT_DEST} not found after download."
        exit 1
    fi
fi

# ── TensorRT engine status ────────────────────────────────────────────────────

if [ "$PLATFORM" = "jetson" ] || [ "$PLATFORM" = "gpu" ]; then
    ENGINE_DEST="models/${ENGINE_FILENAME}"
    if [ -f "$ENGINE_DEST" ]; then
        success "Found ${ENGINE_DEST} — TensorRT engine ready."
    else
        info "${ENGINE_FILENAME} not found — will be auto-built from ${MODEL_FILENAME} on first pipeline start."
        [ "$PLATFORM" = "jetson" ] && \
            info "(Export runs inside the l4t-ml container using its bundled TensorRT version.)"
    fi
fi

# ── Write docker-compose.override.yml ────────────────────────────────────────

OVERRIDE="docker-compose.override.yml"
info "Writing ${OVERRIDE}..."

case "$PLATFORM" in
    jetson)
        cat > "$OVERRIDE" <<EOF
# IcarusEye v2 — Jetson overlay (generated by setup.sh)
services:
  pipeline:
    build:
      context: .
      dockerfile: docker/pipeline/Dockerfile.jetson
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
      - NVIDIA_DRIVER_CAPABILITIES=all
      - MODEL_PATH=${MODEL_PATH}
    devices:
      - /dev/v4l2-nvdec:/dev/v4l2-nvdec
      - /dev/v4l2-nvenc:/dev/v4l2-nvenc
      - /dev/nvhost-nvdec:/dev/nvhost-nvdec
EOF
        ;;
    gpu)
        cat > "$OVERRIDE" <<EOF
# IcarusEye v2 — x86_64 NVIDIA GPU overlay (generated by setup.sh)
services:
  pipeline:
    build:
      context: .
      dockerfile: docker/pipeline/Dockerfile.gpu
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
      - NVIDIA_DRIVER_CAPABILITIES=all
      - MODEL_PATH=${MODEL_PATH}
EOF
        ;;
    cpu)
        cat > "$OVERRIDE" <<EOF
# IcarusEye v2 — CPU / cross-platform overlay (generated by setup.sh)
services:
  pipeline:
    environment:
      - MODEL_PATH=${MODEL_PATH}
EOF
        ;;
esac

success "Wrote ${OVERRIDE}"

# ── Write .env if missing ─────────────────────────────────────────────────────

if [ ! -f .env ]; then
    info "Creating .env from .env.example..."
    cp .env.example .env
    success "Created .env — edit it to set HLS_SOURCE, CAPTURE_TYPE, etc."
else
    info ".env already exists — skipping."
fi

# ── Sample directory hint ─────────────────────────────────────────────────────

if [ ! -d sample ]; then
    echo ""
    warn "No sample/ directory. For file-based capture:"
    warn "  mkdir sample && cp your_video.mp4 sample/"
    warn "  Set CAPTURE_TYPE=file and HLS_SOURCE=/app/sample/your_video.mp4 in .env"
fi

# ── Summary ───────────────────────────────────────────────────────────────────

echo ""
echo "─────────────────────────────────────────"
success "Setup complete!"
echo ""
echo -e "  Platform:  ${BOLD}${PLATFORM}${NC}"
echo -e "  Model:     ${BOLD}${MODEL_NAME}${NC}  (${MODEL_PATH_NOTE})"
echo ""
echo -e "  Start:  ${BOLD}docker-compose up --build${NC}"
echo -e "  UI:     ${BOLD}http://localhost:8080${NC}"
echo ""
