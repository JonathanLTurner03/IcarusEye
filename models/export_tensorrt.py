"""
IcarusEye v2 — TensorRT Engine Export
models/export_tensorrt.py

Converts a YOLO .pt checkpoint to a TensorRT .engine file for Jetson deployment.

Run this DIRECTLY ON THE JETSON — TensorRT engines are hardware-specific.
An engine built on x86 will NOT work on Jetson, and vice versa.

Usage:
    # Basic export (INT8, recommended for Orin)
    python models/export_tensorrt.py

    # Custom model or output path
    python models/export_tensorrt.py --model models/best.pt --output models/custom.engine

    # FP16 instead of INT8 (faster export, slightly lower accuracy)
    python models/export_tensorrt.py --fp16

    # Different image size (must match what you train/infer with)
    python models/export_tensorrt.py --imgsz 1280

Requirements:
    pip install ultralytics
    TensorRT 8.x (included in JetPack 6.x — already on your Orin)

Output:
    models/yolo26s_visdrone.engine   (or --output path)

After export, verify pipeline.yaml points to the right file:
    models:
      - engine: models/yolo26s_visdrone.engine
        mock: false
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export YOLO .pt checkpoint to TensorRT .engine for Jetson"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="models/yolo26s_visdrone.pt",
        help="Path to input .pt checkpoint (default: models/yolo26s_visdrone.pt)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="models/yolo26s_visdrone.engine",
        help="Output .engine path (default: models/yolo26s_visdrone.engine)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Inference image size — must match training size (default: 640)",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Use FP16 precision instead of INT8 (faster export, slightly lower accuracy)",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=1,
        help="TensorRT batch size (default: 1 — pipeline processes one frame at a time)",
    )
    parser.add_argument(
        "--workspace",
        type=int,
        default=4,
        help="TensorRT workspace size in GB (default: 4)",
    )
    return parser.parse_args()


def check_prerequisites() -> None:
    """Verify ultralytics and TensorRT are available before starting."""
    try:
        import ultralytics
        print(f"  ultralytics {ultralytics.__version__} — OK")
    except ImportError:
        print("ERROR: ultralytics not installed.")
        print("  Run: pip install ultralytics")
        sys.exit(1)

    try:
        import tensorrt as trt
        print(f"  TensorRT {trt.__version__} — OK")
    except ImportError:
        # TensorRT may be present via system install without Python bindings
        # Ultralytics will still find it — this is just a warning not a hard fail
        print("  TensorRT Python bindings not found — Ultralytics will use system TRT")
        print("  This is normal on Jetson with JetPack 6.x")

    try:
        import torch
        cuda_ok = torch.cuda.is_available()
        device_name = torch.cuda.get_device_name(0) if cuda_ok else "none"
        print(f"  PyTorch {torch.__version__}, CUDA={cuda_ok}, device={device_name}")
        if not cuda_ok:
            print("ERROR: CUDA not available. Export must run on the Jetson.")
            sys.exit(1)
    except ImportError:
        print("ERROR: PyTorch not installed.")
        sys.exit(1)


def export(args: argparse.Namespace) -> Path:
    """Run the export and return the path to the output engine file."""
    from ultralytics import YOLO

    model_path  = Path(args.model)
    output_path = Path(args.output)

    if not model_path.exists():
        print(f"\nERROR: Model not found at {model_path}")
        print("Make sure you have a trained .pt checkpoint.")
        print("Download pretrained weights:  yolo26s.pt  (COCO pretrained)")
        print("Or copy your VisDrone fine-tuned best.pt here and rename it.")
        sys.exit(1)

    precision = "INT8" if not args.fp16 else "FP16"
    print(f"\nExporting {model_path.name} → TensorRT {precision}")
    print(f"  imgsz={args.imgsz}  batch={args.batch}  workspace={args.workspace}GB")
    print(f"  Output: {output_path}")
    print()

    t0 = time.monotonic()

    model = YOLO(str(model_path))
    model.export(
        format="engine",
        device=0,
        imgsz=args.imgsz,
        int8=not args.fp16,
        half=args.fp16,
        batch=args.batch,
        workspace=args.workspace,
        verbose=True,
    )

    # Ultralytics saves the engine alongside the .pt file by default
    # e.g. models/yolo26s_visdrone.pt → models/yolo26s_visdrone.engine
    auto_output = model_path.with_suffix(".engine")

    elapsed = time.monotonic() - t0
    print(f"\nExport complete in {elapsed:.0f}s")

    # Move to requested output path if different
    if auto_output != output_path and auto_output.exists():
        output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(auto_output), str(output_path))
        print(f"Moved engine to {output_path}")

    return output_path


def verify(engine_path: Path, imgsz: int) -> None:
    """
    Quick smoke test — load the engine and run one dummy inference.
    Confirms the engine is valid before you update pipeline.yaml.
    """
    from ultralytics import YOLO
    import numpy as np

    print(f"\nVerifying engine: {engine_path}")
    model = YOLO(str(engine_path), task="detect")

    dummy = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)

    # First run triggers TRT kernel compilation — expect it to be slow
    print("  Warmup run 1/3...")
    model(dummy, verbose=False)
    print("  Warmup run 2/3...")
    model(dummy, verbose=False)
    print("  Warmup run 3/3...")
    model(dummy, verbose=False)

    # Timed run
    import time
    t0 = time.monotonic()
    model(dummy, verbose=False)
    inf_ms = (time.monotonic() - t0) * 1000

    print(f"  Inference time (post-warmup): {inf_ms:.1f}ms")
    print(f"  Engine verified OK")


def print_next_steps(engine_path: Path) -> None:
    print("\n" + "─" * 60)
    print("Next steps:")
    print()
    print(f"1. Engine is at: {engine_path}")
    print()
    print("2. Update configs/pipeline.yaml:")
    print("     models:")
    print(f"       - engine: {engine_path}")
    print("         mock: false")
    print()
    print("3. Update .env:")
    print("     DEV_MODE=false")
    print()
    print("4. Rebuild and run:")
    print("     docker compose up -d --build")
    print()
    print("5. Watch logs for TensorRTEngine loading:")
    print("     docker compose logs -f pipeline")
    print("─" * 60)


def main() -> None:
    args = parse_args()

    print("IcarusEye v2 — TensorRT Export")
    print("─" * 60)
    print("Checking prerequisites...")
    check_prerequisites()

    engine_path = export(args)

    if engine_path.exists():
        verify(engine_path, args.imgsz)
        print_next_steps(engine_path)
    else:
        print(f"\nERROR: Engine file not found at {engine_path} after export.")
        print("Check the ultralytics output above for errors.")
        sys.exit(1)


if __name__ == "__main__":
    main()