"""
IcarusEye v2 — Mark Annotator
pipeline/mark/annotator.py

Draws detection bounding boxes and labels onto frames.
Receives DetectionResult list directly (not via Redis — same process).

Design:
  - Mark runs in the same hawk_mark_thread as hawk.
  - Hawk infers, gets InferenceResult, passes it directly to mark.
  - Mark draws onto a copy of the frame and returns the annotated Frame.
  - No Redis round-trip for frames — Redis only carries the DetectionMessage
    (metadata) so tower's stats panel can display detection counts.

Visual style:
  - Colour per class (consistent hue derived from class_id)
  - Label: "<class> <conf%>" in a filled pill above the box
  - Boxes are 2px, rounded-feel via line type AA
  - HUD overlay: detection count + inference time (bottom-left corner)
"""

from __future__ import annotations

import colorsys
import time

import cv2
import numpy as np

from pipeline.hawk.base import InferenceResult
from pipeline.lens.base import Frame


# ── Colour palette ────────────────────────────────────────────────────────────
# One saturated colour per class_id, consistent across frames
_MAX_CLASSES = 80
_PALETTE: list[tuple[int, int, int]] = []


def _build_palette(n: int) -> list[tuple[int, int, int]]:
    """Generate n evenly-spaced saturated colours in BGR."""
    colours = []
    for i in range(n):
        hue = i / n
        r, g, b = colorsys.hsv_to_rgb(hue, 0.85, 0.95)
        colours.append((int(b * 255), int(g * 255), int(r * 255)))  # BGR
    return colours


_PALETTE = _build_palette(_MAX_CLASSES)


def _class_colour(class_id: int) -> tuple[int, int, int]:
    return _PALETTE[class_id % _MAX_CLASSES]


# ── Font settings ─────────────────────────────────────────────────────────────
_FONT       = cv2.FONT_HERSHEY_SIMPLEX
_FONT_SCALE = 0.45
_FONT_THICK = 1
_LINE_TYPE  = cv2.LINE_AA
_BOX_THICK  = 2

# HUD overlay settings
_HUD_FONT_SCALE = 0.4
_HUD_COLOUR     = (0, 212, 255)   # IcarusEye accent cyan — BGR


def draw(frame: Frame, result: InferenceResult) -> Frame:
    """
    Draw detections from InferenceResult onto frame.

    Returns a new Frame with annotations drawn.
    The original frame.data is not mutated — we copy before drawing.
    """
    img = frame.data.copy()
    h, w = img.shape[:2]

    for det in result.detections:
        x1, y1, x2, y2 = det.bbox
        # Denormalise
        px1 = int(x1 * w)
        py1 = int(y1 * h)
        px2 = int(x2 * w)
        py2 = int(y2 * h)

        colour = _class_colour(det.class_id)

        # Bounding box
        cv2.rectangle(img, (px1, py1), (px2, py2), colour, _BOX_THICK, _LINE_TYPE)

        # Label text
        label_text = f"{det.label} {det.confidence:.0%}"
        (tw, th), baseline = cv2.getTextSize(label_text, _FONT, _FONT_SCALE, _FONT_THICK)

        # Label pill background (above the box, clamped to frame)
        label_y1 = max(0, py1 - th - baseline - 4)
        label_y2 = max(th + baseline + 4, py1)
        label_x2 = min(w, px1 + tw + 6)

        cv2.rectangle(img, (px1, label_y1), (label_x2, label_y2), colour, cv2.FILLED)
        cv2.putText(
            img, label_text,
            (px1 + 3, label_y2 - baseline - 1),
            _FONT, _FONT_SCALE,
            (0, 0, 0),  # black text on coloured pill
            _FONT_THICK, _LINE_TYPE,
        )

    # HUD overlay — bottom-left
    n_det = len(result.detections)
    hud_lines = [
        f"DET: {n_det}",
        f"INF: {result.inference_ms:.1f}ms",
    ]
    hud_y = h - (len(hud_lines) * 16) - 8
    for line in hud_lines:
        cv2.putText(
            img, line,
            (8, hud_y),
            _FONT, _HUD_FONT_SCALE,
            _HUD_COLOUR,
            _FONT_THICK, _LINE_TYPE,
        )
        hud_y += 16

    return Frame(
        data=img,
        frame_id=frame.frame_id,
        timestamp=frame.timestamp,
        width=frame.width,
        height=frame.height,
        source=frame.source,
        meta={
            **frame.meta,
            "annotated": True,
            "detection_count": n_det,
            "inference_ms": result.inference_ms,
        },
    )