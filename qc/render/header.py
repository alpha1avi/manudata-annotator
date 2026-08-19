"""The burned-in header bar.

One line, semi-transparent over the top of the composited frame rather
than a solid strip above it, so it costs no vertical space and reads as
an overlay on the footage instead of chrome around it.

Carries the provenance an evaluator needs without being asked: which
site and task, the true source resolution and frame rate, and exactly
which frame they are looking at. The status tag on the right names the
*reason* a skeleton is dimmed or absent, so a viewer never has to guess
whether they are seeing occlusion or a tracking failure.
"""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np

from qc.config import (
    HEADER_BG,
    HEADER_H,
    TEXT_DIM,
    TEXT_LOST,
    TEXT_PRIMARY,
    TEXT_WARN,
    WORDMARK,
)
from qc.pose.gapfill import TAG_NO_HAND, TAG_UNRECOVERED
from qc.render.text import draw as text_draw
from qc.render.text import width_of

FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.44
FONT_WEIGHT = 1
BAR_OPACITY = 0.72
PAD_X = 14
SEPARATOR = "·"

TAG_COLORS = {
    TAG_UNRECOVERED: TEXT_WARN,
    TAG_NO_HAND: TEXT_LOST,
}


def draw_header(
    canvas: np.ndarray,
    site: str,
    task: str,
    width: int,
    height: int,
    fps: float,
    frame_index: int,
    total_frames: int,
    status_tag: str = "",
    bar_height: int = HEADER_H,
) -> np.ndarray:
    """Blend the header bar over the top of *canvas* in place."""
    strip = canvas[:bar_height]
    tint = np.empty_like(strip)
    tint[:] = HEADER_BG
    cv2.addWeighted(tint, BAR_OPACITY, strip, 1.0 - BAR_OPACITY, 0.0, dst=strip)

    baseline = int(bar_height * 0.68)

    # Left: wordmark, then site and task.
    x = PAD_X
    x = _text(canvas, WORDMARK, x, baseline, TEXT_PRIMARY)
    x = _text(canvas, SEPARATOR, x + 8, baseline, TEXT_DIM)
    x = _text(canvas, site or "unknown site", x + 8, baseline, TEXT_PRIMARY)
    x = _text(canvas, SEPARATOR, x + 8, baseline, TEXT_DIM)
    x = _text(canvas, task or "unlabelled task", x + 8, baseline, TEXT_PRIMARY)

    if status_tag:
        x = _text(canvas, SEPARATOR, x + 10, baseline, TEXT_DIM)
        _text(canvas, status_tag, x + 8,
              baseline, TAG_COLORS.get(status_tag, TEXT_WARN))

    # Right: source properties and the frame counter, right-aligned.
    digits = max(len(str(total_frames)), 1)
    right = (
        f"{width}x{height} {SEPARATOR} {fps:.2f} fps {SEPARATOR} "
        f"frame {frame_index:0{digits}d} / {total_frames}"
    )
    tw = width_of(right, FONT_SCALE, FONT_WEIGHT)
    _text(canvas, right, canvas.shape[1] - tw - PAD_X, baseline, TEXT_DIM)

    return canvas


def _text(
    canvas: np.ndarray,
    text: str,
    x: int,
    y: int,
    color: Tuple[int, int, int],
) -> int:
    """Draw *text* and return the x coordinate just past it."""
    # No halo: the header already sits on its own tinted bar, and a halo
    # on every field would muddy a 28px line.
    return text_draw(canvas, text, (x, y), FONT_SCALE, color, FONT_WEIGHT,
                     halo=False)
