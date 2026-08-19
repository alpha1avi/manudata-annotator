"""Text drawing with a legible halo.

The obvious way to outline Hershey text — draw it thick in black, then
thin in colour at the same origin — is subtly wrong. OpenCV's glyph
*advance* scales with stroke thickness, so the two passes drift apart
along the string and a long caption ends with a ghost of its own last
few characters sticking out to the right.

Drawing the halo as offset copies at the *same* thickness keeps both
passes on identical glyph positions, so the outline stays registered to
the fill no matter how long the string is.
"""

from __future__ import annotations

from typing import Sequence, Tuple

import cv2
import numpy as np

FONT = cv2.FONT_HERSHEY_SIMPLEX
HALO_COLOR = (0, 0, 0)

# 4-neighbour halo: enough to lift text off busy footage, and a third
# the cost of the 8-neighbour version at this stroke weight.
_OFFSETS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def draw(
    img: np.ndarray,
    text: str,
    org: Tuple[int, int],
    scale: float,
    color: Sequence[int],
    thickness: int = 1,
    halo: bool = True,
    halo_px: int = 1,
) -> int:
    """Draw *text* at *org*; return the x coordinate just past it."""
    x, y = int(org[0]), int(org[1])
    if halo:
        for dx, dy in _OFFSETS:
            cv2.putText(img, text, (x + dx * halo_px, y + dy * halo_px), FONT,
                        scale, HALO_COLOR, thickness, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), FONT, scale, tuple(int(c) for c in color),
                thickness, cv2.LINE_AA)
    return x + width_of(text, scale, thickness)


def width_of(text: str, scale: float, thickness: int = 1) -> int:
    (w, _), _ = cv2.getTextSize(text, FONT, scale, thickness)
    return int(w)


def size_of(text: str, scale: float, thickness: int = 1) -> Tuple[int, int]:
    (w, h), _ = cv2.getTextSize(text, FONT, scale, thickness)
    return int(w), int(h)


def draw_centered(
    img: np.ndarray,
    text: str,
    center: Tuple[int, int],
    scale: float,
    color: Sequence[int],
    thickness: int = 1,
) -> None:
    """Draw *text* centred on *center*."""
    w, h = size_of(text, scale, thickness)
    draw(img, text, (int(center[0] - w / 2), int(center[1] + h / 2)),
         scale, color, thickness)
