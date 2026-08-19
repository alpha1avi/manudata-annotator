"""Whole-video visibility timeline, with a playhead.

A 16:9 source letterboxed into the left column leaves a band above and
below it. Rather than pad it with black, it carries a strip that shows,
for the entire video at once, where each hand was tracked, where it was
held as a ghost, where it was visible but unrecovered, and where no hand
was in view.

This is the single most useful thing a technical evaluator can be handed:
it turns "how often does this fail, and is it clustered or spread?" from
a question requiring a full watch-through into something answerable at a
glance, and it makes the gaps impossible to miss rather than easy to
overlook. Volunteering that is the point.

The strip is rendered once per video and then blitted, so the per-frame
cost is a memcpy plus one line for the playhead.
"""

from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np

from qc.config import HAND_LABELS, TEXT_DIM, WRIST_COLORS
from qc.pose.gapfill import ABSENT, FADING, HELD, TRACKED, UNRECOVERED, RenderPlan
from qc.pose.schema import N_HANDS

ROW_H = 12
ROW_GAP = 4
LABEL_W = 20
AXIS_H = 14

COLOR_ABSENT = (38, 36, 34)
COLOR_UNRECOVERED = (70, 150, 210)
PLAYHEAD = (250, 250, 250)

FONT = cv2.FONT_HERSHEY_SIMPLEX


def strip_height() -> int:
    return N_HANDS * ROW_H + (N_HANDS - 1) * ROW_GAP + AXIS_H


class VisibilityTimeline:
    """Pre-rendered per-hand status strip for one video."""

    def __init__(self, plan: RenderPlan, width: int, fps: float) -> None:
        self.width = int(width)
        self.height = strip_height()
        self.n_frames = plan.n_frames
        self.fps = fps
        self._strip = self._render_strip(plan)

    def _render_strip(self, plan: RenderPlan) -> np.ndarray:
        strip = np.zeros((self.height, self.width, 3), np.uint8)
        bar_w = max(1, self.width - LABEL_W)

        # Map each output column to a frame range, then let the worst
        # status in that range win. A single dropped frame must stay
        # visible after downsampling 20 000 frames into 1 100 pixels —
        # averaging would hide exactly what this strip exists to show.
        edges = np.linspace(0, self.n_frames, bar_w + 1).astype(np.int64)

        for h in range(N_HANDS):
            y0 = h * (ROW_H + ROW_GAP)
            status_h = plan.status[:, h]
            row = strip[y0:y0 + ROW_H, LABEL_W:]

            for x in range(bar_w):
                lo, hi = edges[x], max(edges[x] + 1, edges[x + 1])
                row[:, x] = _worst_color(status_h[lo:hi], h)

            cv2.putText(strip, HAND_LABELS[h], (2, y0 + ROW_H - 2), FONT, 0.33,
                        WRIST_COLORS[h], 1, cv2.LINE_AA)

        self._draw_time_axis(strip, bar_w)
        return strip

    def _draw_time_axis(self, strip: np.ndarray, bar_w: int) -> None:
        y = self.height - AXIS_H
        duration = self.n_frames / self.fps if self.fps else 0.0
        if duration <= 0:
            return

        step = _tick_step(duration)
        t = 0.0
        while t <= duration + 1e-6:
            x = LABEL_W + int(bar_w * (t / duration))
            x = min(x, self.width - 1)
            cv2.line(strip, (x, y), (x, y + 3), TEXT_DIM, 1)
            label = f"{int(t)}s"
            (tw, _), _ = cv2.getTextSize(label, FONT, 0.3, 1)
            tx = min(max(x - tw // 2, 0), self.width - tw)
            cv2.putText(strip, label, (tx, y + 12), FONT, 0.3, TEXT_DIM, 1, cv2.LINE_AA)
            t += step

    def base_strip(self) -> np.ndarray:
        """The static strip, without a playhead.

        The compositor blits this once into a prebuilt band and draws only
        the playhead per frame, so the strip is never re-rendered.
        """
        return self._strip

    def playhead_x(self, frame_index: int) -> int:
        """Playhead offset in strip-local x coordinates."""
        bar_w = max(1, self.width - LABEL_W)
        frac = frame_index / max(1, self.n_frames - 1)
        return LABEL_W + int(round(frac * (bar_w - 1)))

    def playhead_height(self) -> int:
        return self.height - AXIS_H - 2

    def render(self, frame_index: int) -> np.ndarray:
        """The strip with a playhead at *frame_index*."""
        out = self._strip.copy()
        cv2.line(out, (self.playhead_x(frame_index), 0),
                 (self.playhead_x(frame_index), self.playhead_height()), PLAYHEAD, 1)
        return out


def _worst_color(statuses: np.ndarray, hand: int) -> Tuple[int, int, int]:
    """Colour for a column, biased toward reporting problems."""
    if statuses.size == 0:
        return COLOR_ABSENT
    if (statuses == UNRECOVERED).any():
        return COLOR_UNRECOVERED
    if ((statuses == HELD) | (statuses == FADING)).any():
        return _dim(WRIST_COLORS[hand], 0.45)
    if (statuses == TRACKED).any():
        return WRIST_COLORS[hand]
    return COLOR_ABSENT


def _dim(color, factor: float) -> Tuple[int, int, int]:
    return tuple(int(c * factor) for c in color)


def _tick_step(duration: float) -> float:
    for step in (1, 2, 5, 10, 15, 30, 60, 120, 300):
        if duration / step <= 12:
            return float(step)
    return 600.0
