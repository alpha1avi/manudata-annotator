"""Assemble one output frame from the source frame and the pose track.

This module is where frame-exactness is enforced. Every panel is drawn
from a single ``frame_index`` argument, and the pose for that index is
looked up through the render plan rather than carried along in any
per-panel state. There is no independent counter anywhere in the render
path that could drift, and no panel can advance without the others.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from qc.config import (
    BG_CANVAS,
    BG_VIDEO_PANEL,
    HEADER_H,
    LAYOUT_2PANEL,
    LAYOUT_3PANEL,
    PANEL_DIVIDER,
    TEXT_DIM,
    TEXT_LOST,
    TEXT_WARN,
    RenderConfig,
    VideoMeta,
)
from qc.pose.gapfill import (
    ABSENT,
    TAG_NO_HAND,
    TAG_UNRECOVERED,
    RenderPlan,
    UNRECOVERED,
)
from qc.pose.schema import N_HANDS, PoseTrack
from qc.render.header import draw_header
from qc.render.overlay2d import draw_hands
from qc.render.text import draw as text_draw
from qc.render.text import draw_centered as text_draw_centered
from qc.render.text import width_of as text_width
from qc.render.timeline import PLAYHEAD as PLAYHEAD_COLOR
from qc.render.timeline import VisibilityTimeline, strip_height
from qc.render.viewport3d import Viewport3D, view_center_from_track

CAPTION_FONT = cv2.FONT_HERSHEY_SIMPLEX
CAPTION_SCALE = 0.42
CAPTION_COLOR = TEXT_DIM
CAPTION_PAD = 14


@dataclass
class PanelRect:
    x: int
    y: int
    w: int
    h: int

    def slice_of(self, canvas: np.ndarray) -> np.ndarray:
        return canvas[self.y:self.y + self.h, self.x:self.x + self.w]


class FrameComposer:
    """Builds output frames for one video."""

    def __init__(
        self,
        meta: VideoMeta,
        track: PoseTrack,
        plan: RenderPlan,
        cfg: RenderConfig,
        site: str,
        task: str,
        slam_panel=None,
    ) -> None:
        self.meta = meta
        self.track = track
        self.plan = plan
        self.cfg = cfg
        self.site = site
        self.task = task
        self.slam_panel = slam_panel

        widths = LAYOUT_3PANEL if cfg.with_slam else LAYOUT_2PANEL
        if sum(widths) != cfg.canvas_w:
            raise ValueError(
                f"Panel widths {widths} sum to {sum(widths)}, "
                f"not the canvas width {cfg.canvas_w}"
            )

        self.rects: List[PanelRect] = []
        x = 0
        for w in widths:
            self.rects.append(PanelRect(x=x, y=0, w=w, h=cfg.canvas_h))
            x += w

        self.video_rect = self.rects[0]
        self.view_rect = self.rects[1]
        self.slam_rect = self.rects[2] if cfg.with_slam else None

        # Letterbox the source into the video panel, preserving aspect.
        self.video_scale = min(
            self.video_rect.w / meta.width, self.video_rect.h / meta.height
        )
        self.video_w = int(round(meta.width * self.video_scale))
        self.video_h = int(round(meta.height * self.video_scale))
        self.video_x = self.video_rect.x + (self.video_rect.w - self.video_w) // 2
        self.video_y = self.video_rect.y + (self.video_rect.h - self.video_h) // 2

        self.viewport = Viewport3D(
            width=self.view_rect.w,
            height=self.view_rect.h,
            half_extent=cfg.view_half_extent_m,
            orbit_deg_per_sec=cfg.orbit_deg_per_sec,
        )
        # Fixed for the whole video — see viewport3d for why.
        self.viewport.set_center(view_center_from_track(track.kp3d, track.valid))

        # Letterbox bands around the footage, filled with real information
        # rather than black: quality headline above, visibility timeline
        # below. A 16:9 source in this column leaves ~200px of each.
        self.top_band_h = self.video_y - self.video_rect.y
        bottom_band_h = (
            self.video_rect.y + self.video_rect.h - (self.video_y + self.video_h)
        )
        self.summary_line = _summary_line(track)

        self.timeline = None
        self.timeline_y = 0
        if bottom_band_h >= strip_height() + 30:
            self.timeline = VisibilityTimeline(
                plan, width=self.video_w, fps=meta.fps
            )
            self.timeline_y = self.video_y + self.video_h + 18

        # The letterbox bands are identical on every frame apart from the
        # timeline playhead, so they are drawn once here and blitted. That
        # removes both a full-panel fill and all of the left panel's text
        # rendering from the per-frame path.
        self._top_band = self._build_top_band(self.top_band_h)
        self._bottom_band = self._build_bottom_band(bottom_band_h)

    # ── per-frame assembly ────────────────────────────────────────────

    def compose(self, frame_index: int, frame: np.ndarray) -> np.ndarray:
        """Build the output frame for absolute source index *frame_index*."""
        if not 0 <= frame_index < self.plan.n_frames:
            raise IndexError(
                f"Frame index {frame_index} is outside the pose track "
                f"(0..{self.plan.n_frames - 1}). The video and keypoints "
                "have come apart; refusing to guess."
            )

        # No background fill: the panels tile the canvas exactly, so every
        # pixel is written below. Filling first would cost a 6 MB memset
        # per frame for nothing.
        canvas = np.empty((self.cfg.canvas_h, self.cfg.canvas_w, 3), np.uint8)

        hands = self._hands_at(frame_index)

        self._draw_video_panel(canvas, frame, hands, frame_index)
        self._draw_view_panel(canvas, frame_index, hands)
        if self.slam_rect is not None:
            self._draw_slam_panel(canvas, frame_index)

        for rect in self.rects[1:]:
            cv2.line(canvas, (rect.x, 0), (rect.x, self.cfg.canvas_h),
                     PANEL_DIVIDER, 1, cv2.LINE_AA)

        draw_header(
            canvas,
            site=self.site,
            task=self.task,
            width=self.meta.width,
            height=self.meta.height,
            fps=self.meta.fps,
            frame_index=frame_index,
            total_frames=self.plan.n_frames,
            status_tag=self.plan.header_tag(frame_index),
        )
        return canvas

    def _hands_at(self, frame_index: int) -> List[Tuple[int, np.ndarray, np.ndarray, float]]:
        """``(hand, kp2d, kp3d, alpha)`` for each hand drawable at this frame."""
        out: List[Tuple[int, np.ndarray, np.ndarray, float]] = []
        for h in range(N_HANDS):
            src = int(self.plan.src_index[frame_index, h])
            alpha = float(self.plan.alpha[frame_index, h])
            if src < 0 or alpha <= 0.01:
                continue
            out.append((h, self.track.kp2d[src, h], self.track.kp3d[src, h], alpha))
        return out

    def _draw_video_panel(self, canvas, frame, hands, frame_index) -> None:
        rect = self.video_rect
        x0, x1 = rect.x, rect.x + rect.w
        top = self.video_y
        bottom = self.video_y + self.video_h

        if frame.shape[:2] != (self.meta.height, self.meta.width):
            raise ValueError(
                f"Decoded frame is {frame.shape[1]}x{frame.shape[0]} but the "
                f"video probed as {self.meta.width}x{self.meta.height}"
            )

        if self._top_band.size:
            canvas[rect.y:top, x0:x1] = self._top_band
        if self._bottom_band.size:
            canvas[bottom:rect.y + rect.h, x0:x1] = self._bottom_band

        # Side pillars, when the source is narrower than the column.
        if self.video_x > x0:
            canvas[top:bottom, x0:self.video_x] = BG_VIDEO_PANEL
        if self.video_x + self.video_w < x1:
            canvas[top:bottom, self.video_x + self.video_w:x1] = BG_VIDEO_PANEL

        interp = cv2.INTER_AREA if self.video_scale < 1.0 else cv2.INTER_LINEAR
        region = cv2.resize(frame, (self.video_w, self.video_h), interpolation=interp)

        # Draw into the video region itself, not the canvas, so a hand
        # leaving the frame cannot paint bones across the letterbox bands
        # or up into the header.
        draw_hands(
            region,
            [(h, kp2d, alpha) for h, kp2d, _, alpha in hands],
            scale=self.video_scale,
            offset=(0, 0),
        )
        canvas[top:bottom, self.video_x:self.video_x + self.video_w] = region

        self._draw_playhead(canvas, frame_index)

    def _build_top_band(self, height: int) -> np.ndarray:
        """Static band above the footage: the whole-video quality headline."""
        band = np.empty((max(0, height), self.video_rect.w, 3), np.uint8)
        if not band.size:
            return band
        band[:] = BG_VIDEO_PANEL

        if self.summary_line and height >= 24:
            y = height - max(10, (height - 12) // 2)
            if y > HEADER_H + 10:
                tw = text_width(self.summary_line, CAPTION_SCALE)
                text_draw(band, self.summary_line, ((self.video_rect.w - tw) // 2, y),
                          CAPTION_SCALE, CAPTION_COLOR)
        return band

    def _build_bottom_band(self, height: int) -> np.ndarray:
        """Static band below the footage: visibility timeline and caption."""
        band = np.empty((max(0, height), self.video_rect.w, 3), np.uint8)
        if not band.size:
            return band
        band[:] = BG_VIDEO_PANEL

        if self.timeline is not None:
            strip = self.timeline.base_strip()
            y = self.timeline_y - (self.video_y + self.video_h)
            x = self.video_x - self.video_rect.x
            if 0 <= y and y + strip.shape[0] <= height:
                band[y:y + strip.shape[0], x:x + strip.shape[1]] = strip

        caption_y = height - CAPTION_PAD
        if caption_y > 0:
            text_draw(band, "RGB + 2D KEYPOINTS", (CAPTION_PAD, caption_y),
                      CAPTION_SCALE, CAPTION_COLOR)
        return band

    def _draw_playhead(self, canvas, frame_index: int) -> None:
        """The one part of the timeline that changes per frame."""
        if self.timeline is None:
            return
        x = self.video_x + self.timeline.playhead_x(frame_index)
        y0 = self.timeline_y
        y1 = y0 + self.timeline.playhead_height()
        cv2.line(canvas, (x, y0), (x, y1), PLAYHEAD_COLOR, 1)

    def _draw_view_panel(self, canvas, frame_index, hands) -> None:
        rect = self.view_rect
        panel = self.viewport.render(
            frame_index=frame_index,
            fps=self.meta.fps,
            hands=[(h, kp3d, alpha) for h, _, kp3d, alpha in hands],
        )
        rect.slice_of(canvas)[:] = panel

        if not hands:
            _center_label(canvas, rect, self._absence_label(frame_index))

        _caption(
            canvas, rect,
            f"3D SKELETON {chr(183)} ORBIT {self.cfg.orbit_deg_per_sec:.0f}°/s",
        )

    def _draw_slam_panel(self, canvas, frame_index) -> None:
        rect = self.slam_rect
        assert rect is not None
        if self.slam_panel is None:
            rect.slice_of(canvas)[:] = BG_VIDEO_PANEL
            _center_label(canvas, rect, "no trajectory")
        else:
            rect.slice_of(canvas)[:] = self.slam_panel.render(frame_index)
        _caption(canvas, rect, "CAMERA TRAJECTORY")

    def _absence_label(self, frame_index: int) -> str:
        """Say *why* the panel is empty, never just that it is."""
        row = self.plan.status[frame_index]
        if (row == UNRECOVERED).any():
            return TAG_UNRECOVERED
        if (row == ABSENT).all():
            return TAG_NO_HAND
        return ""


# ── small drawing helpers ─────────────────────────────────────────────


def _summary_line(track: PoseTrack) -> str:
    """The headline quality number, burned into every frame.

    Pose recovery is quoted over *visible-hand* slots, because that is
    the only figure that measures the tracker rather than the factory.
    The visibility figure sits next to it so the denominator is never
    hidden — a viewer can see both what we recovered and how much of the
    footage had a hand to recover.
    """
    total = track.hand_visible.size
    visible = track.n_visible_slots
    if visible == 0:
        return f"NO HAND VISIBLE IN ANY FRAME {chr(183)} 0 / {total} HAND-SLOTS"
    recovery = 100.0 * track.pose_recovery_rate
    vis_pct = 100.0 * visible / total if total else 0.0
    return (
        f"POSE RECOVERY {recovery:.1f}% OF VISIBLE-HAND SLOTS "
        f"{chr(183)} HAND VISIBLE IN {visible} / {total} SLOTS ({vis_pct:.1f}%)"
    )


def _caption(canvas: np.ndarray, rect: PanelRect, text: str) -> None:
    y = rect.y + rect.h - CAPTION_PAD
    text_draw(canvas, text, (rect.x + CAPTION_PAD, y), CAPTION_SCALE, CAPTION_COLOR)


def _center_label(canvas: np.ndarray, rect: PanelRect, text: str) -> None:
    if not text:
        return
    color = TEXT_WARN if text == TAG_UNRECOVERED else TEXT_LOST
    text_draw_centered(
        canvas, text, (rect.x + rect.w // 2, rect.y + rect.h // 2), 0.62, color
    )
