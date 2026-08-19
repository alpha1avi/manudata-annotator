"""Fixed-scale 3D skeleton viewport, drawn with numpy and cv2 primitives.

Deliberately not matplotlib. At 1080p60 across 25 videos the 3D panel is
drawn a few hundred thousand times, and an Agg canvas costs ~60-120 ms a
frame against ~1-2 ms here. It also gives exact control over the two
things that make a 3D panel look untrustworthy:

*Scale is fixed.* The view cube half-extent is a constant, and the view
centre is computed once per video from the whole track — never per frame.
Autoscaling per frame is what makes a skeleton appear to breathe.

*Depth is real.* Bones are depth-sorted and drawn far-to-near with
width and brightness falling off with distance, so the orbit reads as
rotation of a solid object rather than a wobbling 2D projection.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from qc.render.overlay2d import label_anchor
from qc.render.text import draw as text_draw

from qc.config import (
    BG_3D_PANEL,
    GROUND_GRID_DIVS,
    PALM_BONES,
    PALM_COLORS,
    TEXT_DIM,
    VIEW_ELEV_DEG,
    WRIST,
    WRIST_COLORS,
    finger_bones,
    finger_color,
)

# Camera sits this many view-cube half-extents back from the centre.
CAMERA_DISTANCE_FACTOR = 3.2
# Fraction of the shorter panel dimension a half-extent should occupy.
FRAME_FILL = 0.34

GRID_COLOR = (58, 55, 52)
GRID_COLOR_MAJOR = (78, 74, 70)
AXIS_COLORS = ((80, 80, 235), (90, 210, 110), (235, 170, 70))  # X red, Y green, Z blue


@dataclass
class _Camera:
    """A resolved view: world points in, pixel coordinates and depth out."""

    eye: np.ndarray
    basis: np.ndarray  # rows: right, up, forward
    focal: float
    cx: float
    cy: float

    def project(self, points: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Project ``(N, 3)`` world points.

        Returns ``(pixels, depth, in_front)``. Points at or behind the
        camera plane are flagged rather than producing a wild projection.
        """
        rel = points - self.eye
        view = rel @ self.basis.T
        depth = view[:, 2]
        in_front = depth > 1e-4
        safe = np.where(in_front, depth, 1.0)
        px = self.cx + self.focal * view[:, 0] / safe
        py = self.cy - self.focal * view[:, 1] / safe
        return np.stack([px, py], axis=1), depth, in_front


def make_camera(
    center: np.ndarray,
    half_extent: float,
    azim_deg: float,
    elev_deg: float,
    width: int,
    height: int,
    frame_fill: float = FRAME_FILL,
) -> _Camera:
    """Build an orbiting camera looking at *center*.

    Shared by the hand viewport and the SLAM trajectory panel so both
    panels orbit with identical geometry and a viewer can read them as
    the same 3D space seen at the same angle.
    """
    az = math.radians(azim_deg)
    el = math.radians(elev_deg)
    dist = half_extent * CAMERA_DISTANCE_FACTOR
    center = np.asarray(center, np.float32).reshape(3)

    # World is a camera-space frame: +Y points down, so "up" is -Y.
    offset = np.array(
        [dist * math.cos(el) * math.sin(az),
         -dist * math.sin(el),
         -dist * math.cos(el) * math.cos(az)],
        np.float32,
    )
    eye = center + offset

    forward = center - eye
    forward = forward / (np.linalg.norm(forward) + 1e-9)
    world_up = np.array([0.0, -1.0, 0.0], np.float32)
    right = np.cross(forward, world_up)
    norm = np.linalg.norm(right)
    right = np.array([1.0, 0.0, 0.0], np.float32) if norm < 1e-6 else right / norm
    up = np.cross(right, forward)

    focal = frame_fill * min(width, height) * dist / max(half_extent, 1e-6)
    return _Camera(
        eye=eye,
        basis=np.stack([right, up, forward]).astype(np.float32),
        focal=float(focal),
        cx=width / 2.0,
        cy=height / 2.0,
    )


class Viewport3D:
    """Renders the right-hand 3D panel for one video."""

    def __init__(
        self,
        width: int,
        height: int,
        half_extent: float,
        orbit_deg_per_sec: float,
        elev_deg: float = VIEW_ELEV_DEG,
        bg_color: Sequence[int] = BG_3D_PANEL,
    ) -> None:
        self.width = width
        self.height = height
        self.half_extent = float(half_extent)
        self.orbit_deg_per_sec = float(orbit_deg_per_sec)
        self.elev_deg = float(elev_deg)
        self.bg_color = tuple(int(c) for c in bg_color)
        # Overwritten by set_center(); the origin is only a safe default.
        self.center = np.zeros(3, np.float32)
        self._background_cache: Optional[Tuple[int, np.ndarray]] = None

    # ── view setup ────────────────────────────────────────────────────

    def set_center(self, center: np.ndarray) -> None:
        """Fix the view centre for the whole video.

        Called once, from a statistic over the entire track. Anything
        per-frame here would reintroduce the jitter this class exists to
        avoid.
        """
        self.center = np.asarray(center, np.float32).reshape(3)
        self._background_cache = None

    def azimuth_deg(self, frame_index: int, fps: float) -> float:
        """Continuous orbit angle for an absolute frame index."""
        return (frame_index / fps) * self.orbit_deg_per_sec

    def _camera(self, azim_deg: float) -> _Camera:
        return make_camera(
            center=self.center,
            half_extent=self.half_extent,
            azim_deg=azim_deg,
            elev_deg=self.elev_deg,
            width=self.width,
            height=self.height,
        )

    # ── drawing ───────────────────────────────────────────────────────

    def render(
        self,
        frame_index: int,
        fps: float,
        hands: Sequence[Tuple[int, np.ndarray, float]],
    ) -> np.ndarray:
        """Draw the panel.

        *hands* is a sequence of ``(hand_index, joints_3d, alpha)`` where
        ``joints_3d`` is ``(21, 3)`` in metres and *alpha* is opacity.
        """
        camera = self._camera(self.azimuth_deg(frame_index, fps))
        panel = self._background(camera)

        for hand_index, joints, alpha in hands:
            if joints is None or alpha <= 0.01:
                continue
            if not np.isfinite(joints).all():
                continue
            layer = panel.copy()
            self._draw_hand(layer, camera, hand_index, np.asarray(joints, np.float32))
            if alpha >= 0.999:
                panel = layer
            else:
                cv2.addWeighted(layer, alpha, panel, 1.0 - alpha, 0.0, dst=panel)

        return panel

    def _background(self, camera: _Camera) -> np.ndarray:
        """Ground grid and axis triad for the current orbit angle."""
        panel = np.empty((self.height, self.width, 3), np.uint8)
        panel[:] = self.bg_color
        self._draw_ground(panel, camera)
        self._draw_axis_triad(panel, camera)
        return panel

    def _draw_ground(self, panel: np.ndarray, camera: _Camera) -> None:
        """A square grid on the plane below the hands, for depth reference."""
        extent = self.half_extent * 1.5
        y = float(self.center[1]) + self.half_extent  # +Y is down
        divs = GROUND_GRID_DIVS
        steps = np.linspace(-extent, extent, divs + 1, dtype=np.float32)

        segments: List[Tuple[np.ndarray, np.ndarray, bool]] = []
        for i, s in enumerate(steps):
            major = i in (0, divs, divs // 2)
            segments.append((
                np.array([self.center[0] + s, y, self.center[2] - extent], np.float32),
                np.array([self.center[0] + s, y, self.center[2] + extent], np.float32),
                major,
            ))
            segments.append((
                np.array([self.center[0] - extent, y, self.center[2] + s], np.float32),
                np.array([self.center[0] + extent, y, self.center[2] + s], np.float32),
                major,
            ))

        starts = np.stack([s[0] for s in segments])
        ends = np.stack([s[1] for s in segments])
        p0, _, front0 = camera.project(starts)
        p1, _, front1 = camera.project(ends)

        for i, (_, _, major) in enumerate(segments):
            if not (front0[i] and front1[i]):
                continue
            color = GRID_COLOR_MAJOR if major else GRID_COLOR
            cv2.line(
                panel,
                _pt(p0[i]), _pt(p1[i]),
                color, 1, cv2.LINE_AA,
            )

    def _draw_axis_triad(self, panel: np.ndarray, camera: _Camera) -> None:
        """Small XYZ triad so the viewer can name the axes."""
        length = self.half_extent * 0.42
        origin = np.array(
            [self.center[0], self.center[1] + self.half_extent, self.center[2]],
            np.float32,
        )
        tips = np.stack([
            origin + np.array([length, 0, 0], np.float32),
            origin + np.array([0, -length, 0], np.float32),
            origin + np.array([0, 0, length], np.float32),
        ])
        pts, _, front = camera.project(np.concatenate([origin[None], tips]))
        if not front.all():
            return

        o = _pt(pts[0])
        for i, label in enumerate(("X", "Y", "Z")):
            tip = _pt(pts[i + 1])
            cv2.line(panel, o, tip, AXIS_COLORS[i], 2, cv2.LINE_AA)
            text_draw(panel, label, (tip[0] + 4, tip[1] + 4), 0.4, AXIS_COLORS[i])

    def _draw_hand(
        self,
        panel: np.ndarray,
        camera: _Camera,
        hand_index: int,
        joints: np.ndarray,
    ) -> None:
        pts, depth, front = camera.project(joints)
        if not front.any():
            return

        near = float(np.min(depth[front]))
        far = float(np.max(depth[front]))
        span = max(far - near, 1e-6)

        # Collect every mark with its depth, then paint back-to-front.
        marks: List[Tuple[float, str, tuple]] = []

        palm_color = PALM_COLORS[hand_index]
        for a, b in PALM_BONES:
            if not (front[a] and front[b]):
                continue
            marks.append((
                (depth[a] + depth[b]) / 2.0, "bone",
                (_pt(pts[a]), _pt(pts[b]), palm_color, 2),
            ))

        for finger, a, b in finger_bones():
            if not (front[a] and front[b]):
                continue
            z = (depth[a] + depth[b]) / 2.0
            width = _depth_width(z, near, span, base=5, drop=2)
            marks.append((
                z, "bone",
                (_pt(pts[a]), _pt(pts[b]), finger_color(hand_index, finger), width),
            ))

        for j in range(joints.shape[0]):
            if not front[j]:
                continue
            if j == WRIST:
                continue
            finger = _finger_of_joint(j)
            radius = _depth_width(depth[j], near, span, base=4, drop=1.5)
            marks.append((
                depth[j], "joint",
                (_pt(pts[j]), int(radius), finger_color(hand_index, finger)),
            ))

        # Painter's algorithm: farthest first.
        marks.sort(key=lambda m: -m[0])
        for _, kind, args in marks:
            if kind == "bone":
                p0, p1, color, width = args
                cv2.line(panel, p0, p1, color, int(width), cv2.LINE_AA)
            else:
                center, radius, color = args
                cv2.circle(panel, center, radius, color, -1, cv2.LINE_AA)

        # Wrist last and larger — it anchors the hand for the viewer.
        if front[WRIST]:
            wrist_px = _pt(pts[WRIST])
            cv2.circle(panel, wrist_px, 8, (0, 0, 0), -1, cv2.LINE_AA)
            cv2.circle(panel, wrist_px, 7, WRIST_COLORS[hand_index], -1, cv2.LINE_AA)
            # Same far-side placement as the 2D panel, so the badge does
            # not end up under the fingers at some orbit angles.
            label_anchor(panel, pts[WRIST], pts[front], hand_index,
                         wrist_radius=7, scale=0.5)


# ── helpers ───────────────────────────────────────────────────────────


def _pt(xy: np.ndarray) -> Tuple[int, int]:
    """Clamp to a range cv2 will draw without integer overflow."""
    x = float(xy[0])
    y = float(xy[1])
    limit = 1 << 14
    return (
        int(max(-limit, min(limit, x))),
        int(max(-limit, min(limit, y))),
    )


def _depth_width(z: float, near: float, span: float, base: float, drop: float) -> float:
    """Line width / radius that shrinks with distance."""
    k = (z - near) / span
    return max(1.0, base - drop * k)


_JOINT_TO_FINGER = {}
for _finger, _a, _b in finger_bones():
    _JOINT_TO_FINGER.setdefault(_b, _finger)


def _finger_of_joint(joint: int) -> str:
    return _JOINT_TO_FINGER.get(joint, "index")


def view_center_from_track(kp3d: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """A single view centre for a whole video.

    Uses the median over every recovered joint rather than the mean, so a
    handful of wild frames from a bad detection cannot drag the camera
    off the hands for the entire render.
    """
    mask = np.broadcast_to(valid[:, :, None], kp3d.shape[:3])
    pts = kp3d[mask]
    pts = pts.reshape(-1, 3) if pts.size else pts
    finite = pts[np.isfinite(pts).all(axis=1)] if pts.size else pts
    if finite.size == 0:
        return np.zeros(3, np.float32)
    return np.median(finite, axis=0).astype(np.float32)
