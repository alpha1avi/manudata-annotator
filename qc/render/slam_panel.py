"""Optional third panel: the camera trajectory as a growing 3D polyline.

This renders a trajectory; it does not compute one. MASt3R-SLAM is a
separate, expensive stage, and pretending to derive a camera path from
hand keypoints would put a fabricated figure in front of a technical
evaluator. So the panel consumes a trajectory file produced by the SLAM
stage, and if none is supplied it says so on the panel rather than
drawing something plausible.

The path is drawn in full but dimmed ahead of the playhead and bright
behind it, so the viewer sees both where the camera has been and the
shape of the whole take at once.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from qc.config import BG_SLAM_PANEL, TEXT_DIM, TEXT_LOST
from qc.render.text import draw_centered as text_draw_centered
from qc.render.viewport3d import make_camera

logger = logging.getLogger(__name__)

PATH_AHEAD = (74, 70, 66)
PATH_BEHIND = (215, 170, 90)
CURRENT = (250, 250, 250)
GRID = (52, 50, 48)

TRAJECTORY_ORBIT_DEG_PER_SEC = 9.0
ELEV_DEG = 24.0
TRAJECTORY_FRAME_FILL = 0.44


class TrajectoryUnavailable(RuntimeError):
    pass


def load_trajectory(path: Path, n_frames: int) -> np.ndarray:
    """Load camera positions as ``(n_frames, 3)``.

    Accepts a few shapes because SLAM stages disagree on output format:
    an ``(N, 3)`` position array, an ``(N, 4, 4)`` stack of camera-to-
    world matrices, or a TUM-style text file whose columns start
    ``timestamp tx ty tz``.

    A trajectory of a different length than the video is resampled onto
    the video's frames rather than silently truncated — SLAM commonly
    runs at a lower rate than the source footage.
    """
    path = Path(path)
    if not path.exists():
        raise TrajectoryUnavailable(f"No trajectory file at {path}")

    suffix = path.suffix.lower()
    if suffix == ".npy":
        raw = np.load(path)
    elif suffix == ".npz":
        with np.load(path) as data:
            key = next(
                (k for k in ("positions", "poses", "trajectory", "traj") if k in data),
                None,
            )
            if key is None:
                raise TrajectoryUnavailable(
                    f"{path.name} has no positions/poses/trajectory array "
                    f"(found: {list(data.keys())})"
                )
            raw = data[key]
    else:
        raw = np.loadtxt(path, comments="#")

    raw = np.asarray(raw, np.float32)
    if raw.ndim == 3 and raw.shape[1:] == (4, 4):
        positions = raw[:, :3, 3]
    elif raw.ndim == 2 and raw.shape[1] >= 8:
        positions = raw[:, 1:4]  # TUM: timestamp tx ty tz qx qy qz qw
    elif raw.ndim == 2 and raw.shape[1] == 3:
        positions = raw
    else:
        raise TrajectoryUnavailable(
            f"Unrecognised trajectory shape {raw.shape} in {path.name}"
        )

    positions = positions.astype(np.float32)
    if len(positions) == 0:
        raise TrajectoryUnavailable(f"{path.name} contains no poses")

    if len(positions) != n_frames:
        logger.info(
            "Resampling trajectory %s from %d to %d poses to match the video.",
            path.name, len(positions), n_frames,
        )
        src = np.linspace(0.0, 1.0, len(positions))
        dst = np.linspace(0.0, 1.0, n_frames)
        positions = np.stack(
            [np.interp(dst, src, positions[:, i]) for i in range(3)], axis=1
        ).astype(np.float32)

    return positions


class TrajectoryPanel:
    """Draws the growing camera path for one video."""

    def __init__(
        self,
        width: int,
        height: int,
        positions: np.ndarray,
        fps: float,
        orbit_deg_per_sec: float = TRAJECTORY_ORBIT_DEG_PER_SEC,
    ) -> None:
        self.width = width
        self.height = height
        self.fps = fps
        self.orbit_deg_per_sec = orbit_deg_per_sec
        self.positions = np.asarray(positions, np.float32).reshape(-1, 3)

        self.center = self.positions.mean(axis=0)
        spread = float(np.abs(self.positions - self.center).max())
        # Fixed for the whole video, like the hand viewport.
        self.half_extent = max(spread * 1.25, 0.05)

    def render(self, frame_index: int) -> np.ndarray:
        panel = np.empty((self.height, self.width, 3), np.uint8)
        panel[:] = BG_SLAM_PANEL

        azim = (frame_index / self.fps) * self.orbit_deg_per_sec if self.fps else 0.0
        camera = make_camera(
            center=self.center, half_extent=self.half_extent,
            azim_deg=azim, elev_deg=ELEV_DEG,
            width=self.width, height=self.height,
            # This panel is narrow and tall, so the path is sized against
            # its width; a larger fill keeps it from reading as a speck.
            frame_fill=TRAJECTORY_FRAME_FILL,
        )

        pts, _, front = camera.project(self.positions)
        index = int(np.clip(frame_index, 0, len(self.positions) - 1))

        self._draw_polyline(panel, pts, front, 0, len(pts) - 1, PATH_AHEAD, 1)
        self._draw_polyline(panel, pts, front, 0, index, PATH_BEHIND, 2)

        if front[index]:
            here = _pt(pts[index])
            cv2.circle(panel, here, 6, (0, 0, 0), -1, cv2.LINE_AA)
            cv2.circle(panel, here, 4, CURRENT, -1, cv2.LINE_AA)

        return panel

    @staticmethod
    def _draw_polyline(panel, pts, front, start, end, color, width) -> None:
        for i in range(start, min(end, len(pts) - 1)):
            if not (front[i] and front[i + 1]):
                continue
            cv2.line(panel, _pt(pts[i]), _pt(pts[i + 1]), color, width, cv2.LINE_AA)


class MissingTrajectoryPanel:
    """Stand-in that states plainly that no trajectory was supplied."""

    def __init__(self, width: int, height: int, reason: str = "") -> None:
        self.width = width
        self.height = height
        self.reason = reason

    def render(self, frame_index: int) -> np.ndarray:
        panel = np.empty((self.height, self.width, 3), np.uint8)
        panel[:] = BG_SLAM_PANEL
        _center_text(panel, "no SLAM trajectory", TEXT_LOST, 0.5, dy=-10)
        if self.reason:
            _center_text(panel, self.reason[:40], TEXT_DIM, 0.36, dy=12)
        return panel


def _center_text(panel, text, color, scale, dy=0) -> None:
    text_draw_centered(
        panel, text, (panel.shape[1] // 2, panel.shape[0] // 2 + dy), scale, color
    )


def _pt(xy: np.ndarray):
    limit = 1 << 14
    return (
        int(max(-limit, min(limit, float(xy[0])))),
        int(max(-limit, min(limit, float(xy[1])))),
    )
