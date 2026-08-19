"""2D skeleton overlay on the source RGB frame.

Drawn after the frame has been scaled to panel size rather than before,
so line weights stay constant on screen regardless of whether the source
is 1080p or 4K, and so we are not downscaling anti-aliased strokes.

Left and right hands take the warm and cool palettes from
``qc.config``; within a hand each finger gets its own colour so an
evaluator can follow an individual digit through an occlusion.
"""

from __future__ import annotations

from typing import Sequence, Tuple

import cv2
import numpy as np

from qc.render.text import draw_centered as text_draw_centered

from qc.config import (
    HAND_LABELS,
    PALM_BONES,
    PALM_COLORS,
    WRIST,
    WRIST_COLORS,
    finger_bones,
    finger_color,
)

BONE_WIDTH = 3
JOINT_RADIUS = 4
WRIST_RADIUS = 8
# Dark casing under every stroke so the skeleton survives a bright
# workbench or a pale glove without a colour clash.
OUTLINE_COLOR = (0, 0, 0)


def draw_hands(
    panel: np.ndarray,
    hands: Sequence[Tuple[int, np.ndarray, float]],
    scale: float,
    offset: Tuple[int, int] = (0, 0),
) -> np.ndarray:
    """Draw 2D skeletons onto *panel* in place and return it.

    *hands* is ``(hand_index, kp2d, alpha)`` with ``kp2d`` of shape
    ``(21, 2)`` in **source** pixel coordinates; *scale* and *offset*
    map those into panel coordinates.
    """
    for hand_index, kp2d, alpha in hands:
        if kp2d is None or alpha <= 0.01:
            continue
        pts = np.asarray(kp2d, np.float32)
        if pts.shape != (21, 2) or not np.isfinite(pts).all():
            continue

        px = pts * scale + np.asarray(offset, np.float32)

        if alpha >= 0.999:
            _draw_one(panel, hand_index, px)
        else:
            layer = panel.copy()
            _draw_one(layer, hand_index, px)
            cv2.addWeighted(layer, alpha, panel, 1.0 - alpha, 0.0, dst=panel)

    return panel


def _draw_one(panel: np.ndarray, hand_index: int, px: np.ndarray) -> None:
    palm_color = PALM_COLORS[hand_index]

    for a, b in PALM_BONES:
        cv2.line(panel, _pt(px[a]), _pt(px[b]), OUTLINE_COLOR, BONE_WIDTH + 2, cv2.LINE_AA)
    for a, b in PALM_BONES:
        cv2.line(panel, _pt(px[a]), _pt(px[b]), palm_color, BONE_WIDTH - 1, cv2.LINE_AA)

    bones = list(finger_bones())
    for _, a, b in bones:
        cv2.line(panel, _pt(px[a]), _pt(px[b]), OUTLINE_COLOR, BONE_WIDTH + 2, cv2.LINE_AA)
    for finger, a, b in bones:
        cv2.line(
            panel, _pt(px[a]), _pt(px[b]),
            finger_color(hand_index, finger), BONE_WIDTH, cv2.LINE_AA,
        )

    for _, _, joint in bones:
        cv2.circle(panel, _pt(px[joint]), JOINT_RADIUS + 1, OUTLINE_COLOR, -1, cv2.LINE_AA)
    for finger, _, joint in bones:
        cv2.circle(
            panel, _pt(px[joint]), JOINT_RADIUS,
            finger_color(hand_index, finger), -1, cv2.LINE_AA,
        )

    wrist = _pt(px[WRIST])
    cv2.circle(panel, wrist, WRIST_RADIUS + 2, OUTLINE_COLOR, -1, cv2.LINE_AA)
    cv2.circle(panel, wrist, WRIST_RADIUS, WRIST_COLORS[hand_index], -1, cv2.LINE_AA)

    label_anchor(panel, px[WRIST], px, hand_index, WRIST_RADIUS)


def label_anchor(
    panel: np.ndarray,
    wrist_xy: np.ndarray,
    cloud: np.ndarray,
    hand_index: int,
    wrist_radius: int,
    scale: float = 0.62,
) -> None:
    """Draw the L/R badge on the far side of the wrist from the fingers.

    Anchoring it at a fixed offset buries the letter under whichever
    bones happen to radiate that way. Pushing it along the wrist-minus-
    centroid direction keeps it clear of the hand at any orientation.
    """
    wrist_xy = np.asarray(wrist_xy, np.float32).reshape(2)
    cloud = np.asarray(cloud, np.float32).reshape(-1, 2)
    finite = cloud[np.isfinite(cloud).all(axis=1)]
    centroid = finite.mean(axis=0) if len(finite) else wrist_xy
    away = wrist_xy - centroid
    norm = float(np.linalg.norm(away))
    away = away / norm if norm > 1e-3 else np.array([0.0, -1.0], np.float32)

    label = HAND_LABELS[hand_index]
    offset = wrist_xy + away * (wrist_radius + 14)
    center = (
        int(max(-(1 << 13), min(1 << 13, offset[0]))),
        int(max(-(1 << 13), min(1 << 13, offset[1]))),
    )
    text_draw_centered(panel, label, center, scale, WRIST_COLORS[hand_index],
                       thickness=2)


def _pt(xy: np.ndarray) -> Tuple[int, int]:
    """Clamp off-frame keypoints to a range cv2 can draw safely.

    Hands leaving the frame is normal on a factory line, and the
    projected joint can sit well outside the panel. cv2 clips lines
    fine but overflows on extreme integers, hence the bound.
    """
    limit = 1 << 14
    return (
        int(max(-limit, min(limit, float(xy[0])))),
        int(max(-limit, min(limit, float(xy[1])))),
    )
