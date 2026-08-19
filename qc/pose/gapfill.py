"""Turn visibility flags into a per-frame drawing plan.

A QC render must never blink a skeleton in and out — that reads as a
broken pipeline even when the data is fine.  It must also never claim a
hand is present when the detector says it is not.  This module resolves
that tension by treating the two missing-pose cases differently:

``hand_visible and not valid`` — the hand is there and we lost it.
    Hold the last recovered pose as a dimmed ghost for up to
    ``hold_limit_s``, then fade it out and say ``pose unrecovered``.
    Holding is legitimate here: the hand really is in that vicinity.

``not hand_visible`` — no hand present, or fully occluded.
    Dissolve the last pose over a fraction of a second so the skeleton
    leaves rather than snaps off, then draw nothing and say ``no hand in
    view``.  Holding a stale skeleton here would assert something the
    detector explicitly denies, so the dissolve is deliberately far too
    short to read as "a hand is still there".

The plan stores a *source frame index* rather than copied keypoints, so
the renderer always reads the real pose array and there is no second
copy of the data to fall out of sync.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from qc.config import RenderConfig
from qc.pose.schema import N_HANDS, PoseTrack

# Per-hand status codes, stored in an int8 array.
TRACKED = 0      # live pose from this very frame
HELD = 1         # ghosted last-known pose, hand still detected
UNRECOVERED = 2  # hand detected, pose lost beyond the hold limit
FADING = 3       # hand gone, last pose dissolving out
ABSENT = 4       # nothing detected, nothing drawn

STATUS_NAMES = {
    TRACKED: "tracked",
    HELD: "held",
    UNRECOVERED: "unrecovered",
    FADING: "fading",
    ABSENT: "absent",
}

# Seconds to ramp a live pose down to ghost opacity.  Short enough to be
# immediate, long enough not to strobe.
GHOST_RAMP_S = 0.12
# Seconds to dissolve a ghost away to nothing.
FADE_OUT_S = 0.15

TAG_UNRECOVERED = "pose unrecovered"
TAG_NO_HAND = "no hand in view"


@dataclass
class RenderPlan:
    """Per-frame, per-hand drawing directives for one video."""

    src_index: np.ndarray  # (T, 2) int32, frame to read the pose from; -1 = draw nothing
    alpha: np.ndarray      # (T, 2) float32, 0..1 opacity
    status: np.ndarray     # (T, 2) int8, one of the codes above

    @property
    def n_frames(self) -> int:
        return int(self.src_index.shape[0])

    def header_tag(self, t: int) -> str:
        """Short status word for the burned-in header, or "" when clean."""
        row = self.status[t]
        if (row == TRACKED).any():
            return ""
        if ((row == HELD) | (row == UNRECOVERED)).any():
            return TAG_UNRECOVERED
        return TAG_NO_HAND

    def is_ghosted(self, t: int) -> bool:
        """True when nothing on screen at *t* is a live pose."""
        return not (self.status[t] == TRACKED).any()


def build_render_plan(track: PoseTrack, fps: float, cfg: RenderConfig) -> RenderPlan:
    """Resolve *track*'s visibility flags into a :class:`RenderPlan`."""
    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps}")

    n = track.n_frames
    src_index = np.full((n, N_HANDS), -1, np.int32)
    alpha = np.zeros((n, N_HANDS), np.float32)
    status = np.full((n, N_HANDS), ABSENT, np.int8)

    ghost = float(cfg.ghost_alpha)
    hold_limit_s = float(cfg.hold_limit_s)

    for h in range(N_HANDS):
        valid_h = track.valid[:, h]
        visible_h = track.hand_visible[:, h]
        last_valid = -1

        for t in range(n):
            if valid_h[t]:
                src_index[t, h] = t
                alpha[t, h] = 1.0
                status[t, h] = TRACKED
                last_valid = t
                continue

            if last_valid < 0:
                # Nothing recovered yet in this video; nothing to hold.
                status[t, h] = UNRECOVERED if visible_h[t] else ABSENT
                continue

            gap_s = (t - last_valid) / fps

            if visible_h[t]:
                if gap_s <= hold_limit_s:
                    src_index[t, h] = last_valid
                    alpha[t, h] = _ghost_ramp(gap_s, ghost)
                    status[t, h] = HELD
                elif gap_s - hold_limit_s < FADE_OUT_S:
                    over = gap_s - hold_limit_s
                    src_index[t, h] = last_valid
                    alpha[t, h] = ghost * (1.0 - over / FADE_OUT_S)
                    status[t, h] = HELD
                else:
                    status[t, h] = UNRECOVERED
            else:
                if gap_s < FADE_OUT_S:
                    src_index[t, h] = last_valid
                    alpha[t, h] = _ghost_ramp(gap_s, ghost) * (1.0 - gap_s / FADE_OUT_S)
                    status[t, h] = FADING
                else:
                    status[t, h] = ABSENT

    return RenderPlan(src_index=src_index, alpha=alpha, status=status)


def _ghost_ramp(gap_s: float, ghost: float) -> float:
    """Opacity as a live pose ages into a ghost."""
    k = min(1.0, gap_s / GHOST_RAMP_S) if GHOST_RAMP_S > 0 else 1.0
    return float(1.0 + (ghost - 1.0) * k)
