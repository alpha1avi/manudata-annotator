"""Constants and run configuration for the ManuData hand-pose QC renderer.

Skeleton topology, palette and layout live here so the 2D overlay and the
3D viewport always draw the same hand the same way.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterator, Optional, Sequence, Tuple

# ── skeleton topology ─────────────────────────────────────────────────
# 21-joint MANO / OpenPose hand ordering, shared by WiLoR and MediaPipe:
#   0 wrist, 1-4 thumb, 5-8 index, 9-12 middle, 13-16 ring, 17-20 pinky.

N_JOINTS = 21
WRIST = 0

HAND_L = 0
HAND_R = 1
HAND_NAMES = ("left", "right")
HAND_LABELS = ("L", "R")

FINGER_CHAINS: Dict[str, Tuple[int, ...]] = {
    "thumb": (0, 1, 2, 3, 4),
    "index": (0, 5, 6, 7, 8),
    "middle": (0, 9, 10, 11, 12),
    "ring": (0, 13, 14, 15, 16),
    "pinky": (0, 17, 18, 19, 20),
}

# Bones outlining the palm — drawn muted so the fingers stay legible.
PALM_BONES: Tuple[Tuple[int, int], ...] = ((5, 9), (9, 13), (13, 17), (0, 17))

FINGER_ORDER = ("thumb", "index", "middle", "ring", "pinky")


def finger_bones() -> Iterator[Tuple[str, int, int]]:
    """Yield ``(finger, joint_a, joint_b)`` for every finger bone."""
    for finger in FINGER_ORDER:
        chain = FINGER_CHAINS[finger]
        for a, b in zip(chain[:-1], chain[1:]):
            yield finger, a, b


# ── palette (BGR, as OpenCV wants) ────────────────────────────────────
# Warm family = left hand, cool family = right hand.  Within a hand the
# five fingers ramp across the family so each is individually readable.

WARM_FINGER_COLORS: Dict[str, Tuple[int, int, int]] = {
    "thumb": (48, 62, 235),    # red
    "index": (40, 120, 248),   # orange
    "middle": (60, 180, 250),  # amber
    "ring": (95, 215, 248),    # gold
    "pinky": (140, 170, 252),  # salmon
}

COOL_FINGER_COLORS: Dict[str, Tuple[int, int, int]] = {
    "thumb": (200, 90, 130),   # violet
    "index": (225, 120, 70),   # indigo
    "middle": (235, 165, 55),  # blue
    "ring": (230, 205, 70),    # cyan
    "pinky": (205, 225, 130),  # pale teal
}

HAND_PALETTES = (WARM_FINGER_COLORS, COOL_FINGER_COLORS)

# Muted palm links, one per hand.
PALM_COLORS = ((70, 95, 150), (150, 115, 70))

# Wrist joint marker, one per hand.
WRIST_COLORS = ((70, 110, 255), (240, 170, 60))


def finger_color(hand: int, finger: str) -> Tuple[int, int, int]:
    """BGR colour for one finger of one hand."""
    return HAND_PALETTES[hand][finger]


# ── canvas layout ─────────────────────────────────────────────────────

CANVAS_W = 1920
CANVAS_H = 1080
HEADER_H = 28

# Column widths.  Two-panel is the default; --with-slam adds a narrow third.
LAYOUT_2PANEL: Tuple[int, ...] = (1152, 768)
LAYOUT_3PANEL: Tuple[int, ...] = (1088, 512, 320)

BG_CANVAS = (20, 19, 18)
BG_VIDEO_PANEL = (14, 13, 12)
BG_3D_PANEL = (34, 32, 30)
BG_SLAM_PANEL = (28, 27, 26)
HEADER_BG = (16, 15, 14)
PANEL_DIVIDER = (54, 52, 50)

TEXT_PRIMARY = (238, 238, 238)
TEXT_DIM = (150, 150, 150)
TEXT_WARN = (90, 190, 250)
TEXT_LOST = (120, 120, 200)

# ── 3D viewport ───────────────────────────────────────────────────────

ORBIT_DEG_PER_SEC = 15.0
VIEW_ELEV_DEG = 16.0
# Half-extent of the view cube in metres.  FIXED — never autoscale per
# frame, that is what makes a 3D panel jitter.  Sized to hold both hands at
# their true separation (~0.15 m median, up to ~0.4 m) plus a hand's own
# reach; a single hand centred in it is still large and legible.
VIEW_HALF_EXTENT_M = 0.35
GROUND_GRID_DIVS = 8

# ── visibility / gap handling ─────────────────────────────────────────

GHOST_ALPHA = 0.25
# A pose held longer than this fades to an explicit "no pose" label
# rather than sitting on screen as a stale skeleton.
HOLD_LIMIT_S = 0.5

# ── clip selection ────────────────────────────────────────────────────

CLIP_MIN_S = 20.0
CLIP_MAX_S = 30.0
CLIP_SEARCH_STRIDE_S = 0.5
REEL_CLIP_COUNT = 4
REEL_GAP_S = 0.5

WORDMARK = "ManuData"


# ── run configuration ─────────────────────────────────────────────────


@dataclass
class RenderConfig:
    """Everything that affects the bytes of a rendered output.

    ``fingerprint()`` feeds the ``--resume`` check: an output produced
    under a different configuration is not a valid resume target.
    """

    with_slam: bool = False
    clips_only: bool = False
    max_size_mb: Optional[float] = 50.0
    orbit_deg_per_sec: float = ORBIT_DEG_PER_SEC
    view_half_extent_m: float = VIEW_HALF_EXTENT_M
    hold_limit_s: float = HOLD_LIMIT_S
    ghost_alpha: float = GHOST_ALPHA
    canvas_w: int = CANVAS_W
    canvas_h: int = CANVAS_H
    encoder: str = "auto"  # auto | nvenc | x264
    pose_backend: str = "wilor"

    def fingerprint(self) -> str:
        """Stable short hash of the render-affecting settings."""
        blob = json.dumps(asdict(self), sort_keys=True).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()[:16]


@dataclass
class VideoMeta:
    """Probed source-video properties.

    ``n_frames_exact`` records whether the container actually reported a
    frame count or we derived one from the duration. The difference
    matters: only an exact count can be used to hard-fail a keypoint
    track on length mismatch, and an estimate must be verified against
    the decoder instead.
    """

    path: Path
    width: int
    height: int
    fps: float
    n_frames: int
    n_frames_exact: bool = True

    @property
    def duration_s(self) -> float:
        return self.n_frames / self.fps if self.fps > 0 else 0.0
