"""Generate a synthetic video + PoseTrack for developing and testing.

Not part of the shipped pipeline. It exists so the render path, the
analysis pass and the visibility state machine can be exercised without
a GPU, WiLoR weights, or access to real factory footage.

The synthetic track deliberately reproduces the two missing-pose cases
separately — stretches where the hand leaves frame (``hand_visible``
false) and stretches where the hand is present but the pose is lost
(``hand_visible`` true, ``valid`` false) — because telling those apart
is the property the report and the render are built around.

Usage:
    python -m tests.make_fixture OUTDIR [--seconds 12] [--fps 30]
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np

from qc.config import FINGER_CHAINS, N_JOINTS
from qc.io.sizing import EncodeQuality
from qc.io.video_writer import VideoWriter
from qc.pose.schema import N_HANDS, PoseTrack

FOCAL_PX = 1100.0

# Canonical right-hand rest pose, metres, wrist at the origin.
# +X across the palm, -Y along the fingers, +Z out of the palm.
_MCP = {
    "thumb": (-0.030, -0.022, 0.014),
    "index": (-0.020, -0.082, 0.002),
    "middle": (0.000, -0.088, 0.000),
    "ring": (0.019, -0.084, -0.002),
    "pinky": (0.037, -0.074, -0.004),
}
_SEGMENTS = {
    "thumb": (0.036, 0.032, 0.026),
    "index": (0.040, 0.026, 0.020),
    "middle": (0.044, 0.028, 0.021),
    "ring": (0.041, 0.026, 0.020),
    "pinky": (0.033, 0.021, 0.018),
}
_SPREAD = {"thumb": -0.85, "index": -0.16, "middle": 0.0, "ring": 0.15, "pinky": 0.30}


def canonical_hand(curl: float, is_left: bool) -> np.ndarray:
    """A 21-joint hand at the given curl (0 = open, 1 = fist)."""
    joints = np.zeros((N_JOINTS, 3), np.float32)

    for finger, chain in FINGER_CHAINS.items():
        base = np.array(_MCP[finger], np.float32)
        joints[chain[1]] = base

        spread = _SPREAD[finger]
        direction = np.array([math.sin(spread), -math.cos(spread), 0.0], np.float32)
        # The thumb curls across the palm rather than into it.
        axis = np.array([0.0, 0.0, 1.0], np.float32) if finger == "thumb" else \
            np.array([1.0, 0.0, 0.0], np.float32)

        point = base.copy()
        for i, length in enumerate(_SEGMENTS[finger]):
            angle = curl * (0.9 + 0.35 * i)
            direction = _rotate(direction, axis, angle)
            point = point + direction * length
            joints[chain[i + 2]] = point

    if is_left:
        joints[:, 0] *= -1.0
    return joints


def _rotate(vec: np.ndarray, axis: np.ndarray, angle: float) -> np.ndarray:
    """Rodrigues rotation of *vec* about a unit *axis*."""
    axis = axis / (np.linalg.norm(axis) + 1e-9)
    c, s = math.cos(angle), math.sin(angle)
    return vec * c + np.cross(axis, vec) * s + axis * np.dot(axis, vec) * (1.0 - c)


def _euler(yaw: float, pitch: float, roll: float) -> np.ndarray:
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cr, sr = math.cos(roll), math.sin(roll)
    rz = np.array([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]], np.float32)
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], np.float32)
    rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]], np.float32)
    return (rz @ ry @ rx).astype(np.float32)


def project(points3d: np.ndarray, width: int, height: int) -> np.ndarray:
    """Pinhole projection into source pixel coordinates."""
    z = np.maximum(points3d[:, 2], 1e-3)
    x = width / 2.0 + FOCAL_PX * points3d[:, 0] / z
    y = height / 2.0 + FOCAL_PX * points3d[:, 1] / z
    return np.stack([x, y], axis=1).astype(np.float32)


def build_track(
    n_frames: int, fps: float, width: int, height: int, seed: int = 7
) -> PoseTrack:
    """Two hands doing a repetitive assembly motion, with realistic dropouts."""
    rng = np.random.default_rng(seed)
    track = PoseTrack.empty(n_frames, meta={"source": "synthetic fixture"})

    # Occlusion windows: hand genuinely out of view or fully hidden.
    absent = np.zeros((n_frames, N_HANDS), bool)
    # Pose-loss windows: hand visible, tracker failed. Motion blur, glare.
    lost = np.zeros((n_frames, N_HANDS), bool)

    for h in range(N_HANDS):
        for _ in range(max(1, n_frames // (int(fps) * 5))):
            start = rng.integers(0, n_frames)
            span = int(rng.uniform(0.35, 1.8) * fps)
            absent[start:start + span, h] = True
        for _ in range(max(1, n_frames // (int(fps) * 3))):
            start = rng.integers(0, n_frames)
            span = int(rng.uniform(0.08, 0.9) * fps)
            lost[start:start + span, h] = True
    lost &= ~absent

    for t in range(n_frames):
        phase = 2 * math.pi * t / (fps * 2.4)
        curl = 0.45 + 0.42 * math.sin(phase)

        for h in range(N_HANDS):
            side = -1.0 if h == 0 else 1.0
            hand = canonical_hand(curl if h == 0 else 1.0 - curl * 0.7, is_left=(h == 0))
            # Egocentric framing: a cap-mounted camera sees the hands enter
            # from the lower edge with the fingers reaching up and away, so
            # the wrists sit low in frame and the fingers point toward -Y.
            rot = _euler(
                yaw=side * 0.30 + 0.22 * math.sin(phase * 0.7),
                pitch=0.34 + 0.16 * math.sin(phase + h),
                roll=side * 0.45,
            )
            translation = np.array([
                side * 0.075 + 0.018 * math.sin(phase + h * 1.3),
                0.055 + 0.025 * math.cos(phase * 0.8 + h),
                0.46 + 0.05 * math.sin(phase * 0.5 + h),
            ], np.float32)

            posed = (hand @ rot.T) + translation

            visible = not absent[t, h]
            recovered = visible and not lost[t, h]

            track.hand_visible[t, h] = visible
            track.det_conf[t, h] = (
                float(rng.uniform(0.72, 0.98)) if visible else 0.0
            )
            if recovered:
                jitter = rng.normal(0.0, 0.0012, posed.shape).astype(np.float32)
                posed = posed + jitter
                track.kp3d[t, h] = posed
                track.kp2d[t, h] = project(posed, width, height)
                track.conf[t, h] = float(rng.uniform(0.70, 0.97))
                track.valid[t, h] = True

    track.assert_consistent()
    return track


# ── synthetic footage ─────────────────────────────────────────────────


def _backdrop(width: int, height: int, seed: int = 3) -> np.ndarray:
    """A workbench-ish background so the overlay lands on something."""
    rng = np.random.default_rng(seed)
    grad = np.linspace(60, 26, height, dtype=np.float32)[:, None]
    frame = np.repeat(grad[:, :, None], 3, axis=2)
    frame = np.repeat(frame, width, axis=1)
    frame[:, :, 0] *= 1.14  # cool steel cast
    frame[:, :, 2] *= 0.86

    for _ in range(14):
        x = int(rng.uniform(0, width * 0.9))
        y = int(rng.uniform(height * 0.45, height * 0.95))
        w = int(rng.uniform(40, 190))
        h = int(rng.uniform(24, 90))
        shade = rng.uniform(0.55, 1.5)
        frame[y:y + h, x:x + w] *= shade

    frame += rng.normal(0, 3.0, frame.shape)
    return np.clip(frame, 0, 255).astype(np.uint8)


def _draw_synthetic_hand(frame: np.ndarray, kp2d: np.ndarray) -> None:
    """A soft skin-toned blob following the keypoints."""
    pts = kp2d[np.isfinite(kp2d).all(axis=1)]
    if len(pts) < 3:
        return
    hull = cv2.convexHull(pts.astype(np.int32))
    mask = np.zeros(frame.shape[:2], np.uint8)
    cv2.fillConvexPoly(mask, hull, 255)
    cv2.dilate(mask, np.ones((21, 21), np.uint8), dst=mask)
    mask = cv2.GaussianBlur(mask, (31, 31), 0)

    skin = np.empty_like(frame)
    skin[:] = (96, 122, 158)
    alpha = (mask.astype(np.float32) / 255.0)[:, :, None]
    np.copyto(frame, (frame * (1 - alpha) + skin * alpha).astype(np.uint8))


def render_fixture_video(
    path: Path, track: PoseTrack, width: int, height: int, fps: float
) -> None:
    backdrop = _backdrop(width, height)
    quality = EncodeQuality(crf=20, max_bitrate_kbps=None)
    with VideoWriter(path, width, height, fps, quality, encoder_preference="x264") as w:
        for t in range(track.n_frames):
            frame = backdrop.copy()
            for h in range(N_HANDS):
                # The blob follows the hand whenever the hand is *visible*,
                # including frames where the pose was not recovered — that is
                # exactly the case the QC render has to communicate.
                if not track.hand_visible[t, h]:
                    continue
                src = t if track.valid[t, h] else _last_valid(track, t, h)
                if src is None:
                    continue
                _draw_synthetic_hand(frame, track.kp2d[src, h])
            w.write(frame)


def _last_valid(track: PoseTrack, t: int, h: int):
    idx = np.flatnonzero(track.valid[:t + 1, h])
    return int(idx[-1]) if len(idx) else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("outdir", type=Path)
    parser.add_argument("--seconds", type=float, default=12.0)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--name", default="fixture")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    n_frames = int(round(args.seconds * args.fps))

    track = build_track(n_frames, args.fps, args.width, args.height, seed=args.seed)
    video_path = args.outdir / f"{args.name}.mp4"
    render_fixture_video(video_path, track, args.width, args.height, args.fps)

    track.meta.update({
        "n_frames": n_frames, "fps": args.fps,
        "width": args.width, "height": args.height,
        "model": "synthetic-fixture", "model_version": "n/a",
    })
    track.save(args.outdir / f"{args.name}.npz")

    visible = track.n_visible_slots
    print(f"wrote {video_path} ({n_frames} frames @ {args.fps} fps)")
    print(f"  hand-slots visible:   {visible} / {n_frames * N_HANDS}")
    print(f"  pose recovery:        {100 * track.pose_recovery_rate:.1f}% of visible")


if __name__ == "__main__":
    main()
