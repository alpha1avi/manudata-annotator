"""Per-video keypoint cache — the unit of resumable work.

WiLoR inference is by far the most expensive stage, so it is checkpointed
per video rather than per batch. A rented instance that dies at hour
three has kept every ``.npz`` it finished, and the next run re-infers
only what is genuinely missing.

The cache is also the render's fast path: re-rendering with different
layout or encoding settings never re-runs the network.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Protocol

from qc.config import VideoMeta
from qc.pose.schema import PoseTrack, PoseTrackError

logger = logging.getLogger(__name__)


class PoseBackend(Protocol):
    """What a keypoint producer has to offer."""

    name: str
    version: str

    def infer(self, meta: VideoMeta) -> PoseTrack:
        ...


def npz_path(keypoints_dir: Path, video_path: Path) -> Path:
    return Path(keypoints_dir) / f"{Path(video_path).stem}.npz"


def load_cached(
    keypoints_dir: Path,
    meta: VideoMeta,
) -> Optional[PoseTrack]:
    """Return a usable cached track, or None.

    A cached track whose length disagrees with an *exact* probed frame
    count is not silently discarded and not silently used — it raises,
    because either answer could quietly desynchronise the render and the
    operator needs to know which video is affected.
    """
    path = npz_path(keypoints_dir, meta.path)
    if not path.exists():
        return None

    try:
        track = PoseTrack.load(path)
    except (PoseTrackError, OSError, ValueError) as exc:
        logger.warning(
            "Ignoring unreadable keypoint cache %s (%s); it will be recomputed.",
            path.name, exc,
        )
        return None

    if meta.n_frames_exact:
        track.check_alignment(meta.n_frames, meta.path.name)
    elif track.n_frames != meta.n_frames:
        # The probe only estimated, so a small disagreement is expected;
        # the render loop verifies exactly while streaming.
        logger.info(
            "%s: cached track has %d frames, probe estimated %d. Trusting the "
            "cache; the render will verify against the decoder.",
            meta.path.name, track.n_frames, meta.n_frames,
        )

    logger.info("Using cached keypoints: %s", path.name)
    return track


def get_track(
    meta: VideoMeta,
    keypoints_dir: Path,
    backend: Optional[PoseBackend],
    force: bool = False,
) -> PoseTrack:
    """Load the cached track for *meta*, or produce and cache it."""
    keypoints_dir = Path(keypoints_dir)

    if not force:
        cached = load_cached(keypoints_dir, meta)
        if cached is not None:
            return cached

    if backend is None:
        raise PoseTrackError(
            f"No cached keypoints for {meta.path.name} and no pose backend is "
            f"available. Expected {npz_path(keypoints_dir, meta.path)}."
        )

    logger.info("Running %s inference on %s", backend.name, meta.path.name)
    track = backend.infer(meta)
    track.assert_consistent()

    track.meta.update({
        "model": backend.name,
        "model_version": backend.version,
        "source_video": meta.path.name,
        "source_fps": meta.fps,
        "source_width": meta.width,
        "source_height": meta.height,
        "n_frames": track.n_frames,
        "coordinate_frame": "camera-space metric, metres, +X right / +Y down / +Z forward",
    })

    out = npz_path(keypoints_dir, meta.path)
    track.save(out)
    logger.info("Cached keypoints to %s", out.name)
    return track
