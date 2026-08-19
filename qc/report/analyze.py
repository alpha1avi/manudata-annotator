"""Per-video quality statistics and best-clip selection.

The central reporting decision lives here: the two ways a frame can lack
a pose are counted separately and never summed into a single "NaN rate".

* Slots where no hand was detected are a fact about the factory floor —
  hands go behind workpieces, out of frame, under tooling. Reporting
  them as tracking failure would understate our tracker and misdescribe
  the data.
* Slots where a hand *was* detected and no pose came back are ours.
  ``pose_recovery_pct`` is computed over exactly those, and it is the
  number the ranking sorts on.

A video that is 60% occluded but recovers a pose on 97% of the frames
where a hand is actually visible is good data. A single blended rate
would rank it below a video that is barely occluded and recovers 80%,
which is the wrong way round.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from qc.config import (
    CLIP_MAX_S,
    CLIP_MIN_S,
    CLIP_SEARCH_STRIDE_S,
    VideoMeta,
)
from qc.pose.schema import N_HANDS, PoseTrack


@dataclass
class ClipWindow:
    """A recommended excerpt, in seconds and in frames."""

    start_s: float
    end_s: float
    start_frame: int
    end_frame: int
    both_hands_pct: float

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s


@dataclass
class VideoStats:
    """One row of ``qc_report.csv``."""

    filename: str
    site: str
    task: str
    duration_s: float
    total_frames: int

    # Frames by how many hands we could actually draw.
    frames_both_hands: int
    frames_one_hand: int
    frames_zero_hands: int

    # Case (b): a hand was detected and the pose did not come back.
    frames_visible_no_pose: int
    # Case (a): no hand detected, as a share of all hand-slots.
    occluded_or_absent_pct: float
    # The quality measure: recovered / visible.
    pose_recovery_pct: float

    longest_gap_s: float
    mean_confidence: float
    recommended_clip_start_s: float
    recommended_clip_end_s: float

    # Not written to CSV; used for ranking and reel assembly.
    clip: Optional[ClipWindow] = None
    source_path: Optional[Path] = None

    @property
    def rank_key(self) -> Tuple[float, float, float]:
        """Sort key: recovery first, then clip quality, then coverage.

        Recovery leads because it is the tracker measure. Two videos with
        equal recovery are then separated by how good a 20-30s excerpt we
        can actually cut from them, which is what the reel needs.
        """
        recovery = self.pose_recovery_pct if not math.isnan(self.pose_recovery_pct) else -1.0
        clip_quality = self.clip.both_hands_pct if self.clip else 0.0
        coverage = 100.0 - self.occluded_or_absent_pct
        return (recovery, clip_quality, coverage)


def analyze(
    track: PoseTrack,
    meta: VideoMeta,
    site: str,
    task: str,
) -> VideoStats:
    """Compute the report row for one video."""
    n = track.n_frames
    fps = meta.fps

    drawn = track.valid.sum(axis=1)
    frames_both = int((drawn == 2).sum())
    frames_one = int((drawn == 1).sum())
    frames_zero = int((drawn == 0).sum())

    total_slots = int(track.hand_visible.size)
    visible_slots = track.n_visible_slots
    recovered_slots = track.n_recovered_slots
    visible_no_pose = int((track.hand_visible & ~track.valid).sum())

    occluded_pct = (
        100.0 * (total_slots - visible_slots) / total_slots if total_slots else 0.0
    )
    recovery_pct = (
        100.0 * recovered_slots / visible_slots if visible_slots else float("nan")
    )

    conf = track.conf[track.valid]
    mean_conf = float(conf.mean()) if conf.size else float("nan")

    longest_gap_s = longest_run(drawn == 0) / fps if fps else 0.0
    clip = best_window(track, fps)

    return VideoStats(
        filename=meta.path.name,
        site=site,
        task=task,
        duration_s=n / fps if fps else 0.0,
        total_frames=n,
        frames_both_hands=frames_both,
        frames_one_hand=frames_one,
        frames_zero_hands=frames_zero,
        frames_visible_no_pose=visible_no_pose,
        occluded_or_absent_pct=occluded_pct,
        pose_recovery_pct=recovery_pct,
        longest_gap_s=longest_gap_s,
        mean_confidence=mean_conf,
        recommended_clip_start_s=clip.start_s if clip else 0.0,
        recommended_clip_end_s=clip.end_s if clip else 0.0,
        clip=clip,
        source_path=meta.path,
    )


def longest_run(mask: np.ndarray) -> int:
    """Length of the longest run of True in a 1-D boolean array."""
    if mask.size == 0 or not mask.any():
        return 0
    # Difference of run-boundary indices, via a padded edge detector.
    padded = np.concatenate(([False], mask.astype(bool), [False]))
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return int((edges[1::2] - edges[0::2]).max())


def best_window(track: PoseTrack, fps: float) -> Optional[ClipWindow]:
    """Find the strongest contiguous 20-30s excerpt.

    Scores on the fraction of frames with *both* hands drawable, because
    a two-handed assembly shot is what demonstrates the tracking. Ties
    break toward the longer window — given equal quality, more footage is
    a better sample.
    """
    n = track.n_frames
    if fps <= 0 or n == 0:
        return None

    both = (track.valid.sum(axis=1) == 2).astype(np.int64)
    prefix = np.concatenate(([0], np.cumsum(both)))

    min_len = int(round(CLIP_MIN_S * fps))
    max_len = int(round(CLIP_MAX_S * fps))
    stride = max(1, int(round(CLIP_SEARCH_STRIDE_S * fps)))

    # Videos shorter than the minimum window get their whole length back
    # rather than nothing — a 14s clip is still usable in the reel.
    if n < min_len:
        score = 100.0 * float(both.mean())
        return ClipWindow(0.0, n / fps, 0, n, score)

    best: Optional[ClipWindow] = None
    best_score = -1.0

    lengths = sorted({min_len, (min_len + max_len) // 2, min(max_len, n)}, reverse=True)
    for length in lengths:
        if length > n:
            continue
        for start in range(0, n - length + 1, stride):
            end = start + length
            score = (prefix[end] - prefix[start]) / length
            # Strictly greater keeps the first (longest) length on a tie.
            if score > best_score:
                best_score = score
                best = ClipWindow(
                    start_s=start / fps,
                    end_s=end / fps,
                    start_frame=start,
                    end_frame=end,
                    both_hands_pct=100.0 * score,
                )

    return best


def rank(stats: List[VideoStats]) -> List[VideoStats]:
    """Best first."""
    return sorted(stats, key=lambda s: s.rank_key, reverse=True)
