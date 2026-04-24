"""ManuData Annotator — Frame Quality Filter (Stage 2).

Filters out bad frames BEFORE any API calls, saving 40-60% of VLM cost.
Egocentric helmet-cam footage has lots of blur, duplicates, and head-turn
transitions that add cost without adding annotation value.

Filter order (cheapest → most expensive):
    1. Darkness / occlusion check  (histogram stats)
    2. Blur detection              (Laplacian variance)
    3. Duplicate detection         (SSIM against previous good frame)
    4. Transition detection        (dense optical flow on sliding window)
"""

import logging
import sys
from dataclasses import dataclass, field
from typing import List, Tuple

import cv2
import numpy as np
from tqdm import tqdm

# Lazy import — skimage may not be installed
try:
    from skimage.metrics import structural_similarity as _ssim_func
except ImportError:
    _ssim_func = None

from config import AnnotatorConfig
from utils.cost_estimator import estimate_cost

logger = logging.getLogger(__name__)


# ── result dataclasses ────────────────────────────────────────────────


@dataclass
class FilterResult:
    """Outcome of running all quality filters on a frame sequence."""

    good_frames: List[Tuple[float, str]]          # (timestamp, filepath)
    skipped_blur: List[Tuple[float, str]]          # blurry frames
    skipped_duplicate: List[Tuple[float, str]]     # near-duplicate frames
    skipped_transition: List[Tuple[float, float]]  # (start_ts, end_ts) ranges
    skipped_dark: List[Tuple[float, str]]          # too dark / overexposed / occluded
    stats: dict = field(default_factory=dict)


# ── filter implementation ─────────────────────────────────────────────


class FrameQualityFilter:
    """Multi-stage quality filter for egocentric manufacturing video frames."""

    def __init__(self, config: AnnotatorConfig) -> None:
        # Configurable thresholds
        self.blur_threshold: float = config.blur_threshold          # default 50
        self.duplicate_threshold: float = config.duplicate_threshold  # default 0.95

        # Transition detection (optical flow)
        self.transition_flow_threshold: float = 15.0
        self.transition_min_frames: int = 3

        # Brightness / entropy
        self.brightness_low: int = 30
        self.brightness_high: int = 240
        self.entropy_threshold: float = 3.0

    # ── individual detectors ──────────────────────────────────────────

    def detect_darkness(self, frame: np.ndarray) -> Tuple[bool, str]:
        """Check mean brightness and histogram entropy.

        Returns:
            (is_bad, reason) where reason is one of
            ``"too_dark"``, ``"overexposed"``, ``"occluded"``, or ``""``.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        mean_brightness = float(np.mean(gray))

        if mean_brightness < self.brightness_low:
            return True, "too_dark"
        if mean_brightness > self.brightness_high:
            return True, "overexposed"

        # Histogram entropy — low entropy means uniform / occluded
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
        hist = hist / (hist.sum() + 1e-8)
        hist = hist[hist > 0]
        entropy = float(-np.sum(hist * np.log2(hist)))

        if entropy < self.entropy_threshold:
            return True, "occluded"

        return False, ""

    def detect_blur(self, frame: np.ndarray) -> Tuple[bool, float]:
        """Compute Laplacian variance for blur detection.

        Returns:
            (is_blurry, laplacian_variance).
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        return variance < self.blur_threshold, variance

    def detect_duplicate(
        self, frame: np.ndarray, prev_frame: np.ndarray
    ) -> Tuple[bool, float]:
        """Compute SSIM between current and previous good frame.

        Both frames are downscaled to 256x256 grayscale for speed.

        Returns:
            (is_duplicate, ssim_score).
        """
        size = (256, 256)
        gray_curr = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        gray_prev = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY) if prev_frame.ndim == 3 else prev_frame

        gray_curr = cv2.resize(gray_curr, size, interpolation=cv2.INTER_AREA)
        gray_prev = cv2.resize(gray_prev, size, interpolation=cv2.INTER_AREA)

        if _ssim_func is not None:
            score = float(_ssim_func(gray_curr, gray_prev))
        else:
            # Numpy fallback: normalised cross-correlation
            a = gray_curr.astype(np.float64)
            b = gray_prev.astype(np.float64)
            a = (a - a.mean()) / max(a.std(), 1e-8)
            b = (b - b.mean()) / max(b.std(), 1e-8)
            score = float(np.mean(a * b))
            score = max(0.0, min(1.0, score))

        return score > self.duplicate_threshold, score

    def detect_transition(self, frames_window: List[np.ndarray]) -> bool:
        """Detect head-turn / walking transitions via dense optical flow.

        Computes Farneback optical flow between consecutive frames in the
        window.  If mean flow magnitude exceeds the threshold for *all*
        pairs, the window is flagged as a transition.

        Frames are downscaled to 320x240 grayscale for speed.

        Args:
            frames_window: List of BGR frames (length >= 2).

        Returns:
            ``True`` if transition detected.
        """
        if len(frames_window) < 2:
            return False

        size = (320, 240)
        grays = []
        for f in frames_window:
            g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) if f.ndim == 3 else f
            grays.append(cv2.resize(g, size, interpolation=cv2.INTER_AREA))

        high_flow_count = 0
        for i in range(len(grays) - 1):
            flow = cv2.calcOpticalFlowFarneback(
                grays[i], grays[i + 1],
                None,
                pyr_scale=0.5, levels=3, winsize=15,
                iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
            )
            mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            mean_mag = float(np.mean(mag))
            if mean_mag > self.transition_flow_threshold:
                high_flow_count += 1

        return high_flow_count >= (len(grays) - 1)

    # ── main filter pipeline ──────────────────────────────────────────

    def filter(self, frame_paths: List[Tuple[float, str]]) -> FilterResult:
        """Run all quality checks on a list of frames.

        Processing order (cheapest first):
            1. Darkness / occlusion
            2. Blur detection
            3. Duplicate detection (SSIM vs. previous good frame)
            4. Transition detection (optical flow sliding window)

        Args:
            frame_paths: Sorted list of ``(timestamp_s, filepath)`` tuples.

        Returns:
            :class:`FilterResult` with categorised frames and stats.
        """
        good_frames: List[Tuple[float, str]] = []
        skipped_blur: List[Tuple[float, str]] = []
        skipped_duplicate: List[Tuple[float, str]] = []
        skipped_dark: List[Tuple[float, str]] = []
        transition_ranges: List[Tuple[float, float]] = []

        prev_good_frame: np.ndarray | None = None
        # Sliding window buffer for transition detection
        window_buf: List[Tuple[float, np.ndarray]] = []
        transition_streak: List[float] = []

        total = len(frame_paths)
        logger.info("Filtering %d frames through quality checks …", total)

        for ts, fpath in tqdm(frame_paths, desc="Quality filter", unit="frame"):
            frame = cv2.imread(fpath, cv2.IMREAD_COLOR)
            if frame is None:
                logger.warning("Cannot read frame: %s", fpath)
                skipped_dark.append((ts, fpath))
                continue

            # 1. Darkness / occlusion
            is_dark, dark_reason = self.detect_darkness(frame)
            if is_dark:
                logger.debug("Frame t=%.2f skipped (%s)", ts, dark_reason)
                skipped_dark.append((ts, fpath))
                continue

            # 2. Blur
            is_blurry, blur_var = self.detect_blur(frame)
            if is_blurry:
                logger.debug("Frame t=%.2f skipped (blur=%.1f)", ts, blur_var)
                skipped_blur.append((ts, fpath))
                continue

            # 3. Duplicate
            if prev_good_frame is not None:
                is_dup, ssim_score = self.detect_duplicate(frame, prev_good_frame)
                if is_dup:
                    logger.debug("Frame t=%.2f skipped (dup SSIM=%.3f)", ts, ssim_score)
                    skipped_duplicate.append((ts, fpath))
                    continue

            # 4. Transition detection (sliding window)
            window_buf.append((ts, frame))
            if len(window_buf) > self.transition_min_frames:
                window_buf.pop(0)

            if len(window_buf) == self.transition_min_frames:
                window_frames = [f for _, f in window_buf]
                if self.detect_transition(window_frames):
                    transition_streak.append(ts)
                    continue
                else:
                    # Flush any accumulated transition streak
                    if transition_streak:
                        transition_ranges.append(
                            (transition_streak[0], transition_streak[-1])
                        )
                        transition_streak.clear()

            # Frame passed all checks
            good_frames.append((ts, fpath))
            prev_good_frame = frame

        # Close any trailing transition streak
        if transition_streak:
            transition_ranges.append(
                (transition_streak[0], transition_streak[-1])
            )

        stats = self._compute_stats(
            total, good_frames, skipped_blur, skipped_duplicate,
            transition_ranges, skipped_dark,
        )

        result = FilterResult(
            good_frames=good_frames,
            skipped_blur=skipped_blur,
            skipped_duplicate=skipped_duplicate,
            skipped_transition=transition_ranges,
            skipped_dark=skipped_dark,
            stats=stats,
        )
        return result

    # ── stats & reporting ─────────────────────────────────────────────

    @staticmethod
    def _compute_stats(
        total: int,
        good: list,
        blur: list,
        dup: list,
        trans: list,
        dark: list,
    ) -> dict:
        n_good = len(good)
        n_blur = len(blur)
        n_dup = len(dup)
        # Count individual transition frames (estimate from ranges)
        n_trans = sum(1 for _ in trans)  # number of transition ranges
        n_dark = len(dark)
        n_filtered = total - n_good

        pct = lambda n: (n / total * 100) if total > 0 else 0.0

        return {
            "total_frames": total,
            "good_frames": n_good,
            "good_pct": round(pct(n_good), 1),
            "skipped_blur": n_blur,
            "blur_pct": round(pct(n_blur), 1),
            "skipped_duplicate": n_dup,
            "dup_pct": round(pct(n_dup), 1),
            "skipped_transition_ranges": n_trans,
            "skipped_dark": n_dark,
            "dark_pct": round(pct(n_dark), 1),
            "total_filtered": n_filtered,
            "filter_pct": round(pct(n_filtered), 1),
        }

    def print_stats(self, result: FilterResult) -> None:
        """Print a summary table of filtering results."""
        s = result.stats
        total = s["total_frames"]

        lines = [
            "",
            "╔══════════════════════════════════════════╗",
            "║       Frame Quality Filter Results       ║",
            "╠══════════════════════════════════════════╣",
            f"║  Total frames:    {total:>6}                 ║",
            f"║  Good frames:     {s['good_frames']:>6}  ({s['good_pct']:>5.1f}%)       ║",
            f"║  Skipped blur:    {s['skipped_blur']:>6}  ({s['blur_pct']:>5.1f}%)       ║",
            f"║  Skipped dup:     {s['skipped_duplicate']:>6}  ({s['dup_pct']:>5.1f}%)       ║",
            f"║  Skipped trans:   {s['skipped_transition_ranges']:>6}  ranges            ║",
            f"║  Skipped dark:    {s['skipped_dark']:>6}  ({s['dark_pct']:>5.1f}%)       ║",
            "╠══════════════════════════════════════════╣",
            f"║  Total filtered:  {s['total_filtered']:>6}  ({s['filter_pct']:>5.1f}%)       ║",
            "╚══════════════════════════════════════════╝",
        ]

        # Estimate cost saved
        cost_all = estimate_cost(total, total, "gemini", 5)
        cost_filtered = estimate_cost(total, s["good_frames"], "gemini", 5)
        saved = cost_all["estimated_cost_inr"] - cost_filtered["estimated_cost_inr"]
        lines.append(f"  Estimated API cost saved: ₹{saved:.2f} (Gemini Flash)")

        summary = "\n".join(lines)
        print(summary)
        logger.info(summary)


# ── standalone test ───────────────────────────────────────────────────

if __name__ == "__main__":
    import os
    from pathlib import Path

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from utils.logging_config import setup_logging

    setup_logging(verbose=True)

    if len(sys.argv) < 2:
        print("Usage: python frame_quality_filter.py <frames_dir>")
        print("  frames_dir should contain frame_*.jpg files")
        sys.exit(1)

    frames_dir = Path(sys.argv[1])
    if not frames_dir.is_dir():
        print(f"Not a directory: {frames_dir}")
        sys.exit(1)

    # Build frame list sorted by filename
    import re

    frame_files = sorted(frames_dir.glob("frame_*.jpg"))
    frame_paths: List[Tuple[float, str]] = []
    for fp in frame_files:
        m = re.search(r"frame_(\d+)", fp.stem)
        if m:
            ts_ms = int(m.group(1))
            frame_paths.append((ts_ms / 1000.0, str(fp)))

    print(f"Found {len(frame_paths)} frames in {frames_dir}")

    config = AnnotatorConfig()
    filt = FrameQualityFilter(config)
    result = filt.filter(frame_paths)
    filt.print_stats(result)
