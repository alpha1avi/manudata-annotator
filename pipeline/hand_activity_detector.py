"""ManuData Annotator — Hand Activity Detection (Stage 3a).

Combines hand pose estimation with temporal activity scoring to decide
which frames should be sent to the VLM and which can be auto-labelled
as idle.
"""

import logging
import math
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from tqdm import tqdm

from config import AnnotatorConfig
from models.hand_pose import HandDetection, HandPoseEstimator, HandPoseResult, WRIST

logger = logging.getLogger(__name__)


# ── dataclasses ───────────────────────────────────────────────────────


@dataclass
class FrameHandMeta:
    """Hand metadata attached to a single frame destined for VLM."""

    timestamp: float
    filepath: str
    hand_result: HandPoseResult
    activity_score: float
    activity_class: str  # active_manipulation | possible_activity | idle_or_observing


@dataclass
class ActivityResult:
    """Aggregated activity-detection result for a frame sequence."""

    frames_to_vlm: List[FrameHandMeta] = field(default_factory=list)
    frames_idle: List[Tuple[float, str]] = field(default_factory=list)
    per_frame_activity: Dict[float, float] = field(default_factory=dict)
    per_frame_hands: Dict[float, HandPoseResult] = field(default_factory=dict)
    stats: Dict[str, Any] = field(default_factory=dict)


# ── detector ──────────────────────────────────────────────────────────


class HandActivityDetector:
    """Score per-frame hand activity and decide VLM vs. idle routing."""

    # Activity classification thresholds
    ACTIVE_THRESHOLD = 0.6
    POSSIBLE_THRESHOLD = 0.2  # == config.activity_threshold default

    def __init__(self, config: AnnotatorConfig) -> None:
        self.config = config
        self.hand_estimator = HandPoseEstimator()
        self.activity_threshold = config.activity_threshold  # frames below → idle
        self.POSSIBLE_THRESHOLD = self.activity_threshold

    # ── public API ────────────────────────────────────────────────────

    def process_frames(
        self, good_frames: List[Tuple[float, str]]
    ) -> ActivityResult:
        """Run hand detection + activity scoring on every good frame.

        For each frame:
            1. Detect hands via MediaPipe.
            2. Compute activity score (temporal comparison with previous).
            3. Classify: ``active_manipulation`` | ``possible_activity``
               | ``idle_or_observing``.

        Frames with score >= ``activity_threshold`` go to VLM; the rest
        are auto-labelled idle.

        Args:
            good_frames: Sorted ``(timestamp, filepath)`` list from the
                quality filter.

        Returns:
            :class:`ActivityResult`.
        """
        frames_to_vlm: List[FrameHandMeta] = []
        frames_idle: List[Tuple[float, str]] = []
        per_frame_activity: Dict[float, float] = {}
        per_frame_hands: Dict[float, HandPoseResult] = {}

        prev_hands: Optional[HandPoseResult] = None

        logger.info("Running hand-activity detection on %d frames …", len(good_frames))

        for ts, fpath in tqdm(good_frames, desc="Activity detection", unit="frame"):
            frame = cv2.imread(fpath, cv2.IMREAD_COLOR)
            if frame is None:
                logger.warning("Cannot read frame: %s", fpath)
                frames_idle.append((ts, fpath))
                continue

            hand_result = self.hand_estimator.detect(frame)
            per_frame_hands[ts] = hand_result

            score = self.compute_activity_score(hand_result, prev_hands)
            per_frame_activity[ts] = score
            prev_hands = hand_result

            activity_class = self._classify(score)

            if activity_class == "idle_or_observing":
                frames_idle.append((ts, fpath))
            else:
                meta = FrameHandMeta(
                    timestamp=ts,
                    filepath=fpath,
                    hand_result=hand_result,
                    activity_score=score,
                    activity_class=activity_class,
                )
                frames_to_vlm.append(meta)

        stats = self._build_stats(
            len(good_frames), frames_to_vlm, frames_idle, per_frame_activity,
        )

        result = ActivityResult(
            frames_to_vlm=frames_to_vlm,
            frames_idle=frames_idle,
            per_frame_activity=per_frame_activity,
            per_frame_hands=per_frame_hands,
            stats=stats,
        )

        self._log_stats(stats)
        return result

    # ── activity scoring ──────────────────────────────────────────────

    def compute_activity_score(
        self,
        current_hands: HandPoseResult,
        prev_hands: Optional[HandPoseResult],
    ) -> float:
        """Compute a 0-1 activity score from hand observations.

        Weighted components:
            - 0.4 — hand presence
            - 0.3 — hand movement speed (wrist displacement)
            - 0.2 — finger movement (mean keypoint displacement)
            - 0.1 — bounding-box area change (grasping signal)

        Each component is normalised to 0-1 before weighting.
        """
        # --- hand presence (0.4) ---
        if current_hands.num_hands == 0:
            presence = 0.0
        elif current_hands.num_hands == 1:
            presence = 0.5
        else:
            presence = 1.0

        # If no previous frame, can't compute motion — use presence only
        if prev_hands is None or prev_hands.num_hands == 0 or current_hands.num_hands == 0:
            return 0.4 * presence

        # Match hands by handedness
        curr_map = {h.handedness: h for h in current_hands.hands}
        prev_map = {h.handedness: h for h in prev_hands.hands}
        common = set(curr_map.keys()) & set(prev_map.keys())

        if not common:
            return 0.4 * presence

        # --- wrist movement (0.3) ---
        wrist_disps = []
        for label in common:
            cw = curr_map[label].keypoints_pixel[WRIST]
            pw = prev_map[label].keypoints_pixel[WRIST]
            wrist_disps.append(self._pixel_dist(cw, pw))
        mean_wrist = sum(wrist_disps) / len(wrist_disps)
        # Normalise: 50 px displacement → 1.0
        wrist_score = min(1.0, mean_wrist / 50.0)

        # --- finger movement (0.2) ---
        kp_disps = []
        for label in common:
            ckp = curr_map[label].keypoints_pixel
            pkp = prev_map[label].keypoints_pixel
            for ci, pi in zip(ckp, pkp):
                kp_disps.append(self._pixel_dist(ci, pi))
        mean_kp = sum(kp_disps) / (len(kp_disps) + 1e-8)
        finger_score = min(1.0, mean_kp / 30.0)

        # --- bbox area change (0.1) ---
        area_changes = []
        for label in common:
            ca = self._bbox_area(curr_map[label].bbox)
            pa = self._bbox_area(prev_map[label].bbox)
            if pa > 0:
                area_changes.append(abs(ca - pa) / pa)
        mean_area_change = sum(area_changes) / (len(area_changes) + 1e-8)
        area_score = min(1.0, mean_area_change / 0.3)

        score = (
            0.4 * presence
            + 0.3 * wrist_score
            + 0.2 * finger_score
            + 0.1 * area_score
        )
        return min(1.0, score)

    # ── internals ─────────────────────────────────────────────────────

    def _classify(self, score: float) -> str:
        if score >= self.ACTIVE_THRESHOLD:
            return "active_manipulation"
        if score >= self.POSSIBLE_THRESHOLD:
            return "possible_activity"
        return "idle_or_observing"

    @staticmethod
    def _pixel_dist(a: Tuple[int, int], b: Tuple[int, int]) -> float:
        return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

    @staticmethod
    def _bbox_area(bbox: Tuple[int, int, int, int]) -> float:
        x1, y1, x2, y2 = bbox
        return max(0, x2 - x1) * max(0, y2 - y1)

    @staticmethod
    def _build_stats(
        total: int,
        vlm_frames: list,
        idle_frames: list,
        scores: dict,
    ) -> dict:
        n_active = sum(1 for f in vlm_frames if f.activity_class == "active_manipulation")
        n_possible = sum(1 for f in vlm_frames if f.activity_class == "possible_activity")
        n_idle = len(idle_frames)
        pct = lambda n: round(n / total * 100, 1) if total > 0 else 0.0

        all_scores = list(scores.values())
        return {
            "total_frames": total,
            "active_manipulation": n_active,
            "active_pct": pct(n_active),
            "possible_activity": n_possible,
            "possible_pct": pct(n_possible),
            "idle_or_observing": n_idle,
            "idle_pct": pct(n_idle),
            "frames_to_vlm": len(vlm_frames),
            "vlm_pct": pct(len(vlm_frames)),
            "mean_activity_score": round(sum(all_scores) / (len(all_scores) + 1e-8), 3),
        }

    @staticmethod
    def _log_stats(stats: dict) -> None:
        logger.info(
            "Activity detection complete: %d total | "
            "%d active (%.1f%%) | %d possible (%.1f%%) | %d idle (%.1f%%) | "
            "%d frames → VLM (%.1f%%)",
            stats["total_frames"],
            stats["active_manipulation"], stats["active_pct"],
            stats["possible_activity"], stats["possible_pct"],
            stats["idle_or_observing"], stats["idle_pct"],
            stats["frames_to_vlm"], stats["vlm_pct"],
        )

    def close(self) -> None:
        """Release MediaPipe resources."""
        self.hand_estimator.close()


# ── standalone test ───────────────────────────────────────────────────

if __name__ == "__main__":
    import os
    import re
    from pathlib import Path

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from utils.logging_config import setup_logging

    setup_logging(verbose=True)

    if len(sys.argv) < 2:
        print("Usage: python hand_activity_detector.py <frames_dir>")
        print("  frames_dir should contain frame_*.jpg files")
        sys.exit(1)

    frames_dir = Path(sys.argv[1])
    if not frames_dir.is_dir():
        print(f"Not a directory: {frames_dir}")
        sys.exit(1)

    frame_files = sorted(frames_dir.glob("frame_*.jpg"))
    frame_paths: List[Tuple[float, str]] = []
    for fp in frame_files:
        m = re.search(r"frame_(\d+)", fp.stem)
        if m:
            ts_ms = int(m.group(1))
            frame_paths.append((ts_ms / 1000.0, str(fp)))

    print(f"Found {len(frame_paths)} frames")

    config = AnnotatorConfig()
    detector = HandActivityDetector(config)
    result = detector.process_frames(frame_paths)

    print(f"\nFrames to VLM:  {len(result.frames_to_vlm)}")
    print(f"Frames idle:    {len(result.frames_idle)}")
    print(f"Stats:          {result.stats}")

    detector.close()
