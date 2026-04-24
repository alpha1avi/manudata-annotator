"""ManuData Annotator — Quality Scorer (Stage 7).

Computes composite quality scores for each segment based on pose
completeness, object visibility, trajectory smoothness, action clarity,
frame quality, and depth consistency.  All computation is NumPy-based.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from pipeline.local_vision_pipeline import FrameAnnotation
from pipeline.segment_merger import Segment
from pipeline.trajectory_extractor import Trajectory

logger = logging.getLogger(__name__)


# ── dataclass ─────────────────────────────────────────────────────────


@dataclass
class QualityScore:
    """Composite quality score for a single segment."""

    composite: float               # weighted average 0-1
    pose_completeness: float
    object_visibility: float
    trajectory_smoothness: float
    action_clarity: float
    frame_quality: float
    depth_consistency: float
    may_discard: bool              # composite < 0.2


# ── scorer ────────────────────────────────────────────────────────────


class QualityScorer:
    """Compute per-segment and per-video quality scores."""

    def __init__(self) -> None:
        self.weights: Dict[str, float] = {
            "pose_completeness": 0.25,
            "object_visibility": 0.20,
            "trajectory_smoothness": 0.20,
            "action_clarity": 0.15,
            "frame_quality": 0.10,
            "depth_consistency": 0.10,
        }

    # ── single segment ────────────────────────────────────────────────

    def score_segment(
        self,
        segment: Segment,
        frame_annotations: List[FrameAnnotation],
        trajectory: Optional[Trajectory],
    ) -> QualityScore:
        """Compute quality sub-scores and a weighted composite.

        Args:
            segment: The segment to score.
            frame_annotations: Frame annotations within this segment's
                time window.
            trajectory: Hand trajectory for the active hand (may be None).

        Returns:
            :class:`QualityScore`.
        """
        # Filter annotations to segment window
        seg_anns = [
            a for a in frame_annotations
            if segment.start_time <= a.timestamp <= segment.end_time
        ]

        pose = self._score_pose_completeness(seg_anns)
        obj_vis = self._score_object_visibility(seg_anns, segment.objects_involved)
        smooth = self._score_trajectory_smoothness(trajectory)
        clarity = self._score_action_clarity(segment)
        fq = self._score_frame_quality(seg_anns)
        depth = self._score_depth_consistency(seg_anns)

        composite = (
            self.weights["pose_completeness"] * pose
            + self.weights["object_visibility"] * obj_vis
            + self.weights["trajectory_smoothness"] * smooth
            + self.weights["action_clarity"] * clarity
            + self.weights["frame_quality"] * fq
            + self.weights["depth_consistency"] * depth
        )
        composite = round(min(1.0, max(0.0, composite)), 4)

        return QualityScore(
            composite=composite,
            pose_completeness=round(pose, 4),
            object_visibility=round(obj_vis, 4),
            trajectory_smoothness=round(smooth, 4),
            action_clarity=round(clarity, 4),
            frame_quality=round(fq, 4),
            depth_consistency=round(depth, 4),
            may_discard=composite < 0.2,
        )

    # ── full video ────────────────────────────────────────────────────

    def score_video(
        self,
        segments: List[Segment],
        frame_annotations: List[FrameAnnotation],
        trajectories: Dict,
    ) -> Dict[str, Any]:
        """Score all segments and return aggregate statistics.

        Returns:
            ``{"scores": {segment_id: QualityScore, ...},
              "aggregate": {"mean": …, "median": …, "min": …,
                            "distribution": …}}``
        """
        scores: Dict[int, QualityScore] = {}

        for seg in segments:
            # Pick the best matching trajectory
            traj: Optional[Trajectory] = None
            if seg.hand_used in ("left", "right"):
                traj = trajectories.get(seg.hand_used)
            elif seg.hand_used == "both":
                traj = trajectories.get("right") or trajectories.get("left")

            qs = self.score_segment(seg, frame_annotations, traj)
            scores[seg.id] = qs

        # Aggregate
        composites = [qs.composite for qs in scores.values()] or [0.0]
        bins = {"excellent": 0, "good": 0, "fair": 0, "poor": 0, "discard": 0}
        for c in composites:
            if c >= 0.8:
                bins["excellent"] += 1
            elif c >= 0.6:
                bins["good"] += 1
            elif c >= 0.4:
                bins["fair"] += 1
            elif c >= 0.2:
                bins["poor"] += 1
            else:
                bins["discard"] += 1

        aggregate = {
            "mean": round(float(np.mean(composites)), 4),
            "median": round(float(np.median(composites)), 4),
            "min": round(float(np.min(composites)), 4),
            "max": round(float(np.max(composites)), 4),
            "std": round(float(np.std(composites)), 4),
            "distribution": bins,
            "total_segments": len(segments),
            "may_discard_count": sum(1 for qs in scores.values() if qs.may_discard),
        }

        logger.info(
            "Quality scores: mean=%.3f, median=%.3f, min=%.3f, discard=%d/%d",
            aggregate["mean"], aggregate["median"], aggregate["min"],
            aggregate["may_discard_count"], aggregate["total_segments"],
        )

        return {"scores": scores, "aggregate": aggregate}

    # ── sub-score implementations ─────────────────────────────────────

    @staticmethod
    def _score_pose_completeness(annotations: List[FrameAnnotation]) -> float:
        """Percentage of frames with hand pose, weighted by keypoint count.

        21 keypoints visible = 1.0 per frame; no hands = 0.0.
        """
        if not annotations:
            return 0.0

        per_frame: List[float] = []
        for ann in annotations:
            if ann.hand_pose.num_hands == 0:
                per_frame.append(0.0)
                continue
            # Best hand's keypoint fraction
            best = 0.0
            for h in ann.hand_pose.hands:
                n_kp = len(h.keypoints_pixel)
                best = max(best, n_kp / 21.0)
            per_frame.append(min(1.0, best))

        return float(np.mean(per_frame))

    @staticmethod
    def _score_object_visibility(
        annotations: List[FrameAnnotation],
        expected_objects: List[str],
    ) -> float:
        """Average object detection confidence; penalise missing expected objects."""
        if not annotations:
            return 0.0

        per_frame: List[float] = []
        for ann in annotations:
            if not ann.objects:
                per_frame.append(0.0)
                continue
            avg_conf = float(np.mean([o.confidence for o in ann.objects]))

            # Penalty: expected objects not detected
            if expected_objects:
                detected_names = {o.class_name for o in ann.objects}
                found = sum(1 for e in expected_objects if e in detected_names)
                ratio = found / len(expected_objects)
                avg_conf *= (0.5 + 0.5 * ratio)  # 50% penalty for missing all

            per_frame.append(min(1.0, avg_conf))

        return float(np.mean(per_frame))

    @staticmethod
    def _score_trajectory_smoothness(trajectory: Optional[Trajectory]) -> float:
        """Inverse of mean jerk magnitude, normalised to 0-1.

        Smooth trajectory → 1.0; very jerky → 0.0.
        Default 0.5 if no trajectory data.
        """
        if trajectory is None:
            return 0.5

        acc = trajectory.acceleration
        if acc.size == 0 or len(acc) < 2:
            return 0.5

        # Jerk = derivative of acceleration
        jerk = np.diff(acc, axis=0)
        jerk_mag = np.linalg.norm(jerk, axis=1)
        mean_jerk = float(np.mean(jerk_mag))

        # Normalise: 0 jerk → 1.0, jerk >= 100 → 0.0
        score = max(0.0, 1.0 - mean_jerk / 100.0)
        return score

    @staticmethod
    def _score_action_clarity(segment: Segment) -> float:
        """Task label confidence with bonus for valid phase sequence.

        Returns:
            Score in [0, 1].
        """
        from pipeline.segment_merger import SegmentMerger

        score = segment.confidence_avg
        if SegmentMerger.validate_phase_sequence(segment.manipulation_phases):
            score = min(1.0, score + 0.1)
        return score

    @staticmethod
    def _score_frame_quality(annotations: List[FrameAnnotation]) -> float:
        """Average Laplacian variance + brightness consistency.

        Reads frames from disk to compute Laplacian variance.
        Falls back to 0.5 if frames are unavailable.
        """
        if not annotations:
            return 0.5

        variances: List[float] = []
        brightnesses: List[float] = []

        for ann in annotations:
            frame = cv2.imread(ann.frame_path, cv2.IMREAD_GRAYSCALE)
            if frame is None:
                continue
            var = float(cv2.Laplacian(frame, cv2.CV_64F).var())
            variances.append(var)
            brightnesses.append(float(np.mean(frame)))

        if not variances:
            return 0.5

        # Sharpness: normalise Laplacian variance (200 = crisp → 1.0)
        mean_var = float(np.mean(variances))
        sharpness = min(1.0, mean_var / 200.0)

        # Brightness consistency: penalise high std
        bright_std = float(np.std(brightnesses))
        consistency = max(0.0, 1.0 - bright_std / 80.0)

        return 0.7 * sharpness + 0.3 * consistency

    @staticmethod
    def _score_depth_consistency(annotations: List[FrameAnnotation]) -> float:
        """Check depth plausibility — no sudden jumps > 0.5 between frames.

        Default 0.5 if no depth data.
        """
        depth_values: List[float] = []
        for ann in annotations:
            if ann.depth is None:
                continue
            # Sample depth at frame centre
            dm = ann.depth.depth_map
            h, w = dm.shape[:2]
            depth_values.append(float(dm[h // 2, w // 2]))

        if len(depth_values) < 2:
            return 0.5

        diffs = np.abs(np.diff(depth_values))
        n_jumps = int(np.sum(diffs > 0.5))

        if n_jumps == 0:
            return 1.0

        # Penalise: each jump reduces score
        penalty = n_jumps / len(diffs)
        return max(0.0, 1.0 - penalty)


# ── standalone test ───────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    import os

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from utils.logging_config import setup_logging
    from models.hand_pose import HandPoseResult

    setup_logging(verbose=True)

    # Synthetic segment
    seg = Segment(
        id=1, start_time=0.0, end_time=5.0, duration=5.0,
        action="pick_up", action_description="Picking up wrench",
        task_hierarchy={"high_level": "assembly"},
        objects_involved=["wrench"],
        grasp_type="power", hand_used="right",
        manipulation_phases=["reach", "grasp", "manipulate"],
        trajectory_summary={}, confidence_avg=0.85, confidence_min=0.7,
        frames_analyzed=10, frames_skipped=0,
        labelling_method="vlm", needs_review=False,
    )

    # Synthetic annotations (no real frames)
    anns = []
    for i in range(10):
        anns.append(FrameAnnotation(
            timestamp=i * 0.5,
            frame_path="nonexistent.jpg",
            hand_pose=HandPoseResult(hands=[], num_hands=0, hand_visibility="no_hands"),
            objects=[], depth=None, interactions=[],
            activity_score=0.5, hand_visibility="no_hands",
            processing_time_ms=5.0,
        ))

    scorer = QualityScorer()
    qs = scorer.score_segment(seg, anns, None)

    print(f"\nQuality Score for segment '{seg.action}':")
    print(f"  Composite:      {qs.composite:.3f}")
    print(f"  Pose complete:  {qs.pose_completeness:.3f}")
    print(f"  Object vis:     {qs.object_visibility:.3f}")
    print(f"  Traj smooth:    {qs.trajectory_smoothness:.3f}")
    print(f"  Action clarity: {qs.action_clarity:.3f}")
    print(f"  Frame quality:  {qs.frame_quality:.3f}")
    print(f"  Depth consist:  {qs.depth_consistency:.3f}")
    print(f"  May discard:    {qs.may_discard}")
