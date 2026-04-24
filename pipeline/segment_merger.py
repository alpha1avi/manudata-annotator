"""ManuData Annotator — Segment Merger (Stage 6).

Merges per-batch :class:`TaskLabel` results into coherent temporal
segments, inserts transition segments, enforces minimum durations,
validates manipulation-phase sequences, and flags segments that need
human review.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from config import AnnotatorConfig
from pipeline.task_labeller import TaskLabel
from pipeline.trajectory_extractor import Trajectory

logger = logging.getLogger(__name__)

# Valid manipulation phase ordering (any contiguous subset is OK)
VALID_PHASE_ORDER = [
    "reach", "grasp", "manipulate", "hold", "release", "retract",
]
_PHASE_INDEX = {p: i for i, p in enumerate(VALID_PHASE_ORDER)}


# ── dataclass ─────────────────────────────────────────────────────────


@dataclass
class Segment:
    """A temporally coherent action segment."""

    id: int
    start_time: float
    end_time: float
    duration: float
    action: str
    action_description: str
    task_hierarchy: Dict[str, str]
    objects_involved: List[str]
    grasp_type: str
    hand_used: str
    manipulation_phases: List[str]
    trajectory_summary: Dict[str, Any]   # path_length, max_velocity, grasp_events
    confidence_avg: float
    confidence_min: float
    frames_analyzed: int
    frames_skipped: int
    labelling_method: str
    needs_review: bool
    review_reason: Optional[str] = None  # low_confidence | phase_sequence_broken | hand_not_visible


# ── merger ────────────────────────────────────────────────────────────


class SegmentMerger:
    """Merge per-batch labels into coherent temporal segments."""

    def __init__(self, config: AnnotatorConfig) -> None:
        self.min_segment_duration = config.min_segment_duration   # 0.5 s
        self.idle_threshold = config.idle_segment_threshold       # 2.0 s

    # ── public API ────────────────────────────────────────────────────

    def merge(
        self,
        task_labels: List[TaskLabel],
        trajectories: Dict,
        transition_ranges: List[Tuple[float, float]],
    ) -> List[Segment]:
        """Merge task labels into segments.

        Steps:
            1. Group consecutive labels with the same action.
            2. Insert transition segments from Stage 2 ranges.
            3. Merge segments shorter than ``min_segment_duration``.
            4. Split long idle segments.
            5. Enrich each segment (confidence, phases, trajectory, review flag).
            6. Assign sequential IDs.

        Returns:
            List of :class:`Segment` sorted by ``start_time``.
        """
        if not task_labels:
            return []

        # Sort labels by timestamp
        labels = sorted(task_labels, key=lambda l: l.timestamp_start)

        # 1. Group consecutive same-action labels
        raw_groups = self._group_consecutive(labels)

        # 2. Insert transition segments
        raw_groups = self._insert_transitions(raw_groups, transition_ranges)

        # 3. Merge short segments into neighbours
        raw_groups = self._merge_short(raw_groups)

        # 4. Split long idle segments
        raw_groups = self._split_long_idle(raw_groups)

        # 5. Build Segment objects with enrichment
        segments = self._build_segments(raw_groups, trajectories)

        # 6. Assign sequential IDs, sort
        segments.sort(key=lambda s: s.start_time)
        for idx, seg in enumerate(segments):
            seg.id = idx + 1

        logger.info(
            "Segment merger: %d labels → %d segments "
            "(%d need review)",
            len(task_labels),
            len(segments),
            sum(1 for s in segments if s.needs_review),
        )
        return segments

    # ── phase validation ──────────────────────────────────────────────

    @staticmethod
    def validate_phase_sequence(phases: List[str]) -> bool:
        """Check whether manipulation phases follow a valid progression.

        Valid: any contiguous subset of
        ``reach → grasp → manipulate → hold → release → retract``
        appearing in order.

        Returns:
            ``True`` if valid (or empty / single phase).
        """
        known = [p for p in phases if p in _PHASE_INDEX]
        if len(known) <= 1:
            return True

        last_idx = -1
        for p in known:
            cur_idx = _PHASE_INDEX[p]
            if cur_idx < last_idx:
                return False
            last_idx = cur_idx
        return True

    # ── internal helpers ──────────────────────────────────────────────

    @staticmethod
    def _group_consecutive(
        labels: List[TaskLabel],
    ) -> List[Dict[str, Any]]:
        """Group consecutive labels with the same action."""
        groups: List[Dict[str, Any]] = []
        current_action: Optional[str] = None
        current_labels: List[TaskLabel] = []

        for lbl in labels:
            if lbl.action != current_action:
                if current_labels:
                    groups.append({
                        "action": current_action,
                        "labels": list(current_labels),
                    })
                current_action = lbl.action
                current_labels = [lbl]
            else:
                current_labels.append(lbl)

        if current_labels:
            groups.append({"action": current_action, "labels": list(current_labels)})

        return groups

    def _insert_transitions(
        self,
        groups: List[Dict],
        transition_ranges: List[Tuple[float, float]],
    ) -> List[Dict]:
        """Insert transition-type groups from the quality filter's ranges."""
        if not transition_ranges:
            return groups

        # Flatten into a timeline
        merged: List[Dict] = []
        tr_idx = 0
        sorted_ranges = sorted(transition_ranges, key=lambda r: r[0])

        for grp in groups:
            g_start = grp["labels"][0].timestamp_start
            g_end = grp["labels"][-1].timestamp_end

            # Insert any transition ranges that fall before this group
            while tr_idx < len(sorted_ranges) and sorted_ranges[tr_idx][0] < g_start:
                tr_s, tr_e = sorted_ranges[tr_idx]
                merged.append({
                    "action": "transition",
                    "labels": [],
                    "_start": tr_s,
                    "_end": tr_e,
                })
                tr_idx += 1

            merged.append(grp)

        # Trailing transitions
        while tr_idx < len(sorted_ranges):
            tr_s, tr_e = sorted_ranges[tr_idx]
            merged.append({
                "action": "transition",
                "labels": [],
                "_start": tr_s,
                "_end": tr_e,
            })
            tr_idx += 1

        return merged

    def _merge_short(self, groups: List[Dict]) -> List[Dict]:
        """Merge groups shorter than ``min_segment_duration`` into neighbours."""
        if len(groups) <= 1:
            return groups

        merged: List[Dict] = [groups[0]]

        for grp in groups[1:]:
            dur = self._group_duration(grp)
            if dur < self.min_segment_duration and merged:
                # Absorb into the previous group
                prev = merged[-1]
                prev["labels"].extend(grp.get("labels", []))
                # Keep the action of the longer group
            else:
                merged.append(grp)

        return merged

    def _split_long_idle(self, groups: List[Dict]) -> List[Dict]:
        """Split idle groups longer than ``idle_threshold`` into chunks."""
        result: List[Dict] = []
        for grp in groups:
            if grp["action"] != "idle":
                result.append(grp)
                continue

            dur = self._group_duration(grp)
            if dur <= self.idle_threshold:
                result.append(grp)
                continue

            # Split labels into chunks of ~idle_threshold duration
            labels = grp.get("labels", [])
            if not labels:
                result.append(grp)
                continue

            chunk: List[TaskLabel] = []
            chunk_start = labels[0].timestamp_start
            for lbl in labels:
                if lbl.timestamp_end - chunk_start > self.idle_threshold and chunk:
                    result.append({"action": "idle", "labels": list(chunk)})
                    chunk = [lbl]
                    chunk_start = lbl.timestamp_start
                else:
                    chunk.append(lbl)
            if chunk:
                result.append({"action": "idle", "labels": list(chunk)})

        return result

    def _build_segments(
        self, groups: List[Dict], trajectories: Dict
    ) -> List[Segment]:
        """Convert raw groups into enriched :class:`Segment` objects."""
        segments: List[Segment] = []

        for grp in groups:
            labels: List[TaskLabel] = grp.get("labels", [])
            action = grp["action"]

            # Timestamps
            if labels:
                start_t = labels[0].timestamp_start
                end_t = labels[-1].timestamp_end
            else:
                start_t = grp.get("_start", 0.0)
                end_t = grp.get("_end", 0.0)

            duration = round(end_t - start_t, 4)

            # Confidence
            confs = [l.confidence for l in labels] if labels else [0.0]
            conf_avg = round(sum(confs) / len(confs), 4)
            conf_min = round(min(confs), 4)

            # Description & hierarchy (from first label)
            desc = labels[0].action_description if labels else ""
            hierarchy = labels[0].task_hierarchy if labels else {}
            objects = []
            for l in labels:
                for o in l.objects_involved:
                    if o not in objects:
                        objects.append(o)

            grasp = labels[0].grasp_type if labels else "none"
            hand = labels[0].hand_used if labels else "none"
            method = labels[0].labelling_method if labels else "unknown"

            # Phases
            phases = []
            for l in labels:
                if l.manipulation_phase and l.manipulation_phase not in phases:
                    phases.append(l.manipulation_phase)

            # Trajectory summary
            traj_summary = self._trajectory_summary(trajectories, start_t, end_t)

            # Review flags
            needs_review = False
            review_reason: Optional[str] = None

            if conf_avg < 0.4:
                needs_review = True
                review_reason = "low_confidence"
            elif not self.validate_phase_sequence(phases):
                needs_review = True
                review_reason = "phase_sequence_broken"
            elif labels:
                # Check hand visibility
                hands_missing = sum(
                    1 for l in labels if l.hand_used == "none"
                )
                if hands_missing > len(labels) * 0.5:
                    needs_review = True
                    review_reason = "hand_not_visible"

            seg = Segment(
                id=0,  # assigned later
                start_time=round(start_t, 4),
                end_time=round(end_t, 4),
                duration=round(duration, 4),
                action=action,
                action_description=desc,
                task_hierarchy=hierarchy,
                objects_involved=objects,
                grasp_type=grasp,
                hand_used=hand,
                manipulation_phases=phases,
                trajectory_summary=traj_summary,
                confidence_avg=conf_avg,
                confidence_min=conf_min,
                frames_analyzed=len(labels),
                frames_skipped=0,
                labelling_method=method,
                needs_review=needs_review,
                review_reason=review_reason,
            )
            segments.append(seg)

        return segments

    @staticmethod
    def _group_duration(grp: Dict) -> float:
        labels = grp.get("labels", [])
        if labels:
            return labels[-1].timestamp_end - labels[0].timestamp_start
        return grp.get("_end", 0.0) - grp.get("_start", 0.0)

    @staticmethod
    def _trajectory_summary(
        trajectories: Dict, start_t: float, end_t: float
    ) -> Dict[str, Any]:
        """Extract trajectory stats within the segment time window."""
        summary: Dict[str, Any] = {
            "path_length": 0.0,
            "max_velocity": 0.0,
            "grasp_events": 0,
        }

        for hand_label in ("left", "right"):
            traj: Optional[Trajectory] = trajectories.get(hand_label)
            if traj is None:
                continue

            # Find indices within time window
            indices = [
                i for i, ts in enumerate(traj.timestamps)
                if start_t <= ts <= end_t
            ]
            if not indices:
                continue

            summary["path_length"] = max(summary["path_length"], traj.path_length)
            summary["max_velocity"] = max(summary["max_velocity"], traj.max_velocity)

        # Count grasp events in window
        grasp_events = trajectories.get("grasp_events", [])
        summary["grasp_events"] = sum(
            1 for e in grasp_events if start_t <= e.timestamp <= end_t
        )

        return summary


# ── standalone test ───────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    import os

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from utils.logging_config import setup_logging

    setup_logging(verbose=True)

    # Quick smoke test with synthetic labels
    labels = []
    for i in range(10):
        action = "pick_up" if i < 5 else "idle"
        labels.append(TaskLabel(
            action=action,
            action_description=f"Test {action}",
            objects_involved=["wrench"],
            grasp_type="power",
            hand_used="right",
            manipulation_phase="manipulate",
            task_hierarchy={"high_level": "assembly", "mid_level": action},
            confidence=0.8 if i < 5 else 0.95,
            is_idle=(action == "idle"),
            is_transition=False,
            labelling_method="vlm",
            was_fallback=False,
            vlm_backend="gemini",
            raw_response={},
            timestamp_start=i * 2.0,
            timestamp_end=(i + 1) * 2.0,
        ))

    config = AnnotatorConfig()
    merger = SegmentMerger(config)
    segments = merger.merge(labels, {}, [])

    print(f"\n{len(segments)} segments:")
    for s in segments:
        print(
            f"  [{s.id}] {s.start_time:.1f}-{s.end_time:.1f}s "
            f"{s.action:15s} conf={s.confidence_avg:.2f} "
            f"review={s.needs_review}"
        )
