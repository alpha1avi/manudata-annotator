"""ManuData Annotator — Output Writer (Stage 8).

Generates all output formats: JSON, CSV, timeline, review queue,
and optionally RLDS (TFRecord) or HDF5.
"""

import csv
import json
import logging
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from config import AnnotatorConfig
from pipeline.local_vision_pipeline import FrameAnnotation
from pipeline.segment_merger import Segment
from pipeline.quality_scorer import QualityScore
from pipeline.trajectory_extractor import Trajectory

logger = logging.getLogger(__name__)


class OutputWriter:
    """Generate all annotation output files."""

    def __init__(self, config: AnnotatorConfig) -> None:
        self.output_format = config.output_format  # json, rlds, hdf5

    # ── main entry point ──────────────────────────────────────────────

    def write_all(
        self,
        output_dir: str,
        video_info: Dict[str, Any],
        segments: List[Segment],
        frame_annotations: List[FrameAnnotation],
        trajectories: Dict,
        quality_scores: Dict,
        processing_stats: Dict[str, Any],
    ) -> List[str]:
        """Generate all outputs and return the list of written file paths.

        Always writes: JSON, CSV, timeline, review queue.
        Additionally writes RLDS or HDF5 based on ``self.output_format``.
        """
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        written: List[str] = []

        # Always produce these four
        p = str(out / "annotations.json")
        self.write_json(p, video_info, segments, quality_scores, processing_stats)
        written.append(p)

        p = str(out / "annotations.csv")
        self.write_csv(p, segments, quality_scores)
        written.append(p)

        p = str(out / "timeline.txt")
        self.write_timeline(p, segments)
        written.append(p)

        p = str(out / "review_queue.json")
        self.write_review_queue(p, segments, frame_annotations)
        written.append(p)

        # Format-specific
        if self.output_format == "rlds":
            p = str(out / "rlds")
            self.write_rlds(p, segments, frame_annotations, trajectories)
            written.append(p)

        if self.output_format == "hdf5":
            p = str(out / "annotations.hdf5")
            self.write_hdf5(p, segments, frame_annotations, trajectories)
            written.append(p)

        for fp in written:
            size = _file_size_str(fp)
            logger.info("Written: %s (%s)", fp, size)

        return written

    # ── JSON ──────────────────────────────────────────────────────────

    def write_json(
        self,
        output_path: str,
        video_info: Dict,
        segments: List[Segment],
        quality_scores: Dict,
        processing_stats: Dict,
    ) -> None:
        """Full structured JSON annotation file."""

        # Action summary
        action_summary = self._action_summary(segments)

        # Manipulation profile
        manip_profile = self._manipulation_profile(segments)

        # Segments as dicts
        seg_dicts = []
        for seg in segments:
            qs = quality_scores.get("scores", {}).get(seg.id)
            seg_d = {
                "segment_id": seg.id,
                "start_time": seg.start_time,
                "end_time": seg.end_time,
                "duration": seg.duration,
                "action": seg.action,
                "action_description": seg.action_description,
                "task_hierarchy": seg.task_hierarchy,
                "objects_involved": seg.objects_involved,
                "grasp_type": seg.grasp_type,
                "hand_used": seg.hand_used,
                "manipulation_phases": seg.manipulation_phases,
                "trajectory_summary": seg.trajectory_summary,
                "confidence_avg": seg.confidence_avg,
                "confidence_min": seg.confidence_min,
                "frames_analyzed": seg.frames_analyzed,
                "frames_skipped": seg.frames_skipped,
                "labelling_method": seg.labelling_method,
                "needs_review": seg.needs_review,
                "review_reason": seg.review_reason,
                "quality_score": qs.composite if qs else None,
            }
            seg_dicts.append(seg_d)

        doc = {
            "video_file": video_info.get("video_file", ""),
            "video_type": video_info.get("video_type", "egocentric"),
            "camera_mount": video_info.get("camera_mount", "helmet"),
            "duration": video_info.get("duration", 0.0),
            "fps": video_info.get("fps", 0.0),
            "vlm_backend": video_info.get("vlm_backend", ""),
            "processing_stats": processing_stats,
            "segments": seg_dicts,
            "action_summary": action_summary,
            "manipulation_profile": manip_profile,
            "quality_aggregate": quality_scores.get("aggregate", {}),
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(doc, f, indent=2, ensure_ascii=False, default=str)

    # ── CSV ───────────────────────────────────────────────────────────

    def write_csv(
        self,
        output_path: str,
        segments: List[Segment],
        quality_scores: Dict,
    ) -> None:
        """Flat CSV with one row per segment."""
        fieldnames = [
            "segment_id", "start_time", "end_time", "duration",
            "action", "description", "objects", "grasp_type", "hand",
            "manipulation_phases", "confidence_avg", "quality_score",
            "needs_review", "review_reason", "labelling_method",
        ]

        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for seg in segments:
                qs = quality_scores.get("scores", {}).get(seg.id)
                writer.writerow({
                    "segment_id": seg.id,
                    "start_time": seg.start_time,
                    "end_time": seg.end_time,
                    "duration": seg.duration,
                    "action": seg.action,
                    "description": seg.action_description,
                    "objects": "; ".join(seg.objects_involved),
                    "grasp_type": seg.grasp_type,
                    "hand": seg.hand_used,
                    "manipulation_phases": " -> ".join(seg.manipulation_phases),
                    "confidence_avg": seg.confidence_avg,
                    "quality_score": qs.composite if qs else "",
                    "needs_review": seg.needs_review,
                    "review_reason": seg.review_reason or "",
                    "labelling_method": seg.labelling_method,
                })

    # ── timeline ──────────────────────────────────────────────────────

    def write_timeline(self, output_path: str, segments: List[Segment]) -> None:
        """Human-readable timeline file."""
        lines: List[str] = ["ManuData Annotation Timeline", "=" * 80, ""]

        for seg in segments:
            t_start = _fmt_time(seg.start_time)
            t_end = _fmt_time(seg.end_time)

            if seg.action == "transition":
                line = f"{t_start} - {t_end}  [---]  TRANSITION -- Head movement / walking"
            elif seg.needs_review:
                line = (
                    f"{t_start} - {t_end}  [{seg.confidence_avg:.2f}] "
                    f"{seg.action} -- NEEDS REVIEW ({seg.review_reason})"
                )
            else:
                phases = " -> ".join(seg.manipulation_phases) if seg.manipulation_phases else ""
                line = (
                    f"{t_start} - {t_end}  [{seg.confidence_avg:.2f}] "
                    f"{seg.action} -- {seg.action_description} "
                    f"({seg.hand_used}"
                )
                if phases:
                    line += f", {phases}"
                line += ")"

            lines.append(line)

        lines.append("")
        lines.append("=" * 80)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

    # ── review queue ──────────────────────────────────────────────────

    def write_review_queue(
        self,
        output_path: str,
        segments: List[Segment],
        frame_annotations: List[FrameAnnotation],
    ) -> None:
        """JSON file listing segments that need human review."""
        review_clips: List[Dict] = []

        for seg in segments:
            if not seg.needs_review:
                continue

            # Collect frame paths within segment window
            fps_in_seg = [
                a.frame_path for a in frame_annotations
                if seg.start_time <= a.timestamp <= seg.end_time
            ]

            review_clips.append({
                "segment_id": seg.id,
                "start_time": seg.start_time,
                "end_time": seg.end_time,
                "reason": seg.review_reason,
                "suggested_action": seg.action,
                "confidence": seg.confidence_avg,
                "frame_paths": fps_in_seg,
            })

        # Estimate review time: ~5 min per clip
        est_minutes = len(review_clips) * 5

        doc = {
            "review_clips": review_clips,
            "total_review_needed": len(review_clips),
            "estimated_review_time_minutes": est_minutes,
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(doc, f, indent=2, ensure_ascii=False)

    # ── RLDS (TFRecord) ──────────────────────────────────────────────

    def write_rlds(
        self,
        output_path: str,
        segments: List[Segment],
        frame_annotations: List[FrameAnnotation],
        trajectories: Dict,
    ) -> None:
        """RLDS format matching Open X-Embodiment.

        Each segment → episode; each frame → step.
        Skipped with a warning if TensorFlow is not installed.
        """
        try:
            import tensorflow as tf
        except ImportError:
            logger.warning(
                "TensorFlow not installed — skipping RLDS output. "
                "Install with: pip install tensorflow"
            )
            return

        out = Path(output_path)
        out.mkdir(parents=True, exist_ok=True)

        ann_by_ts = {a.timestamp: a for a in frame_annotations}

        for seg in segments:
            seg_anns = sorted(
                [a for a in frame_annotations if seg.start_time <= a.timestamp <= seg.end_time],
                key=lambda a: a.timestamp,
            )
            if not seg_anns:
                continue

            record_path = str(out / f"episode_{seg.id:04d}.tfrecord")
            with tf.io.TFRecordWriter(record_path) as writer:
                for step_idx, ann in enumerate(seg_anns):
                    # Image
                    frame = cv2.imread(ann.frame_path, cv2.IMREAD_COLOR)
                    if frame is None:
                        continue
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame_256 = cv2.resize(frame_rgb, (256, 256))
                    img_bytes = frame_256.tobytes()

                    # Depth
                    depth_bytes = b""
                    if ann.depth is not None:
                        depth_256 = cv2.resize(ann.depth.depth_map, (256, 256))
                        depth_bytes = depth_256.astype(np.float32).tobytes()

                    # Action (7-D delta)
                    action = self._compute_action_delta(
                        ann, seg_anns[step_idx - 1] if step_idx > 0 else None, trajectories
                    )

                    feature = {
                        "observation/image": tf.train.Feature(
                            bytes_list=tf.train.BytesList(value=[img_bytes])
                        ),
                        "observation/depth": tf.train.Feature(
                            bytes_list=tf.train.BytesList(value=[depth_bytes])
                        ),
                        "action": tf.train.Feature(
                            float_list=tf.train.FloatList(value=action.tolist())
                        ),
                        "language_instruction": tf.train.Feature(
                            bytes_list=tf.train.BytesList(
                                value=[seg.action_description.encode("utf-8")]
                            )
                        ),
                        "is_first": tf.train.Feature(
                            int64_list=tf.train.Int64List(value=[int(step_idx == 0)])
                        ),
                        "is_last": tf.train.Feature(
                            int64_list=tf.train.Int64List(
                                value=[int(step_idx == len(seg_anns) - 1)]
                            )
                        ),
                        "is_terminal": tf.train.Feature(
                            int64_list=tf.train.Int64List(
                                value=[int(step_idx == len(seg_anns) - 1)]
                            )
                        ),
                    }
                    example = tf.train.Example(
                        features=tf.train.Features(feature=feature)
                    )
                    writer.write(example.SerializeToString())

        logger.info("RLDS output: %d episodes written to %s", len(segments), output_path)

    # ── HDF5 ──────────────────────────────────────────────────────────

    def write_hdf5(
        self,
        output_path: str,
        segments: List[Segment],
        frame_annotations: List[FrameAnnotation],
        trajectories: Dict,
    ) -> None:
        """HDF5 format matching DROID / robomimic conventions."""
        import h5py

        with h5py.File(output_path, "w") as hf:
            for seg in segments:
                seg_anns = sorted(
                    [a for a in frame_annotations
                     if seg.start_time <= a.timestamp <= seg.end_time],
                    key=lambda a: a.timestamp,
                )
                if not seg_anns:
                    continue

                T = len(seg_anns)
                ep = hf.create_group(f"episode_{seg.id}")

                # ── observations ──
                obs = ep.create_group("obs")
                images_grp = obs.create_group("images")
                img_ds = images_grp.create_dataset(
                    "cam_head", shape=(T, 256, 256, 3), dtype=np.uint8
                )

                has_depth = any(a.depth is not None for a in seg_anns)
                depth_ds = None
                if has_depth:
                    depth_grp = obs.create_group("depth")
                    depth_ds = depth_grp.create_dataset(
                        "cam_head", shape=(T, 256, 256), dtype=np.float32
                    )

                # Hand pose: (T, 2, 21, 2)
                hand_pose_ds = obs.create_dataset(
                    "hand_pose", shape=(T, 2, 21, 2), dtype=np.float32
                )

                # ── action (T, 7) ──
                action_ds = ep.create_dataset("action", shape=(T, 7), dtype=np.float32)

                # ── quality scores (T,) ──
                quality_ds = ep.create_dataset(
                    "quality_scores", shape=(T,), dtype=np.float32
                )

                # ── per-step labels ──
                dt = h5py.string_dtype()
                label_ds = ep.create_dataset("action_labels", shape=(T,), dtype=dt)

                # ── language instruction ──
                ep.attrs["language_instruction"] = seg.action_description

                # Fill data
                for i, ann in enumerate(seg_anns):
                    # Image
                    frame = cv2.imread(ann.frame_path, cv2.IMREAD_COLOR)
                    if frame is not None:
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        img_ds[i] = cv2.resize(frame_rgb, (256, 256))

                    # Depth
                    if depth_ds is not None and ann.depth is not None:
                        depth_ds[i] = cv2.resize(ann.depth.depth_map, (256, 256))

                    # Hand pose
                    kp_array = np.zeros((2, 21, 2), dtype=np.float32)
                    for h_idx, hand in enumerate(ann.hand_pose.hands[:2]):
                        for k_idx, (kx, ky) in enumerate(hand.keypoints_2d[:21]):
                            kp_array[h_idx, k_idx] = [kx, ky]
                    hand_pose_ds[i] = kp_array

                    # Action delta
                    prev_ann = seg_anns[i - 1] if i > 0 else None
                    action_ds[i] = self._compute_action_delta(ann, prev_ann, trajectories)

                    # Quality
                    quality_ds[i] = ann.activity_score

                    # Label
                    label_ds[i] = seg.action

        logger.info("HDF5 output: %d episodes written to %s", len(segments), output_path)

    # ── helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _compute_action_delta(
        current: FrameAnnotation,
        previous: Optional[FrameAnnotation],
        trajectories: Dict,
    ) -> np.ndarray:
        """Compute 7-D action: [dx, dy, dz, droll, dpitch, dyaw, grasp].

        Falls back to zeros when trajectory data is unavailable.
        """
        action = np.zeros(7, dtype=np.float32)

        if previous is None:
            return action

        # Try to get wrist positions from the dominant hand
        for hand_label in ("right", "left"):
            curr_wrist = _get_wrist(current, hand_label)
            prev_wrist = _get_wrist(previous, hand_label)
            if curr_wrist is not None and prev_wrist is not None:
                action[0] = curr_wrist[0] - prev_wrist[0]  # dx
                action[1] = curr_wrist[1] - prev_wrist[1]  # dy
                break

        # Depth delta
        if current.depth is not None and previous.depth is not None:
            curr_w = _get_wrist(current, "right") or _get_wrist(current, "left")
            prev_w = _get_wrist(previous, "right") or _get_wrist(previous, "left")
            if curr_w and prev_w:
                cz = current.depth.depth_at_point(int(curr_w[0]), int(curr_w[1]))
                pz = previous.depth.depth_at_point(int(prev_w[0]), int(prev_w[1]))
                action[2] = cz - pz  # dz

        # Grasp signal (simplified: thumb-index distance)
        grasp = 0.0
        for h in current.hand_pose.hands:
            if len(h.keypoints_pixel) >= 21:
                thumb = np.array(h.keypoints_pixel[4], dtype=np.float32)
                index = np.array(h.keypoints_pixel[8], dtype=np.float32)
                dist = float(np.linalg.norm(thumb - index))
                grasp = max(grasp, 1.0 - min(dist / 100.0, 1.0))
                break
        action[6] = grasp

        return action

    @staticmethod
    def _action_summary(segments: List[Segment]) -> List[Dict]:
        """Per-action counts, durations, avg confidence."""
        action_data: Dict[str, Dict] = defaultdict(
            lambda: {"count": 0, "total_duration": 0.0, "confidences": []}
        )
        for seg in segments:
            d = action_data[seg.action]
            d["count"] += 1
            d["total_duration"] += seg.duration
            d["confidences"].append(seg.confidence_avg)

        summary = []
        for action, d in sorted(action_data.items(), key=lambda x: -x[1]["total_duration"]):
            confs = d["confidences"]
            summary.append({
                "action": action,
                "count": d["count"],
                "total_duration": round(d["total_duration"], 2),
                "avg_confidence": round(sum(confs) / len(confs), 4) if confs else 0.0,
            })
        return summary

    @staticmethod
    def _manipulation_profile(segments: List[Segment]) -> Dict[str, Any]:
        """Compute manipulation profile from segments."""
        total_dur = sum(s.duration for s in segments) or 1.0
        idle_dur = sum(s.duration for s in segments if s.action == "idle")

        hands = [s.hand_used for s in segments if s.hand_used != "none"]
        hand_counts = Counter(hands)
        dominant = hand_counts.most_common(1)[0][0] if hand_counts else "unknown"
        bimanual_count = sum(1 for s in segments if s.hand_used == "both")

        grasps = [s.grasp_type for s in segments if s.grasp_type != "none"]
        grasp_counts = Counter(grasps)
        top_grasp = grasp_counts.most_common(1)[0][0] if grasp_counts else "none"

        return {
            "dominant_hand": dominant,
            "bimanual_pct": round(bimanual_count / max(len(segments), 1) * 100, 1),
            "idle_pct": round(idle_dur / total_dur * 100, 1),
            "most_used_grasp": top_grasp,
            "grasp_distribution": dict(grasp_counts),
            "hand_distribution": dict(hand_counts),
            "total_segments": len(segments),
            "total_duration": round(total_dur, 2),
        }


# ── module-level helpers ──────────────────────────────────────────────


def _get_wrist(ann: FrameAnnotation, hand: str):
    """Get wrist pixel coords for a hand, or None."""
    for h in ann.hand_pose.hands:
        if h.handedness == hand and len(h.keypoints_pixel) >= 1:
            return h.keypoints_pixel[0]
    return None


def _fmt_time(seconds: float) -> str:
    """Format seconds as MM:SS."""
    m = int(seconds) // 60
    s = int(seconds) % 60
    return f"{m:02d}:{s:02d}"


def _file_size_str(path: str) -> str:
    """Human-readable file/dir size."""
    p = Path(path)
    if p.is_file():
        size = p.stat().st_size
    elif p.is_dir():
        size = sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
    else:
        return "N/A"
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


# ── standalone test ───────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from utils.logging_config import setup_logging

    setup_logging(verbose=True)
    print("OutputWriter module loaded successfully.")
    print("  Formats: JSON, CSV, timeline, review queue, RLDS, HDF5")
    print("  Use write_all() with segments and annotations.")
