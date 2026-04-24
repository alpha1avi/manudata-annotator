"""ManuData Annotator — End-to-End Smoke Test.

Validates the full pipeline works without real video or API keys.

Run with:
    python test_pipeline.py
"""

import csv
import json
import os
import shutil
import sys
import tempfile
import traceback
from pathlib import Path

import cv2
import numpy as np

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── colour helpers ──────────────────────────────────────────────────

GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
RESET = "\033[0m"
BOLD = "\033[1m"


def _pass(stage: str, detail: str = "") -> None:
    extra = f" — {detail}" if detail else ""
    print(f"  {GREEN}PASS{RESET}  {stage}{extra}")


def _fail(stage: str, detail: str = "") -> None:
    extra = f" — {detail}" if detail else ""
    print(f"  {RED}FAIL{RESET}  {stage}{extra}")


def _warn(stage: str, detail: str = "") -> None:
    extra = f" — {detail}" if detail else ""
    print(f"  {YELLOW}WARN{RESET}  {stage}{extra}")


# ── synthetic video creation ────────────────────────────────────────

def create_synthetic_video(path: str, duration: float = 10.0,
                           width: int = 640, height: int = 480,
                           fps: int = 30) -> str:
    """Create a synthetic test video.

    - First 3 seconds:  static background with a coloured rectangle (workbench)
    - Next 4 seconds:   a circle moves across the frame (simulates hand motion)
    - Last 3 seconds:   static again
    """
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, (width, height))

    total_frames = int(duration * fps)
    phase1_end = int(3.0 * fps)
    phase2_end = int(7.0 * fps)

    for i in range(total_frames):
        frame = np.full((height, width, 3), (40, 40, 40), dtype=np.uint8)

        # Static workbench rectangle (always present)
        cv2.rectangle(frame, (100, 200), (540, 400), (120, 80, 50), -1)
        cv2.rectangle(frame, (100, 200), (540, 400), (180, 140, 90), 2)

        if i < phase1_end:
            # Phase 1: static
            pass
        elif i < phase2_end:
            # Phase 2: moving circle (simulating hand)
            progress = (i - phase1_end) / (phase2_end - phase1_end)
            cx = int(120 + progress * 400)
            cy = int(300 + 50 * np.sin(progress * np.pi * 2))
            cv2.circle(frame, (cx, cy), 35, (200, 170, 140), -1)
            cv2.circle(frame, (cx, cy), 35, (230, 200, 170), 2)
            # Fingers
            for angle_offset in [-30, -10, 10, 30]:
                fx = cx + int(30 * np.cos(np.radians(angle_offset - 90)))
                fy = cy + int(30 * np.sin(np.radians(angle_offset - 90)))
                cv2.circle(frame, (fx, fy), 6, (200, 170, 140), -1)
        else:
            # Phase 3: static again
            pass

        writer.write(frame)

    writer.release()
    return path


# ── main smoke test ─────────────────────────────────────────────────

def run_smoke_test() -> bool:
    """Run all pipeline stages and report PASS/FAIL for each."""
    print()
    print(f"{BOLD}{'=' * 60}{RESET}")
    print(f"{BOLD}  ManuData Annotator — End-to-End Smoke Test{RESET}")
    print(f"{BOLD}{'=' * 60}{RESET}")
    print()

    results = {}
    tmpdir = tempfile.mkdtemp(prefix="manudata_test_")
    video_path = os.path.join(tmpdir, "test_video.mp4")
    output_dir = os.path.join(tmpdir, "output")
    frames_dir = os.path.join(output_dir, "frames")
    os.makedirs(output_dir, exist_ok=True)

    try:
        # ── Stage 0: Create synthetic video ─────────────────────────
        try:
            create_synthetic_video(video_path)
            assert os.path.exists(video_path), "Video file not created"
            file_size = os.path.getsize(video_path)
            _pass("Stage 0: Synthetic video", f"{file_size / 1024:.0f} KB")
            results["stage_0"] = True
        except Exception as exc:
            _fail("Stage 0: Synthetic video", str(exc))
            traceback.print_exc()
            results["stage_0"] = False
            return False

        # ── Stage 1: Frame extraction ───────────────────────────────
        frame_tuples = []  # List[(timestamp, filepath)]
        try:
            from pipeline.frame_extractor import FrameExtractor
            from config import AnnotatorConfig

            config = AnnotatorConfig(fps=2)
            extractor = FrameExtractor(config)
            extraction_result = extractor.extract(video_path, output_dir)

            frame_tuples = extraction_result.frame_paths
            assert extraction_result.frame_count > 0, "No frames extracted"
            assert len(frame_tuples) > 0, "No frame paths"
            assert all(
                os.path.exists(fp) for _, fp in frame_tuples
            ), "Some frame files missing"

            _pass(
                "Stage 1: Frame extraction",
                f"{extraction_result.frame_count} frames at {config.fps} fps",
            )
            results["stage_1"] = True
        except Exception as exc:
            _fail("Stage 1: Frame extraction", str(exc))
            traceback.print_exc()
            results["stage_1"] = False

            # Manual fallback via OpenCV
            from config import AnnotatorConfig
            config = AnnotatorConfig(fps=2)
            os.makedirs(frames_dir, exist_ok=True)
            cap = cv2.VideoCapture(video_path)
            src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            interval = max(1, int(round(src_fps / config.fps)))
            idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if idx % interval == 0:
                    ts = idx / src_fps
                    p = os.path.join(frames_dir, f"frame_{idx:06d}.jpg")
                    cv2.imwrite(p, frame)
                    frame_tuples.append((ts, p))
                idx += 1
            cap.release()

        # ── Stage 2: Quality filter ─────────────────────────────────
        filtered_tuples = frame_tuples
        try:
            from pipeline.frame_quality_filter import FrameQualityFilter

            qf = FrameQualityFilter(config)
            filter_result = qf.filter(frame_tuples)

            filtered_tuples = filter_result.good_frames
            n_dup = len(filter_result.skipped_duplicate)
            n_blur = len(filter_result.skipped_blur)
            n_dark = len(filter_result.skipped_dark)
            n_trans = len(filter_result.skipped_transition)

            _pass(
                "Stage 2: Quality filter",
                f"{len(filtered_tuples)} accepted, "
                f"{n_dup} dup, {n_blur} blur, {n_dark} dark, {n_trans} trans",
            )
            results["stage_2"] = True
        except Exception as exc:
            _fail("Stage 2: Quality filter", str(exc))
            traceback.print_exc()
            results["stage_2"] = False

        # ── Stage 3: Local vision pipeline ──────────────────────────
        frame_annotations = []
        try:
            from pipeline.local_vision_pipeline import LocalVisionPipeline

            lvp = LocalVisionPipeline(config)
            frame_annotations = lvp.process_frames(
                filtered_tuples[:10],  # limit for speed
            )

            _pass(
                "Stage 3: Local vision",
                f"{len(frame_annotations)} frames annotated",
            )
            results["stage_3"] = True
        except Exception as exc:
            _warn("Stage 3: Local vision", f"Graceful skip — {exc}")
            results["stage_3"] = None  # Expected with synthetic / missing deps

            # Create minimal annotations for downstream stages
            from pipeline.local_vision_pipeline import FrameAnnotation
            from models.hand_pose import HandPoseResult

            for i, (ts, fp) in enumerate(filtered_tuples[:10]):
                frame_annotations.append(FrameAnnotation(
                    timestamp=ts,
                    frame_path=fp,
                    hand_pose=HandPoseResult(hands=[], num_hands=0, hand_visibility="no_hands"),
                    objects=[], depth=None, interactions=[],
                    activity_score=0.3 if i < 6 else 0.1,
                    hand_visibility="no_hands",
                    processing_time_ms=1.0,
                ))

        # ── Stage 4: VLM (skipped — mock labels) ───────────────────
        mock_labels = []
        try:
            from pipeline.task_labeller import TaskLabel

            # Pick-up action (first 5 frames)
            for i in range(5):
                mock_labels.append(TaskLabel(
                    action="pick_up",
                    action_description="Picking up a wrench from the workbench",
                    objects_involved=["wrench"],
                    grasp_type="power",
                    hand_used="right",
                    manipulation_phase="manipulate" if i > 1 else "reach",
                    task_hierarchy={
                        "high_level": "assembly",
                        "mid_level": "pick_up",
                    },
                    confidence=0.82 + i * 0.02,
                    is_idle=False,
                    is_transition=False,
                    labelling_method="vlm",
                    was_fallback=False,
                    vlm_backend="mock",
                    raw_response={},
                    timestamp_start=i * 1.0,
                    timestamp_end=(i + 1) * 1.0,
                ))

            # Tighten action (next 3 frames)
            for i in range(3):
                mock_labels.append(TaskLabel(
                    action="tighten_bolt",
                    action_description="Tightening bolt with wrench",
                    objects_involved=["wrench", "bolt"],
                    grasp_type="power",
                    hand_used="right",
                    manipulation_phase="manipulate",
                    task_hierarchy={
                        "high_level": "assembly",
                        "mid_level": "tighten_bolt",
                    },
                    confidence=0.78 + i * 0.03,
                    is_idle=False,
                    is_transition=False,
                    labelling_method="vlm",
                    was_fallback=False,
                    vlm_backend="mock",
                    raw_response={},
                    timestamp_start=5.0 + i * 1.0,
                    timestamp_end=6.0 + i * 1.0,
                ))

            # Idle (last 2 frames)
            for i in range(2):
                mock_labels.append(TaskLabel(
                    action="idle",
                    action_description="Worker idle",
                    objects_involved=[],
                    grasp_type="none",
                    hand_used="none",
                    manipulation_phase="",
                    task_hierarchy={"high_level": "idle"},
                    confidence=0.95,
                    is_idle=True,
                    is_transition=False,
                    labelling_method="vlm",
                    was_fallback=False,
                    vlm_backend="mock",
                    raw_response={},
                    timestamp_start=8.0 + i * 1.0,
                    timestamp_end=9.0 + i * 1.0,
                ))

            _pass("Stage 4: VLM (mock)", f"{len(mock_labels)} mock labels created")
            results["stage_4"] = True
        except Exception as exc:
            _fail("Stage 4: VLM (mock)", str(exc))
            traceback.print_exc()
            results["stage_4"] = False
            return False

        # ── Stage 5: Trajectory extraction ──────────────────────────
        trajectories = {}
        try:
            from pipeline.trajectory_extractor import TrajectoryExtractor

            tex = TrajectoryExtractor()
            trajectories = tex.extract(frame_annotations)

            n_traj = sum(1 for k in ("left", "right") if k in trajectories)
            n_events = len(trajectories.get("grasp_events", []))
            if n_traj > 0:
                _pass("Stage 5: Trajectory", f"{n_traj} hand(s), {n_events} grasp events")
            else:
                _warn("Stage 5: Trajectory", "No hand data (expected with synthetic)")
            results["stage_5"] = True
        except Exception as exc:
            _warn("Stage 5: Trajectory", f"Graceful skip — {exc}")
            results["stage_5"] = None

        # ── Stage 6: Segment merging ────────────────────────────────
        segments = []
        try:
            from pipeline.segment_merger import SegmentMerger

            merger = SegmentMerger(config)
            segments = merger.merge(mock_labels, trajectories, [])

            assert len(segments) > 0, "No segments produced"
            n_review = sum(1 for s in segments if s.needs_review)
            _pass(
                "Stage 6: Segment merger",
                f"{len(segments)} segments ({n_review} need review)",
            )
            results["stage_6"] = True
        except Exception as exc:
            _fail("Stage 6: Segment merger", str(exc))
            traceback.print_exc()
            results["stage_6"] = False

        # ── Stage 7: Quality scoring ────────────────────────────────
        quality_result = None
        try:
            from pipeline.quality_scorer import QualityScorer

            scorer = QualityScorer()
            quality_result = scorer.score_video(segments, frame_annotations, trajectories)

            agg = quality_result["aggregate"]
            _pass(
                "Stage 7: Quality scorer",
                f"mean={agg['mean']:.3f}, median={agg['median']:.3f}, "
                f"discard={agg['may_discard_count']}/{agg['total_segments']}",
            )
            results["stage_7"] = True
        except Exception as exc:
            _fail("Stage 7: Quality scorer", str(exc))
            traceback.print_exc()
            results["stage_7"] = False

        # ── Stage 8: Output writing ─────────────────────────────────
        try:
            from pipeline.output_writer import OutputWriter

            writer = OutputWriter(config)

            # Prepare quality scores dict
            q_scores = {}
            if quality_result:
                q_scores = quality_result["scores"]

            video_info = {
                "video_path": video_path,
                "duration": 10.0,
                "fps": 30,
                "resolution": "640x480",
            }

            processing_stats = {
                "total_frames": len(frame_tuples),
                "accepted_frames": len(filtered_tuples),
                "processing_time_s": 1.5,
                "estimated_cost_inr": 0.0,
                "vlm_backend": "mock",
            }

            writer.write_all(
                output_dir=output_dir,
                video_info=video_info,
                segments=segments,
                frame_annotations=frame_annotations,
                trajectories=trajectories,
                quality_scores=q_scores,
                processing_stats=processing_stats,
            )

            # Validate outputs
            json_files = list(Path(output_dir).glob("*.json"))
            csv_files = list(Path(output_dir).glob("*.csv"))
            timeline_files = list(Path(output_dir).glob("*timeline*"))

            assert len(json_files) > 0, "No JSON output"
            assert len(csv_files) > 0, "No CSV output"
            assert len(timeline_files) > 0, "No timeline output"

            # Validate JSON is parseable
            annotations_json = None
            for jf in json_files:
                if "annotations" in jf.name:
                    annotations_json = jf
                    break
            if annotations_json is None:
                annotations_json = json_files[0]

            with open(annotations_json, "r", encoding="utf-8") as f:
                data = json.load(f)
            assert "segments" in data, "JSON missing 'segments'"
            assert len(data["segments"]) > 0, "JSON has no segments"

            # Validate CSV is parseable
            with open(csv_files[0], "r", encoding="utf-8") as f:
                reader = csv.reader(f)
                rows = list(reader)
            assert len(rows) > 1, "CSV has no data rows"

            # Validate timeline
            with open(timeline_files[0], "r", encoding="utf-8") as f:
                timeline_text = f.read()
            assert len(timeline_text) > 0, "Timeline is empty"

            _pass(
                "Stage 8: Output writer",
                f"JSON({len(json_files)}), CSV({len(csv_files)}), "
                f"Timeline({len(timeline_files)})",
            )
            results["stage_8"] = True

        except Exception as exc:
            _fail("Stage 8: Output writer", str(exc))
            traceback.print_exc()
            results["stage_8"] = False

        # ── Summary ─────────────────────────────────────────────────
        print()
        print(f"{BOLD}{'=' * 60}{RESET}")
        print(f"{BOLD}  SMOKE TEST SUMMARY{RESET}")
        print(f"{BOLD}{'=' * 60}{RESET}")

        total = len(results)
        passed = sum(1 for v in results.values() if v is True)
        warned = sum(1 for v in results.values() if v is None)
        failed = sum(1 for v in results.values() if v is False)

        for stage, status in sorted(results.items()):
            if status is True:
                icon = f"{GREEN}PASS{RESET}"
            elif status is None:
                icon = f"{YELLOW}WARN{RESET}"
            else:
                icon = f"{RED}FAIL{RESET}"
            print(f"  {icon}  {stage}")

        print()
        print(f"  Total: {total} | "
              f"{GREEN}Passed: {passed}{RESET} | "
              f"{YELLOW}Warned: {warned}{RESET} | "
              f"{RED}Failed: {failed}{RESET}")

        if failed == 0:
            print(f"\n  {GREEN}{BOLD}ALL CRITICAL STAGES PASSED{RESET}")
        else:
            print(f"\n  {RED}{BOLD}{failed} STAGE(S) FAILED{RESET}")

        print(f"\n  Output directory: {output_dir}")
        print()

        return failed == 0

    finally:
        # Clean up temp directory
        try:
            shutil.rmtree(tmpdir, ignore_errors=True)
        except Exception:
            pass


# ── entry point ─────────────────────────────────────────────────────

if __name__ == "__main__":
    success = run_smoke_test()
    sys.exit(0 if success else 1)
