"""ManuData Annotator — Main CLI & Orchestrator.

Wires all pipeline stages together for end-to-end video annotation.

Subcommands:
    bootstrap  — Extract frames, filter, run local CV, annotate via VLM.
    annotate   — Run local fine-tuned model with optional VLM fallback.

Usage:
    python annotator.py bootstrap video.mp4 --vlm gemini --fps 2
    python annotator.py annotate video.mp4 --model-path weights/best.pt
    python annotator.py bootstrap ./videos/ --batch --vlm claude_haiku
"""

import argparse
import asyncio
import logging
import signal
import sys
import time
from pathlib import Path
from typing import List

from config import AnnotatorConfig
from pipeline.frame_extractor import FrameExtractor
from pipeline.frame_quality_filter import FrameQualityFilter
from pipeline.hand_activity_detector import HandActivityDetector
from pipeline.local_vision_pipeline import LocalVisionPipeline
from pipeline.task_labeller import TaskLabeller
from pipeline.trajectory_extractor import TrajectoryExtractor
from pipeline.segment_merger import SegmentMerger
from pipeline.quality_scorer import QualityScorer
from pipeline.output_writer import OutputWriter
from utils.cost_estimator import estimate_cost
from utils.logging_config import setup_logging
from utils.progress_tracker import ProgressTracker

logger = logging.getLogger(__name__)

# ── graceful shutdown ─────────────────────────────────────────────────

_shutdown_requested = False


def _signal_handler(sig, frame):
    global _shutdown_requested
    if _shutdown_requested:
        logger.warning("Force quit — exiting immediately")
        sys.exit(1)
    _shutdown_requested = True
    logger.warning("Ctrl+C received — saving progress and shutting down …")


signal.signal(signal.SIGINT, _signal_handler)


# ── CLI ───────────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser with bootstrap and annotate subcommands."""
    parser = argparse.ArgumentParser(
        prog="manudata-annotator",
        description="ManuData Annotator — manufacturing video annotation pipeline",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # ── bootstrap ──────────────────────────────────────────────────────
    bp = subparsers.add_parser(
        "bootstrap",
        help="Extract frames from video, filter, and annotate via VLM API",
    )
    bp.add_argument("input_path", help="Video file or directory (with --batch)")
    bp.add_argument("--vlm", default="gemini",
                    choices=["gemini", "claude_haiku", "claude_sonnet", "openai"],
                    help="VLM backend for annotation (default: gemini)")
    bp.add_argument("--fps", type=int, default=2, help="Frame extraction rate (default: 2)")
    bp.add_argument("--batch-size", type=int, default=5, help="Frames per VLM batch (default: 5)")
    bp.add_argument("--output-dir", default=None, help="Output directory (default: auto)")
    bp.add_argument("--max-concurrent", type=int, default=3,
                    help="Max concurrent VLM requests (default: 3)")
    bp.add_argument("--cost-estimate", action="store_true", help="Print cost estimate and exit")
    bp.add_argument("--resume", action="store_true", help="Resume from previous progress")
    bp.add_argument("--object-taxonomy", nargs="+", default=None,
                    help="Custom object taxonomy list")
    bp.add_argument("--duration", type=float, default=None,
                    help="Only process first N seconds")
    bp.add_argument("--verbose", action="store_true", help="Enable debug logging")
    bp.add_argument("--batch", action="store_true",
                    help="Process all videos in a directory")
    bp.add_argument("--save-filtered-frames", action="store_true",
                    help="Save frames that pass quality filtering")

    # ── annotate ───────────────────────────────────────────────────────
    ap = subparsers.add_parser(
        "annotate",
        help="Run local model inference with optional VLM fallback",
    )
    ap.add_argument("input_path", help="Video file or directory (with --batch)")
    ap.add_argument("--model-path", required=True, help="Path to fine-tuned model weights")
    ap.add_argument("--output-dir", default=None, help="Output directory (default: auto)")
    ap.add_argument("--output-format", default="json", choices=["json", "rlds", "hdf5"],
                    help="Output format (default: json)")
    ap.add_argument("--fallback-vlm", default="gemini",
                    choices=["gemini", "claude_haiku", "claude_sonnet", "openai"],
                    help="VLM backend for low-confidence fallback (default: gemini)")
    ap.add_argument("--fallback-threshold", type=float, default=0.4,
                    help="Confidence below which VLM fallback is triggered (default: 0.4)")
    ap.add_argument("--no-fallback", action="store_true",
                    help="Disable VLM fallback entirely")
    ap.add_argument("--review-only", action="store_true",
                    help="Only review and correct existing annotations")
    ap.add_argument("--gpu", default=None, help="GPU device ID (default: auto)")
    ap.add_argument("--batch", action="store_true",
                    help="Process all videos in a directory")
    ap.add_argument("--verbose", action="store_true", help="Enable debug logging")

    return parser


def build_config(args: argparse.Namespace) -> AnnotatorConfig:
    """Build an AnnotatorConfig from parsed CLI arguments."""
    kwargs = {}

    if hasattr(args, "fps"):
        kwargs["fps"] = args.fps
    if hasattr(args, "batch_size"):
        kwargs["batch_size"] = args.batch_size
    if hasattr(args, "max_concurrent"):
        kwargs["max_concurrent"] = args.max_concurrent
    if hasattr(args, "vlm"):
        kwargs["vlm_backend"] = args.vlm
    if hasattr(args, "fallback_vlm"):
        kwargs["vlm_backend"] = args.fallback_vlm
    if hasattr(args, "output_format"):
        kwargs["output_format"] = args.output_format
    if hasattr(args, "fallback_threshold"):
        kwargs["fallback_threshold"] = args.fallback_threshold
    if hasattr(args, "model_path") and args.model_path:
        kwargs["model_path"] = args.model_path
    if hasattr(args, "object_taxonomy") and args.object_taxonomy:
        kwargs["object_taxonomy"] = args.object_taxonomy

    return AnnotatorConfig(**kwargs)


def resolve_input_paths(input_path: str, batch: bool) -> List[Path]:
    """Resolve input to a list of video file paths."""
    p = Path(input_path)
    video_exts = {".mp4", ".avi", ".mkv", ".mov", ".webm", ".flv"}

    if batch:
        if not p.is_dir():
            logger.error("--batch requires a directory, got: %s", input_path)
            sys.exit(1)
        paths = sorted(f for f in p.iterdir() if f.suffix.lower() in video_exts)
        if not paths:
            logger.error("No video files found in %s", input_path)
            sys.exit(1)
        return paths
    else:
        if not p.exists():
            logger.error("Input not found: %s", input_path)
            sys.exit(1)
        return [p]


def resolve_output_dir(video_path: Path, output_dir_arg: str = None) -> str:
    """Determine the output directory for a single video."""
    if output_dir_arg:
        out = Path(output_dir_arg)
    else:
        out = video_path.parent / f"{video_path.stem}_annotations"
    out.mkdir(parents=True, exist_ok=True)
    return str(out)


# ── bootstrap pipeline ───────────────────────────────────────────────


async def run_bootstrap_single(
    video_path: str, config: AnnotatorConfig, output_dir: str,
    cost_estimate_only: bool = False, resume: bool = False,
    duration: float = None,
) -> dict:
    """Bootstrap mode: VLM-based labelling with local CV enrichment."""
    global _shutdown_requested
    start_time = time.time()

    logger.info("Starting bootstrap annotation: %s", video_path)

    # Progress tracker for resume support
    tracker = ProgressTracker(output_dir, video_path)
    if resume:
        logger.info("Resuming from previous progress …")

    # Stage 1: Extract frames
    if not tracker.is_stage_completed("extraction") or not resume:
        tracker.set_stage("extraction")
        extractor = FrameExtractor(config)
        extraction = extractor.extract(video_path, output_dir, duration=duration)
        tracker.save_partial_result("extraction_total", extraction.total_frames)
        tracker.complete_stage("extraction")
    else:
        # Reload extraction result from frames on disk
        extractor = FrameExtractor(config)
        extraction = extractor.extract(video_path, output_dir, duration=duration)
        logger.info("Reusing existing extraction: %d frames", extraction.total_frames)

    logger.info("Extracted %d frames at %d fps", extraction.total_frames, config.fps)

    if _shutdown_requested:
        logger.info("Shutdown after extraction — progress saved")
        return {"status": "interrupted", "stage": "extraction"}

    # Stage 2: Filter bad frames
    tracker.set_stage("filtering")
    quality_filter = FrameQualityFilter(config)
    filter_result = quality_filter.filter(extraction.frame_paths)
    quality_filter.print_stats(filter_result)
    tracker.complete_stage("filtering")

    # Cost estimate checkpoint
    cost = estimate_cost(
        total_frames=extraction.total_frames,
        frames_after_filter=len(filter_result.good_frames),
        vlm_backend=config.vlm_backend,
        batch_size=config.batch_size,
    )
    logger.info("Estimated VLM cost: INR %.2f (%s)", cost["estimated_cost_inr"], config.vlm_backend)

    if cost_estimate_only:
        print("\n=== Cost Estimate ===")
        print(f"  Total frames:     {cost['total_frames']}")
        print(f"  After filtering:  {cost['frames_after_filter']} ({cost['filter_ratio']:.1%})")
        print(f"  Batches:          {cost['num_batches']}")
        print(f"  Backend:          {cost['vlm_backend']}")
        print(f"  Est. cost:        INR {cost['estimated_cost_inr']:.2f}")
        print("\n  All backends:")
        for backend, c in cost["all_backends"].items():
            print(f"    {backend:16s}  INR {c:.2f}")
        return {"status": "cost_estimate", "cost": cost}

    if _shutdown_requested:
        logger.info("Shutdown after filtering — progress saved")
        return {"status": "interrupted", "stage": "filtering"}

    # Stage 3: Hand activity detection (lightweight, decides VLM routing)
    tracker.set_stage("activity_detection")
    activity_detector = HandActivityDetector(config)
    activity_result = activity_detector.process_frames(filter_result.good_frames)
    activity_detector.close()

    frames_for_vlm = [(m.timestamp, m.filepath) for m in activity_result.frames_to_vlm]
    idle_frames = activity_result.frames_idle
    logger.info(
        "Activity routing: %d frames -> VLM, %d idle",
        len(frames_for_vlm), len(idle_frames),
    )
    tracker.complete_stage("activity_detection")

    if _shutdown_requested:
        logger.info("Shutdown after activity detection — progress saved")
        return {"status": "interrupted", "stage": "activity_detection"}

    # Stage 3 (full): Local vision pipeline on VLM-bound frames
    tracker.set_stage("local_vision")
    vision = LocalVisionPipeline(config, use_gpu=True)
    frame_annotations = vision.process_frames(frames_for_vlm, depth_every_n=5)
    tracker.complete_stage("local_vision")

    if _shutdown_requested:
        logger.info("Shutdown after local vision — progress saved")
        vision.close()
        return {"status": "interrupted", "stage": "local_vision"}

    # Stage 4: VLM task labelling with enriched context
    tracker.set_stage("vlm_labelling")
    labeller = TaskLabeller("bootstrap", config)
    labeller._vision_pipeline = vision

    task_labels = await labeller.label_video(
        frames_for_vlm, frame_annotations, idle_frames
    )
    labeller_cost = labeller.get_cost_summary()
    logger.info("VLM cost: INR %.2f (%d calls)", labeller_cost["total_cost_inr"], labeller_cost["total_calls"])
    tracker.complete_stage("vlm_labelling")

    if _shutdown_requested:
        logger.info("Shutdown after VLM labelling — saving partial results")
        await labeller.close()
        vision.close()
        return {"status": "interrupted", "stage": "vlm_labelling"}

    # Stage 5: Trajectory extraction
    tracker.set_stage("trajectory")
    traj_extractor = TrajectoryExtractor()
    trajectories = traj_extractor.extract(frame_annotations)
    tracker.complete_stage("trajectory")

    # Stage 6: Segment merging
    tracker.set_stage("merging")
    merger = SegmentMerger(config)
    segments = merger.merge(task_labels, trajectories, filter_result.skipped_transition)
    tracker.complete_stage("merging")

    # Stage 7: Quality scoring
    tracker.set_stage("scoring")
    scorer = QualityScorer()
    quality_scores = scorer.score_video(segments, frame_annotations, trajectories)
    tracker.complete_stage("scoring")

    # Stage 8: Output
    tracker.set_stage("output")
    writer = OutputWriter(config)
    processing_stats = {
        "total_frames_extracted": extraction.total_frames,
        "frames_after_filter": len(filter_result.good_frames),
        "frames_skipped_blur": len(filter_result.skipped_blur),
        "frames_skipped_duplicate": len(filter_result.skipped_duplicate),
        "frames_skipped_transition": len(filter_result.skipped_transition),
        "frames_skipped_dark": len(filter_result.skipped_dark),
        "frames_sent_to_vlm": len(frames_for_vlm),
        "frames_idle": len(idle_frames),
        "vlm_calls_made": labeller_cost["total_calls"],
        "vlm_cost_inr": labeller_cost["total_cost_inr"],
        "processing_time_seconds": round(time.time() - start_time, 2),
    }

    video_info = {
        "video_file": str(video_path),
        "video_type": "egocentric",
        "camera_mount": "helmet",
        "duration": extraction.video_metadata.get("duration", 0),
        "fps": extraction.video_metadata.get("fps", 0),
        "resolution": extraction.video_metadata.get("resolution", ""),
        "vlm_backend": config.vlm_backend,
    }

    output_files = writer.write_all(
        output_dir, video_info, segments, frame_annotations,
        trajectories, quality_scores, processing_stats,
    )
    tracker.complete_stage("output")

    # Cleanup
    await labeller.close()
    vision.close()

    n_review = sum(1 for s in segments if s.needs_review)
    elapsed = time.time() - start_time

    logger.info(
        "Done! %d segments, %d need review, %.1fs elapsed",
        len(segments), n_review, elapsed,
    )
    logger.info("Output files: %s", output_files)

    return {
        "status": "complete",
        "segments": len(segments),
        "needs_review": n_review,
        "cost_inr": labeller_cost["total_cost_inr"],
        "quality_mean": quality_scores.get("aggregate", {}).get("mean", 0),
        "elapsed_s": round(elapsed, 2),
        "output_files": output_files,
    }


# ── annotate pipeline ────────────────────────────────────────────────


async def run_annotate_single(
    video_path: str, config: AnnotatorConfig, output_dir: str,
    duration: float = None,
) -> dict:
    """Production mode: local model + optional VLM fallback."""
    start_time = time.time()
    logger.info("Starting annotate: %s (model: %s)", video_path, config.model_path)

    # Stages 1-2: same as bootstrap
    extractor = FrameExtractor(config)
    extraction = extractor.extract(video_path, output_dir, duration=duration)
    logger.info("Extracted %d frames", extraction.total_frames)

    quality_filter = FrameQualityFilter(config)
    filter_result = quality_filter.filter(extraction.frame_paths)
    quality_filter.print_stats(filter_result)

    # Stage 3: Activity detection + local vision
    activity_detector = HandActivityDetector(config)
    activity_result = activity_detector.process_frames(filter_result.good_frames)
    activity_detector.close()

    frames_for_model = [(m.timestamp, m.filepath) for m in activity_result.frames_to_vlm]
    idle_frames = activity_result.frames_idle

    vision = LocalVisionPipeline(config, use_gpu=True)
    frame_annotations = vision.process_frames(frames_for_model, depth_every_n=5)

    # Stage 4: Local model labelling (with optional VLM fallback)
    labeller = TaskLabeller("annotate", config)
    labeller._vision_pipeline = vision

    task_labels = await labeller.label_video(
        frames_for_model, frame_annotations, idle_frames
    )
    labeller_cost = labeller.get_cost_summary()

    # Stages 5-8: same structure
    traj_extractor = TrajectoryExtractor()
    trajectories = traj_extractor.extract(frame_annotations)

    merger = SegmentMerger(config)
    segments = merger.merge(task_labels, trajectories, filter_result.skipped_transition)

    scorer = QualityScorer()
    quality_scores = scorer.score_video(segments, frame_annotations, trajectories)

    writer = OutputWriter(config)
    processing_stats = {
        "total_frames_extracted": extraction.total_frames,
        "frames_after_filter": len(filter_result.good_frames),
        "frames_skipped_blur": len(filter_result.skipped_blur),
        "frames_skipped_duplicate": len(filter_result.skipped_duplicate),
        "frames_skipped_transition": len(filter_result.skipped_transition),
        "frames_skipped_dark": len(filter_result.skipped_dark),
        "frames_sent_to_model": len(frames_for_model),
        "frames_idle": len(idle_frames),
        "vlm_fallback_calls": labeller_cost["total_calls"],
        "vlm_fallback_cost_inr": labeller_cost["total_cost_inr"],
        "processing_time_seconds": round(time.time() - start_time, 2),
    }

    video_info = {
        "video_file": str(video_path),
        "video_type": "egocentric",
        "camera_mount": "helmet",
        "duration": extraction.video_metadata.get("duration", 0),
        "fps": extraction.video_metadata.get("fps", 0),
        "resolution": extraction.video_metadata.get("resolution", ""),
        "vlm_backend": config.vlm_backend,
        "model_path": config.model_path,
    }

    output_files = writer.write_all(
        output_dir, video_info, segments, frame_annotations,
        trajectories, quality_scores, processing_stats,
    )

    await labeller.close()
    vision.close()

    n_review = sum(1 for s in segments if s.needs_review)
    elapsed = time.time() - start_time

    logger.info(
        "Done! %d segments, %d need review, %.1fs elapsed",
        len(segments), n_review, elapsed,
    )

    return {
        "status": "complete",
        "segments": len(segments),
        "needs_review": n_review,
        "cost_inr": labeller_cost["total_cost_inr"],
        "quality_mean": quality_scores.get("aggregate", {}).get("mean", 0),
        "elapsed_s": round(elapsed, 2),
        "output_files": output_files,
    }


# ── batch processing ─────────────────────────────────────────────────


async def handle_batch(
    input_dir: str, config: AnnotatorConfig, mode: str,
    args: argparse.Namespace,
) -> None:
    """Process all video files in a directory sequentially."""
    videos = resolve_input_paths(input_dir, batch=True)
    logger.info("Batch processing: %d videos in %s (mode=%s)", len(videos), input_dir, mode)

    results = []
    for idx, vpath in enumerate(videos):
        logger.info("\n{'='*60}")
        logger.info("Video %d/%d: %s", idx + 1, len(videos), vpath.name)

        output_dir = resolve_output_dir(vpath, args.output_dir)

        # Skip if already processed (resume support)
        if args.resume and (Path(output_dir) / "annotations.json").exists():
            logger.info("Skipping %s — already processed (use without --resume to reprocess)", vpath.name)
            results.append({"status": "skipped", "video": str(vpath)})
            continue

        try:
            if mode == "bootstrap":
                result = await run_bootstrap_single(
                    str(vpath), config, output_dir,
                    cost_estimate_only=getattr(args, "cost_estimate", False),
                    resume=args.resume,
                    duration=getattr(args, "duration", None),
                )
            else:
                result = await run_annotate_single(
                    str(vpath), config, output_dir,
                    duration=getattr(args, "duration", None),
                )
            result["video"] = str(vpath)
            results.append(result)
        except Exception as exc:
            logger.error("Failed to process %s: %s", vpath.name, exc, exc_info=True)
            results.append({"status": "error", "video": str(vpath), "error": str(exc)})

        if _shutdown_requested:
            logger.info("Batch interrupted — %d/%d videos processed", idx + 1, len(videos))
            break

    # Print batch summary
    _print_batch_summary(results)


def _print_batch_summary(results: list) -> None:
    """Print a summary table for batch processing."""
    completed = [r for r in results if r.get("status") == "complete"]
    skipped = [r for r in results if r.get("status") == "skipped"]
    errors = [r for r in results if r.get("status") == "error"]

    total_segments = sum(r.get("segments", 0) for r in completed)
    total_cost = sum(r.get("cost_inr", 0) for r in completed)
    total_review = sum(r.get("needs_review", 0) for r in completed)
    qualities = [r.get("quality_mean", 0) for r in completed if r.get("quality_mean")]
    avg_quality = sum(qualities) / len(qualities) if qualities else 0

    print("\n" + "=" * 60)
    print("          BATCH PROCESSING SUMMARY")
    print("=" * 60)
    print(f"  Videos processed:     {len(completed)}")
    print(f"  Videos skipped:       {len(skipped)}")
    print(f"  Videos failed:        {len(errors)}")
    print(f"  Total segments:       {total_segments}")
    print(f"  Total VLM cost:       INR {total_cost:.2f}")
    print(f"  Avg quality score:    {avg_quality:.3f}")
    print(f"  Segments need review: {total_review}")
    print("=" * 60)

    if errors:
        print("\n  Failed videos:")
        for r in errors:
            print(f"    {r['video']}: {r.get('error', 'unknown')}")


# ── main ──────────────────────────────────────────────────────────────


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    setup_logging(verbose=args.verbose)
    config = build_config(args)
    print(config.summary())

    if args.command == "bootstrap":
        if args.batch:
            asyncio.run(handle_batch(args.input_path, config, "bootstrap", args))
        else:
            output_dir = resolve_output_dir(Path(args.input_path), args.output_dir)
            asyncio.run(run_bootstrap_single(
                args.input_path, config, output_dir,
                cost_estimate_only=args.cost_estimate,
                resume=args.resume,
                duration=args.duration,
            ))

    elif args.command == "annotate":
        if args.batch:
            asyncio.run(handle_batch(args.input_path, config, "annotate", args))
        else:
            output_dir = resolve_output_dir(Path(args.input_path), args.output_dir)
            asyncio.run(run_annotate_single(
                args.input_path, config, output_dir,
            ))


if __name__ == "__main__":
    main()
