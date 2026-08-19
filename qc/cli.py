"""``manudata-qc-render`` — batch hand-pose QC renderer.

Two subcommands:

``init-manifest``
    Scan the video folders and write a manifest template with the site
    column pre-filled from the folder names.

``run``
    Analyse every video, print the ranked summary, write
    ``qc_report.csv``, then render. ``--analyze-only`` stops after the
    report; ``--resume`` skips completed outputs.

The analysis pass always runs before any rendering, because the ranking
is what decides which videos are worth the GPU time.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional, Sequence

from qc import __version__
from qc.config import REEL_CLIP_COUNT, RenderConfig
from qc.io.video_reader import FrameWindow, discover_videos, probe
from qc.logging_setup import setup as setup_logging
from qc.manifest import DEFAULT_MANIFEST_NAME, ManifestError
from qc.manifest import init as init_manifest
from qc.manifest import load as load_manifest
from qc.manifest import resolve as resolve_label
from qc.parallel import AnalyzeJob, BackendSpec, RenderJob
from qc.parallel import execute as parallel_execute
from qc.parallel import run_analyze, run_render, shard
from qc.reel import ReelError, build_reel
from qc.report import analyze as analyze_mod
from qc.report import csv_writer, ranking
from qc.runner import output_is_complete

logger = logging.getLogger("qc.cli")

REPORT_NAME = "qc_report.csv"
REEL_NAME = "manudata_qc_reel.mp4"

# Below this, a 1080p render is too soft to judge keypoint accuracy from.
QUALITY_ADVISORY_KBPS = 2500


# ── argument parsing ──────────────────────────────────────────────────


def _parse_shard(text: str) -> tuple:
    """Parse ``I/N`` into a 1-based (index, total) pair."""
    try:
        index_s, total_s = text.split("/", 1)
        index, total = int(index_s), int(total_s)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"--shard expects I/N, e.g. 2/3 (got {text!r})"
        ) from None
    if total < 1 or not 1 <= index <= total:
        raise argparse.ArgumentTypeError(
            f"--shard {text} is out of range; need 1 <= I <= N and N >= 1"
        )
    return (index, total)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="manudata-qc-render",
        description="Render hand-pose QC videos and a batch quality report.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    init = sub.add_parser(
        "init-manifest",
        help="Write a manifest template with site pre-filled from folder names.",
    )
    init.add_argument("inputs", nargs="+", type=Path,
                      help="Video files or directories to scan.")
    init.add_argument("--out", type=Path, required=True,
                      help=f"Manifest path to write (e.g. E:/manudata_qc/{DEFAULT_MANIFEST_NAME}).")

    run = sub.add_parser("run", help="Analyse and render.")
    run.add_argument("inputs", nargs="+", type=Path,
                     help="Video files or directories to process.")
    run.add_argument("--out", type=Path, required=True,
                     help="Output directory; created if missing.")
    run.add_argument("--manifest", type=Path, default=None,
                     help=f"Manifest CSV (default: <out>/{DEFAULT_MANIFEST_NAME}).")

    run.add_argument("--analyze-only", action="store_true",
                     help="Write the report and stop; render nothing.")
    run.add_argument("--clips-only", action="store_true",
                     help="Render only the recommended window of each video.")
    run.add_argument("--top", type=int, default=None, metavar="N",
                     help="Render only the N best-ranked videos.")
    run.add_argument("--resume", action="store_true",
                     help="Skip videos whose output already exists and verifies.")
    run.add_argument("--max-size-mb", type=float, default=50.0,
                     help="Target size per output MP4 (default: 50).")
    run.add_argument("--with-slam", action="store_true",
                     help="Add the MASt3R-SLAM camera-trajectory panel.")
    run.add_argument("--slam-dir", type=Path, default=None,
                     help="Directory of trajectory files named <video-stem>.npy/"
                          ".npz/.txt, produced by the MASt3R-SLAM stage. This tool "
                          "renders trajectories; it does not compute them.")
    run.add_argument("--no-reel", action="store_true",
                     help="Skip building the combined customer reel.")
    run.add_argument("--reel-clips", type=int, default=REEL_CLIP_COUNT,
                     help=f"Clips in the reel (default: {REEL_CLIP_COUNT}).")

    run.add_argument("--pose-backend", choices=("wilor", "cached"), default="wilor",
                     help="Keypoint source. 'cached' requires existing .npz files.")
    run.add_argument("--wilor-weights", type=Path, default=Path("pretrained_models"),
                     help="Directory holding the WiLoR checkpoint and detector.")
    run.add_argument("--keypoints-dir", type=Path, default=None,
                     help="Keypoint cache directory (default: <out>/keypoints).")
    run.add_argument("--device", default=None, help="Torch device, e.g. cuda:0.")
    run.add_argument("--batch-size", type=int, default=8,
                     help="Frames per inference batch (default: 8).")
    run.add_argument("--force-inference", action="store_true",
                     help="Recompute keypoints even when a cache exists.")

    run.add_argument("--encoder", choices=("auto", "nvenc", "x264"), default="auto",
                     help="H.264 encoder (default: auto-detect NVENC).")
    run.add_argument("--limit", type=int, default=None,
                     help="Process at most N videos; useful for a trial run.")
    run.add_argument("--workers", type=int, default=1, metavar="N",
                     help="Process N videos concurrently, one process each "
                          "(default: 1). Each WiLoR worker needs roughly 6 GB of "
                          "VRAM, so keep N x 6 GB under the card's memory.")
    run.add_argument("--shard", type=_parse_shard, default=None, metavar="I/N",
                     help="Take only shard I of N, e.g. --shard 2/3. Splits the "
                          "video list round-robin so several pods can cover one "
                          "batch without overlapping.")
    run.add_argument("--smoke-test", type=float, default=None, metavar="SECONDS",
                     help="Render only the first N seconds of each video.")
    run.add_argument("--no-progress", action="store_true", help="Disable progress bars.")
    run.add_argument("-v", "--verbose", action="store_true", help="Debug logging.")
    return parser


# ── commands ──────────────────────────────────────────────────────────


def cmd_init_manifest(args: argparse.Namespace) -> int:
    setup_logging(None)
    videos = discover_videos(args.inputs)
    if not videos:
        logger.error("No video files found under: %s",
                     ", ".join(str(p) for p in args.inputs))
        return 2
    init_manifest(videos, args.out)

    labels = load_manifest(args.out)
    blank = [name for name, label in labels.items() if not label.task]

    print(f"\nWrote {args.out} with {len(videos)} rows.")
    if blank:
        print(f"{len(blank)} row(s) need a 'task' before rendering, starting with:")
        for name in sorted(blank)[:5]:
            print(f"  {name}")
        print("\nFill those in, then run:")
    else:
        print("Every row is labelled from the filename convention. Run:")
    print(f"  manudata-qc-render run <dirs...> --out {args.out.parent}")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    out_dir = Path(args.out)
    renders_dir = out_dir / "renders"
    keypoints_dir = Path(args.keypoints_dir) if args.keypoints_dir else out_dir / "keypoints"
    for directory in (out_dir, renders_dir, keypoints_dir, out_dir / "logs"):
        directory.mkdir(parents=True, exist_ok=True)

    setup_logging(out_dir / "logs", verbose=args.verbose)

    manifest_path = Path(args.manifest) if args.manifest else out_dir / DEFAULT_MANIFEST_NAME
    videos = discover_videos(args.inputs)
    if not videos:
        logger.error("No video files found under: %s",
                     ", ".join(str(p) for p in args.inputs))
        return 2
    if args.limit:
        videos = videos[: args.limit]

    if args.shard:
        index, total = args.shard
        videos = shard(videos, index, total)
        logger.info("Shard %d of %d: %d video(s) on this pod", index, total, len(videos))
        for path in videos:
            logger.info("  %s", path.name)
        if not videos:
            logger.error("This shard is empty — fewer videos than shards.")
            return 2

    logger.info("Found %d video(s)", len(videos))

    try:
        labels = load_manifest(manifest_path)
    except ManifestError as exc:
        logger.error("%s", exc)
        return 2

    cfg = RenderConfig(
        with_slam=args.with_slam,
        clips_only=args.clips_only,
        max_size_mb=args.max_size_mb,
        encoder=args.encoder,
        pose_backend=args.pose_backend,
    )

    workers = max(1, args.workers)

    backend_spec = None
    if args.pose_backend == "wilor":
        backend_spec = BackendSpec(
            kind="wilor", weights_dir=Path(args.wilor_weights),
            device=args.device, batch_size=args.batch_size,
        )
        # Validate the weight files without constructing the model. Building
        # it here would initialise CUDA in the parent, which a spawned pool
        # tolerates but which turns any later fork into an obscure failure —
        # and it would load ~2 GB we do not need in this process.
        if not _wilor_weights_ok(args):
            return 2

    try:
        video_labels = {path: resolve_label(path, labels) for path in videos}
    except ManifestError as exc:
        logger.error("%s", exc)
        return 2

    # ── analysis pass ────────────────────────────────────────────────
    analyze_results = parallel_execute(
        run_analyze,
        [AnalyzeJob(video=path, keypoints_dir=keypoints_dir,
                    label=video_labels[path], backend=backend_spec,
                    force_inference=args.force_inference)
         for path in videos],
        workers=workers,
        label="analyse",
    )

    stats: List[analyze_mod.VideoStats] = [
        r.value for r in analyze_results if r.ok and r.value is not None
    ]
    failures = sum(1 for r in analyze_results if not r.ok)

    if not stats:
        logger.error("No videos could be analysed. Nothing to report or render.")
        return 1

    ranked = analyze_mod.rank(stats)
    report_path = csv_writer.write(ranked, out_dir / REPORT_NAME)
    logger.info("Wrote %s", report_path)

    print()
    print(ranking.format_table(ranked))
    print()

    if args.analyze_only:
        logger.info("--analyze-only: stopping before render.")
        return 0

    # ── render pass ──────────────────────────────────────────────────
    selected = ranked[: args.top] if args.top else ranked
    logger.info("Rendering %d of %d video(s)", len(selected), len(ranked))

    metas = {r.source_path: probe(r.source_path) for r in selected}

    rendered: List[Path] = []
    render_jobs: List[RenderJob] = []

    for row in selected:
        path = row.source_path
        window = _window_for(row, metas[path], args)
        out_path = renders_dir / f"{path.stem}_qc.mp4"

        if args.resume and output_is_complete(out_path, cfg, window.count, path.name):
            logger.info("Skipping %s (already complete)", out_path.name)
            rendered.append(out_path)
            continue

        _warn_if_quality_starved(row, window, metas[path], cfg)
        render_jobs.append(RenderJob(
            video=path, keypoints_dir=keypoints_dir, out_path=out_path,
            label=video_labels[path], cfg=cfg,
            window=(window.start, window.end),
            slam_dir=Path(args.slam_dir) if args.slam_dir else None,
        ))

    render_results = parallel_execute(
        run_render, render_jobs, workers=workers, label="render",
    )
    for result in render_results:
        if result.ok and result.value is not None:
            rendered.append(result.value.path)
        else:
            failures += 1

    if failures:
        logger.warning("%d video(s) failed; see the log.", failures)

    # ── reel ─────────────────────────────────────────────────────────
    if not args.no_reel and rendered:
        try:
            _build_reel(ranked, renders_dir, out_dir, cfg, args, metas, labels)
        except (ReelError, RuntimeError) as exc:
            logger.error("Reel build failed: %s", exc)
            failures += 1

    logger.info("Done. Outputs in %s", out_dir)
    return 1 if failures else 0


def _wilor_weights_ok(args) -> bool:
    from qc.pose.wilor_backend import WiLoRPaths, WiLoRUnavailable

    try:
        WiLoRPaths.under(Path(args.wilor_weights)).check()
        return True
    except WiLoRUnavailable as exc:
        logger.error("%s", exc)
        logger.error(
            "If keypoints were computed elsewhere, copy the .npz files into the "
            "keypoints directory and re-run with --pose-backend cached."
        )
        return False


def _warn_if_quality_starved(row, window, meta, cfg: RenderConfig) -> None:
    """Flag a size ceiling that will visibly wreck a long render.

    A 50 MB cap is sensible for a 25-second clip and meaningless for a
    25-minute one — it works out near 260 kbps, which no evaluator would
    accept for judging keypoint accuracy. Better to say so before the
    render than to hand over a blurry file.
    """
    if not cfg.max_size_mb or meta.fps <= 0:
        return
    duration_s = window.count / meta.fps
    if duration_s <= 0:
        return
    kbps = cfg.max_size_mb * 8 * 1024 * 1024 * 0.94 / duration_s / 1000
    if kbps < QUALITY_ADVISORY_KBPS:
        logger.warning(
            "%s is %.1f min; a %.0f MB ceiling allows only ~%.0f kbps, which will "
            "look bad at 1080p. Use --max-size-mb 0 for no cap, or raise it to "
            "~%.0f MB.",
            row.filename, duration_s / 60, cfg.max_size_mb, kbps,
            QUALITY_ADVISORY_KBPS * duration_s * 1000 / (8 * 1024 * 1024 * 0.94),
        )


def _window_for(row, meta, args) -> FrameWindow:
    """Which frames to render for one video."""
    if args.smoke_test:
        count = min(meta.n_frames, int(round(args.smoke_test * meta.fps)))
        return FrameWindow(0, max(1, count))
    if args.clips_only and row.clip is not None:
        return FrameWindow(row.clip.start_frame, row.clip.end_frame)
    return FrameWindow.full(meta)


def _build_reel(ranked, renders_dir, out_dir, cfg, args, metas, labels) -> None:
    """Render the top clips as standalone excerpts, then concatenate."""
    top = [r for r in ranked if r.clip is not None][: args.reel_clips]
    if not top:
        logger.info("No recommended windows available; skipping the reel.")
        return

    keypoints_dir = (
        Path(args.keypoints_dir) if args.keypoints_dir else out_dir / "keypoints"
    )
    # A reel clip may come from a video the render pass skipped (--top),
    # so probe anything not already known rather than assuming metas has it.
    for row in top:
        if row.source_path not in metas:
            metas[row.source_path] = probe(row.source_path)

    # The reel needs the excerpt, not the full render, so cut each one
    # separately. Size the parts so the concatenated result fits.
    per_clip_mb = (
        (args.max_size_mb / max(1, len(top))) * 0.92 if args.max_size_mb else None
    )
    clip_cfg = RenderConfig(
        with_slam=cfg.with_slam, clips_only=True, max_size_mb=per_clip_mb,
        encoder=cfg.encoder, pose_backend=cfg.pose_backend,
    )

    clips_dir = out_dir / "reel_clips"
    clips_dir.mkdir(parents=True, exist_ok=True)

    clip_paths: List[Path] = []
    jobs: List[RenderJob] = []

    for row in top:
        path = row.source_path
        window = FrameWindow(row.clip.start_frame, row.clip.end_frame)
        clip_path = clips_dir / f"{path.stem}_clip.mp4"

        if args.resume and output_is_complete(clip_path, clip_cfg, window.count, path.name):
            logger.info("Reusing reel clip %s", clip_path.name)
            clip_paths.append(clip_path)
            continue

        jobs.append(RenderJob(
            video=path, keypoints_dir=keypoints_dir, out_path=clip_path,
            label=resolve_label(path, labels), cfg=clip_cfg,
            window=(window.start, window.end),
            slam_dir=Path(args.slam_dir) if args.slam_dir else None,
        ))

    for result in parallel_execute(
        run_render, jobs, workers=max(1, args.workers), label="reel",
    ):
        if result.ok and result.value is not None:
            clip_paths.append(result.value.path)
        else:
            logger.warning("Excluding %s from the reel.", result.video.name)

    if not clip_paths:
        logger.warning("No reel clips were produced.")
        return

    # Workers finish out of order, and resumed clips were collected first,
    # so restore the ranking — the reel must open with the best clip.
    rank_of = {
        (clips_dir / f"{row.source_path.stem}_clip.mp4"): i
        for i, row in enumerate(top)
    }
    clip_paths.sort(key=lambda p: rank_of.get(p, len(top)))

    reference = metas[top[0].source_path]
    build_reel(
        clips=clip_paths,
        out_path=out_dir / REEL_NAME,
        width=cfg.canvas_w,
        height=cfg.canvas_h,
        fps=reference.fps,
        max_size_mb=args.max_size_mb,
        encoder_preference=cfg.encoder,
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "init-manifest":
        return cmd_init_manifest(args)
    if args.command == "run":
        return cmd_run(args)
    return 2


if __name__ == "__main__":
    sys.exit(main())
