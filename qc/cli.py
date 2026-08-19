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
from qc.config import LAYOUT_3PANEL, REEL_CLIP_COUNT, RenderConfig
from qc.io.video_reader import FrameWindow, discover_videos, probe
from qc.logging_setup import setup as setup_logging
from qc.manifest import DEFAULT_MANIFEST_NAME, ManifestError
from qc.manifest import init as init_manifest
from qc.manifest import load as load_manifest
from qc.manifest import resolve as resolve_label
from qc.pose.cache import get_track
from qc.pose.schema import PoseTrackError
from qc.reel import ReelError, build_reel
from qc.report import analyze as analyze_mod
from qc.report import csv_writer, ranking
from qc.runner import output_is_complete, render_video

logger = logging.getLogger("qc.cli")

REPORT_NAME = "qc_report.csv"
REEL_NAME = "manudata_qc_reel.mp4"


# ── argument parsing ──────────────────────────────────────────────────


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

    backend = None
    if args.pose_backend == "wilor":
        backend = _load_wilor(args)
        if backend is None:
            return 2

    # ── analysis pass ────────────────────────────────────────────────
    stats: List[analyze_mod.VideoStats] = []
    tracks = {}
    metas = {}

    for path in videos:
        try:
            meta = probe(path)
            label = resolve_label(path, labels)
            track = get_track(meta, keypoints_dir, backend, force=args.force_inference)
            row = analyze_mod.analyze(track, meta, label.site, label.task)
        except (ManifestError, PoseTrackError, RuntimeError) as exc:
            logger.error("Skipping %s: %s", path.name, exc)
            continue
        stats.append(row)
        tracks[path] = track
        metas[path] = meta

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

    rendered: List[Path] = []
    failures = 0

    for position, row in enumerate(selected):
        path = row.source_path
        meta = metas[path]
        track = tracks[path]
        label = resolve_label(path, labels)

        window = _window_for(row, meta, args)
        out_path = renders_dir / f"{path.stem}_qc.mp4"

        if args.resume and output_is_complete(out_path, cfg, window.count, path.name):
            logger.info("Skipping %s (already complete)", out_path.name)
            rendered.append(out_path)
            continue

        try:
            result = render_video(
                meta=meta, track=track, cfg=cfg, label=label,
                out_path=out_path, window=window,
                show_progress=not args.no_progress,
                slam_panel=_slam_panel_for(meta, cfg, args),
            )
            rendered.append(result.path)
        except Exception as exc:  # keep going; one bad video must not end the batch
            failures += 1
            logger.exception("Failed to render %s: %s", path.name, exc)

    if failures:
        logger.warning("%d video(s) failed to render; see the log.", failures)

    # ── reel ─────────────────────────────────────────────────────────
    if not args.no_reel and rendered:
        try:
            _build_reel(ranked, renders_dir, out_dir, cfg, args, metas, labels)
        except (ReelError, RuntimeError) as exc:
            logger.error("Reel build failed: %s", exc)
            failures += 1

    logger.info("Done. Outputs in %s", out_dir)
    return 1 if failures else 0


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

    # The reel needs the excerpt, not the full render, so cut each one
    # separately. Size the parts so the concatenated result fits.
    per_clip_mb = (args.max_size_mb / max(1, len(top))) * 0.92
    clip_cfg = RenderConfig(
        with_slam=cfg.with_slam, clips_only=True, max_size_mb=per_clip_mb,
        encoder=cfg.encoder, pose_backend=cfg.pose_backend,
    )

    clips_dir = out_dir / "reel_clips"
    clips_dir.mkdir(parents=True, exist_ok=True)
    clip_paths: List[Path] = []

    for row in top:
        path = row.source_path
        meta = metas[path]
        window = FrameWindow(row.clip.start_frame, row.clip.end_frame)
        clip_path = clips_dir / f"{path.stem}_clip.mp4"

        if args.resume and output_is_complete(clip_path, clip_cfg, window.count, path.name):
            logger.info("Reusing reel clip %s", clip_path.name)
            clip_paths.append(clip_path)
            continue

        from qc.pose.cache import load_cached

        track = load_cached(Path(args.keypoints_dir) if args.keypoints_dir
                            else out_dir / "keypoints", meta)
        if track is None:
            logger.warning("No keypoints for %s; excluding it from the reel.", path.name)
            continue

        result = render_video(
            meta=meta, track=track, cfg=clip_cfg,
            label=resolve_label(path, labels), out_path=clip_path,
            window=window, show_progress=not args.no_progress,
        )
        clip_paths.append(result.path)

    if not clip_paths:
        logger.warning("No reel clips were produced.")
        return

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


def _slam_panel_for(meta, cfg: RenderConfig, args):
    """Build the trajectory panel, or a labelled stand-in.

    A missing trajectory never fails the render — the panel says so on
    screen. Silently omitting the panel would be worse: the render would
    look complete while quietly dropping the differentiator it was asked
    to show.
    """
    if not cfg.with_slam:
        return None

    from qc.render.slam_panel import (
        MissingTrajectoryPanel,
        TrajectoryPanel,
        TrajectoryUnavailable,
        load_trajectory,
    )

    width, height = LAYOUT_3PANEL[2], cfg.canvas_h

    if args.slam_dir is None:
        return MissingTrajectoryPanel(width, height, "--slam-dir not supplied")

    stem = Path(meta.path).stem
    for suffix in (".npy", ".npz", ".txt", ".tum"):
        candidate = Path(args.slam_dir) / f"{stem}{suffix}"
        if candidate.exists():
            try:
                positions = load_trajectory(candidate, meta.n_frames)
            except (TrajectoryUnavailable, ValueError) as exc:
                logger.warning("Trajectory %s unusable: %s", candidate.name, exc)
                return MissingTrajectoryPanel(width, height, "trajectory unreadable")
            return TrajectoryPanel(width, height, positions, meta.fps)

    logger.warning("No trajectory file for %s in %s", stem, args.slam_dir)
    return MissingTrajectoryPanel(width, height, "no trajectory for this video")


def _load_wilor(args):
    from qc.pose.wilor_backend import WiLoRBackend, WiLoRUnavailable

    try:
        return WiLoRBackend(
            weights_dir=args.wilor_weights,
            device=args.device,
            batch_size=args.batch_size,
        )
    except WiLoRUnavailable as exc:
        logger.error("%s", exc)
        logger.error(
            "If keypoints were computed on another machine, copy the .npz files "
            "into the keypoints directory and re-run with --pose-backend cached."
        )
        return None


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "init-manifest":
        return cmd_init_manifest(args)
    if args.command == "run":
        return cmd_run(args)
    return 2


if __name__ == "__main__":
    sys.exit(main())
