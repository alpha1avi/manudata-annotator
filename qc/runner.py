"""Render one video end to end, resumably.

The render loop is deliberately boring: decode a frame, look up its
absolute index, compose, encode. There is no buffering, reordering or
frame-dropping anywhere in it, because every one of those is a way for
the two panels to drift apart.

The loop also verifies as it goes. It refuses an index outside the pose
track, and it checks the decoded frame count against what it expected
once the stream ends. A source file that decodes to a different length
than the keypoints were computed from is a hard failure, never a
silently shorter video.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from tqdm import tqdm

from qc.config import RenderConfig, VideoMeta
from qc.io.ffmpeg import ffmpeg_bin, ffprobe_bin, run
from qc.io.sizing import plan_quality
from qc.io.video_reader import FrameWindow, iter_frames
from qc.io.video_writer import VideoWriter, read_sidecar, write_sidecar
from qc.manifest import Label
from qc.pose.gapfill import build_render_plan
from qc.pose.schema import PoseTrack, PoseTrackError
from qc.progress import PeriodicProgress
from qc.render.compositor import FrameComposer

logger = logging.getLogger(__name__)


@dataclass
class RenderResult:
    path: Path
    frames: int
    seconds: float
    size_mb: float
    skipped: bool = False


def render_video(
    meta: VideoMeta,
    track: PoseTrack,
    cfg: RenderConfig,
    label: Label,
    out_path: Path,
    window: Optional[FrameWindow] = None,
    show_progress: bool = True,
    position: int = 0,
    slam_panel=None,
) -> RenderResult:
    """Render *meta* to *out_path*. Returns what was produced."""
    out_path = Path(out_path)
    window = window or FrameWindow.full(meta)

    plan = build_render_plan(track, meta.fps, cfg)
    composer = FrameComposer(
        meta=meta, track=track, plan=plan, cfg=cfg,
        site=label.site, task=label.task, slam_panel=slam_panel,
    )

    expected = window.count
    if expected <= 0:
        raise ValueError(f"Empty render window for {meta.path.name}")

    quality = plan_quality(
        duration_s=expected / meta.fps,
        max_size_mb=cfg.max_size_mb,
    )

    started = time.monotonic()
    written = 0
    last_index = -1

    bar = tqdm(
        total=expected,
        desc=meta.path.name[:28],
        unit="f",
        leave=False,
        disable=not show_progress,
        position=position,
        dynamic_ncols=True,
    )
    # A render inside a worker has no bar, and a full-length render takes
    # tens of minutes. Without this it is completely silent for that whole
    # time and indistinguishable from a hang.
    progress = PeriodicProgress(
        meta.path.name, expected, unit="frames", enabled=not show_progress,
    )

    try:
        with VideoWriter(
            out_path, cfg.canvas_w, cfg.canvas_h, meta.fps, quality,
            encoder_preference=cfg.encoder,
        ) as writer:
            for index, frame in iter_frames(meta, window):
                if index >= window.end:
                    break
                canvas = composer.compose(index, frame)
                writer.write(canvas)
                written += 1
                last_index = index
                bar.update(1)
                progress.update(written)
    finally:
        bar.close()
        progress.close(written, what="render complete")

    if written != expected:
        raise PoseTrackError(
            f"{meta.path.name}: expected {expected} frames in the render window "
            f"but the decoder produced {written} (last index {last_index}). The "
            "video and the keypoint track disagree on length — refusing to "
            "publish a render whose panels may be offset."
        )

    elapsed = time.monotonic() - started
    size_mb = out_path.stat().st_size / 1e6

    write_sidecar(out_path, {
        "source": meta.path.name,
        "config_fingerprint": cfg.fingerprint(),
        "frames": written,
        "window_start": window.start,
        "window_end": window.end,
        "fps": meta.fps,
        "canvas": [cfg.canvas_w, cfg.canvas_h],
        "pose_model": track.meta.get("model", "unknown"),
        "pose_model_version": track.meta.get("model_version", "unknown"),
        "size_mb": round(size_mb, 3),
        # Exact size, for the resume check. The container's frame count is
        # written up front by +faststart, so a truncated file still claims
        # its original length — byte size is what actually catches a
        # half-copied or disk-full output.
        "size_bytes": out_path.stat().st_size,
    })

    if cfg.max_size_mb and size_mb > cfg.max_size_mb * 1.05:
        logger.warning(
            "%s came out at %.1f MB against a %.1f MB target. The bitrate cap "
            "hit its quality floor; shorten the clip or raise --max-size-mb.",
            out_path.name, size_mb, cfg.max_size_mb,
        )

    logger.info(
        "%s: %d frames in %.1fs (%.1f fps), %.1f MB",
        out_path.name, written, elapsed, written / elapsed if elapsed else 0.0, size_mb,
    )
    return RenderResult(out_path, written, elapsed, size_mb)


# ── resume support ────────────────────────────────────────────────────


def output_is_complete(
    out_path: Path,
    cfg: RenderConfig,
    expected_frames: int,
    source_name: str,
) -> bool:
    """Whether ``--resume`` may skip this video.

    Several things must agree, because each has been an actual way to
    resurrect a bad file: the sidecar exists (so the write finished), the
    config fingerprint matches (so it was not rendered with different
    settings), the source matches, the byte size matches, the frame count
    matches, and the *end* of the file still decodes.

    The byte-size and tail-decode checks are not redundant with the frame
    count. ``+faststart`` writes the index at the front of the file, so a
    truncated MP4 keeps advertising its original frame count and sails
    past a header-only check — which is precisely the half-written file
    ``--resume`` is supposed to catch.
    """
    out_path = Path(out_path)
    if not out_path.exists() or out_path.stat().st_size == 0:
        return False

    sidecar = read_sidecar(out_path)
    if sidecar is None:
        logger.info("%s has no completion sidecar; re-rendering.", out_path.name)
        return False

    if sidecar.get("config_fingerprint") != cfg.fingerprint():
        logger.info(
            "%s was rendered with different settings; re-rendering.", out_path.name
        )
        return False

    if sidecar.get("source") != source_name:
        logger.info("%s was rendered from a different source; re-rendering.",
                    out_path.name)
        return False

    if int(sidecar.get("frames") or 0) != expected_frames:
        logger.info(
            "%s holds %s frames, expected %d; re-rendering.",
            out_path.name, sidecar.get("frames"), expected_frames,
        )
        return False

    recorded_size = sidecar.get("size_bytes")
    actual_size = out_path.stat().st_size
    if recorded_size is not None and int(recorded_size) != actual_size:
        logger.info(
            "%s is %d bytes, expected %d — truncated or partially copied; "
            "re-rendering.", out_path.name, actual_size, int(recorded_size),
        )
        return False

    actual = probe_frame_count(out_path)
    if actual is None:
        logger.info("%s is not decodable; re-rendering.", out_path.name)
        return False
    if actual != expected_frames:
        logger.info(
            "%s reports %d frames, expected %d; re-rendering.",
            out_path.name, actual, expected_frames,
        )
        return False

    if not tail_decodes(out_path):
        logger.info("%s does not decode to the end; re-rendering.", out_path.name)
        return False

    return True


def tail_decodes(path: Path, seconds: float = 0.5) -> bool:
    """Whether the last *seconds* of a file actually decode.

    Cheap where a full ``-count_frames`` pass is not: it catches a file
    whose header promises more than its payload delivers, without paying
    to decode the whole render on every resumed run.
    """
    proc = run([
        ffmpeg_bin(), "-nostdin", "-hide_banner", "-v", "error",
        "-sseof", f"-{seconds}", "-i", str(path),
        "-f", "null", "-",
    ], timeout=120)
    return proc.returncode == 0 and not proc.stderr.strip()


def probe_frame_count(path: Path) -> Optional[int]:
    """Frame count of a rendered output, or None if it will not decode."""
    proc = run([
        ffprobe_bin(), "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=nb_frames",
        "-of", "default=nokey=1:noprint_wrappers=1", str(path),
    ])
    if proc.returncode != 0:
        return None
    try:
        return int(proc.stdout.strip())
    except ValueError:
        return None
