"""Build the combined customer reel.

The top clips are concatenated with a half-second of black between them,
in one ffmpeg pass, and the whole thing is bitrate-capped to the size
ceiling. This file is the one that gets attached to an email, so going
over the limit is a failure rather than a rounding issue — the size is
checked after writing and reported plainly if the quality floor stopped
us reaching it.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import List, Optional, Sequence

from qc.config import REEL_GAP_S
from qc.io.ffmpeg import ffmpeg_bin, select_encoder
from qc.io.sizing import plan_quality
from qc.runner import probe_frame_count

logger = logging.getLogger(__name__)


class ReelError(RuntimeError):
    pass


def build_reel(
    clips: Sequence[Path],
    out_path: Path,
    width: int,
    height: int,
    fps: float,
    max_size_mb: Optional[float] = 50.0,
    gap_s: float = REEL_GAP_S,
    encoder_preference: str = "auto",
) -> Path:
    """Concatenate *clips* with black gaps into a single MP4."""
    clips = [Path(c) for c in clips if Path(c).exists()]
    if not clips:
        raise ReelError("No clips available to build a reel from.")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_name(out_path.name + ".part")

    durations = [_duration_s(c, fps) for c in clips]
    total_s = sum(durations) + gap_s * (len(clips) - 1)
    quality = plan_quality(total_s, max_size_mb)
    encoder = select_encoder(encoder_preference)

    cmd = [ffmpeg_bin(), "-nostdin", "-hide_banner", "-loglevel", "error", "-y"]
    for clip in clips:
        cmd += ["-i", str(clip)]

    # One reusable black source; referenced once per gap in the filter.
    gap_input_index = len(clips)
    if len(clips) > 1:
        cmd += [
            "-f", "lavfi",
            "-i", f"color=c=black:s={width}x{height}:r={fps:.6f}:d={gap_s:.3f}",
        ]

    parts: List[str] = []

    # Normalise every clip: one rendered at a different size or rate would
    # otherwise make concat fail outright or silently retime the reel.
    for i in range(len(clips)):
        parts.append(
            f"[{i}:v]scale={width}:{height}:force_original_aspect_ratio=decrease,"
            f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,setsar=1,"
            f"fps={fps:.6f},format=yuv420p[v{i}];"
        )

    # A filter output can only be consumed once, so the single black source
    # is split into one stream per gap.
    n_gaps = len(clips) - 1
    gap_labels = [f"g{g}" for g in range(n_gaps)]
    if n_gaps > 0:
        outs = "".join(f"[{lbl}]" for lbl in gap_labels)
        parts.append(
            f"[{gap_input_index}:v]setsar=1,format=yuv420p,split={n_gaps}{outs};"
        )

    sequence = ""
    for i in range(len(clips)):
        sequence += f"[v{i}]"
        if i < n_gaps:
            sequence += f"[{gap_labels[i]}]"
    parts.append(f"{sequence}concat=n={len(clips) + n_gaps}:v=1:a=0[out]")

    cmd += ["-filter_complex", "".join(parts), "-map", "[out]", "-an",
            "-c:v", encoder, "-pix_fmt", "yuv420p"]

    if encoder == "h264_nvenc":
        cmd += ["-preset", "p5", "-rc", "vbr", "-cq", str(quality.crf)]
    else:
        cmd += ["-preset", "medium", "-crf", str(quality.crf)]
    if quality.max_bitrate_kbps:
        cmd += ["-maxrate", f"{quality.max_bitrate_kbps}k",
                "-bufsize", f"{quality.bufsize_kbps}k"]

    cmd += ["-movflags", "+faststart", "-f", "mp4", str(tmp)]

    logger.info("Building reel from %d clips (%.1fs total)", len(clips), total_s)
    import subprocess

    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          stdin=subprocess.DEVNULL, text=True, errors="replace")
    if proc.returncode != 0:
        tmp.unlink(missing_ok=True)
        raise ReelError(f"Reel encode failed: {proc.stderr.strip()}")

    os.replace(tmp, out_path)
    size_mb = out_path.stat().st_size / 1e6
    if max_size_mb and size_mb > max_size_mb:
        logger.warning(
            "Reel is %.1f MB, over the %.1f MB ceiling. The bitrate cap hit its "
            "quality floor — drop a clip or shorten the windows.",
            size_mb, max_size_mb,
        )
    logger.info("Wrote %s (%.1f MB)", out_path, size_mb)
    return out_path


def _duration_s(clip: Path, fps: float) -> float:
    frames = probe_frame_count(clip)
    return (frames / fps) if frames and fps else 0.0
