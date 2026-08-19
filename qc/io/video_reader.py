"""Streaming frame decode over an ffmpeg pipe, indexed by absolute frame.

Every frame this yields carries its absolute source-frame index. That
index is the single clock the whole renderer runs on — pose lookup, both
panels and the header all key off it — which is what makes the panels
frame-exact by construction rather than by careful bookkeeping.

Frame windows are selected with ffmpeg's ``select`` filter on the frame
*number*, never with a timestamp seek, because a timestamp seek can land
a frame or two off and that offset would silently desynchronise the
skeleton from the video.
"""

from __future__ import annotations

import json
import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional, Tuple

import numpy as np

from qc.config import VideoMeta
from qc.io.ffmpeg import ffmpeg_bin, ffprobe_bin, run

logger = logging.getLogger(__name__)

VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".avi", ".m4v", ".mpg", ".mpeg", ".webm"}


class VideoReadError(RuntimeError):
    pass


def _parse_rate(text: str) -> float:
    """Parse ffprobe's ``num/den`` rate strings."""
    text = (text or "").strip()
    if not text or text == "0/0":
        return 0.0
    if "/" in text:
        num, _, den = text.partition("/")
        try:
            n, d = float(num), float(den)
        except ValueError:
            return 0.0
        return n / d if d else 0.0
    try:
        return float(text)
    except ValueError:
        return 0.0


def probe(path: Path) -> VideoMeta:
    """Read width/height/fps and a best-effort frame count.

    ``n_frames`` here is an estimate used for progress bars and planning.
    The authoritative count is whatever the decoder actually yields; that
    is what gets recorded on the pose track and asserted at render time.
    """
    path = Path(path)
    if not path.exists():
        raise VideoReadError(f"No such video: {path}")

    proc = run([
        ffprobe_bin(), "-v", "error",
        "-select_streams", "v:0",
        "-show_entries",
        "stream=width,height,avg_frame_rate,r_frame_rate,nb_frames:format=duration",
        "-of", "json", str(path),
    ])
    if proc.returncode != 0:
        raise VideoReadError(f"ffprobe failed on {path.name}: {proc.stderr.strip()}")

    try:
        info = json.loads(proc.stdout)
    except ValueError as exc:
        raise VideoReadError(f"Could not parse ffprobe output for {path.name}") from exc

    streams = info.get("streams") or []
    if not streams:
        raise VideoReadError(f"{path.name} has no video stream")
    stream = streams[0]

    width = int(stream.get("width") or 0)
    height = int(stream.get("height") or 0)
    if width <= 0 or height <= 0:
        raise VideoReadError(f"{path.name} reports a {width}x{height} video stream")

    fps = _parse_rate(stream.get("avg_frame_rate", "")) or _parse_rate(
        stream.get("r_frame_rate", "")
    )
    if fps <= 0:
        raise VideoReadError(f"Could not determine frame rate for {path.name}")

    n_frames = int(stream.get("nb_frames") or 0)
    exact = n_frames > 0
    if not exact:
        duration = float((info.get("format") or {}).get("duration") or 0.0)
        n_frames = int(round(duration * fps))
        logger.debug(
            "%s has no nb_frames; estimating %d from duration", path.name, n_frames
        )

    return VideoMeta(
        path=path, width=width, height=height, fps=fps,
        n_frames=n_frames, n_frames_exact=exact,
    )


@dataclass
class FrameWindow:
    """A half-open ``[start, end)`` range of absolute frame indices."""

    start: int
    end: int

    @property
    def count(self) -> int:
        return max(0, self.end - self.start)

    @classmethod
    def full(cls, meta: VideoMeta) -> "FrameWindow":
        return cls(0, meta.n_frames)

    @classmethod
    def from_seconds(cls, start_s: float, end_s: float, fps: float) -> "FrameWindow":
        return cls(int(round(start_s * fps)), int(round(end_s * fps)))


def iter_frames(
    meta: VideoMeta,
    window: Optional[FrameWindow] = None,
) -> Iterator[Tuple[int, np.ndarray]]:
    """Yield ``(absolute_frame_index, bgr_frame)`` for *meta*.

    Raises if the decoder dies mid-stream, so a truncated source file
    fails the render rather than producing a short, silently wrong video.
    """
    cmd = [ffmpeg_bin(), "-nostdin", "-hide_banner", "-loglevel", "error",
           "-i", str(meta.path)]

    start = 0
    if window is not None and (window.start > 0 or window.end < meta.n_frames):
        start = window.start
        # Frame-number selection, not a timestamp seek: exact by construction.
        cmd += ["-vf", f"select=between(n\\,{window.start}\\,{window.end - 1})",
                "-fps_mode", "passthrough"]

    cmd += ["-f", "rawvideo", "-pix_fmt", "bgr24", "-"]

    frame_bytes = meta.width * meta.height * 3
    buf = bytearray(frame_bytes)
    view = memoryview(buf)

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        stdin=subprocess.DEVNULL,
        bufsize=0,
    )
    assert proc.stdout is not None

    index = start
    try:
        while True:
            filled = 0
            while filled < frame_bytes:
                chunk = proc.stdout.readinto(view[filled:])
                if not chunk:
                    break
                filled += chunk

            if filled == 0:
                break
            if filled < frame_bytes:
                raise VideoReadError(
                    f"{meta.path.name} ended mid-frame at index {index} "
                    f"({filled} of {frame_bytes} bytes). The file is likely truncated."
                )

            frame = np.frombuffer(bytes(buf), dtype=np.uint8).reshape(
                meta.height, meta.width, 3
            )
            yield index, frame
            index += 1
    finally:
        if proc.stdout is not None:
            proc.stdout.close()
        stderr = proc.stderr.read().decode("utf-8", "replace") if proc.stderr else ""
        if proc.stderr is not None:
            proc.stderr.close()
        code = proc.wait()
        if code not in (0, None) and index == start:
            raise VideoReadError(
                f"ffmpeg failed to decode {meta.path.name}: {stderr.strip()}"
            )


def count_frames(meta: VideoMeta) -> int:
    """Exact frame count via a full decode. Slow; used only when asked."""
    proc = run([
        ffprobe_bin(), "-v", "error", "-count_frames",
        "-select_streams", "v:0",
        "-show_entries", "stream=nb_read_frames",
        "-of", "default=nokey=1:noprint_wrappers=1",
        str(meta.path),
    ])
    if proc.returncode != 0:
        raise VideoReadError(f"Frame count failed for {meta.path.name}: {proc.stderr}")
    try:
        return int(proc.stdout.strip())
    except ValueError as exc:
        raise VideoReadError(
            f"Unexpected frame-count output for {meta.path.name}: {proc.stdout!r}"
        ) from exc


def discover_videos(roots) -> list:
    """Collect video files under one or more directories, sorted stably."""
    found: list = []
    for root in roots:
        root = Path(root)
        if root.is_file():
            found.append(root)
            continue
        if not root.is_dir():
            raise VideoReadError(f"Not a file or directory: {root}")
        for path in sorted(root.rglob("*")):
            if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES:
                found.append(path)
    # Stable, case-insensitive ordering so runs are reproducible across OSes.
    return sorted(dict.fromkeys(found), key=lambda p: str(p).lower())
