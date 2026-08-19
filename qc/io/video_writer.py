"""Streaming H.264 encode over an ffmpeg pipe, written atomically.

Frames go straight from numpy into ffmpeg's stdin — no PNG dump to disk,
which at 1080p60 would be tens of gigabytes per video and would make the
encoder wait on the filesystem.

The output is built under a ``.part`` name and renamed into place only
after ffmpeg exits cleanly. A Vast.ai instance that dies mid-render
therefore leaves either nothing or a complete file, never a plausible-
looking truncated MP4 that ``--resume`` would skip over.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
from pathlib import Path
from types import TracebackType
from typing import Optional, Type

import numpy as np

from qc.io.ffmpeg import ffmpeg_bin, select_encoder
from qc.io.sizing import EncodeQuality

logger = logging.getLogger(__name__)

SIDECAR_SUFFIX = ".done.json"


class VideoWriteError(RuntimeError):
    pass


class VideoWriter:
    """Context manager that pipes BGR frames into an H.264 MP4.

    Use as::

        with VideoWriter(path, 1920, 1080, 60.0, quality) as w:
            w.write(frame)
    """

    def __init__(
        self,
        path: Path,
        width: int,
        height: int,
        fps: float,
        quality: EncodeQuality,
        encoder_preference: str = "auto",
    ) -> None:
        if width % 2 or height % 2:
            raise VideoWriteError(
                f"H.264 needs even dimensions; got {width}x{height}."
            )
        self.path = Path(path)
        self.width = width
        self.height = height
        self.fps = fps
        self.quality = quality
        self.encoder = select_encoder(encoder_preference)
        self.frames_written = 0
        self._tmp = self.path.with_name(self.path.name + ".part")
        self._proc: Optional[subprocess.Popen] = None
        self._closed = False

    # ── command construction ──────────────────────────────────────────

    def _build_cmd(self) -> list:
        cmd = [
            ffmpeg_bin(), "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-s", f"{self.width}x{self.height}",
            "-r", f"{self.fps:.6f}",
            "-i", "-",
            "-an",
            "-c:v", self.encoder,
            "-pix_fmt", "yuv420p",
        ]

        q = self.quality
        if self.encoder == "h264_nvenc":
            cmd += ["-preset", "p5", "-rc", "vbr", "-cq", str(q.crf)]
        else:
            cmd += ["-preset", "medium", "-crf", str(q.crf)]

        if q.max_bitrate_kbps:
            cmd += [
                "-maxrate", f"{q.max_bitrate_kbps}k",
                "-bufsize", f"{q.bufsize_kbps}k",
            ]

        # faststart puts the index up front so the customer's mail client
        # can start playing before the whole file has downloaded.
        #
        # -f mp4 is required, not optional: we encode to a ".part" name for
        # atomicity, and ffmpeg picks its muxer from the file extension.
        cmd += ["-movflags", "+faststart", "-f", "mp4", str(self._tmp)]
        return cmd

    # ── lifecycle ─────────────────────────────────────────────────────

    def __enter__(self) -> "VideoWriter":
        self._tmp.parent.mkdir(parents=True, exist_ok=True)
        if self._tmp.exists():
            self._tmp.unlink()

        cmd = self._build_cmd()
        logger.debug("encoder command: %s", " ".join(cmd))
        self._proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        logger.info(
            "Encoding %s with %s at %.3f fps", self.path.name, self.encoder, self.fps
        )
        return self

    def write(self, frame: np.ndarray) -> None:
        if self._proc is None or self._proc.stdin is None:
            raise VideoWriteError("VideoWriter used outside its context manager")
        if frame.shape != (self.height, self.width, 3):
            raise VideoWriteError(
                f"Frame {self.frames_written} has shape {frame.shape}, "
                f"expected {(self.height, self.width, 3)}"
            )
        if frame.dtype != np.uint8:
            frame = frame.astype(np.uint8, copy=False)

        try:
            self._proc.stdin.write(np.ascontiguousarray(frame).tobytes())
        except BrokenPipeError as exc:
            raise VideoWriteError(self._encoder_died_message()) from exc
        self.frames_written += 1

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc: Optional[BaseException],
        tb: Optional[TracebackType],
    ) -> None:
        self.close(abort=exc_type is not None)

    def close(self, abort: bool = False) -> None:
        if self._closed or self._proc is None:
            return
        self._closed = True

        if self._proc.stdin is not None:
            try:
                self._proc.stdin.close()
            except BrokenPipeError:
                pass

        stderr = b""
        if self._proc.stderr is not None:
            stderr = self._proc.stderr.read()
            self._proc.stderr.close()
        code = self._proc.wait()

        if abort:
            # The caller is already unwinding; drop the partial file so
            # --resume cannot mistake it for finished work.
            self._tmp.unlink(missing_ok=True)
            return

        if code != 0:
            self._tmp.unlink(missing_ok=True)
            raise VideoWriteError(
                f"Encoder exited {code} while writing {self.path.name}: "
                f"{stderr.decode('utf-8', 'replace').strip()}"
            )
        if self.frames_written == 0:
            self._tmp.unlink(missing_ok=True)
            raise VideoWriteError(f"No frames were written for {self.path.name}")

        os.replace(self._tmp, self.path)
        logger.info(
            "Wrote %s (%d frames, %.1f MB)",
            self.path.name, self.frames_written, self.path.stat().st_size / 1e6,
        )

    def _encoder_died_message(self) -> str:
        stderr = ""
        if self._proc is not None and self._proc.stderr is not None:
            stderr = self._proc.stderr.read().decode("utf-8", "replace").strip()
        return (
            f"Encoder closed the pipe after {self.frames_written} frames "
            f"while writing {self.path.name}: {stderr or '(no stderr)'}"
        )


# ── completion sidecar, for --resume ──────────────────────────────────


def write_sidecar(video_path: Path, payload: dict) -> None:
    """Record what produced *video_path*, atomically, next to it."""
    path = Path(str(video_path) + SIDECAR_SUFFIX)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


def read_sidecar(video_path: Path) -> Optional[dict]:
    path = Path(str(video_path) + SIDECAR_SUFFIX)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return None
