"""Locating ffmpeg/ffprobe and probing what the local build can do.

Kept separate from the reader and writer because both need it, and
because encoder availability is the one thing that reliably differs
between a Windows workstation and a rented Ubuntu GPU box.
"""

from __future__ import annotations

import functools
import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


class FFmpegNotFound(RuntimeError):
    pass


def _find(name: str) -> str:
    # Honour an explicit override before PATH so a Vast.ai image with a
    # hand-built ffmpeg does not need PATH surgery.
    override = os.environ.get(f"MANUDATA_{name.upper()}")
    if override and Path(override).exists():
        return override
    found = shutil.which(name)
    if not found:
        raise FFmpegNotFound(
            f"{name} not found on PATH. Install ffmpeg (see VAST_SETUP.md) or "
            f"set MANUDATA_{name.upper()} to its full path."
        )
    return found


@functools.lru_cache(maxsize=1)
def ffmpeg_bin() -> str:
    return _find("ffmpeg")


@functools.lru_cache(maxsize=1)
def ffprobe_bin() -> str:
    return _find("ffprobe")


def run(cmd: List[str], *, timeout: Optional[float] = None) -> subprocess.CompletedProcess:
    """Run a command with no shell, capturing both streams as text."""
    return subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        stdin=subprocess.DEVNULL,
        text=True,
        errors="replace",
        timeout=timeout,
    )


@functools.lru_cache(maxsize=1)
def has_nvenc() -> bool:
    """True only if h264_nvenc is present *and* actually encodes.

    Listing an encoder proves nothing on a rented box — the driver may be
    missing or the GPU may already be saturated by another tenant. So we
    encode one synthetic frame and check the exit status.
    """
    listed = run([ffmpeg_bin(), "-hide_banner", "-loglevel", "error", "-encoders"])
    if "h264_nvenc" not in listed.stdout:
        logger.debug("h264_nvenc not listed by this ffmpeg build")
        return False

    probe = run(
        [
            ffmpeg_bin(), "-hide_banner", "-loglevel", "error",
            "-f", "lavfi", "-i", "color=c=black:s=256x256:d=0.1:r=10",
            "-c:v", "h264_nvenc", "-frames:v", "1",
            "-f", "null", "-",
        ],
        timeout=120,
    )
    ok = probe.returncode == 0
    if not ok:
        logger.info(
            "h264_nvenc is listed but failed a test encode; falling back to "
            "libx264. ffmpeg said: %s",
            probe.stderr.strip().splitlines()[-1] if probe.stderr.strip() else "(nothing)",
        )
    return ok


def select_encoder(preference: str = "auto") -> str:
    """Resolve ``auto`` / ``nvenc`` / ``x264`` to a concrete encoder name."""
    if preference == "x264":
        return "libx264"
    if preference == "nvenc":
        if not has_nvenc():
            raise RuntimeError(
                "--encoder nvenc was requested but h264_nvenc is unavailable or "
                "non-functional on this machine."
            )
        return "h264_nvenc"
    if preference != "auto":
        raise ValueError(f"unknown encoder preference {preference!r}")
    return "h264_nvenc" if has_nvenc() else "libx264"
