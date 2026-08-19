"""Preflight checks — run this before spending GPU hours.

Every check here corresponds to something that has actually gone wrong,
or would waste a rental if it did: torch installed without CUDA, weights
half-downloaded, a manifest with blank tasks discovered only after
inference finished, a disk too small for the outputs, more workers than
VRAM.

The point is to fail in thirty seconds rather than three hours.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from qc.config import RenderConfig

PASS = "PASS"
WARN = "WARN"
FAIL = "FAIL"

# Measured on 1080p60: compositing per frame per worker, and the whole
# decode -> composite -> encode loop with a software encoder.
COMPOSITE_MS = 18.8
FULL_LOOP_MS = 63.0
# Rough VRAM per concurrent WiLoR worker.
VRAM_PER_WORKER_GB = 6.0


@dataclass
class Check:
    name: str
    status: str
    detail: str = ""

    @property
    def failed(self) -> bool:
        return self.status == FAIL


def _fmt_gb(n_bytes: float) -> str:
    return f"{n_bytes / 1e9:.1f} GB"


def _fmt_hours(hours: float) -> str:
    if hours < 1 / 60:
        return "<1 min"
    if hours < 1:
        return f"{hours * 60:.0f} min"
    return f"{hours:.1f} h"


# ── individual checks ─────────────────────────────────────────────────


def check_ffmpeg() -> List[Check]:
    from qc.io.ffmpeg import FFmpegNotFound, ffmpeg_bin, ffprobe_bin, run

    out: List[Check] = []
    try:
        proc = run([ffmpeg_bin(), "-version"])
        version = proc.stdout.splitlines()[0] if proc.stdout else "unknown"
        out.append(Check("ffmpeg", PASS, version[:60]))
    except FFmpegNotFound as exc:
        out.append(Check("ffmpeg", FAIL, str(exc)))
        return out

    try:
        ffprobe_bin()
        out.append(Check("ffprobe", PASS))
    except FFmpegNotFound as exc:
        out.append(Check("ffprobe", FAIL, str(exc)))
    return out


def check_encoder(preference: str) -> tuple:
    """Returns ``(check, encoder_name)``.

    The name is returned rather than parsed back out of the message —
    sniffing the text matches "NVENC unavailable" as readily as
    "h264_nvenc", which silently produced an optimistic runtime estimate.
    """
    try:
        from qc.io.ffmpeg import has_nvenc, select_encoder

        encoder = select_encoder(preference)
    except Exception as exc:  # noqa: BLE001
        return Check("encoder", FAIL, str(exc)), None

    if encoder == "h264_nvenc":
        return Check("encoder", PASS, "h264_nvenc (GPU) — encode is off the CPU"), encoder
    if has_nvenc():
        return Check("encoder", PASS, f"{encoder} (requested)"), encoder
    return (
        Check("encoder", WARN,
              "libx264 — NVENC unavailable, so rendering is slower. Correct output, "
              "just slower."),
        encoder,
    )


def check_torch(pose_backend: str, workers: int) -> List[Check]:
    if pose_backend != "wilor":
        return [Check("torch / CUDA", PASS, "not needed for --pose-backend cached")]

    try:
        import torch
    except ImportError:
        return [Check("torch / CUDA", FAIL,
                      "torch is not installed; see VAST_SETUP.md step 3")]

    out = [Check("torch", PASS, f"{torch.__version__} (cuda build {torch.version.cuda})")]

    if not torch.cuda.is_available():
        out.append(Check(
            "CUDA", FAIL,
            "torch.cuda.is_available() is False — WiLoR would run on CPU at roughly "
            "1/100th speed. Reinstall torch from the index matching the driver.",
        ))
        return out

    name = torch.cuda.get_device_name(0)
    total_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    out.append(Check("CUDA", PASS, f"{name}, {total_gb:.0f} GB VRAM"))

    needed = workers * VRAM_PER_WORKER_GB
    if needed > total_gb:
        out.append(Check(
            "workers vs VRAM", FAIL,
            f"--workers {workers} needs ~{needed:.0f} GB but the card has "
            f"{total_gb:.0f} GB. Use --workers {max(1, int(total_gb // VRAM_PER_WORKER_GB))}.",
        ))
    elif needed > total_gb * 0.85:
        out.append(Check(
            "workers vs VRAM", WARN,
            f"--workers {workers} needs ~{needed:.0f} GB of {total_gb:.0f} GB — tight.",
        ))
    else:
        out.append(Check("workers vs VRAM", PASS,
                         f"--workers {workers} needs ~{needed:.0f} GB of {total_gb:.0f} GB"))
    return out


def check_weights(weights_dir: Optional[Path], pose_backend: str) -> Check:
    if pose_backend != "wilor":
        return Check("WiLoR weights", PASS, "not needed for --pose-backend cached")
    if weights_dir is None:
        return Check("WiLoR weights", FAIL, "--wilor-weights not given")

    from qc.pose.wilor_backend import WiLoRPaths, WiLoRUnavailable

    try:
        WiLoRPaths.under(weights_dir).check()
    except WiLoRUnavailable as exc:
        return Check("WiLoR weights", FAIL, str(exc).replace("\n", " ")[:200])

    total = sum(p.stat().st_size for p in Path(weights_dir).glob("*") if p.is_file())
    return Check("WiLoR weights", PASS, f"all three present ({_fmt_gb(total)})")


def check_videos(inputs, limit: Optional[int]) -> tuple:
    """Returns (checks, videos, probed metas)."""
    from qc.io.video_reader import VideoReadError, discover_videos, probe

    try:
        videos = discover_videos(inputs)
    except VideoReadError as exc:
        return [Check("source videos", FAIL, str(exc))], [], []

    if not videos:
        return [Check("source videos", FAIL, "no video files found")], [], []
    if limit:
        videos = videos[:limit]

    total_bytes = sum(p.stat().st_size for p in videos)
    checks = [Check("source videos", PASS,
                    f"{len(videos)} file(s), {_fmt_gb(total_bytes)} total")]

    metas = []
    unreadable = []
    for path in videos:
        try:
            metas.append(probe(path))
        except VideoReadError as exc:
            unreadable.append(f"{path.name}: {exc}")

    if unreadable:
        checks.append(Check(
            "probe", FAIL,
            f"{len(unreadable)} unreadable: " + "; ".join(unreadable[:2])[:180],
        ))
    else:
        frames = sum(m.n_frames for m in metas)
        minutes = sum(m.duration_s for m in metas) / 60
        shapes = {f"{m.width}x{m.height}@{m.fps:.2f}" for m in metas}
        checks.append(Check(
            "probe", PASS,
            f"{frames:,} frames, {minutes:.1f} min total, formats: "
            + ", ".join(sorted(shapes)[:3]),
        ))
    return checks, videos, metas


def check_manifest(manifest_path: Path, videos) -> Check:
    from qc.manifest import ManifestError, load

    try:
        labels = load(manifest_path)
    except ManifestError as exc:
        return Check("manifest", FAIL, str(exc).replace("\n", " ")[:200])

    missing = [p.name for p in videos if p.name.lower() not in labels]
    blank = [
        p.name for p in videos
        if p.name.lower() in labels and not labels[p.name.lower()].task
    ]

    if missing:
        return Check("manifest", FAIL,
                     f"{len(missing)} video(s) have no row, e.g. {missing[0]}")
    if blank:
        return Check("manifest", FAIL,
                     f"{len(blank)} row(s) have a blank task, e.g. {blank[0]}")
    return Check("manifest", PASS, f"{len(videos)} video(s) labelled")


def check_disk(out_dir: Path, metas, cfg: RenderConfig) -> Check:
    try:
        usage = shutil.disk_usage(out_dir if out_dir.exists() else out_dir.parent)
    except OSError as exc:
        return Check("disk", WARN, f"could not stat: {exc}")

    frames = sum(m.n_frames for m in metas)
    minutes = sum(m.duration_s for m in metas) / 60
    # Uncapped 1080p renders land near 30 MB/min; keypoints near 2.5 MB/min.
    render_bytes = (
        minutes * 30e6 if not cfg.max_size_mb
        else min(minutes * 30e6, len(metas) * cfg.max_size_mb * 1e6)
    )
    keypoint_bytes = minutes * 2.5e6
    needed = render_bytes + keypoint_bytes

    detail = (
        f"{_fmt_gb(usage.free)} free; outputs need about "
        f"{_fmt_gb(needed)} (renders + keypoints)"
    )
    if needed > usage.free:
        return Check("disk", FAIL, detail + " — will not fit")
    if needed > usage.free * 0.8:
        return Check("disk", WARN, detail + " — tight")
    return Check("disk", PASS, detail)


def estimate_runtime(metas, workers: int, nvenc: bool) -> Check:
    frames = sum(m.n_frames for m in metas)
    if not frames:
        return Check("render estimate", WARN, "no frames to estimate from")

    per_frame_ms = COMPOSITE_MS if nvenc else FULL_LOOP_MS
    hours = frames * per_frame_ms / 1000 / 3600 / max(1, workers)
    return Check(
        "render estimate", PASS,
        f"{frames:,} frames -> ~{_fmt_hours(hours)} at --workers {workers} "
        f"({'NVENC' if nvenc else 'libx264'}). Inference is extra — time it in the "
        "smoke test.",
    )


# ── driver ────────────────────────────────────────────────────────────


def run_checks(
    inputs,
    out_dir: Path,
    manifest_path: Path,
    cfg: RenderConfig,
    workers: int,
    weights_dir: Optional[Path],
    limit: Optional[int] = None,
) -> List[Check]:
    checks: List[Check] = []
    checks.extend(check_ffmpeg())
    encoder_check, encoder = check_encoder(cfg.encoder)
    checks.append(encoder_check)
    checks.extend(check_torch(cfg.pose_backend, workers))
    checks.append(check_weights(weights_dir, cfg.pose_backend))

    video_checks, videos, metas = check_videos(inputs, limit)
    checks.extend(video_checks)

    if videos:
        checks.append(check_manifest(manifest_path, videos))
    if metas:
        checks.append(check_disk(out_dir, metas, cfg))
        checks.append(estimate_runtime(
            metas, workers, nvenc=(encoder == "h264_nvenc"),
        ))
    return checks


def format_report(checks: List[Check]) -> str:
    width = max(len(c.name) for c in checks) if checks else 10
    symbols = {PASS: "ok  ", WARN: "warn", FAIL: "FAIL"}
    lines = ["", "Preflight", "-" * 72]
    for check in checks:
        lines.append(f"  [{symbols[check.status]}] {check.name:<{width}}  {check.detail}")
    lines.append("-" * 72)

    failed = [c for c in checks if c.status == FAIL]
    warned = [c for c in checks if c.status == WARN]
    if failed:
        lines.append(f"{len(failed)} blocking problem(s). Fix these before rendering:")
        lines.extend(f"  - {c.name}: {c.detail}" for c in failed)
    elif warned:
        lines.append(f"Ready, with {len(warned)} warning(s) above.")
    else:
        lines.append("All checks passed. Safe to start the batch.")
    lines.append("")
    return "\n".join(lines)
