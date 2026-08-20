"""Process-level parallelism across videos.

One video is the unit of work: inference, analysis and rendering for a
single file all happen inside one worker process. That keeps the
frame-exactness guarantees intact — nothing about a video's pipeline is
split across processes, so there is no shared cursor to desynchronise.

Workers are spawned, never forked. A forked child inherits a CUDA
context it cannot use, which fails deep inside torch with an error that
looks nothing like its cause; spawning costs a couple of seconds of
interpreter startup and avoids the entire class of problem. It is also
why the parent never constructs a WiLoR backend when running in
parallel — it validates the weight *paths* instead, and each worker
builds its own model lazily.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from qc.config import RenderConfig
from qc.io.video_reader import FrameWindow
from qc.manifest import Label

logger = logging.getLogger(__name__)

# One WiLoR model per worker process, built on first use and reused.
_BACKEND = None


@dataclass
class BackendSpec:
    """Everything needed to build a pose backend inside a worker."""

    kind: str  # "wilor", "wilor_mini" or "cached"
    weights_dir: Optional[Path] = None
    device: Optional[str] = None
    batch_size: int = 8
    # None means the backend's default assumption. Set it once the camera's
    # true field of view is known — depth is linear in focal, so this is the
    # single number that converts assumed depth into measured depth.
    assumed_hfov_deg: Optional[float] = None

    def build(self):
        if self.kind == "wilor":
            from qc.pose.wilor_backend import WiLoRBackend

            return WiLoRBackend(
                weights_dir=self.weights_dir,
                device=self.device,
                batch_size=self.batch_size,
            )
        if self.kind == "wilor_mini":
            from qc.pose.wilor_mini_backend import (
                DEFAULT_ASSUMED_HFOV_DEG,
                WiLoRMiniBackend,
            )

            return WiLoRMiniBackend(
                weights_dir=self.weights_dir,
                device=self.device,
                assumed_hfov_deg=(
                    self.assumed_hfov_deg
                    if self.assumed_hfov_deg is not None
                    else DEFAULT_ASSUMED_HFOV_DEG
                ),
            )
        return None


_MODEL_BACKENDS = ("wilor", "wilor_mini")


def _backend_for(spec: Optional[BackendSpec]):
    global _BACKEND
    if spec is None or spec.kind not in _MODEL_BACKENDS:
        return None
    if _BACKEND is None:
        _BACKEND = spec.build()
    return _BACKEND


@dataclass
class AnalyzeJob:
    video: Path
    keypoints_dir: Path
    label: Label
    backend: Optional[BackendSpec]
    force_inference: bool = False


@dataclass
class RenderJob:
    video: Path
    keypoints_dir: Path
    out_path: Path
    label: Label
    cfg: RenderConfig
    window: Tuple[int, int]
    slam_dir: Optional[Path] = None


@dataclass
class JobResult:
    """Outcome of one worker task — never raises across the pool boundary."""

    video: Path
    value: Any = None
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.error is None


def _worker_logging() -> None:
    """Minimal logging inside a spawned worker."""
    root = logging.getLogger()
    if root.handlers:
        return
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter(f"%(asctime)s [pid {os.getpid()}] %(levelname)-7s %(message)s",
                          "%H:%M:%S")
    )
    root.addHandler(handler)
    root.setLevel(logging.INFO)


def run_analyze(job: AnalyzeJob) -> JobResult:
    """Probe, obtain keypoints (inferring if needed), and compute stats."""
    _worker_logging()
    try:
        from qc.io.video_reader import probe
        from qc.pose.cache import get_track
        from qc.report.analyze import analyze

        meta = probe(job.video)
        track = get_track(
            meta, job.keypoints_dir, _backend_for(job.backend),
            force=job.force_inference,
        )
        stats = analyze(track, meta, job.label.site, job.label.task)
        return JobResult(video=job.video, value=stats)
    except Exception as exc:  # noqa: BLE001 — reported, never propagated
        logging.getLogger(__name__).exception("Analysis failed for %s", job.video.name)
        return JobResult(video=job.video, error=f"{type(exc).__name__}: {exc}")


def run_render(job: RenderJob) -> JobResult:
    """Render one video from its cached keypoints."""
    _worker_logging()
    try:
        from qc.io.video_reader import probe
        from qc.pose.cache import load_cached
        from qc.pose.schema import PoseTrackError
        from qc.runner import render_video

        meta = probe(job.video)
        track = load_cached(job.keypoints_dir, meta)
        if track is None:
            raise PoseTrackError(
                f"No cached keypoints for {job.video.name}; the analysis pass "
                "should have produced them."
            )

        slam_panel = _slam_panel(job, meta)
        result = render_video(
            meta=meta, track=track, cfg=job.cfg, label=job.label,
            out_path=job.out_path,
            window=FrameWindow(job.window[0], job.window[1]),
            # Several bars writing to one terminal interleave into noise,
            # so parallel workers report on completion instead.
            show_progress=False,
            slam_panel=slam_panel,
        )
        return JobResult(video=job.video, value=result)
    except Exception as exc:  # noqa: BLE001
        logging.getLogger(__name__).exception("Render failed for %s", job.video.name)
        return JobResult(video=job.video, error=f"{type(exc).__name__}: {exc}")


def _slam_panel(job: RenderJob, meta):
    if not job.cfg.with_slam:
        return None
    from qc.config import LAYOUT_3PANEL
    from qc.render.slam_panel import (
        MissingTrajectoryPanel,
        TrajectoryPanel,
        TrajectoryUnavailable,
        load_trajectory,
    )

    width, height = LAYOUT_3PANEL[2], job.cfg.canvas_h
    if job.slam_dir is None:
        return MissingTrajectoryPanel(width, height, "--slam-dir not supplied")

    for suffix in (".npy", ".npz", ".txt", ".tum"):
        candidate = Path(job.slam_dir) / f"{job.video.stem}{suffix}"
        if candidate.exists():
            try:
                positions = load_trajectory(candidate, meta.n_frames)
            except (TrajectoryUnavailable, ValueError):
                return MissingTrajectoryPanel(width, height, "trajectory unreadable")
            return TrajectoryPanel(width, height, positions, meta.fps)
    return MissingTrajectoryPanel(width, height, "no trajectory for this video")


def execute(
    func: Callable[[Any], JobResult],
    jobs: Sequence[Any],
    workers: int,
    label: str,
) -> List[JobResult]:
    """Run *jobs* through *func*, sequentially or across a spawned pool.

    A failing job is reported and the batch continues — one unreadable
    video must not cost the other nine their GPU time.
    """
    if not jobs:
        return []

    if workers <= 1:
        results = []
        for i, job in enumerate(jobs, start=1):
            logger.info("[%s %d/%d] %s", label, i, len(jobs), _name(job))
            results.append(func(job))
            _report(results[-1])
        return results

    logger.info("[%s] %d job(s) across %d worker process(es)",
                label, len(jobs), workers)

    results: List[JobResult] = []
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as pool:
        futures = {pool.submit(func, job): job for job in jobs}
        for i, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            results.append(result)
            logger.info("[%s %d/%d] finished %s",
                        label, i, len(jobs), result.video.name)
            _report(result)
    return results


def _report(result: JobResult) -> None:
    if not result.ok:
        logger.error("%s failed: %s", result.video.name, result.error)


def _name(job: Any) -> str:
    video = getattr(job, "video", None)
    return Path(video).name if video else str(job)


def shard(items: Sequence[Any], index: int, total: int) -> List[Any]:
    """Deterministic 1-based slice of *items* for pod *index* of *total*.

    Round-robin rather than contiguous blocks, so a run split across pods
    divides long and short videos evenly instead of loading one pod with
    every big file.
    """
    if total < 1:
        raise ValueError("shard total must be >= 1")
    if not 1 <= index <= total:
        raise ValueError(f"shard index {index} is outside 1..{total}")
    return [item for i, item in enumerate(items) if i % total == index - 1]
