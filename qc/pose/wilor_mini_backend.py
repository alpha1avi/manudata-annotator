"""WiLoR-mini keypoint inference — the backend that matches what is installed.

Upstream WiLoR (:mod:`qc.pose.wilor_backend`) is not pip-installable and
needs a licence-gated ``MANO_RIGHT.pkl``.  What is actually installed on
the pod is **WiLoR-mini**, a repackaging with a different API:

    from wilor_mini.pipelines.wilor_hand_pose3d_estimation_pipeline import (
        WiLorHandPose3dEstimationPipeline,
    )
    pipe = WiLorHandPose3dEstimationPipeline(device=..., dtype=torch.float16)
    dets = pipe.predict(rgb_image)   # RGB, not BGR

``predict`` returns a *list*, one dict per detected hand::

    {"hand_bbox": [x1, y1, x2, y2], "is_right": 0.0|1.0, "wilor_preds": {...}}

where ``wilor_preds`` carries ``pred_keypoints_3d`` (21x3, wrist-relative
metres), ``pred_cam_t_full`` (the full-image camera translation) and
``pred_keypoints_2d`` (21x2, full-image pixels).  Camera-space metric
keypoints are ``pred_keypoints_3d + pred_cam_t_full``, matching the
coordinate frame the rest of the pipeline assumes.

    A NOTE ON THE TWO VISIBILITY FLAGS.  The report's headline number,
    ``pose_recovery_pct``, rests on separating *hand detected* from *pose
    recovered*.  WiLoR-mini has a real two-stage shape — YOLO detection
    then MANO regression — but the regressor returns **no per-hand
    confidence and never reports failure**: every detected hand yields a
    pose.  So the case that defines a tracking defect, "hand was visible
    but we lost its pose", cannot be observed with this backend.  We
    therefore set ``hand_visible == valid`` for every hand and record
    that fact in the track metadata.  ``pose_recovery_pct`` will read
    100%% (or n/a); it is *not* a measured quality figure here, and the
    customer README must not present it as one.  This is flagged rather
    than papered over, per the handoff.

The only confidence signal available is the YOLO detection score, which
``predict`` discards.  So this backend runs the detector itself — which it
must do anyway, to keep the single best box per hand side for the
two-slot :class:`PoseTrack` — and then calls ``predict_with_bboxes`` so
the score survives into ``conf``/``det_conf``.
"""

from __future__ import annotations

import logging
import math
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from qc.config import HAND_L, HAND_R, N_JOINTS, VideoMeta
from qc.io.video_reader import FrameWindow, iter_frames
from qc.pose.checkpoint import InferenceCheckpoint, strip_checkpoint_meta
from qc.pose.schema import PoseTrack

logger = logging.getLogger(__name__)

DEFAULT_DET_THRESHOLD = 0.30
DEFAULT_RESCALE_FACTOR = 2.5   # WiLoR-mini's own default crop padding

# WiLoR estimates a hand's *shape* metrically but its absolute *depth* is
# unobservable from one image — the model resolves that ambiguity with a
# fixed nominal focal length (5000 px at a 256 px crop). Scaled to a 1920 px
# frame that is 37500 px, tens of times a real action-camera focal, which
# places the hand at ~12 m instead of arm's length and makes the delivered
# `kp3d` depths physically wrong.
#
# We have no calibration for these SJCAM cap cameras, so we assume a
# horizontal field of view and derive a focal from it. Depth scales linearly
# with focal and the 2D projection is invariant to it, so rescaling z by
# focal_assumed / focal_nominal makes `kp3d` plausible (~0.5 m, arm's length)
# and shrinks the per-frame depth jitter by the same factor, without moving a
# single 2D keypoint. This is an *assumption*, recorded in the track meta and
# stated in README_FOR_CUSTOMER; intra-hand geometry stays exactly metric.
DEFAULT_ASSUMED_HFOV_DEG = 65.0

# How often the inference loop reports progress. Logged rather than drawn
# as a bar because inference usually runs inside a spawned worker, where
# several tqdm bars would fight over one terminal — and because a log line
# survives being piped through `tee` into a batch log.
PROGRESS_LOG_INTERVAL_S = 30.0


class WiLoRMiniUnavailable(RuntimeError):
    """Raised with instructions when the WiLoR-mini stack cannot be loaded."""


@dataclass
class WiLoRMiniPaths:
    """Where WiLoR-mini's four weight files live.

    WiLoR-mini looks for everything under ``<root>/pretrained_models/``,
    so the conventional ``--wilor-weights`` value for this backend is a
    directory literally named ``pretrained_models`` (e.g.
    ``~/pretrained_models``).  ``MANO_RIGHT.pkl`` and
    ``mano_mean_params.npz`` are auto-downloaded from HuggingFace on first
    load if absent, so only the two large files are hard requirements.
    """

    checkpoint: Path      # wilor_final.ckpt
    detector: Path        # detector.pt
    mano: Path            # MANO_RIGHT.pkl        (auto-downloaded if missing)
    mano_mean: Path       # mano_mean_params.npz  (auto-downloaded if missing)

    @classmethod
    def under(cls, root: Path) -> "WiLoRMiniPaths":
        root = Path(root)
        return cls(
            checkpoint=root / "wilor_final.ckpt",
            detector=root / "detector.pt",
            mano=root / "MANO_RIGHT.pkl",
            mano_mean=root / "mano_mean_params.npz",
        )

    def check(self) -> None:
        missing = [str(p) for p in (self.checkpoint, self.detector) if not p.exists()]
        if missing:
            raise WiLoRMiniUnavailable(
                "Missing WiLoR-mini weight files:\n  " + "\n  ".join(missing) +
                "\nPoint --wilor-weights at the directory that holds "
                "wilor_final.ckpt and detector.pt."
            )
        for extra in (self.mano, self.mano_mean):
            if not extra.exists():
                logger.warning(
                    "%s is absent; WiLoR-mini will download it from HuggingFace "
                    "on load (needs network).", extra.name,
                )


class WiLoRMiniBackend:
    """Runs WiLoR-mini over a video and returns a :class:`PoseTrack`."""

    name = "WiLoR-mini"

    def __init__(
        self,
        weights_dir: Path,
        device: Optional[str] = None,
        det_threshold: float = DEFAULT_DET_THRESHOLD,
        rescale_factor: float = DEFAULT_RESCALE_FACTOR,
        assumed_hfov_deg: float = DEFAULT_ASSUMED_HFOV_DEG,
        use_fp16: bool = True,
    ) -> None:
        self.paths = WiLoRMiniPaths.under(weights_dir)
        self.paths.check()
        self.det_threshold = det_threshold
        self.rescale_factor = rescale_factor
        self.assumed_hfov_deg = assumed_hfov_deg

        self._torch = _import_torch()
        self.device = device or ("cuda" if self._torch.cuda.is_available() else "cpu")
        if self.device == "cpu":
            logger.warning(
                "No CUDA device visible — WiLoR-mini will run on CPU, which is "
                "roughly two orders of magnitude slower. Check nvidia-smi."
            )
        dtype = (
            self._torch.float16
            if use_fp16 and self.device.startswith("cuda")
            else self._torch.float32
        )

        self._pipe = self._build_pipeline(weights_dir, dtype)
        self.version = f"WiLoR-mini (ckpt={self.paths.checkpoint.name})"
        # Recorded into every .npz so the report generator and README can
        # see that this backend cannot separate detection from pose.
        self.pose_recovery_measurable = False

    # ── loading ───────────────────────────────────────────────────────

    def _build_pipeline(self, weights_dir: Path, dtype):
        try:
            import wilor_mini.pipelines.wilor_hand_pose3d_estimation_pipeline as wmp
            from wilor_mini.pipelines.wilor_hand_pose3d_estimation_pipeline import (
                WiLorHandPose3dEstimationPipeline,
            )
        except ImportError as exc:
            raise WiLoRMiniUnavailable(
                "The 'wilor_mini' package is not importable. Install it "
                "(pip install wilor-mini) or choose a different --pose-backend. "
                f"Import error: {exc}"
            ) from exc

        # WiLoR-mini anti-aliases each crop by Gaussian-blurring the *entire*
        # source frame with skimage, once per detected hand — ~60 ms on a
        # 1080p frame, which dominated inference (~172 ms/frame). cv2's
        # Gaussian is the same operation ~30x faster; swapping it in cuts
        # inference to ~60 ms/frame (2.9x) with sub-2px keypoint difference.
        wmp.gaussian = _fast_gaussian

        root = self._pretrained_root(weights_dir)
        return WiLorHandPose3dEstimationPipeline(
            device=self._torch.device(self.device),
            dtype=dtype,
            wilor_pretrained_dir=str(root),
            verbose=False,   # silence per-frame logs across tens of thousands of frames
        )

    @staticmethod
    def _pretrained_root(weights_dir: Path) -> Path:
        """A root such that ``<root>/pretrained_models`` is *weights_dir*.

        WiLoR-mini hard-codes the ``pretrained_models`` subfolder.  When
        the weights already live in a directory of that name we just hand
        it the parent; otherwise we stage a throwaway directory whose
        ``pretrained_models`` symlinks to the real weights, so no files
        are copied and no re-download happens.
        """
        weights_dir = Path(weights_dir).resolve()
        if weights_dir.name == "pretrained_models":
            return weights_dir.parent
        staging = Path(tempfile.mkdtemp(prefix="wilor_mini_weights_"))
        link = staging / "pretrained_models"
        if not link.exists():
            link.symlink_to(weights_dir, target_is_directory=True)
        return staging

    # ── inference ─────────────────────────────────────────────────────

    def infer(self, meta: VideoMeta, checkpoint: Optional[Path] = None) -> PoseTrack:
        """Run detection and pose regression across the whole video.

        With *checkpoint*, the partially-filled track is written there
        every few thousand frames and a compatible partial is resumed
        from — so a killed instance loses a minute of work rather than
        the whole pass. See :mod:`qc.pose.checkpoint`.
        """
        self._prepare_depth_scale(meta)

        ckpt = (
            InferenceCheckpoint(checkpoint, self._checkpoint_fingerprint(meta))
            if checkpoint is not None else None
        )
        track, start = ckpt.resume(meta.n_frames) if ckpt else (None, 0)
        if track is None:
            track, start = PoseTrack.empty(meta.n_frames), 0
        seen = start

        # Inference is the long pole — tens of minutes on a full-length
        # video — so it reports progress. Without this the operator has no
        # way to tell a working run from a wedged one, which on a rented
        # instance is the difference between waiting and paying for nothing.
        progress = _InferenceProgress(
            meta, log_every_s=PROGRESS_LOG_INTERVAL_S, start=start,
        )

        # Frame-number selection on resume, never a timestamp seek: frame
        # k here must be the same frame k an uninterrupted pass would have
        # processed, or the resumed half of the track is offset from the
        # half already on disk.
        window = FrameWindow(start, meta.n_frames) if start > 0 else None

        for index, frame in iter_frames(meta, window):
            seen = index + 1
            if index >= track.n_frames:
                track = _grow(track, index + 1)
            # iter_frames yields BGR; WiLoR-mini expects RGB.
            rgb = np.ascontiguousarray(frame[:, :, ::-1])
            self._infer_frame(track, index, rgb)
            progress.update(seen, int(track.valid[index].sum()))
            if ckpt is not None:
                ckpt.maybe_save(track, seen)

        progress.close(seen)

        if seen != track.n_frames:
            track = _truncate(track, seen)

        track.meta["frames_decoded"] = seen
        track.meta["det_threshold"] = self.det_threshold
        track.meta["rescale_factor"] = self.rescale_factor
        track.meta["pose_recovery_measurable"] = False
        track.meta["pose_confidence_source"] = (
            "yolo_detection_score (WiLoR-mini exposes no pose confidence; "
            "hand_visible == valid)"
        )
        # Absolute-depth calibration — see _prepare_depth_scale and the module
        # header. Recorded so the render, report and README can be honest that
        # depth rests on an assumed field of view, not a measured one.
        track.meta["absolute_depth_calibrated"] = False
        track.meta["assumed_hfov_deg"] = self.assumed_hfov_deg
        track.meta["wilor_nominal_focal_px"] = self._focal_nominal_px
        track.meta["depth_scale_applied"] = self._depth_scale
        # Names the customer README documents: the focal/principal point the
        # delivered kp3d is consistent with (the assumed-FOV focal, not the
        # nominal one).
        track.meta["focal_length_px"] = self._focal_assumed_px
        track.meta["principal_point_px"] = [meta.width / 2.0, meta.height / 2.0]

        # The partial is deliberately *not* removed here. The caller still
        # has to write the real .npz from what we return, and a crash in
        # that window should still be resumable.  cache.get_track drops it
        # once the durable file exists.
        strip_checkpoint_meta(track)
        return track

    def _checkpoint_fingerprint(self, meta: VideoMeta) -> Dict[str, Any]:
        """Every input that changes the numbers a resumed pass would produce.

        Deliberately includes the depth scale rather than just the
        assumed FOV: mixing frames inferred at two different scales would
        produce a track that looks entirely plausible and is wrong in its
        second half.
        """
        return {
            "backend": self.name,
            "version": self.version,
            "source": Path(meta.path).name,
            "width": meta.width,
            "height": meta.height,
            "n_frames": meta.n_frames,
            "det_threshold": float(self.det_threshold),
            "rescale_factor": float(self.rescale_factor),
            "assumed_hfov_deg": float(self.assumed_hfov_deg),
            "depth_scale": round(float(self._depth_scale), 9),
        }

    def _prepare_depth_scale(self, meta: VideoMeta) -> None:
        """Factor to rescale WiLoR's nominal-focal depth to an assumed FOV.

        WiLoR's ``pred_cam_t_full`` depth is ``2 * focal / (box_size * s)``,
        linear in the focal length; its x/y are focal-invariant. So one
        multiplier on z alone converts the nominal-focal translation to our
        assumed-focal one, and the 2D projection (already computed by the
        pipeline) is untouched.
        """
        long_edge = max(meta.width, meta.height)
        self._focal_nominal_px = float(
            self._pipe.FOCAL_LENGTH / self._pipe.IMAGE_SIZE * long_edge
        )
        # Horizontal FOV over the frame width gives the assumed focal.
        self._focal_assumed_px = float(
            (meta.width / 2.0) / math.tan(math.radians(self.assumed_hfov_deg / 2.0))
        )
        self._depth_scale = self._focal_assumed_px / self._focal_nominal_px
        logger.info(
            "Depth rescale: assumed HFOV %.0f deg -> focal %.0f px (nominal %.0f px), "
            "z x %.4f. Absolute depth is an assumption, not a calibration.",
            self.assumed_hfov_deg, self._focal_assumed_px,
            self._focal_nominal_px, self._depth_scale,
        )

    def _infer_frame(self, track: PoseTrack, index: int, rgb: np.ndarray) -> None:
        detections = self._detect(rgb)
        if not detections:
            return

        slots = list(detections.keys())
        bboxes = np.stack([detections[s][1] for s in slots])
        is_rights = [detections[s][2] for s in slots]

        preds = self._pipe.predict_with_bboxes(
            rgb, bboxes, is_rights, rescale_factor=self.rescale_factor
        )

        for slot, pred in zip(slots, preds):
            wp = pred["wilor_preds"]
            kp3d_rel = np.asarray(wp["pred_keypoints_3d"][0], np.float32)  # (21,3)
            cam_t = np.asarray(wp["pred_cam_t_full"][0], np.float32).copy()  # (3,)
            # Rescale depth from WiLoR's nominal focal to the assumed FOV.
            # Only z is focal-dependent; x/y are invariant. 2D is untouched.
            cam_t[2] *= self._depth_scale
            if kp3d_rel.shape[0] != N_JOINTS:
                raise WiLoRMiniUnavailable(
                    f"WiLoR-mini returned {kp3d_rel.shape[0]} joints, expected "
                    f"{N_JOINTS}. The installed model does not match this tool."
                )
            # Camera-space metric keypoints: wrist-relative + camera translation.
            kp3d = kp3d_rel + cam_t[None, :]
            kp2d = np.asarray(wp["pred_keypoints_2d"][0], np.float32)      # (21,2)
            conf = detections[slot][0]

            track.kp3d[index, slot] = kp3d
            track.kp2d[index, slot] = kp2d
            track.conf[index, slot] = conf
            track.det_conf[index, slot] = conf
            # WiLoR-mini has no independent pose-failure signal, so a
            # detected hand is also a recovered pose.  See module docstring.
            track.hand_visible[index, slot] = True
            track.valid[index, slot] = True

    def _detect(self, rgb: np.ndarray) -> Dict[int, Tuple[float, np.ndarray, float]]:
        """Best YOLO box per hand side.

        Returns ``{slot: (det_conf, xyxy, is_right)}``.  Two boxes for the
        same side means the detector is confused; the higher-confidence
        one wins rather than both being written into a two-slot array in
        arbitrary order.  Handedness is the detector's class, never
        x-position.
        """
        result = self._pipe.hand_detector(rgb, conf=self.det_threshold, verbose=False)[0]
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return {}

        data = boxes.data.detach().cpu().numpy()  # (N, 6): x1,y1,x2,y2,conf,cls
        best: Dict[int, Tuple[float, np.ndarray, float]] = {}
        for row in data:
            conf = float(row[4])
            # WiLoR-mini convention: class 1 is the right hand, 0 the left.
            is_right = 1.0 if int(round(row[5])) == 1 else 0.0
            slot = HAND_R if is_right else HAND_L
            if slot not in best or conf > best[slot][0]:
                best[slot] = (conf, row[:4].astype(np.float32), is_right)
        return best


# ── helpers, kept free of any torch types ─────────────────────────────


class _InferenceProgress:
    """Periodic progress lines for the long inference pass.

    Reports throughput and a projected finish time from the rate measured
    so far, plus a running detection count — so a run that is alive but
    finding no hands looks different from one that is simply slow.
    """

    def __init__(
        self,
        meta: VideoMeta,
        log_every_s: float = PROGRESS_LOG_INTERVAL_S,
        start: int = 0,
    ):
        self.name = Path(meta.path).name
        self.total = max(1, meta.n_frames)
        self.fps_source = meta.fps
        self.log_every_s = log_every_s
        # Frames already done by an earlier, checkpointed run. They count
        # towards the percentage but not towards the measured rate, or a
        # resumed run would report a throughput it never achieved and an
        # ETA derived from it.
        self.start = max(0, start)
        self.started = time.monotonic()
        self._last_log = self.started
        self._hands = 0

    def _rate(self, done: int, elapsed: float) -> float:
        fresh = done - self.start
        return fresh / elapsed if elapsed > 0 and fresh > 0 else 0.0

    def update(self, done: int, hands_this_frame: int) -> None:
        self._hands += hands_this_frame
        now = time.monotonic()
        if now - self._last_log < self.log_every_s:
            return
        self._last_log = now

        rate = self._rate(done, now - self.started)
        remaining = (self.total - done) / rate if rate > 0 else float("inf")
        logger.info(
            "%s: %d/%d frames (%.1f%%) | %.1f fps | %s left | %d hand-detections",
            self.name, done, self.total, 100.0 * done / self.total,
            rate, _human(remaining), self._hands,
        )

    def close(self, done: int) -> None:
        elapsed = time.monotonic() - self.started
        resumed = " (resumed at %d)" % self.start if self.start else ""
        logger.info(
            "%s: inference done — %d frames in %s (%.1f fps)%s, %d hand-detections",
            self.name, done, _human(elapsed), self._rate(done, elapsed),
            resumed, self._hands,
        )


def _human(seconds: float) -> str:
    if seconds == float("inf"):
        return "unknown"
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m{seconds % 60:02d}s"
    return f"{seconds // 3600}h{(seconds % 3600) // 60:02d}m"


def _fast_gaussian(image, sigma=1.0, channel_axis=None, preserve_range=True, **kwargs):
    """Drop-in for ``skimage.filters.gaussian`` backed by cv2.

    Matches the call WiLoR-mini makes — ``gaussian(img, sigma=...,
    channel_axis=2, preserve_range=True)`` — but runs ~30x faster. cv2
    handles the per-channel blur for an HxWxC array natively; a
    non-positive sigma is a no-op, as in skimage.
    """
    import cv2

    if sigma is None or sigma <= 0:
        return image.astype(np.float32)
    return cv2.GaussianBlur(
        image.astype(np.float32), (0, 0), sigmaX=float(sigma), sigmaY=float(sigma)
    )


def _import_torch():
    try:
        import torch  # type: ignore
    except ImportError as exc:
        raise WiLoRMiniUnavailable(
            "PyTorch is not installed. WiLoR-mini pins torch<=2.5; install the "
            f"CUDA build before using this backend. Import error: {exc}"
        ) from exc
    return torch


def _grow(track: PoseTrack, n: int) -> PoseTrack:
    """Extend a track when the decoder yields more frames than probed."""
    extra = n - track.n_frames
    if extra <= 0:
        return track
    tail = PoseTrack.empty(extra)
    return PoseTrack(
        kp3d=np.concatenate([track.kp3d, tail.kp3d]),
        kp2d=np.concatenate([track.kp2d, tail.kp2d]),
        conf=np.concatenate([track.conf, tail.conf]),
        det_conf=np.concatenate([track.det_conf, tail.det_conf]),
        hand_visible=np.concatenate([track.hand_visible, tail.hand_visible]),
        valid=np.concatenate([track.valid, tail.valid]),
        meta=track.meta,
    )


def _truncate(track: PoseTrack, n: int) -> PoseTrack:
    return PoseTrack(
        kp3d=track.kp3d[:n], kp2d=track.kp2d[:n], conf=track.conf[:n],
        det_conf=track.det_conf[:n], hand_visible=track.hand_visible[:n],
        valid=track.valid[:n], meta=track.meta,
    )
