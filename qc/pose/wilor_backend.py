"""WiLoR keypoint inference: YOLO hand detection, then ViT pose regression.

The two-stage shape of WiLoR is what makes the report's central
distinction possible. The detector answers "is a hand visible here?" and
the regressor answers "can we recover its pose?" — so a detection with
no usable pose is recorded as ``hand_visible and not valid`` rather than
collapsing into a single missing-data rate.

Confidence gating is deliberately one-sided: a detection below the
detector threshold is treated as *no hand*, but a detection above it
whose pose comes back below the pose threshold is treated as *visible,
unrecovered*. Gating both the same way would hide our own failures
inside the occlusion number.

    Verification note: this module talks to upstream WiLoR and
    ultralytics, which need a GPU and ~2 GB of weights. It has not been
    executed in the environment it was written in. Run the smoke test in
    VAST_SETUP.md before committing GPU hours to a full batch — it
    exercises exactly this path on ten seconds of footage.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

from qc.config import HAND_L, HAND_R, N_JOINTS, VideoMeta
from qc.io.video_reader import iter_frames
from qc.pose.schema import PoseTrack

logger = logging.getLogger(__name__)

DEFAULT_DET_THRESHOLD = 0.30
DEFAULT_POSE_THRESHOLD = 0.20
DEFAULT_BATCH_SIZE = 8


class WiLoRUnavailable(RuntimeError):
    """Raised with instructions when the WiLoR stack cannot be loaded."""


@dataclass
class WiLoRPaths:
    checkpoint: Path
    model_config: Path
    detector: Path

    @classmethod
    def under(cls, root: Path) -> "WiLoRPaths":
        root = Path(root)
        return cls(
            checkpoint=root / "wilor_final.ckpt",
            model_config=root / "model_config.yaml",
            detector=root / "detector.pt",
        )

    def check(self) -> None:
        missing = [
            str(p) for p in (self.checkpoint, self.model_config, self.detector)
            if not p.exists()
        ]
        if missing:
            raise WiLoRUnavailable(
                "Missing WiLoR weight files:\n  " + "\n  ".join(missing) +
                "\nDownload them as described in VAST_SETUP.md, or point "
                "--wilor-weights at the directory that holds them."
            )


class WiLoRBackend:
    """Runs WiLoR over a video and returns a :class:`PoseTrack`."""

    name = "WiLoR"

    def __init__(
        self,
        weights_dir: Path,
        device: Optional[str] = None,
        det_threshold: float = DEFAULT_DET_THRESHOLD,
        pose_threshold: float = DEFAULT_POSE_THRESHOLD,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> None:
        self.paths = WiLoRPaths.under(weights_dir)
        self.paths.check()
        self.det_threshold = det_threshold
        self.pose_threshold = pose_threshold
        self.batch_size = max(1, batch_size)

        self._torch = _import_torch()
        self.device = device or ("cuda" if self._torch.cuda.is_available() else "cpu")
        if self.device == "cpu":
            logger.warning(
                "No CUDA device visible — WiLoR will run on CPU, which is "
                "roughly two orders of magnitude slower. Check nvidia-smi."
            )

        self._model, self._model_cfg = self._load_model()
        self._detector = self._load_detector()
        self.version = self._resolve_version()

    # ── loading ───────────────────────────────────────────────────────

    def _load_model(self):
        try:
            from wilor.models import load_wilor  # type: ignore
        except ImportError as exc:
            raise WiLoRUnavailable(
                "The 'wilor' package is not importable. Install it as described "
                "in VAST_SETUP.md (pip install from the upstream repository), "
                f"or choose a different --pose-backend. Import error: {exc}"
            ) from exc

        model, cfg = load_wilor(
            checkpoint_path=str(self.paths.checkpoint),
            cfg_path=str(self.paths.model_config),
        )
        model = model.to(self.device)
        model.eval()
        return model, cfg

    def _load_detector(self):
        try:
            from ultralytics import YOLO  # type: ignore
        except ImportError as exc:
            raise WiLoRUnavailable(
                "The 'ultralytics' package is not importable; WiLoR needs it for "
                f"hand detection. Import error: {exc}"
            ) from exc
        return YOLO(str(self.paths.detector))

    def _resolve_version(self) -> str:
        try:
            import wilor  # type: ignore

            version = getattr(wilor, "__version__", "unknown")
        except Exception:
            version = "unknown"
        return f"{version} (ckpt={self.paths.checkpoint.name})"

    # ── inference ─────────────────────────────────────────────────────

    def infer(self, meta: VideoMeta, checkpoint: Optional[Path] = None) -> PoseTrack:
        """Run detection and pose regression across the whole video.

        *checkpoint* is accepted for interface compatibility and ignored:
        this backend batches frames, so mid-video resume would need the
        batch boundary recorded too. WiLoR-mini is what runs in
        production and it does implement it.
        """
        torch = self._torch
        track = PoseTrack.empty(meta.n_frames)

        focal = self._scaled_focal_length(meta)
        principal = np.array([meta.width / 2.0, meta.height / 2.0], np.float32)

        batch_frames: List[Tuple[int, np.ndarray]] = []
        seen = 0

        for index, frame in iter_frames(meta):
            seen = index + 1
            if index >= track.n_frames:
                track = _grow(track, index + 1)
            batch_frames.append((index, frame.copy()))
            if len(batch_frames) >= self.batch_size:
                self._process_batch(track, batch_frames, focal, principal)
                batch_frames.clear()

        if batch_frames:
            self._process_batch(track, batch_frames, focal, principal)

        if seen != track.n_frames:
            track = _truncate(track, seen)

        track.meta["frames_decoded"] = seen
        track.meta["det_threshold"] = self.det_threshold
        track.meta["pose_threshold"] = self.pose_threshold
        track.meta["focal_length_px"] = float(focal)
        track.meta["principal_point_px"] = [float(principal[0]), float(principal[1])]
        return track

    def _process_batch(
        self,
        track: PoseTrack,
        frames: Sequence[Tuple[int, np.ndarray]],
        focal: float,
        principal: np.ndarray,
    ) -> None:
        for index, frame in frames:
            detections = self._detect(frame)
            for hand_slot, box, det_conf in detections:
                track.hand_visible[index, hand_slot] = True
                track.det_conf[index, hand_slot] = det_conf

            if not detections:
                continue

            poses = self._regress(frame, detections, focal, principal)
            for hand_slot, kp3d, kp2d, score in poses:
                if score < self.pose_threshold:
                    # Detected but not recovered — case (b). Leave `valid`
                    # False so it is reported as our failure, not occlusion.
                    continue
                track.kp3d[index, hand_slot] = kp3d
                track.kp2d[index, hand_slot] = kp2d
                track.conf[index, hand_slot] = score
                track.valid[index, hand_slot] = True

    def _detect(self, frame: np.ndarray) -> List[Tuple[int, np.ndarray, float]]:
        """Run YOLO and keep the best detection per hand side.

        Two boxes for the same side means the detector is confused; the
        higher-confidence one wins rather than both being written into a
        two-slot array in arbitrary order.
        """
        result = self._detector(frame, conf=self.det_threshold, verbose=False)[0]
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return []

        xyxy = boxes.xyxy.detach().cpu().numpy()
        conf = boxes.conf.detach().cpu().numpy()
        cls = boxes.cls.detach().cpu().numpy().astype(int)

        best: dict = {}
        for i in range(len(xyxy)):
            # Upstream convention: class 0 is the left hand, 1 the right.
            slot = HAND_R if cls[i] == 1 else HAND_L
            if slot not in best or conf[i] > best[slot][2]:
                best[slot] = (slot, xyxy[i].astype(np.float32), float(conf[i]))
        return list(best.values())

    def _regress(
        self,
        frame: np.ndarray,
        detections: Sequence[Tuple[int, np.ndarray, float]],
        focal: float,
        principal: np.ndarray,
    ) -> List[Tuple[int, np.ndarray, np.ndarray, float]]:
        """Pose-regress the detected boxes.

        Everything version-specific about upstream WiLoR is confined to
        this method, so an API change surfaces here with a clear message
        instead of as silently wrong keypoints.
        """
        torch = self._torch
        from wilor.datasets.vitdet_dataset import ViTDetDataset  # type: ignore

        boxes = np.stack([d[1] for d in detections])
        is_right = np.array(
            [1.0 if d[0] == HAND_R else 0.0 for d in detections], np.float32
        )

        dataset = ViTDetDataset(
            self._model_cfg, frame, boxes, is_right, rescale_factor=2.0
        )
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=len(detections), shuffle=False
        )

        out: List[Tuple[int, np.ndarray, np.ndarray, float]] = []
        for batch in loader:
            batch = {
                k: (v.to(self.device) if hasattr(v, "to") else v)
                for k, v in batch.items()
            }
            with torch.no_grad():
                pred = self._model(batch)

            kp3d = _to_numpy(pred["pred_keypoints_3d"])          # (B, 21, 3)
            pred_cam = _to_numpy(pred["pred_cam"])               # (B, 3)
            box_center = _to_numpy(batch["box_center"])          # (B, 2)
            box_size = _to_numpy(batch["box_size"])              # (B,)
            right = _to_numpy(batch["right"]).reshape(-1)        # (B,)

            if kp3d.shape[1] != N_JOINTS:
                raise WiLoRUnavailable(
                    f"WiLoR returned {kp3d.shape[1]} joints, expected {N_JOINTS}. "
                    "The installed model does not match what this tool assumes; "
                    "check the checkpoint named in VAST_SETUP.md."
                )

            # Left hands are regressed as mirrored right hands upstream.
            mirror = np.where(right > 0.5, 1.0, -1.0).reshape(-1, 1)
            kp3d = kp3d.copy()
            kp3d[:, :, 0] *= mirror

            cam_t = _cam_crop_to_full(
                pred_cam, box_center, box_size,
                img_w=frame.shape[1], img_h=frame.shape[0], focal=focal,
                is_right=right,
            )
            camera_space = kp3d + cam_t[:, None, :]
            kp2d = _project(camera_space, focal, principal)

            scores = _pose_scores(pred, len(detections))
            for i, (slot, _, _) in enumerate(detections):
                out.append((
                    slot,
                    camera_space[i].astype(np.float32),
                    kp2d[i].astype(np.float32),
                    float(scores[i]),
                ))
        return out

    def _scaled_focal_length(self, meta: VideoMeta) -> float:
        """Focal length in pixels, from the model's assumed intrinsics.

        WiLoR is trained with a fixed nominal focal length relative to
        its crop size; scaling by the image's long edge is the upstream
        convention for putting predictions into real image pixels. We do
        not have per-camera intrinsics for these recordings, so this is
        an assumption, and it is recorded in the track metadata and
        stated in README_FOR_CUSTOMER.md rather than left implicit.
        """
        extra = getattr(self._model_cfg, "EXTRA", None)
        nominal = float(getattr(extra, "FOCAL_LENGTH", 5000.0)) if extra else 5000.0
        image_size = float(
            getattr(getattr(self._model_cfg, "MODEL", None), "IMAGE_SIZE", 256) or 256
        )
        return nominal / image_size * max(meta.width, meta.height)


# ── numeric helpers, kept free of any torch types ─────────────────────


def _import_torch():
    try:
        import torch  # type: ignore
    except ImportError as exc:
        raise WiLoRUnavailable(
            "PyTorch is not installed. Install the CUDA build pinned in "
            f"VAST_SETUP.md before using the WiLoR backend. Import error: {exc}"
        ) from exc
    return torch


def _to_numpy(value) -> np.ndarray:
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _cam_crop_to_full(
    pred_cam: np.ndarray,
    box_center: np.ndarray,
    box_size: np.ndarray,
    img_w: int,
    img_h: int,
    focal: float,
    is_right: np.ndarray,
) -> np.ndarray:
    """Weak-perspective crop camera -> full-image metric translation."""
    box_size = np.asarray(box_size, np.float32).reshape(-1)
    s = pred_cam[:, 0]
    tx = pred_cam[:, 1]
    ty = pred_cam[:, 2]

    # Mirror the horizontal offset for left hands, matching the mirrored
    # keypoints above.
    tx = np.where(is_right > 0.5, tx, -tx)

    z = 2.0 * focal / (np.maximum(box_size, 1e-6) * np.maximum(s, 1e-6))
    cx = box_center[:, 0] - img_w / 2.0
    cy = box_center[:, 1] - img_h / 2.0
    x = tx + cx * z / focal
    y = ty + cy * z / focal
    return np.stack([x, y, z], axis=1).astype(np.float32)


def _project(points: np.ndarray, focal: float, principal: np.ndarray) -> np.ndarray:
    """Pinhole projection of ``(B, 21, 3)`` camera-space metres to pixels."""
    z = np.maximum(points[:, :, 2], 1e-4)
    x = principal[0] + focal * points[:, :, 0] / z
    y = principal[1] + focal * points[:, :, 1] / z
    return np.stack([x, y], axis=2)


def _pose_scores(pred: dict, count: int) -> np.ndarray:
    """Per-hand pose confidence, if the model exposes one.

    WiLoR does not always return a scalar confidence. When it does not,
    every regressed pose counts as recovered — which is the honest
    default, because inventing a score would let us quietly discard
    poses we actually produced and flatter the recovery rate.
    """
    for key in ("pred_keypoints_scores", "pred_score", "scores"):
        if key in pred:
            arr = _to_numpy(pred[key]).reshape(count, -1)
            return arr.mean(axis=1)
    return np.ones(count, np.float32)


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
