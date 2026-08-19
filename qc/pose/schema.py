"""Per-video keypoint container and its on-disk ``.npz`` form.

The two visibility flags are the heart of this schema and of the QC report:

``hand_visible[t, h]``
    The hand detector fired for hand *h* at frame *t* — a hand is in
    frame and not fully occluded.  This is a property of the scene.

``valid[t, h]``
    The pose regressor recovered a pose for that detection.  This is a
    property of our tracker.

Their difference is what the customer actually cares about:

* ``not hand_visible``            → no hand present or fully occluded.
  Ground truth about a factory line, not a tracking failure.
* ``hand_visible and not valid``  → hand was there and we lost it.
  This, and only this, is a quality defect.

Both arrays ship inside the ``.npz`` so they are delivered metadata the
customer can filter on, not something we computed and discarded.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from qc.config import HAND_NAMES, N_JOINTS

N_HANDS = 2

# Field name -> (trailing shape, dtype)
_ARRAY_SPEC = {
    "kp3d": ((N_HANDS, N_JOINTS, 3), np.float32),
    "kp2d": ((N_HANDS, N_JOINTS, 2), np.float32),
    "conf": ((N_HANDS,), np.float32),
    "det_conf": ((N_HANDS,), np.float32),
    "hand_visible": ((N_HANDS,), np.bool_),
    "valid": ((N_HANDS,), np.bool_),
}


class PoseTrackError(RuntimeError):
    """Raised when a track is internally inconsistent or misaligned."""


@dataclass
class PoseTrack:
    """21-keypoint hand track for one video, indexed by absolute frame.

    Every array's first axis is the source frame index.  Nothing in this
    pipeline ever re-bases that index, which is what keeps the two panels
    frame-exact.
    """

    kp3d: np.ndarray           # (T, 2, 21, 3) float32, metres, wrist-relative
    kp2d: np.ndarray           # (T, 2, 21, 2) float32, source pixel coords
    conf: np.ndarray           # (T, 2) float32, pose regressor confidence
    det_conf: np.ndarray       # (T, 2) float32, hand detector confidence
    hand_visible: np.ndarray   # (T, 2) bool, detector fired
    valid: np.ndarray          # (T, 2) bool, pose recovered
    meta: Dict[str, Any] = field(default_factory=dict)

    # ── construction ──────────────────────────────────────────────────

    @classmethod
    def empty(cls, n_frames: int, meta: Optional[Dict[str, Any]] = None) -> "PoseTrack":
        """An all-missing track of *n_frames*, ready to be filled in."""
        return cls(
            kp3d=np.full((n_frames, N_HANDS, N_JOINTS, 3), np.nan, np.float32),
            kp2d=np.full((n_frames, N_HANDS, N_JOINTS, 2), np.nan, np.float32),
            conf=np.zeros((n_frames, N_HANDS), np.float32),
            det_conf=np.zeros((n_frames, N_HANDS), np.float32),
            hand_visible=np.zeros((n_frames, N_HANDS), np.bool_),
            valid=np.zeros((n_frames, N_HANDS), np.bool_),
            meta=dict(meta or {}),
        )

    def __post_init__(self) -> None:
        self._check_shapes()

    # ── invariants ────────────────────────────────────────────────────

    @property
    def n_frames(self) -> int:
        return int(self.kp3d.shape[0])

    def _check_shapes(self) -> None:
        n = self.kp3d.shape[0]
        for name, (tail, dtype) in _ARRAY_SPEC.items():
            arr = getattr(self, name)
            want = (n,) + tail
            if arr.shape != want:
                raise PoseTrackError(
                    f"PoseTrack.{name} has shape {arr.shape}, expected {want}"
                )
            if arr.dtype != dtype:
                setattr(self, name, arr.astype(dtype, copy=False))

    def check_alignment(self, n_video_frames: int, source: str) -> None:
        """Abort loudly if the track does not match the video frame count.

        Silent misalignment is the worst failure this tool can have: the
        skeleton would be drawn over the wrong frame and the whole
        artifact would be quietly dishonest.  So this raises rather than
        truncating or padding to fit.
        """
        if self.n_frames != n_video_frames:
            raise PoseTrackError(
                f"Frame-count mismatch for {source}: keypoint track has "
                f"{self.n_frames} frames but the video has {n_video_frames}. "
                "Refusing to render — the panels would be out of sync. "
                "Delete the cached .npz to re-run inference, or check that "
                "the keypoints were produced from this exact video file."
            )

    def assert_consistent(self) -> None:
        """A pose cannot be recovered for a hand that was never detected."""
        impossible = self.valid & ~self.hand_visible
        if impossible.any():
            n = int(impossible.sum())
            raise PoseTrackError(
                f"{n} hand-slots are marked valid but not hand_visible. "
                "A recovered pose implies a detection; the track is corrupt."
            )

    # ── derived visibility metrics ────────────────────────────────────

    @property
    def n_visible_slots(self) -> int:
        """Hand-slots where a hand was actually in view."""
        return int(self.hand_visible.sum())

    @property
    def n_recovered_slots(self) -> int:
        """Visible hand-slots where we recovered a pose."""
        return int(self.valid.sum())

    @property
    def pose_recovery_rate(self) -> float:
        """Recovered / visible — the true tracker quality measure.

        Returns NaN when no hand was ever visible, because a recovery
        rate over zero opportunities is undefined, not 0% and not 100%.
        """
        visible = self.n_visible_slots
        if visible == 0:
            return float("nan")
        return self.n_recovered_slots / visible

    def hands_drawn(self) -> np.ndarray:
        """(T,) count of hands with a renderable pose, per frame."""
        return self.valid.sum(axis=1).astype(np.int32)

    # ── persistence ───────────────────────────────────────────────────

    def save(self, path: Path) -> None:
        """Write atomically so a killed instance never leaves a torn .npz."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        arrays = {name: getattr(self, name) for name in _ARRAY_SPEC}
        with open(tmp, "wb") as fh:
            np.savez_compressed(fh, meta_json=np.array(_dumps(self.meta)), **arrays)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)

    @classmethod
    def load(cls, path: Path) -> "PoseTrack":
        with np.load(Path(path), allow_pickle=False) as data:
            missing = [n for n in _ARRAY_SPEC if n not in data]
            if missing:
                raise PoseTrackError(
                    f"{path} is missing arrays {missing} — it was written by an "
                    "older version of this tool. Delete it to re-run inference."
                )
            meta = _loads(str(data["meta_json"])) if "meta_json" in data else {}
            track = cls(
                kp3d=data["kp3d"],
                kp2d=data["kp2d"],
                conf=data["conf"],
                det_conf=data["det_conf"],
                hand_visible=data["hand_visible"],
                valid=data["valid"],
                meta=meta,
            )
        track.assert_consistent()
        return track


def _dumps(meta: Dict[str, Any]) -> str:
    import json

    return json.dumps(meta, sort_keys=True, default=str)


def _loads(blob: str) -> Dict[str, Any]:
    import json

    try:
        return json.loads(blob)
    except (ValueError, TypeError):
        return {}


def hand_name(index: int) -> str:
    return HAND_NAMES[index]
