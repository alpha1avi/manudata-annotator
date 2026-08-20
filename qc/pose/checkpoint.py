"""Mid-inference checkpointing, so a killed run does not lose an hour.

The keypoint cache (:mod:`qc.pose.cache`) checkpoints at *video*
granularity: a batch that dies keeps every ``.npz`` it finished. That is
the right unit for a ten-video batch, and useless for the case that
actually bites on a rented instance — a single 25-minute video killed at
95%, which loses the whole pass because the ``.npz`` is written once, at
the end.

This module adds a second, finer checkpoint *inside* one video's
inference pass. Every few thousand frames the partially-filled track is
written next to its eventual ``.npz`` as ``<stem>.npz.partial``; the next
run picks it up and resumes at the first unprocessed frame.

Two things make that safe to trust:

* **The fingerprint.** A partial is only reused when the source video,
  the backend and every parameter that changes the numbers (detection
  threshold, the assumed field of view behind the depth scale) match the
  current run exactly. Anything else and it is discarded rather than
  silently mixed — half a track at one depth scale and half at another
  would be a quiet, plausible-looking corruption, which is the worst
  kind.
* **Frame-number resume.** Decoding restarts through the same
  ``select=between(n,...)`` frame-number filter the renderer uses, never
  a timestamp seek, so frame *k* on resume is the same frame *k* it
  would have been in one continuous pass.

Partials are written uncompressed: they are transient, rewritten every
couple of minutes, and the compression time is pure overhead against the
inference we are trying to protect.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from qc.pose.schema import PoseTrack, PoseTrackError

logger = logging.getLogger(__name__)

# ~90 seconds of inference at the ~22 fps a 1080p video runs at, so a
# kill costs at most a minute or two of recomputation while the periodic
# write stays a fraction of a percent of total runtime.
CHECKPOINT_EVERY_FRAMES = 2000

_FINGERPRINT_KEY = "_checkpoint_fingerprint"
_PROGRESS_KEY = "_checkpoint_frames_done"


def partial_path(npz: Path) -> Path:
    """Where the partial for a given final ``.npz`` lives."""
    return Path(npz).with_name(Path(npz).name + ".partial")


class InferenceCheckpoint:
    """Periodic partial saves for one video's inference pass.

    Construct with the destination and a *fingerprint* — any dict of
    parameters that would change the resulting keypoints. Call
    :meth:`resume` before the decode loop, :meth:`maybe_save` inside it,
    and :meth:`clear` once the real ``.npz`` has been written.
    """

    def __init__(
        self,
        path: Path,
        fingerprint: Dict[str, Any],
        every: int = CHECKPOINT_EVERY_FRAMES,
    ) -> None:
        self.path = Path(path)
        self.fingerprint = dict(fingerprint)
        self.every = max(1, int(every))
        self._next_save = self.every

    # ── resume ────────────────────────────────────────────────────────

    def resume(self, n_frames: int) -> Tuple[Optional[PoseTrack], int]:
        """Return ``(track, first_unprocessed_frame)`` from a usable partial.

        ``(None, 0)`` when there is nothing to resume from — no partial,
        an unreadable one, or one written under different parameters.
        Never raises: a bad partial costs recomputation, and that is
        always preferable to failing a run over a cache file.
        """
        if not self.path.exists():
            return None, 0

        try:
            track = PoseTrack.load(self.path)
        except (PoseTrackError, OSError, ValueError, EOFError) as exc:
            logger.warning(
                "Ignoring unreadable inference checkpoint %s (%s); "
                "this video will be re-inferred from the start.",
                self.path.name, exc,
            )
            self._discard()
            return None, 0

        stored = track.meta.get(_FINGERPRINT_KEY)
        if stored != self.fingerprint:
            logger.info(
                "Discarding inference checkpoint %s: it was written under "
                "different settings and cannot be mixed with this run.",
                self.path.name,
            )
            self._discard()
            return None, 0

        if track.n_frames != n_frames:
            logger.info(
                "Discarding inference checkpoint %s: %d frames, this run "
                "expects %d.", self.path.name, track.n_frames, n_frames,
            )
            self._discard()
            return None, 0

        done = int(track.meta.get(_PROGRESS_KEY) or 0)
        if not 0 < done < n_frames:
            # Zero is nothing to resume; a full count means the pass had
            # finished and the final .npz should exist, so trust that path.
            self._discard()
            return None, 0

        self._next_save = done + self.every
        logger.info(
            "Resuming inference from checkpoint %s at frame %d of %d "
            "(%.1f%% already done).",
            self.path.name, done, n_frames, 100.0 * done / n_frames,
        )
        return track, done

    # ── save ──────────────────────────────────────────────────────────

    def maybe_save(self, track: PoseTrack, done: int) -> None:
        """Write a partial if *done* has crossed the next threshold."""
        if done < self._next_save:
            return
        self._next_save = done + self.every
        self.save(track, done)

    def save(self, track: PoseTrack, done: int) -> None:
        """Write the partial now. Failures are logged, never fatal."""
        previous = {
            key: track.meta.get(key)
            for key in (_FINGERPRINT_KEY, _PROGRESS_KEY)
            if key in track.meta
        }
        track.meta[_FINGERPRINT_KEY] = self.fingerprint
        track.meta[_PROGRESS_KEY] = int(done)
        try:
            # PoseTrack.save is atomic, so a kill during the write leaves
            # the previous partial intact rather than a torn file.
            track.save(self.path, compress=False)
            logger.debug("Checkpointed inference at frame %d -> %s",
                         done, self.path.name)
        except OSError as exc:
            # A full disk must not kill a run that is otherwise working.
            logger.warning("Could not write inference checkpoint %s: %s",
                           self.path.name, exc)
        finally:
            for key in (_FINGERPRINT_KEY, _PROGRESS_KEY):
                track.meta.pop(key, None)
            track.meta.update(previous)

    # ── cleanup ───────────────────────────────────────────────────────

    def clear(self) -> None:
        """Remove the partial — the real ``.npz`` supersedes it."""
        self._discard()

    def _discard(self) -> None:
        try:
            os.remove(self.path)
        except OSError:
            pass


def strip_checkpoint_meta(track: PoseTrack) -> None:
    """Drop checkpoint bookkeeping before a track is delivered."""
    for key in (_FINGERPRINT_KEY, _PROGRESS_KEY):
        track.meta.pop(key, None)
