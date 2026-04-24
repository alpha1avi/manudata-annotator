"""ManuData Annotator — Progress tracker with resume support.

Persists pipeline progress to a JSON file so interrupted runs can resume
without reprocessing already-completed frames.
"""

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class ProgressTracker:
    """Track and persist annotation pipeline progress.

    Progress is saved to a ``.progress`` JSON file in the output directory.
    On resume, previously processed frames are skipped automatically.
    """

    def __init__(self, output_dir: str, video_path: str) -> None:
        self.output_dir = Path(output_dir)
        self.video_path = video_path
        self.progress_file = self.output_dir / ".progress"
        self._state: Dict[str, Any] = self._default_state()

        if self.progress_file.exists():
            self._load()
        else:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self._save()

    def _default_state(self) -> Dict[str, Any]:
        return {
            "video_path": self.video_path,
            "current_stage": "init",
            "started_at": time.time(),
            "updated_at": time.time(),
            "processed_timestamps": [],
            "partial_results": {},
            "completed_stages": [],
        }

    def _load(self) -> None:
        """Load progress from the JSON file."""
        try:
            with open(self.progress_file, "r", encoding="utf-8") as f:
                self._state = json.load(f)
            logger.info(
                "Resumed progress: stage=%s, %d frames already processed",
                self._state.get("current_stage"),
                len(self._state.get("processed_timestamps", [])),
            )
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Corrupt progress file, starting fresh: %s", exc)
            self._state = self._default_state()
            self._save()

    def _save(self) -> None:
        """Persist current state to disk."""
        self._state["updated_at"] = time.time()
        with open(self.progress_file, "w", encoding="utf-8") as f:
            json.dump(self._state, f, indent=2)

    @property
    def current_stage(self) -> str:
        return self._state["current_stage"]

    @property
    def processed_timestamps(self) -> List[float]:
        return self._state["processed_timestamps"]

    def set_stage(self, stage: str) -> None:
        """Update the current pipeline stage."""
        self._state["current_stage"] = stage
        self._save()
        logger.info("Pipeline stage → %s", stage)

    def complete_stage(self, stage: str) -> None:
        """Mark a stage as completed."""
        if stage not in self._state["completed_stages"]:
            self._state["completed_stages"].append(stage)
        self._state["current_stage"] = stage
        self._save()
        logger.info("Stage completed: %s", stage)

    def is_stage_completed(self, stage: str) -> bool:
        return stage in self._state["completed_stages"]

    def mark_frame_processed(self, timestamp: float) -> None:
        """Record that a frame at the given timestamp has been processed."""
        if timestamp not in self._state["processed_timestamps"]:
            self._state["processed_timestamps"].append(timestamp)
        self._save()

    def mark_frames_processed(self, timestamps: List[float]) -> None:
        """Record multiple processed frame timestamps at once."""
        existing = set(self._state["processed_timestamps"])
        for ts in timestamps:
            existing.add(ts)
        self._state["processed_timestamps"] = sorted(existing)
        self._save()

    def is_frame_processed(self, timestamp: float) -> bool:
        return timestamp in self._state["processed_timestamps"]

    def filter_unprocessed(
        self, frames: List[tuple]
    ) -> List[tuple]:
        """Return only frames whose timestamps haven't been processed yet.

        Args:
            frames: List of (timestamp, filepath) tuples.

        Returns:
            Filtered list with only unprocessed frames.
        """
        processed = set(self._state["processed_timestamps"])
        unprocessed = [f for f in frames if f[0] not in processed]
        logger.info(
            "Resume filter: %d total, %d already done, %d remaining",
            len(frames), len(frames) - len(unprocessed), len(unprocessed),
        )
        return unprocessed

    def save_partial_result(self, key: str, value: Any) -> None:
        """Store an intermediate result under the given key."""
        self._state["partial_results"][key] = value
        self._save()

    def get_partial_result(self, key: str, default: Any = None) -> Optional[Any]:
        return self._state["partial_results"].get(key, default)

    def reset(self) -> None:
        """Discard all progress and start fresh."""
        self._state = self._default_state()
        self._save()
        logger.info("Progress reset")
