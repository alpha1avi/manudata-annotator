"""ManuData Annotator — Action Video Dataset (Training).

Loads bootstrap-annotated video segments as a PyTorch Dataset for
fine-tuning video classification models.

Usage:
    dataset = ActionVideoDataset("./output", min_confidence=0.6)
    video_tensor, label = dataset[0]
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

# ImageNet normalisation constants
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class ActionVideoDataset(Dataset):
    """PyTorch Dataset that loads video segments from bootstrap labels."""

    def __init__(
        self,
        data_dir: str,
        min_confidence: float = 0.6,
        num_frames: int = 16,
        frame_size: int = 224,
        augment: bool = False,
        use_metadata: bool = False,
    ) -> None:
        """Load bootstrap labels from *data_dir*.

        Steps:
            1. Find all ``*_labels.json`` files in *data_dir*.
            2. Load segments from each, filter by ``confidence > min_confidence``.
            3. Build class vocabulary from unique action labels.
            4. For each segment, store: video_path, start_time, end_time, action_class.
            5. If *use_metadata*: also load corresponding FrameAnnotation data.

        Args:
            data_dir: Directory containing annotated training data.
            min_confidence: Minimum confidence to include a segment.
            num_frames: Number of frames to sample per segment.
            frame_size: Spatial resolution (square) for each frame.
            augment: Whether to apply egocentric augmentations.
            use_metadata: Whether to load and return metadata feature vectors.
        """
        self.data_dir = Path(data_dir)
        self.min_confidence = min_confidence
        self.num_frames = num_frames
        self.frame_size = frame_size
        self.augment = augment
        self.use_metadata = use_metadata

        # Lazy import augmentations only when needed
        self._augmentor = None
        if self.augment:
            from training.augmentations import EgocentricAugmentation
            self._augmentor = EgocentricAugmentation()

        # Load samples and build class mapping
        self.samples: List[Dict[str, Any]] = []
        self.class_to_idx: Dict[str, int] = {}
        self.idx_to_class: Dict[int, str] = {}

        self._load_labels()
        self._build_class_mapping()

        logger.info(
            "Dataset loaded: %d samples, %d classes from %s",
            len(self.samples), len(self.class_to_idx), data_dir,
        )

    # ── loading ─────────────────────────────────────────────────────────

    def _load_labels(self) -> None:
        """Find all *_labels.json files and extract valid segments."""
        label_files = sorted(self.data_dir.rglob("*_labels.json"))
        if not label_files:
            logger.warning("No *_labels.json files found in %s", self.data_dir)
            return

        for label_path in label_files:
            try:
                with open(label_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning("Skipping %s: %s", label_path, exc)
                continue

            # Determine video path — look for video_path in metadata or infer
            video_path = data.get("video_path", "")
            if not video_path:
                # Try to find a video file alongside the labels
                parent = label_path.parent
                for ext in (".mp4", ".avi", ".mkv", ".mov"):
                    candidates = list(parent.glob(f"*{ext}"))
                    if candidates:
                        video_path = str(candidates[0])
                        break

            segments = data.get("segments", [])
            for seg in segments:
                confidence = seg.get("confidence_avg", seg.get("confidence", 0.0))
                if confidence < self.min_confidence:
                    continue

                action = seg.get("action", "unknown")
                if action in ("idle", "transition", "unknown"):
                    continue

                sample: Dict[str, Any] = {
                    "video_path": video_path,
                    "start_time": seg.get("start_time", 0.0),
                    "end_time": seg.get("end_time", 0.0),
                    "action": action,
                    "confidence": confidence,
                }

                if self.use_metadata:
                    sample["metadata"] = self._extract_metadata_features(seg)

                self.samples.append(sample)

        logger.info(
            "Found %d label files, %d segments passed confidence filter (>%.2f)",
            len(label_files), len(self.samples), self.min_confidence,
        )

    def _build_class_mapping(self) -> None:
        """Build class_to_idx and idx_to_class from unique actions."""
        actions = sorted({s["action"] for s in self.samples})
        self.class_to_idx = {a: i for i, a in enumerate(actions)}
        self.idx_to_class = {i: a for a, i in self.class_to_idx.items()}

    @staticmethod
    def _extract_metadata_features(seg: Dict) -> np.ndarray:
        """Extract a fixed-size feature vector from segment metadata."""
        features = [
            seg.get("confidence_avg", 0.0),
            seg.get("confidence_min", 0.0),
            seg.get("duration", 0.0),
            seg.get("frames_analyzed", 0),
            1.0 if seg.get("hand_used") == "right" else 0.0,
            1.0 if seg.get("hand_used") == "left" else 0.0,
            1.0 if seg.get("hand_used") == "both" else 0.0,
            len(seg.get("objects_involved", [])),
            len(seg.get("manipulation_phases", [])),
        ]
        traj = seg.get("trajectory_summary", {})
        features.extend([
            traj.get("path_length", 0.0),
            traj.get("max_velocity", 0.0),
            traj.get("grasp_events", 0),
        ])
        return np.array(features, dtype=np.float32)

    # ── Dataset interface ───────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        """Return ``(video_tensor, label)`` or ``(video_tensor, metadata, label)``."""
        sample = self.samples[idx]
        label = self.class_to_idx[sample["action"]]

        # Load frames from video
        frames = self._load_video_frames(
            sample["video_path"],
            sample["start_time"],
            sample["end_time"],
        )

        # Apply augmentations
        if self.augment and self._augmentor is not None:
            frames = self._augmentor(frames)

        # Normalise and convert to tensor  (C, T, H, W)
        video_tensor = self._frames_to_tensor(frames)

        if self.use_metadata and "metadata" in sample:
            meta_tensor = torch.from_numpy(sample["metadata"])
            return video_tensor, meta_tensor, label

        return video_tensor, label

    # ── video loading ───────────────────────────────────────────────────

    def _load_video_frames(
        self, video_path: str, start_time: float, end_time: float,
    ) -> List[np.ndarray]:
        """Extract *num_frames* evenly spaced between *start_time* and *end_time*."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.warning("Cannot open video %s — returning black frames", video_path)
            return [
                np.zeros((self.frame_size, self.frame_size, 3), dtype=np.uint8)
                for _ in range(self.num_frames)
            ]

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        start_frame = int(start_time * fps)
        end_frame = int(end_time * fps)
        total_frames = max(1, end_frame - start_frame)

        # Evenly spaced indices
        if total_frames >= self.num_frames:
            indices = np.linspace(start_frame, end_frame - 1, self.num_frames, dtype=int)
        else:
            # Repeat last frame if segment is too short
            indices = list(range(start_frame, end_frame))
            while len(indices) < self.num_frames:
                indices.append(indices[-1])
            indices = np.array(indices[:self.num_frames])

        frames: List[np.ndarray] = []
        for frame_idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (self.frame_size, self.frame_size))
            else:
                frame = np.zeros((self.frame_size, self.frame_size, 3), dtype=np.uint8)
            frames.append(frame)

        cap.release()
        return frames

    def _frames_to_tensor(self, frames: List[np.ndarray]) -> torch.Tensor:
        """Normalise and stack frames into a (C, T, H, W) tensor."""
        normalised: List[np.ndarray] = []
        for frame in frames:
            f = frame.astype(np.float32) / 255.0
            f = (f - IMAGENET_MEAN) / IMAGENET_STD
            normalised.append(f)

        # (T, H, W, C) → (C, T, H, W)
        stacked = np.stack(normalised, axis=0)                # (T, H, W, C)
        tensor = torch.from_numpy(stacked).permute(3, 0, 1, 2)  # (C, T, H, W)
        return tensor.contiguous()

    # ── utilities ───────────────────────────────────────────────────────

    def get_class_distribution(self) -> Dict[str, int]:
        """Return count per class — useful for class-weighted loss."""
        dist: Dict[str, int] = {cls: 0 for cls in self.class_to_idx}
        for s in self.samples:
            dist[s["action"]] += 1
        return dist

    def save_class_mapping(self, path: str) -> None:
        """Save class_mapping.json."""
        mapping = {
            "class_to_idx": self.class_to_idx,
            "idx_to_class": {str(k): v for k, v in self.idx_to_class.items()},
            "num_classes": len(self.class_to_idx),
        }
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(mapping, f, indent=2)
        logger.info("Class mapping saved to %s (%d classes)", path, len(self.class_to_idx))

    @property
    def num_classes(self) -> int:
        return len(self.class_to_idx)

    @property
    def metadata_dim(self) -> int:
        """Dimension of the metadata feature vector."""
        return 12


# ── standalone test ─────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    import tempfile

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from utils.logging_config import setup_logging

    setup_logging(verbose=True)

    # Create a synthetic label file for testing
    with tempfile.TemporaryDirectory() as tmpdir:
        labels = {
            "video_path": "",
            "segments": [
                {
                    "action": "pick_up",
                    "confidence_avg": 0.85,
                    "start_time": 0.0,
                    "end_time": 3.0,
                    "duration": 3.0,
                    "hand_used": "right",
                    "objects_involved": ["wrench"],
                    "manipulation_phases": ["reach", "grasp"],
                    "trajectory_summary": {},
                },
                {
                    "action": "tighten",
                    "confidence_avg": 0.90,
                    "start_time": 3.0,
                    "end_time": 7.0,
                    "duration": 4.0,
                    "hand_used": "right",
                    "objects_involved": ["bolt"],
                    "manipulation_phases": ["manipulate"],
                    "trajectory_summary": {"path_length": 15.0},
                },
                {
                    "action": "idle",
                    "confidence_avg": 0.95,
                    "start_time": 7.0,
                    "end_time": 10.0,
                },
            ],
        }
        label_path = os.path.join(tmpdir, "test_labels.json")
        with open(label_path, "w", encoding="utf-8") as f:
            json.dump(labels, f)

        ds = ActionVideoDataset(tmpdir, min_confidence=0.5)
        print(f"\nDataset: {len(ds)} samples, {ds.num_classes} classes")
        print(f"Classes: {ds.class_to_idx}")
        print(f"Distribution: {ds.get_class_distribution()}")

        if len(ds) > 0:
            item = ds[0]
            if isinstance(item, tuple) and len(item) == 2:
                tensor, label = item
                print(f"Sample shape: {tensor.shape}, label: {label}")
            print("Dataset test PASSED")
        else:
            print("No samples loaded (expected with no video file)")
