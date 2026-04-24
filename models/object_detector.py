"""ManuData Annotator — Object Detection (YOLO-World wrapper).

Open-vocabulary object detection using YOLO-World from the ultralytics
library.  Detects manufacturing objects (tools, parts, fixtures) with a
configurable class vocabulary.
"""

import logging
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ── dataclass ─────────────────────────────────────────────────────────


@dataclass
class ObjectDetection:
    """Single detected object."""

    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]                      # x1, y1, x2, y2  (pixels)
    bbox_normalized: Tuple[float, float, float, float]   # 0-1 relative
    center: Tuple[int, int]                               # pixel centre


# ── detector ──────────────────────────────────────────────────────────


class ObjectDetector:
    """YOLO-World open-vocabulary object detector."""

    MODEL_MAP = {
        "s": "yolov8s-worldv2",
        "m": "yolov8m-worldv2",
        "l": "yolov8l-worldv2",
    }

    def __init__(
        self,
        model_size: str = "s",
        custom_classes: Optional[List[str]] = None,
        confidence_threshold: float = 0.3,
    ) -> None:
        """Load YOLO-World model.

        Args:
            model_size: ``"s"`` | ``"m"`` | ``"l"`` (default ``"s"``).
            custom_classes: List of class names for open-vocab detection.
                Falls back to COCO classes if ``None``.
            confidence_threshold: Minimum confidence to keep a detection.
        """
        from ultralytics import YOLO

        model_name = self.MODEL_MAP.get(model_size, self.MODEL_MAP["s"])
        logger.info("Loading YOLO-World model: %s …", model_name)

        t0 = time.time()
        self.model = YOLO(f"{model_name}.pt")
        load_time = time.time() - t0
        logger.info("YOLO-World loaded in %.2fs", load_time)

        self.confidence_threshold = confidence_threshold

        if custom_classes:
            self.set_classes(custom_classes)
        else:
            self._classes: Optional[List[str]] = None

    # ── public API ────────────────────────────────────────────────────

    def set_classes(self, classes: List[str]) -> None:
        """Update the class vocabulary at runtime.

        Args:
            classes: New list of class names for open-vocabulary detection.
        """
        self._classes = list(classes)
        self.model.set_classes(self._classes)
        logger.info("Object classes updated (%d classes)", len(self._classes))

    def detect(self, frame: np.ndarray) -> List[ObjectDetection]:
        """Run detection on a single frame.

        Args:
            frame: BGR image as NumPy array.

        Returns:
            List of :class:`ObjectDetection` sorted by confidence descending,
            filtered above ``confidence_threshold``.
        """
        t0 = time.time()
        results = self.model.predict(
            frame, conf=self.confidence_threshold, verbose=False
        )
        elapsed_ms = (time.time() - t0) * 1000
        logger.debug("Object detection: %.1fms", elapsed_ms)

        detections = self._parse_results(results, frame.shape[:2])
        return detections

    def detect_batch(
        self, frames: List[np.ndarray]
    ) -> List[List[ObjectDetection]]:
        """Run detection on a batch of frames.

        Args:
            frames: List of BGR images.

        Returns:
            List of detection lists, one per input frame.
        """
        t0 = time.time()
        batch_results = self.model.predict(
            frames, conf=self.confidence_threshold, verbose=False
        )
        elapsed_ms = (time.time() - t0) * 1000
        logger.debug(
            "Batch detection (%d frames): %.1fms (%.1fms/frame)",
            len(frames), elapsed_ms, elapsed_ms / max(len(frames), 1),
        )

        output: List[List[ObjectDetection]] = []
        for idx, result in enumerate(batch_results):
            h, w = frames[idx].shape[:2]
            output.append(self._parse_results([result], (h, w)))
        return output

    # ── internals ─────────────────────────────────────────────────────

    def _parse_results(
        self, results, frame_hw: Tuple[int, int]
    ) -> List[ObjectDetection]:
        """Convert ultralytics results to :class:`ObjectDetection` list."""
        h, w = frame_hw
        detections: List[ObjectDetection] = []

        for result in results:
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue

            for i in range(len(boxes)):
                conf = float(boxes.conf[i])
                if conf < self.confidence_threshold:
                    continue

                cls_id = int(boxes.cls[i])
                class_name = result.names.get(cls_id, f"class_{cls_id}")

                x1, y1, x2, y2 = boxes.xyxy[i].tolist()
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                det = ObjectDetection(
                    class_name=class_name,
                    confidence=round(conf, 4),
                    bbox=(x1, y1, x2, y2),
                    bbox_normalized=(
                        round(x1 / w, 4),
                        round(y1 / h, 4),
                        round(x2 / w, 4),
                        round(y2 / h, 4),
                    ),
                    center=((x1 + x2) // 2, (y1 + y2) // 2),
                )
                detections.append(det)

        detections.sort(key=lambda d: d.confidence, reverse=True)
        return detections


# ── standalone test ───────────────────────────────────────────────────

if __name__ == "__main__":
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from config import AnnotatorConfig, DEFAULT_OBJECT_TAXONOMY
    from utils.image_utils import load_frame
    from utils.logging_config import setup_logging

    setup_logging(verbose=True)

    if len(sys.argv) < 2:
        print("Usage: python object_detector.py <image_path>")
        sys.exit(1)

    frame = load_frame(sys.argv[1])
    detector = ObjectDetector(model_size="s", custom_classes=DEFAULT_OBJECT_TAXONOMY)
    dets = detector.detect(frame)

    print(f"\nDetected {len(dets)} objects:")
    for d in dets:
        print(f"  {d.class_name:20s}  conf={d.confidence:.3f}  bbox={d.bbox}  center={d.center}")
