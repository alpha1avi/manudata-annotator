"""ManuData Annotator — Hand-Object Interaction Detection.

Rule-based detector that determines which hand is interacting with which
object using bounding-box IoU, depth comparison, and fingertip-in-bbox
checks.
"""

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

from models.hand_pose import (
    HandDetection,
    HandPoseEstimator,
    HandPoseResult,
    FINGERTIP_IDS,
    WRIST,
)
from models.object_detector import ObjectDetection
from models.depth_estimator import DepthResult

logger = logging.getLogger(__name__)


# ── dataclass ─────────────────────────────────────────────────────────


@dataclass
class Interaction:
    """Detected interaction between one hand and one object."""

    hand: str                          # "left" | "right"
    object: ObjectDetection
    contact: bool
    contact_score: float               # 0-1 weighted score
    iou: float
    depth_diff: Optional[float]        # None if depth unavailable
    fingertips_inside: int             # count of fingertips inside object bbox
    grasp_type: str                    # e.g. "pinch", "power", … (if contact)
    grasp_confidence: float


# ── detector ──────────────────────────────────────────────────────────


class HandObjectInteractionDetector:
    """Detect hand-object interactions via geometry, depth, and fingertip overlap."""

    # Contact-score weights
    W_IOU = 0.4
    W_DEPTH = 0.25
    W_FINGERTIP = 0.35

    # Thresholds
    CONTACT_SCORE_THRESHOLD = 0.3

    def __init__(self) -> None:
        self._grasp_estimator = HandPoseEstimator(max_hands=2)

    def detect_interactions(
        self,
        hand_result: HandPoseResult,
        objects: List[ObjectDetection],
        depth_result: Optional[DepthResult] = None,
    ) -> List[Interaction]:
        """Determine which hand is interacting with which object.

        For every ``(hand, object)`` pair the following signals are
        computed and combined into a weighted contact score:

        1. **IoU** between hand bbox and object bbox.
        2. **Depth difference** (if available) between wrist and object
           centre — small difference suggests contact.
        3. **Fingertip count** — number of the 5 fingertips inside the
           object's bounding box.

        Only interactions with ``contact_score >= CONTACT_SCORE_THRESHOLD``
        are returned.

        Args:
            hand_result: Output of :class:`HandPoseEstimator.detect`.
            objects: Detected objects in the same frame.
            depth_result: Optional depth map for the frame.

        Returns:
            List of :class:`Interaction` sorted by contact score descending.
        """
        if hand_result.num_hands == 0 or not objects:
            return []

        interactions: List[Interaction] = []

        for hand in hand_result.hands:
            for obj in objects:
                iou = self.compute_iou(hand.bbox, obj.bbox)
                depth_diff = self._compute_depth_diff(hand, obj, depth_result)
                n_tips = self.fingertips_in_bbox(hand, obj.bbox)

                # --- component scores (each 0-1) ---
                iou_score = min(1.0, iou / 0.3) if iou > 0 else 0.0

                if depth_diff is not None:
                    # Closer depth → higher score.  diff=0 → 1, diff>=0.1 → 0
                    depth_score = max(0.0, 1.0 - depth_diff / 0.1)
                else:
                    depth_score = 0.0

                tip_score = min(1.0, n_tips / 3.0)

                # --- weighted contact score ---
                if depth_diff is not None:
                    contact_score = (
                        self.W_IOU * iou_score
                        + self.W_DEPTH * depth_score
                        + self.W_FINGERTIP * tip_score
                    )
                else:
                    # Redistribute depth weight equally
                    w_iou = self.W_IOU + self.W_DEPTH / 2
                    w_tip = self.W_FINGERTIP + self.W_DEPTH / 2
                    contact_score = w_iou * iou_score + w_tip * tip_score

                contact_score = round(min(1.0, contact_score), 4)

                # --- special rules ---
                # IoU > 0.3 alone is a strong signal
                if iou > 0.3:
                    contact_score = max(contact_score, 0.5)
                # IoU > 0.1 AND depth close → confirmed contact
                if iou > 0.1 and depth_diff is not None and depth_diff < 0.05:
                    contact_score = max(contact_score, 0.6)
                # Any fingertip inside → strong signal
                if n_tips >= 1:
                    contact_score = max(contact_score, 0.4)

                if contact_score < self.CONTACT_SCORE_THRESHOLD:
                    continue

                is_contact = contact_score >= self.CONTACT_SCORE_THRESHOLD

                # Grasp classification (only if contact)
                if is_contact:
                    grasp_type, grasp_conf = self._grasp_estimator.get_grasp_type(hand)
                else:
                    grasp_type, grasp_conf = "none", 0.0

                interaction = Interaction(
                    hand=hand.handedness,
                    object=obj,
                    contact=is_contact,
                    contact_score=contact_score,
                    iou=round(iou, 4),
                    depth_diff=round(depth_diff, 4) if depth_diff is not None else None,
                    fingertips_inside=n_tips,
                    grasp_type=grasp_type,
                    grasp_confidence=round(grasp_conf, 3),
                )
                interactions.append(interaction)

        interactions.sort(key=lambda i: i.contact_score, reverse=True)

        logger.debug(
            "Interactions: %d hands x %d objects → %d contacts",
            hand_result.num_hands, len(objects), len(interactions),
        )
        return interactions

    # ── geometry helpers ──────────────────────────────────────────────

    @staticmethod
    def compute_iou(
        bbox1: Tuple[int, int, int, int],
        bbox2: Tuple[int, int, int, int],
    ) -> float:
        """Standard IoU between two ``(x1, y1, x2, y2)`` bounding boxes."""
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])

        inter_w = max(0, x2 - x1)
        inter_h = max(0, y2 - y1)
        inter_area = inter_w * inter_h

        area1 = max(0, bbox1[2] - bbox1[0]) * max(0, bbox1[3] - bbox1[1])
        area2 = max(0, bbox2[2] - bbox2[0]) * max(0, bbox2[3] - bbox2[1])
        union = area1 + area2 - inter_area

        if union <= 0:
            return 0.0
        return inter_area / union

    @staticmethod
    def fingertips_in_bbox(
        hand: HandDetection,
        bbox: Tuple[int, int, int, int],
    ) -> int:
        """Count how many of the 5 fingertip landmarks fall inside *bbox*.

        Args:
            hand: Detected hand with pixel-space keypoints.
            bbox: ``(x1, y1, x2, y2)`` object bounding box.

        Returns:
            Count (0-5).
        """
        x1, y1, x2, y2 = bbox
        count = 0
        for tip_id in FINGERTIP_IDS:
            px, py = hand.keypoints_pixel[tip_id]
            if x1 <= px <= x2 and y1 <= py <= y2:
                count += 1
        return count

    # ── depth helper ──────────────────────────────────────────────────

    @staticmethod
    def _compute_depth_diff(
        hand: HandDetection,
        obj: ObjectDetection,
        depth_result: Optional[DepthResult],
    ) -> Optional[float]:
        """Compute absolute depth difference between hand wrist and object centre."""
        if depth_result is None:
            return None

        wrist = hand.keypoints_pixel[WRIST]
        centre = obj.center

        hand_depth = depth_result.depth_at_point(wrist[0], wrist[1])
        obj_depth = depth_result.depth_at_point(centre[0], centre[1])

        return abs(hand_depth - obj_depth)

    def close(self) -> None:
        """Release resources."""
        self._grasp_estimator.close()


# ── standalone test ───────────────────────────────────────────────────

if __name__ == "__main__":
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from utils.image_utils import load_frame
    from utils.logging_config import setup_logging

    setup_logging(verbose=True)

    if len(sys.argv) < 2:
        print("Usage: python interaction_detector.py <image_path>")
        sys.exit(1)

    frame = load_frame(sys.argv[1])

    # Run hand detection
    from models.hand_pose import HandPoseEstimator

    hand_est = HandPoseEstimator()
    hand_result = hand_est.detect(frame)
    print(f"Hands: {hand_result.num_hands} ({hand_result.hand_visibility})")

    # Run object detection
    from config import DEFAULT_OBJECT_TAXONOMY
    from models.object_detector import ObjectDetector

    obj_det = ObjectDetector(model_size="s", custom_classes=DEFAULT_OBJECT_TAXONOMY)
    objects = obj_det.detect(frame)
    print(f"Objects: {len(objects)}")
    for o in objects:
        print(f"  {o.class_name} (conf={o.confidence:.3f})")

    # Run interaction detection
    detector = HandObjectInteractionDetector()
    interactions = detector.detect_interactions(hand_result, objects)

    print(f"\nInteractions: {len(interactions)}")
    for ix in interactions:
        print(
            f"  {ix.hand} hand ↔ {ix.object.class_name}: "
            f"contact={ix.contact} score={ix.contact_score:.3f} "
            f"iou={ix.iou:.3f} tips={ix.fingertips_inside} "
            f"grasp={ix.grasp_type}({ix.grasp_confidence:.2f})"
        )

    hand_est.close()
    detector.close()
