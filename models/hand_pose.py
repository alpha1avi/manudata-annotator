"""ManuData Annotator — Hand Pose Estimation (MediaPipe Hands wrapper).

Detects hands, extracts 21 keypoints per hand, computes bounding boxes,
and classifies grasp types from keypoint geometry.
"""

import logging
import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np
import mediapipe as mp

logger = logging.getLogger(__name__)

# MediaPipe hand landmark indices
WRIST = 0
THUMB_TIP = 4
INDEX_TIP = 8
MIDDLE_TIP = 12
RING_TIP = 16
PINKY_TIP = 20

# MCP / PIP / DIP indices per finger (for curl computation)
FINGER_JOINTS = {
    "thumb":  [1, 2, 3, 4],
    "index":  [5, 6, 7, 8],
    "middle": [9, 10, 11, 12],
    "ring":   [13, 14, 15, 16],
    "pinky":  [17, 18, 19, 20],
}

FINGERTIP_IDS = [THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP]


# ── dataclasses ───────────────────────────────────────────────────────


@dataclass
class HandDetection:
    """Single hand detection result."""

    handedness: str  # "left" | "right"
    confidence: float
    keypoints_2d: List[Tuple[float, float]]    # 21 keypoints, normalised 0-1
    keypoints_pixel: List[Tuple[int, int]]      # 21 keypoints, pixel coords
    bbox: Tuple[int, int, int, int]              # x1, y1, x2, y2


@dataclass
class HandPoseResult:
    """Aggregated result for all hands in a single frame."""

    hands: List[HandDetection] = field(default_factory=list)
    num_hands: int = 0
    hand_visibility: str = "no_hands"  # both_full | left_only | right_only | partial | no_hands


# ── estimator ─────────────────────────────────────────────────────────


class HandPoseEstimator:
    """MediaPipe Hands wrapper for hand pose estimation and grasp classification."""

    def __init__(
        self,
        max_hands: int = 2,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        self._mp_hands = mp.solutions.hands
        self._hands = self._mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=max_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        logger.info(
            "HandPoseEstimator initialised (max_hands=%d, det_conf=%.2f)",
            max_hands, min_detection_confidence,
        )

    # ── detection ─────────────────────────────────────────────────────

    def detect(self, frame: np.ndarray) -> HandPoseResult:
        """Run MediaPipe Hands on a single BGR frame.

        Args:
            frame: BGR image (H, W, 3).

        Returns:
            :class:`HandPoseResult` with detected hands and visibility info.
        """
        h, w = frame.shape[:2]

        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self._hands.process(rgb)
        except Exception as exc:
            logger.warning("MediaPipe hand detection failed: %s", exc)
            return HandPoseResult()

        if not results.multi_hand_landmarks:
            return HandPoseResult(hands=[], num_hands=0, hand_visibility="no_hands")

        detections: List[HandDetection] = []

        for idx, (hand_lms, hand_cls) in enumerate(
            zip(results.multi_hand_landmarks, results.multi_handedness)
        ):
            # Handedness label and confidence
            cls_info = hand_cls.classification[0]
            # MediaPipe mirrors: "Left" in image is actually the right hand
            handedness = cls_info.label.lower()
            confidence = float(cls_info.score)

            # Extract keypoints
            kp_norm: List[Tuple[float, float]] = []
            kp_pixel: List[Tuple[int, int]] = []
            xs, ys = [], []

            for lm in hand_lms.landmark:
                kp_norm.append((float(lm.x), float(lm.y)))
                px, py = int(lm.x * w), int(lm.y * h)
                kp_pixel.append((px, py))
                xs.append(px)
                ys.append(py)

            # Bounding box with small margin
            margin = 10
            x1 = max(0, min(xs) - margin)
            y1 = max(0, min(ys) - margin)
            x2 = min(w, max(xs) + margin)
            y2 = min(h, max(ys) + margin)

            det = HandDetection(
                handedness=handedness,
                confidence=confidence,
                keypoints_2d=kp_norm,
                keypoints_pixel=kp_pixel,
                bbox=(x1, y1, x2, y2),
            )
            detections.append(det)

        visibility = self._classify_visibility(detections)

        return HandPoseResult(
            hands=detections,
            num_hands=len(detections),
            hand_visibility=visibility,
        )

    def detect_batch(self, frames: List[np.ndarray]) -> List[HandPoseResult]:
        """Process multiple frames sequentially.

        Args:
            frames: List of BGR images.

        Returns:
            List of :class:`HandPoseResult`, one per frame.
        """
        return [self.detect(f) for f in frames]

    # ── grasp classification ──────────────────────────────────────────

    def get_grasp_type(self, hand: HandDetection) -> Tuple[str, float]:
        """Classify grasp type from hand keypoint geometry.

        Rules (evaluated in order):
            1. All fingers extended and spread → ``"no_contact"``
            2. Thumb + index close, others extended → ``"pinch"``
            3. All fingers curled tight, thumb wrapped → ``"power"``
            4. Thumb pressing against side of index → ``"lateral"``
            5. Fingers curled in hook, thumb relaxed → ``"hook"``
            6. Fingers spread around large area → ``"spherical"``

        Args:
            hand: A :class:`HandDetection` with pixel-space keypoints.

        Returns:
            ``(grasp_type, confidence)`` tuple.
        """
        kp = hand.keypoints_pixel
        if len(kp) < 21:
            return "unknown", 0.0

        curl_scores = self._finger_curl_scores(kp)
        thumb_curl = curl_scores["thumb"]
        finger_curls = [curl_scores[f] for f in ["index", "middle", "ring", "pinky"]]
        mean_curl = sum(finger_curls) / 4.0

        thumb_index_dist = self._dist(kp[THUMB_TIP], kp[INDEX_TIP])
        hand_span = self._dist(kp[WRIST], kp[MIDDLE_TIP])
        norm_ti_dist = thumb_index_dist / (hand_span + 1e-6)

        # Finger spread: mean distance between adjacent fingertips
        tip_ids = [INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP]
        spreads = [
            self._dist(kp[tip_ids[i]], kp[tip_ids[i + 1]])
            for i in range(len(tip_ids) - 1)
        ]
        mean_spread = sum(spreads) / (len(spreads) * (hand_span + 1e-6))

        # 1. No contact — all fingers extended and spread
        if mean_curl < 0.3 and mean_spread > 0.25:
            return "no_contact", min(1.0, 0.5 + (0.3 - mean_curl) + mean_spread)

        # 2. Pinch — thumb + index close, others relatively extended
        other_curl = sum(curl_scores[f] for f in ["middle", "ring", "pinky"]) / 3.0
        if norm_ti_dist < 0.25 and other_curl < 0.6:
            conf = min(1.0, (0.25 - norm_ti_dist) * 4)
            return "pinch", max(0.4, conf)

        # 3. Power — all fingers curled tight
        if mean_curl > 0.7 and thumb_curl > 0.5:
            conf = min(1.0, mean_curl)
            return "power", max(0.4, conf)

        # 4. Lateral — thumb pressing against side of index
        thumb_tip = kp[THUMB_TIP]
        index_mcp = kp[5]
        index_pip = kp[6]
        lateral_dist = self._point_to_segment_dist(thumb_tip, index_mcp, index_pip)
        norm_lateral = lateral_dist / (hand_span + 1e-6)
        if norm_lateral < 0.12 and curl_scores["index"] > 0.4:
            conf = min(1.0, (0.12 - norm_lateral) * 8)
            return "lateral", max(0.4, conf)

        # 5. Hook — fingers curled, thumb relaxed
        if mean_curl > 0.5 and thumb_curl < 0.3:
            conf = min(1.0, mean_curl * 0.8)
            return "hook", max(0.4, conf)

        # 6. Spherical — fingers spread around large area
        if mean_curl > 0.3 and mean_spread > 0.2:
            conf = min(1.0, mean_spread * 2)
            return "spherical", max(0.4, conf)

        return "unknown", 0.3

    # ── resource cleanup ──────────────────────────────────────────────

    def close(self) -> None:
        """Release MediaPipe resources."""
        self._hands.close()
        logger.info("HandPoseEstimator closed")

    # ── private helpers ───────────────────────────────────────────────

    @staticmethod
    def _classify_visibility(detections: List[HandDetection]) -> str:
        if not detections:
            return "no_hands"
        labels = {d.handedness for d in detections}
        if "left" in labels and "right" in labels:
            return "both_full"
        if "left" in labels:
            return "left_only"
        if "right" in labels:
            return "right_only"
        return "partial"

    @staticmethod
    def _dist(p1: Tuple[int, int], p2: Tuple[int, int]) -> float:
        return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

    @staticmethod
    def _point_to_segment_dist(
        pt: Tuple[int, int],
        seg_a: Tuple[int, int],
        seg_b: Tuple[int, int],
    ) -> float:
        """Distance from point *pt* to line segment *seg_a*–*seg_b*."""
        ax, ay = seg_a
        bx, by = seg_b
        px, py = pt
        dx, dy = bx - ax, by - ay
        seg_len_sq = dx * dx + dy * dy
        if seg_len_sq == 0:
            return math.sqrt((px - ax) ** 2 + (py - ay) ** 2)
        t = max(0, min(1, ((px - ax) * dx + (py - ay) * dy) / seg_len_sq))
        proj_x = ax + t * dx
        proj_y = ay + t * dy
        return math.sqrt((px - proj_x) ** 2 + (py - proj_y) ** 2)

    @staticmethod
    def _angle_at_joint(
        a: Tuple[int, int], b: Tuple[int, int], c: Tuple[int, int]
    ) -> float:
        """Angle in radians at point *b* formed by segments b→a and b→c."""
        ba = (a[0] - b[0], a[1] - b[1])
        bc = (c[0] - b[0], c[1] - b[1])
        dot = ba[0] * bc[0] + ba[1] * bc[1]
        mag_ba = math.sqrt(ba[0] ** 2 + ba[1] ** 2) + 1e-8
        mag_bc = math.sqrt(bc[0] ** 2 + bc[1] ** 2) + 1e-8
        cos_angle = max(-1.0, min(1.0, dot / (mag_ba * mag_bc)))
        return math.acos(cos_angle)

    def _finger_curl_scores(
        self, kp: List[Tuple[int, int]]
    ) -> dict:
        """Compute curl score (0 = extended, 1 = fully curled) per finger.

        Uses the angle at each joint (MCP, PIP, DIP). A straight finger
        has angles near pi; a curled finger has small angles.
        """
        scores = {}
        for finger, joints in FINGER_JOINTS.items():
            if finger == "thumb":
                # Thumb only has 3 meaningful joints for curl
                angles = [
                    self._angle_at_joint(kp[joints[0]], kp[joints[1]], kp[joints[2]]),
                    self._angle_at_joint(kp[joints[1]], kp[joints[2]], kp[joints[3]]),
                ]
            else:
                angles = [
                    self._angle_at_joint(kp[joints[0]], kp[joints[1]], kp[joints[2]]),
                    self._angle_at_joint(kp[joints[1]], kp[joints[2]], kp[joints[3]]),
                ]
            # Normalise: pi = extended (0), 0 = curled (1)
            curl = 1.0 - (sum(angles) / (len(angles) * math.pi))
            scores[finger] = max(0.0, min(1.0, curl))
        return scores


# ── standalone test ───────────────────────────────────────────────────

if __name__ == "__main__":
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from utils.logging_config import setup_logging
    from utils.image_utils import load_frame

    setup_logging(verbose=True)

    if len(sys.argv) < 2:
        print("Usage: python hand_pose.py <image_path>")
        sys.exit(1)

    frame = load_frame(sys.argv[1])
    estimator = HandPoseEstimator()

    result = estimator.detect(frame)
    print(f"\nHands detected: {result.num_hands}")
    print(f"Visibility: {result.hand_visibility}")

    for i, hand in enumerate(result.hands):
        print(f"\n  Hand {i}: {hand.handedness} (conf={hand.confidence:.2f})")
        print(f"    BBox: {hand.bbox}")
        grasp, g_conf = estimator.get_grasp_type(hand)
        print(f"    Grasp: {grasp} (conf={g_conf:.2f})")
        print(f"    Wrist: {hand.keypoints_pixel[WRIST]}")

    estimator.close()
