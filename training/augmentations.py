"""ManuData Annotator — Egocentric Video Augmentations.

Augmentations designed specifically for egocentric (helmet-cam)
manufacturing video.  All transforms are applied *consistently* across
every frame in a clip to preserve temporal coherence.

Usage:
    aug = EgocentricAugmentation()
    augmented_frames = aug(frames)   # list of np.ndarray (H, W, 3)
"""

import logging
import random
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class EgocentricAugmentation:
    """Augmentations specific to egocentric video.

    Each call randomly selects a subset of transforms and applies
    them **identically** to every frame in the clip.

    Supported transforms:
        - Random rotation ±15° (head tilt)
        - Random brightness ±20% and contrast ±15% (factory lighting)
        - Centre-biased random crop (action is usually centre-bottom)
        - Temporal jitter ±2 frames (applied externally)
        - Horizontal flip with 50% probability (swaps hand labels)
        - Colour jitter: hue ±10, saturation ±20%
        - Random Gaussian blur (slight focus issues)

    NOT applied:
        - Vertical flip (physically impossible in egocentric)
        - Centre cutout / erasing (destroys hand/action region)
        - Extreme rotation > 20° (unrealistic head motion)
    """

    def __init__(
        self,
        rotation_range: float = 15.0,
        brightness_range: float = 0.20,
        contrast_range: float = 0.15,
        crop_scale: Tuple[float, float] = (0.85, 1.0),
        flip_prob: float = 0.5,
        hue_range: int = 10,
        saturation_range: float = 0.20,
        blur_prob: float = 0.3,
        blur_kernel_range: Tuple[int, int] = (3, 7),
    ) -> None:
        self.rotation_range = rotation_range
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.crop_scale = crop_scale
        self.flip_prob = flip_prob
        self.hue_range = hue_range
        self.saturation_range = saturation_range
        self.blur_prob = blur_prob
        self.blur_kernel_range = blur_kernel_range

        # Track whether the last call applied a horizontal flip
        # (caller may need to swap left/right hand labels)
        self.did_flip: bool = False

    def __call__(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """Apply a random subset of augmentations to all frames consistently.

        Args:
            frames: List of (H, W, 3) uint8 arrays (RGB).

        Returns:
            List of augmented frames, same length and dtype.
        """
        if not frames:
            return frames

        self.did_flip = False
        h, w = frames[0].shape[:2]

        # ── decide which augmentations to apply (each with 50% chance) ──
        do_rotate = random.random() < 0.5
        do_brightness_contrast = random.random() < 0.5
        do_crop = random.random() < 0.5
        do_flip = random.random() < self.flip_prob
        do_colour = random.random() < 0.5
        do_blur = random.random() < self.blur_prob

        # ── sample parameters once ──────────────────────────────────────
        angle = random.uniform(-self.rotation_range, self.rotation_range) if do_rotate else 0.0
        brightness_delta = random.uniform(-self.brightness_range, self.brightness_range) if do_brightness_contrast else 0.0
        contrast_factor = 1.0 + random.uniform(-self.contrast_range, self.contrast_range) if do_brightness_contrast else 1.0

        # Centre-biased crop: offset drawn from Gaussian centred on image centre
        crop_scale = random.uniform(*self.crop_scale) if do_crop else 1.0
        crop_h, crop_w = int(h * crop_scale), int(w * crop_scale)
        if do_crop and crop_scale < 1.0:
            # Gaussian centred slightly below centre (egocentric bias)
            cx = w // 2 + int(random.gauss(0, w * 0.05))
            cy = h // 2 + int(random.gauss(h * 0.05, h * 0.05))
            x1 = max(0, min(cx - crop_w // 2, w - crop_w))
            y1 = max(0, min(cy - crop_h // 2, h - crop_h))
        else:
            x1, y1 = 0, 0
            crop_h, crop_w = h, w

        hue_delta = random.randint(-self.hue_range, self.hue_range) if do_colour else 0
        sat_factor = 1.0 + random.uniform(-self.saturation_range, self.saturation_range) if do_colour else 1.0

        blur_k = random.choice(range(self.blur_kernel_range[0], self.blur_kernel_range[1] + 1, 2)) if do_blur else 0

        if do_flip:
            self.did_flip = True

        # ── apply to every frame ────────────────────────────────────────
        augmented: List[np.ndarray] = []
        for frame in frames:
            f = frame.copy()

            # 1. Rotation
            if do_rotate and abs(angle) > 0.5:
                f = self._rotate(f, angle)

            # 2. Brightness / contrast
            if do_brightness_contrast:
                f = self._brightness_contrast(f, brightness_delta, contrast_factor)

            # 3. Centre-biased crop + resize back
            if do_crop and crop_scale < 1.0:
                f = f[y1:y1 + crop_h, x1:x1 + crop_w]
                f = cv2.resize(f, (w, h), interpolation=cv2.INTER_LINEAR)

            # 4. Horizontal flip
            if do_flip:
                f = cv2.flip(f, 1)

            # 5. Colour jitter (hue, saturation)
            if do_colour:
                f = self._colour_jitter(f, hue_delta, sat_factor)

            # 6. Gaussian blur
            if do_blur:
                f = cv2.GaussianBlur(f, (blur_k, blur_k), 0)

            augmented.append(f)

        return augmented

    # ── transform helpers ───────────────────────────────────────────────

    @staticmethod
    def _rotate(frame: np.ndarray, angle: float) -> np.ndarray:
        """Rotate around centre, fill border with edge replication."""
        h, w = frame.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        return cv2.warpAffine(
            frame, M, (w, h),
            borderMode=cv2.BORDER_REPLICATE,
        )

    @staticmethod
    def _brightness_contrast(
        frame: np.ndarray,
        brightness_delta: float,
        contrast_factor: float,
    ) -> np.ndarray:
        """Adjust brightness (additive) and contrast (multiplicative)."""
        f = frame.astype(np.float32)
        f = f * contrast_factor + brightness_delta * 255.0
        return np.clip(f, 0, 255).astype(np.uint8)

    @staticmethod
    def _colour_jitter(
        frame: np.ndarray,
        hue_delta: int,
        sat_factor: float,
    ) -> np.ndarray:
        """Shift hue and scale saturation in HSV space."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[:, :, 0] = (hsv[:, :, 0] + hue_delta) % 180
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * sat_factor, 0, 255)
        hsv = hsv.astype(np.uint8)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    @staticmethod
    def temporal_jitter(
        frame_indices: List[int],
        max_jitter: int = 2,
        total_frames: int = 1000,
    ) -> List[int]:
        """Apply random temporal jitter to frame indices.

        Shifts each index by a random offset in ``[-max_jitter, +max_jitter]``,
        clamped to valid range.  This is applied *before* frame loading.

        Args:
            frame_indices: Original evenly-spaced frame indices.
            max_jitter: Maximum shift per frame.
            total_frames: Total frames in the video (upper bound).

        Returns:
            Jittered frame indices (sorted, deduplicated is NOT enforced
            to preserve temporal ordering).
        """
        jittered: List[int] = []
        for idx in frame_indices:
            offset = random.randint(-max_jitter, max_jitter)
            new_idx = max(0, min(idx + offset, total_frames - 1))
            jittered.append(new_idx)
        return jittered


# ── standalone test ─────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    import os

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    # Create synthetic frames
    frames = [
        np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        for _ in range(16)
    ]

    aug = EgocentricAugmentation()

    print("Testing EgocentricAugmentation...")
    for trial in range(5):
        result = aug(frames)
        assert len(result) == len(frames), "Frame count mismatch"
        assert result[0].shape == frames[0].shape, "Shape mismatch"
        assert result[0].dtype == np.uint8, "Dtype mismatch"
        flip_str = "FLIPPED" if aug.did_flip else "no flip"
        print(f"  Trial {trial + 1}: OK ({flip_str})")

    # Test temporal jitter
    indices = list(range(0, 160, 10))
    jittered = EgocentricAugmentation.temporal_jitter(indices, max_jitter=2, total_frames=200)
    print(f"\nTemporal jitter: {indices[:5]}... → {jittered[:5]}...")

    print("\nAll augmentation tests PASSED")
