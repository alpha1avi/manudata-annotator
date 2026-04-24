"""ManuData Annotator — Monocular Depth Estimation.

Primary backend: Depth Anything V2 (Small) via HuggingFace transformers.
Fallback: MiDaS from ``torch.hub`` if Depth Anything is unavailable.
"""

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch

logger = logging.getLogger(__name__)


# ── result dataclass ──────────────────────────────────────────────────


@dataclass
class DepthResult:
    """Result of monocular depth estimation for a single frame."""

    depth_map: np.ndarray  # (H, W) float32, relative depth 0-1
    _original_hw: Tuple[int, int] = field(default=(0, 0), repr=False)

    def depth_at_point(self, x: int, y: int) -> float:
        """Get relative depth at a pixel coordinate.

        Args:
            x: Pixel column.
            y: Pixel row.

        Returns:
            Depth value in ``[0, 1]`` (0 = near, 1 = far).
        """
        h, w = self.depth_map.shape[:2]
        x = max(0, min(x, w - 1))
        y = max(0, min(y, h - 1))
        return float(self.depth_map[y, x])

    def save_depth_map(self, path: str) -> None:
        """Save the depth map as a ``.npy`` file."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(p), self.depth_map)
        logger.debug("Depth map saved to %s", p)

    def visualize(self) -> np.ndarray:
        """Return a colorised depth map (BGR) for debugging.

        Uses the INFERNO colourmap for good perceptual contrast.
        """
        depth_norm = (self.depth_map * 255).astype(np.uint8)
        coloured = cv2.applyColorMap(depth_norm, cv2.COLORMAP_INFERNO)
        return coloured


# ── estimator ─────────────────────────────────────────────────────────


class DepthEstimator:
    """Monocular depth estimator with Depth Anything V2 / MiDaS backends."""

    def __init__(
        self,
        model_size: str = "small",
        device: str = "cuda",
    ) -> None:
        """Load depth estimation model.

        Tries in order:
            1. Depth Anything V2 Small via ``transformers`` pipeline.
            2. MiDaS Small via ``torch.hub`` (fallback).

        Falls back to CPU if CUDA is not available.
        """
        # Resolve device
        if device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA not available — falling back to CPU for depth estimation")
            device = "cpu"
        self.device = device

        self._backend: Optional[str] = None
        self._pipeline = None       # HF pipeline (Depth Anything)
        self._midas_model = None     # MiDaS model
        self._midas_transform = None

        t0 = time.time()

        # Attempt 1: Depth Anything V2 via transformers pipeline
        if self._try_load_depth_anything(model_size):
            self._backend = "depth_anything_v2"
        # Attempt 2: MiDaS fallback
        elif self._try_load_midas():
            self._backend = "midas"
        else:
            raise RuntimeError(
                "Could not load any depth estimation model. "
                "Install transformers>=4.35 or ensure torch.hub access."
            )

        load_time = time.time() - t0
        logger.info(
            "Depth estimator loaded: backend=%s, device=%s (%.2fs)",
            self._backend, self.device, load_time,
        )

    # ── model loading ─────────────────────────────────────────────────

    def _try_load_depth_anything(self, model_size: str) -> bool:
        """Try loading Depth Anything V2 via HuggingFace transformers."""
        try:
            from transformers import pipeline as hf_pipeline

            model_id = f"depth-anything/Depth-Anything-V2-{model_size.capitalize()}-hf"
            logger.info("Loading Depth Anything V2: %s …", model_id)
            self._pipeline = hf_pipeline(
                "depth-estimation",
                model=model_id,
                device=0 if self.device == "cuda" else -1,
            )
            return True
        except Exception as exc:
            logger.warning("Depth Anything V2 unavailable: %s", exc)
            return False

    def _try_load_midas(self) -> bool:
        """Try loading MiDaS Small via torch.hub as a fallback."""
        try:
            logger.info("Loading MiDaS Small via torch.hub …")
            self._midas_model = torch.hub.load(
                "intel-isl/MiDaS", "MiDaS_small", trust_repo=True
            )
            self._midas_model.to(self.device).eval()

            midas_transforms = torch.hub.load(
                "intel-isl/MiDaS", "transforms", trust_repo=True
            )
            self._midas_transform = midas_transforms.small_transform
            return True
        except Exception as exc:
            logger.warning("MiDaS fallback also failed: %s", exc)
            return False

    # ── public API ────────────────────────────────────────────────────

    def estimate(self, frame: np.ndarray) -> DepthResult:
        """Run monocular depth estimation on a single BGR frame.

        Args:
            frame: BGR image (H, W, 3).

        Returns:
            :class:`DepthResult` with normalised depth map.
        """
        t0 = time.time()

        if self._backend == "depth_anything_v2":
            depth_map = self._estimate_depth_anything(frame)
        elif self._backend == "midas":
            depth_map = self._estimate_midas(frame)
        else:
            raise RuntimeError("No depth backend loaded")

        elapsed_ms = (time.time() - t0) * 1000
        logger.debug("Depth estimation: %.1fms (%s)", elapsed_ms, self._backend)

        return DepthResult(
            depth_map=depth_map,
            _original_hw=frame.shape[:2],
        )

    def estimate_sparse(
        self,
        frame: np.ndarray,
        points: List[Tuple[int, int]],
    ) -> List[float]:
        """Get depth values at specific pixel locations.

        Runs full estimation internally then samples at the requested
        coordinates.

        Args:
            frame: BGR image.
            points: List of ``(x, y)`` pixel coordinates.

        Returns:
            List of depth values in ``[0, 1]``, one per point.
        """
        result = self.estimate(frame)
        return [result.depth_at_point(x, y) for x, y in points]

    # ── backend implementations ───────────────────────────────────────

    def _estimate_depth_anything(self, frame: np.ndarray) -> np.ndarray:
        """Run Depth Anything V2 via HuggingFace pipeline."""
        from PIL import Image

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)

        output = self._pipeline(pil_img)
        depth_pil = output["depth"]

        # Convert PIL depth image to numpy float32 [0, 1]
        depth_np = np.array(depth_pil, dtype=np.float32)

        # Resize to original frame size if different
        h, w = frame.shape[:2]
        if depth_np.shape[:2] != (h, w):
            depth_np = cv2.resize(depth_np, (w, h), interpolation=cv2.INTER_LINEAR)

        # Normalise to 0-1
        d_min, d_max = depth_np.min(), depth_np.max()
        if d_max - d_min > 1e-6:
            depth_np = (depth_np - d_min) / (d_max - d_min)
        else:
            depth_np = np.zeros_like(depth_np)

        return depth_np

    def _estimate_midas(self, frame: np.ndarray) -> np.ndarray:
        """Run MiDaS Small via torch.hub."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_batch = self._midas_transform(rgb).to(self.device)

        with torch.no_grad():
            prediction = self._midas_model(input_batch)

            # Interpolate to original size
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=frame.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        depth_np = prediction.cpu().numpy().astype(np.float32)

        # Normalise to 0-1
        d_min, d_max = depth_np.min(), depth_np.max()
        if d_max - d_min > 1e-6:
            depth_np = (depth_np - d_min) / (d_max - d_min)
        else:
            depth_np = np.zeros_like(depth_np)

        return depth_np


# ── standalone test ───────────────────────────────────────────────────

if __name__ == "__main__":
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from utils.image_utils import load_frame, save_frame
    from utils.logging_config import setup_logging

    setup_logging(verbose=True)

    if len(sys.argv) < 2:
        print("Usage: python depth_estimator.py <image_path> [output_dir]")
        sys.exit(1)

    frame = load_frame(sys.argv[1])
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "."

    estimator = DepthEstimator(model_size="small", device="cuda")
    result = estimator.estimate(frame)

    print(f"\nDepth map shape: {result.depth_map.shape}")
    print(f"Depth range: [{result.depth_map.min():.3f}, {result.depth_map.max():.3f}]")

    # Centre point depth
    h, w = frame.shape[:2]
    centre_depth = result.depth_at_point(w // 2, h // 2)
    print(f"Centre depth: {centre_depth:.3f}")

    # Save outputs
    result.save_depth_map(f"{out_dir}/depth_map.npy")
    vis = result.visualize()
    save_frame(vis, f"{out_dir}/depth_vis.jpg")
    print(f"Saved depth map and visualisation to {out_dir}/")
