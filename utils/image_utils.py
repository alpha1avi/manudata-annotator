"""ManuData Annotator — Image utility functions.

Provides frame resizing, Base64 encoding, and image I/O helpers.
"""

import base64
import logging
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def resize_frame(frame: np.ndarray, max_size: int = 1280) -> np.ndarray:
    """Resize a frame so its longest edge is at most *max_size*, keeping aspect ratio.

    Args:
        frame: Input image as a NumPy array (H, W, C).
        max_size: Maximum allowed dimension in pixels.

    Returns:
        Resized frame (or the original if already within limits).
    """
    h, w = frame.shape[:2]
    if max(h, w) <= max_size:
        return frame

    scale = max_size / max(h, w)
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    logger.debug("Resized frame from %dx%d to %dx%d", w, h, new_w, new_h)
    return resized


def frame_to_base64(frame: np.ndarray, quality: int = 70) -> str:
    """Encode a frame as a JPEG Base64 string.

    Args:
        frame: Input image as a NumPy array (BGR).
        quality: JPEG quality (0-100).

    Returns:
        Base64-encoded JPEG string.
    """
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
    success, buffer = cv2.imencode(".jpg", frame, encode_params)
    if not success:
        raise RuntimeError("Failed to encode frame to JPEG")
    return base64.b64encode(buffer).decode("utf-8")


def load_frame(path: str) -> np.ndarray:
    """Load an image from disk as a NumPy array (BGR).

    Args:
        path: Path to the image file.

    Returns:
        Image as a NumPy array.

    Raises:
        FileNotFoundError: If the image file does not exist.
        RuntimeError: If OpenCV cannot decode the file.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    frame = cv2.imread(str(p), cv2.IMREAD_COLOR)
    if frame is None:
        raise RuntimeError(f"Failed to decode image: {path}")
    logger.debug("Loaded frame %s (%dx%d)", p.name, frame.shape[1], frame.shape[0])
    return frame


def save_frame(frame: np.ndarray, path: str, quality: int = 85) -> None:
    """Save a frame to disk as JPEG.

    Args:
        frame: Image as a NumPy array (BGR).
        path: Destination file path.
        quality: JPEG quality (0-100).
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
    success = cv2.imwrite(str(p), frame, encode_params)
    if not success:
        raise RuntimeError(f"Failed to save frame to {path}")
    logger.debug("Saved frame to %s", p.name)
