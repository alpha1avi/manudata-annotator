"""ManuData Annotator — Video utility functions.

Uses ffprobe to extract video metadata.
"""

import json
import logging
import subprocess
from pathlib import Path
from typing import Dict, Any

logger = logging.getLogger(__name__)


def get_video_metadata(video_path: str) -> Dict[str, Any]:
    """Extract metadata from a video file using ffprobe.

    Args:
        video_path: Path to the video file.

    Returns:
        Dictionary with keys:
            - duration (float): Duration in seconds.
            - fps (float): Frames per second.
            - width (int): Frame width in pixels.
            - height (int): Frame height in pixels.
            - codec (str): Video codec name.
            - resolution (str): "WxH" string.

    Raises:
        FileNotFoundError: If the video file does not exist.
        RuntimeError: If ffprobe is not installed or fails.
    """
    p = Path(video_path)
    if not p.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    cmd = [
        "ffprobe",
        "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        str(p),
    ]

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, check=True, timeout=30
        )
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        # ffprobe not available — fall back to OpenCV
        return _get_video_metadata_opencv(video_path)

    probe = json.loads(result.stdout)

    # Find the first video stream
    video_stream = None
    for stream in probe.get("streams", []):
        if stream.get("codec_type") == "video":
            video_stream = stream
            break

    if video_stream is None:
        raise RuntimeError(f"No video stream found in {video_path}")

    # Parse FPS from r_frame_rate (e.g. "30000/1001")
    fps_str = video_stream.get("r_frame_rate", "0/1")
    num, den = fps_str.split("/")
    fps = float(num) / float(den) if float(den) != 0 else 0.0

    # Duration: prefer stream duration, fall back to format duration
    duration = float(
        video_stream.get("duration", probe.get("format", {}).get("duration", 0))
    )

    width = int(video_stream.get("width", 0))
    height = int(video_stream.get("height", 0))
    codec = video_stream.get("codec_name", "unknown")

    metadata = {
        "duration": duration,
        "fps": round(fps, 2),
        "width": width,
        "height": height,
        "resolution": f"{width}x{height}",
        "codec": codec,
    }

    logger.info(
        "Video metadata for %s: %.1fs, %.1f fps, %s, %s",
        p.name, duration, fps, metadata["resolution"], codec,
    )
    return metadata


def _get_video_metadata_opencv(video_path: str) -> Dict[str, Any]:
    """OpenCV fallback when ffprobe is not available."""
    import cv2

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video with OpenCV: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if fps > 0 else 0.0
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    codec = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])

    cap.release()

    metadata = {
        "duration": round(duration, 2),
        "fps": round(fps, 2),
        "width": width,
        "height": height,
        "resolution": f"{width}x{height}",
        "codec": codec.strip() or "unknown",
    }

    logger.info(
        "Video metadata (OpenCV) for %s: %.1fs, %.1f fps, %s, %s",
        Path(video_path).name, duration, fps, metadata["resolution"], metadata["codec"],
    )
    return metadata
