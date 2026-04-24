"""ManuData Annotator — Frame Extraction (Stage 1).

Extracts frames from video files at a configurable FPS using ffmpeg,
with support for multi-camera manifests and sync offsets.
"""

import json
import logging
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from config import AnnotatorConfig
from utils.video_utils import get_video_metadata

logger = logging.getLogger(__name__)


@dataclass
class ExtractionResult:
    """Result of frame extraction from a single video."""

    video_path: str
    video_metadata: Dict  # duration, fps, resolution, codec
    frames_dir: str
    frame_paths: List[Tuple[float, str]]  # (timestamp_seconds, filepath)
    total_frames: int
    extraction_fps: float
    camera_info: Optional[Dict] = field(default=None)  # for multi-camera

    @property
    def frame_count(self) -> int:
        """Alias for ``total_frames``."""
        return self.total_frames


class FrameExtractor:
    """Extract frames from video files using ffmpeg subprocess."""

    def __init__(self, config: AnnotatorConfig) -> None:
        self.config = config
        self._ffmpeg_available = self._check_ffmpeg()

    # ── internal helpers ───────────────────────────────────────────────

    @staticmethod
    def _check_ffmpeg() -> bool:
        """Check whether ffmpeg is available on the system PATH.

        Returns:
            ``True`` if ffmpeg is available, ``False`` otherwise.
        """
        try:
            subprocess.run(
                ["ffmpeg", "-version"],
                capture_output=True,
                check=True,
                timeout=10,
            )
            return True
        except (FileNotFoundError, subprocess.CalledProcessError):
            logger.warning(
                "ffmpeg not found on PATH. Will use OpenCV fallback for frame extraction."
            )
            return False

    def _build_ffmpeg_cmd(
        self,
        video_path: str,
        output_pattern: str,
        duration: Optional[float] = None,
    ) -> List[str]:
        """Build the ffmpeg frame-extraction command."""
        cmd = ["ffmpeg", "-i", video_path]

        if duration is not None:
            cmd.extend(["-t", str(duration)])

        cmd.extend([
            "-vf", f"fps={self.config.fps}",
            "-q:v", "2",
            "-y",  # overwrite without prompting
            output_pattern,
        ])
        return cmd

    @staticmethod
    def _rename_to_timestamps(
        frames_dir: Path,
        extraction_fps: int,
        sync_offset_s: float = 0.0,
    ) -> List[Tuple[float, str]]:
        """Rename sequential frame files to millisecond-timestamp names.

        ffmpeg outputs ``frame_00000001.jpg``, ``frame_00000002.jpg``, ...
        We rename to ``frame_{timestamp_ms}.jpg`` and return sorted
        ``(timestamp_seconds, filepath)`` pairs.
        """
        pattern = re.compile(r"frame_(\d+)\.jpg")
        frame_files = sorted(frames_dir.glob("frame_*.jpg"))

        result: List[Tuple[float, str]] = []
        for fpath in frame_files:
            m = pattern.match(fpath.name)
            if not m:
                logger.warning("Skipping unexpected file: %s", fpath.name)
                continue

            # ffmpeg 1-indexed frame number → 0-indexed
            frame_idx = int(m.group(1)) - 1
            timestamp_s = (frame_idx / extraction_fps) + sync_offset_s
            timestamp_ms = int(round(timestamp_s * 1000))

            new_name = f"frame_{timestamp_ms:010d}.jpg"
            new_path = fpath.parent / new_name
            fpath.rename(new_path)
            result.append((timestamp_s, str(new_path)))

        result.sort(key=lambda x: x[0])
        return result

    def _extract_opencv(
        self,
        video_path: str,
        frames_dir: Path,
        metadata: Dict,
        duration: Optional[float] = None,
    ) -> List[Tuple[float, str]]:
        """Fallback frame extraction using OpenCV when ffmpeg is unavailable."""
        import cv2 as _cv2

        cap = _cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"OpenCV cannot open video: {video_path}")

        src_fps = cap.get(_cv2.CAP_PROP_FPS) or 30.0
        total_video_frames = int(cap.get(_cv2.CAP_PROP_FRAME_COUNT) or 0)
        frame_interval = max(1, int(round(src_fps / self.config.fps)))

        max_frame = total_video_frames
        if duration is not None:
            max_frame = min(max_frame, int(duration * src_fps))

        result: List[Tuple[float, str]] = []
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret or frame_idx >= max_frame:
                break

            if frame_idx % frame_interval == 0:
                timestamp_s = frame_idx / src_fps
                timestamp_ms = int(round(timestamp_s * 1000))
                fname = f"frame_{timestamp_ms:010d}.jpg"
                fpath = frames_dir / fname
                _cv2.imwrite(str(fpath), frame, [_cv2.IMWRITE_JPEG_QUALITY, 95])
                result.append((timestamp_s, str(fpath)))

            frame_idx += 1

        cap.release()
        result.sort(key=lambda x: x[0])
        logger.info(
            "OpenCV fallback: extracted %d frames at ~%d fps",
            len(result), self.config.fps,
        )
        return result

    # ── public API ─────────────────────────────────────────────────────

    def extract(
        self,
        video_path: str,
        output_dir: str,
        duration: Optional[float] = None,
    ) -> ExtractionResult:
        """Extract frames from a single video at the configured FPS.

        Steps:
            1. Create ``output_dir/frames/`` subdirectory.
            2. Retrieve video metadata via ``ffprobe``.
            3. Run ffmpeg to extract frames.
            4. Rename frames to embed millisecond timestamps.

        Args:
            video_path: Path to the source video file.
            output_dir: Root output directory for this video.
            duration: Optional — only extract the first *N* seconds.

        Returns:
            An :class:`ExtractionResult` with paths, metadata, and stats.
        """
        vpath = Path(video_path)
        if not vpath.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        # 1. Prepare output directory
        frames_dir = Path(output_dir) / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Extracting frames to %s", frames_dir)

        # 2. Get video metadata
        metadata = get_video_metadata(video_path)
        logger.info(
            "Video: %s | %.1fs | %.1f fps | %s",
            vpath.name, metadata["duration"], metadata["fps"], metadata["resolution"],
        )

        # 3. Extract frames via ffmpeg or OpenCV fallback
        if self._ffmpeg_available:
            output_pattern = str(frames_dir / "frame_%08d.jpg")
            effective_duration = duration  # may be None
            cmd = self._build_ffmpeg_cmd(video_path, output_pattern, effective_duration)
            logger.debug("ffmpeg command: %s", " ".join(cmd))

            try:
                proc = subprocess.run(
                    cmd, capture_output=True, text=True, check=True, timeout=600
                )
            except subprocess.CalledProcessError as exc:
                stderr = exc.stderr or ""
                if "Invalid data" in stderr or "corrupt" in stderr.lower():
                    raise RuntimeError(f"Video appears corrupt: {video_path}\n{stderr}")
                if "Unknown encoder" in stderr or "codec" in stderr.lower():
                    raise RuntimeError(
                        f"Unsupported codec in {video_path}\n{stderr}"
                    )
                raise RuntimeError(f"ffmpeg failed for {video_path}\n{stderr}")
            except subprocess.TimeoutExpired:
                raise RuntimeError(
                    f"ffmpeg timed out (>600s) for {video_path}. "
                    "Try using --duration to limit extraction."
                )

            # 4a. Rename to timestamp-based names
            frame_paths = self._rename_to_timestamps(frames_dir, self.config.fps)
        else:
            # 4b. OpenCV fallback extraction
            frame_paths = self._extract_opencv(
                video_path, frames_dir, metadata, duration
            )

        if not frame_paths:
            logger.warning("No frames extracted from %s", video_path)

        result = ExtractionResult(
            video_path=str(vpath.resolve()),
            video_metadata=metadata,
            frames_dir=str(frames_dir),
            frame_paths=frame_paths,
            total_frames=len(frame_paths),
            extraction_fps=float(self.config.fps),
        )

        logger.info(
            "Extracted %d frames at %d fps from %s",
            result.total_frames, self.config.fps, vpath.name,
        )
        return result

    def extract_multi_camera(
        self, manifest_path: str, output_dir: str
    ) -> List[ExtractionResult]:
        """Extract frames from multiple cameras described in a manifest.

        Manifest JSON format::

            {
              "cameras": [
                {"file": "cam_head.mp4", "mount": "helmet", "type": "egocentric"},
                {"file": "cam_chest.mp4", "mount": "chest", "type": "semi_egocentric"}
              ],
              "sync_offset_ms": [0, 15]
            }

        Sync offsets are applied so timestamps across cameras are aligned
        to a common timeline.

        Args:
            manifest_path: Path to the manifest JSON file.
            output_dir: Root output directory.

        Returns:
            List of :class:`ExtractionResult`, one per camera.
        """
        mpath = Path(manifest_path)
        if not mpath.exists():
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")

        with open(mpath, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        cameras = manifest.get("cameras", [])
        offsets_ms = manifest.get("sync_offset_ms", [0] * len(cameras))

        if len(offsets_ms) < len(cameras):
            offsets_ms.extend([0] * (len(cameras) - len(offsets_ms)))

        manifest_dir = mpath.parent
        results: List[ExtractionResult] = []

        for idx, cam in enumerate(cameras):
            cam_file = cam["file"]
            cam_path = manifest_dir / cam_file
            if not cam_path.exists():
                logger.error("Camera file not found: %s", cam_path)
                continue

            cam_label = cam.get("mount", f"cam{idx}")
            cam_output = Path(output_dir) / cam_label
            cam_output.mkdir(parents=True, exist_ok=True)

            logger.info(
                "Extracting camera %d/%d: %s (%s, offset=%dms)",
                idx + 1, len(cameras), cam_file, cam_label, offsets_ms[idx],
            )

            # Extract frames
            frames_dir = cam_output / "frames"
            frames_dir.mkdir(parents=True, exist_ok=True)

            metadata = get_video_metadata(str(cam_path))
            output_pattern = str(frames_dir / "frame_%08d.jpg")
            cmd = self._build_ffmpeg_cmd(str(cam_path), output_pattern)
            logger.debug("ffmpeg command: %s", " ".join(cmd))

            try:
                subprocess.run(
                    cmd, capture_output=True, text=True, check=True, timeout=600
                )
            except subprocess.CalledProcessError as exc:
                logger.error("ffmpeg failed for %s: %s", cam_file, exc.stderr)
                continue
            except subprocess.TimeoutExpired:
                logger.error("ffmpeg timed out for %s", cam_file)
                continue

            # Rename with sync offset applied
            sync_offset_s = offsets_ms[idx] / 1000.0
            frame_paths = self._rename_to_timestamps(
                frames_dir, self.config.fps, sync_offset_s
            )

            er = ExtractionResult(
                video_path=str(cam_path.resolve()),
                video_metadata=metadata,
                frames_dir=str(frames_dir),
                frame_paths=frame_paths,
                total_frames=len(frame_paths),
                extraction_fps=float(self.config.fps),
                camera_info={
                    "index": idx,
                    "mount": cam.get("mount", ""),
                    "type": cam.get("type", ""),
                    "sync_offset_ms": offsets_ms[idx],
                },
            )
            results.append(er)
            logger.info(
                "Camera %s: %d frames extracted (offset %.3fs)",
                cam_label, er.total_frames, sync_offset_s,
            )

        logger.info("Multi-camera extraction complete: %d cameras", len(results))
        return results


# ── standalone test ────────────────────────────────────────────────────

if __name__ == "__main__":
    import os

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from utils.logging_config import setup_logging

    setup_logging(verbose=True)

    if len(sys.argv) < 2:
        print("Usage: python frame_extractor.py <video_path> [output_dir]")
        sys.exit(1)

    video = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else "./extraction_test"

    config = AnnotatorConfig(fps=2)
    extractor = FrameExtractor(config)
    result = extractor.extract(video, out)

    print(f"\n{'='*50}")
    print(f"Video:      {result.video_path}")
    print(f"Duration:   {result.video_metadata['duration']:.1f}s")
    print(f"Source FPS: {result.video_metadata['fps']}")
    print(f"Resolution: {result.video_metadata['resolution']}")
    print(f"Codec:      {result.video_metadata['codec']}")
    print(f"Extracted:  {result.total_frames} frames at {result.extraction_fps} fps")
    print(f"Frames dir: {result.frames_dir}")
    if result.frame_paths:
        print(f"First:      {result.frame_paths[0]}")
        print(f"Last:       {result.frame_paths[-1]}")
    print(f"{'='*50}")
