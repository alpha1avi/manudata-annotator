"""ManuData Annotator — Colab convenience functions.

Simple function interface for calling from Colab cells without argparse.

Usage in Colab:
    from run_colab import bootstrap_video, annotate_video

    result = bootstrap_video("video.mp4", "./output", vlm="gemini", fps=2)
    result = annotate_video("video.mp4", "weights/best.pt", "./output")
"""

import asyncio
import logging
from typing import Optional

from config import AnnotatorConfig
from utils.logging_config import setup_logging

logger = logging.getLogger(__name__)


def bootstrap_video(
    video_path: str,
    output_dir: str,
    vlm: str = "gemini",
    fps: int = 2,
    batch_size: int = 5,
    max_concurrent: int = 3,
    duration: Optional[float] = None,
    cost_estimate_only: bool = False,
    verbose: bool = True,
) -> dict:
    """Bootstrap-annotate a single video via VLM.

    Convenience wrapper for Colab / notebook usage.

    Args:
        video_path: Path to the input video.
        output_dir: Directory for all outputs.
        vlm: VLM backend — ``"gemini"`` | ``"claude_haiku"``
             | ``"claude_sonnet"`` | ``"openai"``.
        fps: Frame extraction rate.
        batch_size: Frames per VLM batch.
        max_concurrent: Max concurrent VLM requests.
        duration: Only process the first *N* seconds (or ``None`` for full video).
        cost_estimate_only: If True, print cost estimate and return without processing.
        verbose: Enable debug logging.

    Returns:
        Result dict with status, segment count, cost, quality, output paths.
    """
    setup_logging(verbose=verbose)

    config = AnnotatorConfig(
        fps=fps,
        batch_size=batch_size,
        max_concurrent=max_concurrent,
        vlm_backend=vlm,
    )
    print(config.summary())

    from annotator import run_bootstrap_single

    result = asyncio.run(
        run_bootstrap_single(
            video_path, config, output_dir,
            cost_estimate_only=cost_estimate_only,
            duration=duration,
        )
    )
    return result


def annotate_video(
    video_path: str,
    model_path: str,
    output_dir: str,
    output_format: str = "rlds",
    fallback_vlm: str = "gemini",
    fallback_threshold: float = 0.4,
    no_fallback: bool = False,
    fps: int = 2,
    duration: Optional[float] = None,
    verbose: bool = True,
) -> dict:
    """Annotate a video using a fine-tuned local model with optional VLM fallback.

    Convenience wrapper for Colab / notebook usage.

    Args:
        video_path: Path to the input video.
        model_path: Path to the fine-tuned model weights.
        output_dir: Directory for all outputs.
        output_format: ``"json"`` | ``"rlds"`` | ``"hdf5"``.
        fallback_vlm: VLM backend for low-confidence fallback.
        fallback_threshold: Confidence threshold for VLM fallback.
        no_fallback: Disable VLM fallback entirely.
        fps: Frame extraction rate.
        duration: Only process the first *N* seconds.
        verbose: Enable debug logging.

    Returns:
        Result dict with status, segment count, cost, quality, output paths.
    """
    setup_logging(verbose=verbose)

    config = AnnotatorConfig(
        fps=fps,
        vlm_backend=fallback_vlm,
        fallback_threshold=fallback_threshold,
        model_path=model_path,
        output_format=output_format,
    )
    if no_fallback:
        config._no_fallback = True

    print(config.summary())

    from annotator import run_annotate_single

    result = asyncio.run(
        run_annotate_single(
            video_path, config, output_dir,
            duration=duration,
        )
    )
    return result


# ── quick test ────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("ManuData Annotator — Colab Interface")
    print()
    print("Usage:")
    print("  from run_colab import bootstrap_video, annotate_video")
    print()
    print('  result = bootstrap_video("video.mp4", "./output")')
    print('  result = annotate_video("video.mp4", "weights/best.pt", "./output")')
