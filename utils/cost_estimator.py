"""ManuData Annotator — API cost estimator.

Estimates VLM API costs in INR based on frame counts and backend selection.
"""

import logging
import math
from typing import Dict, Any

logger = logging.getLogger(__name__)

# Approximate cost per batch (5 frames) in INR
COST_PER_BATCH_INR: Dict[str, float] = {
    "gemini": 0.05,        # Gemini Flash
    "claude_haiku": 0.08,  # Claude Haiku
    "openai": 0.10,        # GPT-4o-mini
    "claude_sonnet": 0.35, # Claude Sonnet
}


def estimate_cost(
    total_frames: int,
    frames_after_filter: int,
    vlm_backend: str,
    batch_size: int = 5,
) -> Dict[str, Any]:
    """Estimate the API cost for annotating the filtered frames.

    Args:
        total_frames: Total frames extracted from the video.
        frames_after_filter: Frames remaining after quality filtering.
        vlm_backend: Selected VLM backend name (e.g. "gemini", "claude_haiku").
        batch_size: Number of frames per API batch.

    Returns:
        Dictionary with:
            - total_frames (int)
            - frames_after_filter (int)
            - filter_ratio (float): Fraction of frames kept.
            - num_batches (int)
            - vlm_backend (str)
            - cost_per_batch_inr (float)
            - estimated_cost_inr (float)
            - all_backends (dict): Cost estimates for every supported backend.
    """
    num_batches = math.ceil(frames_after_filter / batch_size) if frames_after_filter > 0 else 0
    filter_ratio = frames_after_filter / total_frames if total_frames > 0 else 0.0

    cost_per_batch = COST_PER_BATCH_INR.get(vlm_backend, COST_PER_BATCH_INR["gemini"])
    estimated_cost = num_batches * cost_per_batch

    # Cost breakdown for all backends
    all_backends = {}
    for backend, cpb in COST_PER_BATCH_INR.items():
        all_backends[backend] = round(num_batches * cpb, 2)

    result = {
        "total_frames": total_frames,
        "frames_after_filter": frames_after_filter,
        "filter_ratio": round(filter_ratio, 3),
        "num_batches": num_batches,
        "vlm_backend": vlm_backend,
        "cost_per_batch_inr": cost_per_batch,
        "estimated_cost_inr": round(estimated_cost, 2),
        "all_backends": all_backends,
    }

    logger.info(
        "Cost estimate: %d frames -> %d after filter -> %d batches | %s: ₹%.2f",
        total_frames, frames_after_filter, num_batches, vlm_backend, estimated_cost,
    )
    return result
