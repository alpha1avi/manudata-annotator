"""ManuData Annotator — Configuration module.

Loads API keys from .env and provides the AnnotatorConfig dataclass
with all tunable parameters and defaults.
"""

import os
import logging
from dataclasses import dataclass, field
from typing import List

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

# API keys
ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")

DEFAULT_OBJECT_TAXONOMY: List[str] = [
    "wrench", "bolt", "nut", "screwdriver", "hammer", "pliers", "wire",
    "PCB", "circuit board", "motor", "housing", "shaft", "gear", "bearing",
    "bracket", "panel", "switch", "button", "cable", "connector",
    "container", "tray", "bin", "workbench", "clamp", "gauge", "meter",
    "soldering iron", "multimeter", "drill", "saw", "tape", "glue",
]


@dataclass
class AnnotatorConfig:
    """Central configuration for the ManuData annotation pipeline."""

    # Frame extraction
    fps: int = 2
    batch_size: int = 5
    max_concurrent: int = 3

    # Quality filtering thresholds
    blur_threshold: float = 50.0
    duplicate_threshold: float = 0.95
    activity_threshold: float = 0.2

    # Annotation thresholds
    fallback_threshold: float = 0.4

    # Image processing
    frame_max_size: int = 1280
    jpeg_quality: int = 70

    # Temporal segmentation
    min_segment_duration: float = 0.5
    idle_segment_threshold: float = 2.0

    # Output
    output_format: str = "json"  # json, rlds, hdf5

    # VLM backend
    vlm_backend: str = "gemini"

    # Local model path (for fine-tuned model inference)
    model_path: str = ""

    # Object taxonomy
    object_taxonomy: List[str] = field(default_factory=lambda: list(DEFAULT_OBJECT_TAXONOMY))

    def summary(self) -> str:
        """Return a human-readable config summary."""
        lines = [
            "=== ManuData Annotator Config ===",
            f"  FPS:                {self.fps}",
            f"  Batch size:         {self.batch_size}",
            f"  Max concurrent:     {self.max_concurrent}",
            f"  Blur threshold:     {self.blur_threshold}",
            f"  Duplicate thresh:   {self.duplicate_threshold}",
            f"  Activity thresh:    {self.activity_threshold}",
            f"  Fallback thresh:    {self.fallback_threshold}",
            f"  Frame max size:     {self.frame_max_size}",
            f"  JPEG quality:       {self.jpeg_quality}",
            f"  Output format:      {self.output_format}",
            f"  VLM backend:        {self.vlm_backend}",
            f"  Model path:         {self.model_path or '(none)'}",
            f"  Object taxonomy:    {len(self.object_taxonomy)} classes",
            "=================================",
        ]
        return "\n".join(lines)
