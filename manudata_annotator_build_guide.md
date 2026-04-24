# ManuData Annotator — Build Guide

## Sequential Claude Code Prompts (run in order)

Each prompt builds one module. Run them one after another in the same Claude Code session.
If session gets too long, start a new session — Claude Code will see the existing files.

---

## STEP 0: Project Setup

Open PowerShell, create the project folder, start Claude Code.

```powershell
mkdir C:\Projects\manudata-annotator
cd C:\Projects\manudata-annotator
claude
```

---

## STEP 1: Project Skeleton + Config + Utils

### Prompt:

```
Create a Python project called manudata-annotator with this folder structure:

manudata-annotator/
├── annotator.py           # Main CLI (build empty argparse scaffold only)
├── train.py               # Training CLI (empty scaffold only)
├── config.py              # Configuration
├── pipeline/
│   └── __init__.py
├── models/
│   └── __init__.py
├── training/
│   └── __init__.py
├── vlm/
│   └── __init__.py
├── utils/
│   └── __init__.py
├── requirements.txt
├── .env.example
└── README.md

Implement FULLY:

1. config.py:
- Load API keys from .env (ANTHROPIC_API_KEY, GOOGLE_API_KEY, OPENAI_API_KEY)
- Dataclass AnnotatorConfig with all defaults:
  - fps: int = 2
  - batch_size: int = 5
  - max_concurrent: int = 3
  - blur_threshold: float = 50.0
  - duplicate_threshold: float = 0.95
  - activity_threshold: float = 0.2
  - fallback_threshold: float = 0.4
  - frame_max_size: int = 1280
  - jpeg_quality: int = 70
  - min_segment_duration: float = 0.5
  - idle_segment_threshold: float = 2.0
  - output_format: str = "json"  # json, rlds, hdf5
  - vlm_backend: str = "gemini"
  - model_path: str = ""
  - object_taxonomy: list = default factory objects list
- Default object taxonomy list:
  ["wrench", "bolt", "nut", "screwdriver", "hammer", "pliers", "wire",
   "PCB", "circuit board", "motor", "housing", "shaft", "gear", "bearing",
   "bracket", "panel", "switch", "button", "cable", "connector",
   "container", "tray", "bin", "workbench", "clamp", "gauge", "meter",
   "soldering iron", "multimeter", "drill", "saw", "tape", "glue"]

2. utils/image_utils.py:
- resize_frame(frame, max_size=1280) → resized frame maintaining aspect ratio
- frame_to_base64(frame, quality=70) → base64 string
- load_frame(path) → numpy array
- save_frame(frame, path, quality=85)

3. utils/video_utils.py:
- get_video_metadata(video_path) → dict with duration, fps, resolution, codec
- Uses ffprobe subprocess

4. utils/cost_estimator.py:
- estimate_cost(total_frames, frames_after_filter, vlm_backend, batch_size) → dict
- Returns estimated API cost in INR for gemini, claude, openai
- Gemini Flash: ~₹0.05 per 5-frame batch
- Claude Haiku: ~₹0.08 per 5-frame batch
- GPT-4o-mini: ~₹0.10 per 5-frame batch
- Claude Sonnet: ~₹0.35 per 5-frame batch

5. utils/progress_tracker.py:
- ProgressTracker class that saves/loads progress to .progress JSON file
- track processed frame timestamps, current stage, partial results
- Support resume: load previous progress file and skip already-processed frames

6. utils/logging_config.py:
- setup_logging(verbose=False) → configure Python logging
- INFO level by default, DEBUG if verbose
- Format: "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
- Also log to file: annotator.log

7. annotator.py — CLI scaffold with argparse:
- Subcommands: bootstrap, annotate
- bootstrap args: input_path, --vlm, --fps, --batch-size, --output-dir, --max-concurrent, --cost-estimate, --resume, --object-taxonomy, --duration, --verbose, --batch, --save-filtered-frames
- annotate args: input_path, --model-path, --output-dir, --output-format (json/rlds/hdf5), --fallback-vlm, --fallback-threshold, --no-fallback, --review-only, --gpu, --batch, --verbose
- Both subcommands: parse args, build config, print config summary, then call placeholder pipeline function
- Handle both single file and directory (--batch) inputs

8. requirements.txt with ALL dependencies:
opencv-python>=4.8.0
mediapipe>=0.10.8
numpy>=1.24.0
Pillow>=10.0.0
httpx>=0.25.0
python-dotenv>=1.0.0
tqdm>=4.66.0
tenacity>=8.2.0
scikit-image>=0.21.0
scipy>=1.11.0
h5py>=3.9.0
torch>=2.1.0
torchvision>=0.16.0
timm>=0.9.0
transformers>=4.35.0
ultralytics>=8.0.0
einops>=0.7.0
matplotlib>=3.8.0
seaborn>=0.13.0

9. .env.example:
ANTHROPIC_API_KEY=sk-ant-xxx
GOOGLE_API_KEY=AIzaxxx
OPENAI_API_KEY=sk-xxx

10. README.md — brief project description, setup instructions, usage examples for all 3 modes.

Use type hints, docstrings, and structured logging throughout. All code must be complete and runnable.
```

---

## STEP 2: Frame Extraction (Stage 1)

### Prompt:

```
Build pipeline/frame_extractor.py — complete implementation.

This module extracts frames from video files using ffmpeg subprocess.

class FrameExtractor:
    def __init__(self, config: AnnotatorConfig)
    
    def extract(self, video_path: str, output_dir: str) -> ExtractionResult:
        """
        Extract frames from video at configured FPS.
        
        1. Create output_dir/frames/ subdirectory
        2. Get video metadata using utils/video_utils.py
        3. Run ffmpeg to extract frames:
           ffmpeg -i video.mp4 -vf fps={self.config.fps} -q:v 2 output_dir/frames/frame_%08d.jpg
        4. Rename frames to include millisecond timestamps: frame_{timestamp_ms}.jpg
           Calculate timestamp from frame index and extraction FPS
        5. If config.duration is set, only extract first N seconds:
           Add -t {duration} to ffmpeg command
        
        Returns ExtractionResult dataclass:
        - video_path: str
        - video_metadata: dict (duration, fps, resolution)
        - frames_dir: str
        - frame_paths: List[Tuple[float, str]]  # (timestamp_seconds, filepath)
        - total_frames: int
        - extraction_fps: float
        """
    
    def extract_multi_camera(self, manifest_path: str, output_dir: str) -> List[ExtractionResult]:
        """
        Handle multi-camera sync from manifest JSON:
        {
          "cameras": [
            {"file": "cam_head.mp4", "mount": "helmet", "type": "egocentric"},
            {"file": "cam_chest.mp4", "mount": "chest", "type": "semi_egocentric"}
          ],
          "sync_offset_ms": [0, 15]
        }
        
        Extract frames from each camera, apply sync offsets so timestamps align.
        Return list of ExtractionResult, one per camera.
        """

Use proper subprocess calls for ffmpeg. Handle errors (ffmpeg not found, corrupt video, unsupported codec) with clear error messages. Use logging throughout. Include a standalone test:

if __name__ == "__main__":
    # Test with a sample video path passed as sys.argv[1]
    # Print extraction stats
```

---

## STEP 3: Frame Quality Filter (Stage 2)

### Prompt:

```
Build pipeline/frame_quality_filter.py — complete implementation.

This module filters out bad frames BEFORE any API calls, saving 40-60% of VLM cost.
Egocentric helmet-cam footage has lots of blur, duplicates, and head-turn transitions.

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from dataclasses import dataclass
from typing import List, Tuple
import logging

@dataclass
class FilterResult:
    good_frames: List[Tuple[float, str]]         # (timestamp, filepath) — frames to process
    skipped_blur: List[Tuple[float, str]]         # blurry frames
    skipped_duplicate: List[Tuple[float, str]]    # near-duplicate frames
    skipped_transition: List[Tuple[float, float]] # (start_ts, end_ts) transition ranges
    skipped_dark: List[Tuple[float, str]]         # too dark or occluded
    stats: dict                                    # counts and percentages

class FrameQualityFilter:
    def __init__(self, config: AnnotatorConfig):
        self.blur_threshold = config.blur_threshold        # default 50
        self.duplicate_threshold = config.duplicate_threshold  # default 0.95
        self.transition_flow_threshold = 15.0              # optical flow magnitude threshold
        self.transition_min_frames = 3                     # minimum consecutive high-flow frames
        self.brightness_low = 30
        self.brightness_high = 240
        self.entropy_threshold = 3.0                       # low entropy = occluded/uniform

    def filter(self, frame_paths: List[Tuple[float, str]]) -> FilterResult:
        """
        Process all frames through quality checks in order:
        1. Darkness/occlusion check (cheapest — just histogram)
        2. Blur detection (Laplacian — fast)
        3. Duplicate detection (SSIM against previous good frame — moderate cost)
        4. Transition detection (optical flow on sliding window — most expensive)
        
        Process in this order because each filter reduces work for the next.
        Show tqdm progress bar.
        Return FilterResult with all categories and stats.
        """

    def detect_blur(self, frame: np.ndarray) -> Tuple[bool, float]:
        """
        Compute Laplacian variance.
        Convert to grayscale if needed.
        Return (is_blurry: bool, variance: float)
        Threshold: variance < self.blur_threshold → blurry
        """

    def detect_duplicate(self, frame: np.ndarray, prev_frame: np.ndarray) -> Tuple[bool, float]:
        """
        Compute SSIM between current frame and previous good frame.
        Downscale both to 256x256 grayscale first for speed.
        Return (is_duplicate: bool, ssim_score: float)
        Threshold: ssim > self.duplicate_threshold → duplicate
        """

    def detect_transition(self, frames_window: List[np.ndarray]) -> bool:
        """
        Compute Farneback dense optical flow between consecutive frames in window.
        If mean flow magnitude > threshold for all pairs in window → transition.
        Window size = self.transition_min_frames (default 3)
        Downscale frames to 320x240 for speed.
        Return True if this is a transition (head turning, walking).
        """

    def detect_darkness(self, frame: np.ndarray) -> Tuple[bool, str]:
        """
        Check mean brightness and histogram entropy.
        - mean < self.brightness_low → "too_dark"
        - mean > self.brightness_high → "overexposed"
        - entropy < self.entropy_threshold → "occluded" (hand covering camera)
        Return (is_bad: bool, reason: str)
        """

    def print_stats(self, result: FilterResult):
        """Print a nice summary table of filtering results:
        Total frames: 685
        Good frames:  298 (43.5%)
        Skipped blur: 82 (12.0%)
        Skipped dup:  145 (21.2%)
        Skipped trans: 93 (13.6%)
        Skipped dark: 67 (9.8%)
        Estimated API cost saved: ₹XX
        """

All methods must be fully implemented with real OpenCV code. Use logging. Include standalone test.
```

---

## STEP 4: Hand & Activity Detection (Stage 3a)

### Prompt:

```
Build two files:

1. models/hand_pose.py — MediaPipe Hands wrapper

class HandPoseEstimator:
    def __init__(self, max_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        # Initialize MediaPipe Hands
        # Use mp.solutions.hands.Hands()

    def detect(self, frame: np.ndarray) -> HandPoseResult:
        """
        Run MediaPipe Hands on a single BGR frame.
        
        Returns HandPoseResult dataclass:
        - hands: List[HandDetection]
          - HandDetection:
            - handedness: "left" | "right"
            - confidence: float
            - keypoints_2d: List[Tuple[float, float]]  # 21 keypoints, normalized 0-1
            - keypoints_pixel: List[Tuple[int, int]]    # 21 keypoints, pixel coords
            - bbox: Tuple[int, int, int, int]            # x1, y1, x2, y2
        - num_hands: int
        - hand_visibility: "both_full" | "left_only" | "right_only" | "partial" | "no_hands"
        """

    def detect_batch(self, frames: List[np.ndarray]) -> List[HandPoseResult]:
        """Process multiple frames."""

    def get_grasp_type(self, hand: HandDetection) -> Tuple[str, float]:
        """
        Classify grasp type from hand keypoints geometry.
        
        Logic:
        - Compute distances between fingertips (landmarks 4,8,12,16,20)
        - Compute thumb-index angle
        - Compute finger curl angles (MCP→PIP→DIP chain for each finger)
        - Rules:
          - thumb + index close, others extended → "pinch" 
          - all fingers curled tight, thumb wrapped → "power"
          - thumb pressing against side of index → "lateral"
          - fingers curled in hook shape, thumb relaxed → "hook"
          - fingers spread around large area → "spherical"
          - if all fingers extended and spread → "no_contact"
        
        Return (grasp_type: str, confidence: float)
        """

    def close(self):
        """Release MediaPipe resources."""


2. pipeline/hand_activity_detector.py — combines hand detection with activity scoring

class HandActivityDetector:
    def __init__(self, config: AnnotatorConfig):
        self.hand_estimator = HandPoseEstimator()
        self.activity_threshold = config.activity_threshold

    def process_frames(self, good_frames: List[Tuple[float, str]]) -> ActivityResult:
        """
        For each good frame:
        1. Run hand detection
        2. Compute activity score based on:
           - Hand presence (no hands = 0)
           - Hand movement between consecutive frames (wrist displacement)
           - Finger movement (keypoint displacement between frames)
           - Hand bbox area change (grasping = area changes)
        3. Classify frame:
           - activity_score > 0.6 → "active_manipulation" (definitely send to VLM)
           - activity_score 0.2-0.6 → "possible_activity" (send to VLM)
           - activity_score < 0.2 → "idle_or_observing" (skip VLM, label as idle)
        
        Returns ActivityResult:
        - frames_to_vlm: List of (timestamp, filepath, hand_metadata)
        - frames_idle: List of (timestamp, filepath) — auto-labelled as idle
        - per_frame_activity: dict mapping timestamp → activity_score
        - per_frame_hands: dict mapping timestamp → HandPoseResult
        - stats: counts and percentages
        """

    def compute_activity_score(self, current_hands: HandPoseResult, 
                                prev_hands: HandPoseResult) -> float:
        """
        Score 0-1 based on:
        - 0.4 weight: hand presence (0 if no hands, 0.5 if one, 1.0 if both)
        - 0.3 weight: hand movement speed (wrist displacement between frames)
        - 0.2 weight: finger movement (mean keypoint displacement)
        - 0.1 weight: bbox area change (grasping causes size change)
        
        Normalize each component to 0-1, weighted sum.
        """

All methods fully implemented. Use proper MediaPipe API. Handle errors (MediaPipe fails on gloved hands — log warning, set confidence to 0). Include standalone tests.
```

---

## STEP 5: Object Detection + Depth (Stage 3b, 3c, 3d)

### Prompt:

```
Build three model wrappers. These will run on GPU in Colab.

1. models/object_detector.py — YOLO-World wrapper

class ObjectDetector:
    def __init__(self, model_size="s", custom_classes=None):
        """
        Load YOLO-World model using ultralytics.
        Default: yolov8s-worldv2
        Set custom class names from config.object_taxonomy
        """

    def detect(self, frame: np.ndarray) -> List[ObjectDetection]:
        """
        Run YOLO-World open-vocabulary detection.
        
        ObjectDetection dataclass:
        - class_name: str
        - confidence: float
        - bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
        - bbox_normalized: Tuple[float, float, float, float]  # 0-1
        - center: Tuple[int, int]
        
        Filter detections below confidence 0.3.
        Return sorted by confidence descending.
        """

    def detect_batch(self, frames: List[np.ndarray]) -> List[List[ObjectDetection]]:
        """Batch detection for efficiency."""

    def set_classes(self, classes: List[str]):
        """Update the class vocabulary at runtime."""


2. models/depth_estimator.py — Depth Anything V2 wrapper

class DepthEstimator:
    def __init__(self, model_size="small", device="cuda"):
        """
        Load Depth Anything V2 small model.
        Use transformers pipeline or direct model loading.
        Model: depth-anything/Depth-Anything-V2-Small-hf
        Falls back to CPU if CUDA not available.
        """

    def estimate(self, frame: np.ndarray) -> DepthResult:
        """
        Run monocular depth estimation.
        
        DepthResult:
        - depth_map: np.ndarray (H, W) float32, relative depth 0-1
        - depth_at_point(x, y) → float: helper to get depth at pixel
        - save_depth_map(path): save as .npy
        - visualize() → np.ndarray: colorized depth map for debugging
        """

    def estimate_sparse(self, frame: np.ndarray, points: List[Tuple[int,int]]) -> List[float]:
        """
        Get depth at specific points only (faster for hand-object depth comparison).
        Run full estimation, then sample at points.
        """


3. models/interaction_detector.py — Hand-Object Interaction

class HandObjectInteractionDetector:
    def __init__(self):
        pass

    def detect_interactions(self, hand_result: HandPoseResult, 
                           objects: List[ObjectDetection],
                           depth_result: DepthResult = None) -> List[Interaction]:
        """
        Determine which hand is interacting with which object.
        
        For each hand × each object:
        1. Compute IoU between hand bbox and object bbox
        2. If depth available: compute depth difference at hand wrist and object center
        3. Check if any fingertip keypoints are inside object bbox
        4. Scoring:
           - IoU > 0.3 → high contact likelihood
           - IoU > 0.1 AND depth_diff < 0.05 → contact with depth confirmation
           - Fingertip inside object bbox → strong contact signal
           - Weighted score from all three signals
        
        Interaction dataclass:
        - hand: "left" | "right"
        - object: ObjectDetection
        - contact: bool
        - contact_score: float (0-1)
        - iou: float
        - depth_diff: float | None
        - fingertips_inside: int (count of fingertips inside object bbox)
        - grasp_type: str (from HandPoseEstimator.get_grasp_type if contact)
        - grasp_confidence: float
        """

    def compute_iou(self, bbox1, bbox2) -> float:
        """Standard IoU between two bounding boxes."""

    def fingertips_in_bbox(self, hand: HandDetection, bbox: Tuple) -> int:
        """Count how many of the 5 fingertip landmarks fall inside the bbox."""

All fully implemented. Handle model download gracefully (first run downloads weights). 
Log model loading time and inference time per frame. Include standalone tests.

IMPORTANT: For Depth Anything V2, if transformers doesn't have it yet, use the 
huggingface pipeline: pipe = pipeline("depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf")
If that fails too, implement a fallback using MiDaS from torch.hub.
```

---

## STEP 6: Local Vision Pipeline (Stage 3 Integration)

### Prompt:

```
Build pipeline/local_vision_pipeline.py — integrates all Stage 3 models.

This is the orchestrator that runs all local models on each frame.

from models.hand_pose import HandPoseEstimator
from models.object_detector import ObjectDetector
from models.depth_estimator import DepthEstimator
from models.interaction_detector import HandObjectInteractionDetector

@dataclass
class FrameAnnotation:
    timestamp: float
    frame_path: str
    hand_pose: HandPoseResult
    objects: List[ObjectDetection]
    depth: DepthResult | None
    interactions: List[Interaction]
    activity_score: float
    hand_visibility: str
    processing_time_ms: float

class LocalVisionPipeline:
    def __init__(self, config: AnnotatorConfig, use_gpu=True):
        """
        Initialize all models. 
        - HandPoseEstimator: always loads (CPU, lightweight)
        - ObjectDetector: loads with config.object_taxonomy
        - DepthEstimator: loads if GPU available, skip on CPU-only with warning
        - InteractionDetector: no model to load (rule-based)
        
        Log which models loaded and total init time.
        """

    def process_frame(self, frame_path: str, timestamp: float, 
                      prev_annotation: FrameAnnotation = None) -> FrameAnnotation:
        """
        Run all models on a single frame:
        1. Load frame from path
        2. Hand pose detection
        3. Object detection
        4. Depth estimation (every Nth frame based on config, interpolate between)
        5. Hand-object interaction detection
        6. Compute activity score using hand movement from prev_annotation
        7. Return complete FrameAnnotation
        
        Time each step, log if verbose.
        Handle individual model failures gracefully — if hand detection fails,
        still run object detection etc.
        """

    def process_frames(self, good_frames: List[Tuple[float, str]], 
                       depth_every_n: int = 5) -> List[FrameAnnotation]:
        """
        Process all frames with tqdm progress bar.
        
        Optimization: run depth estimation every Nth frame (default 5) 
        and use the same depth map for intermediate frames. Saves 80% of 
        depth computation.
        
        Returns list of FrameAnnotation in timestamp order.
        """

    def generate_vlm_context(self, annotations: List[FrameAnnotation]) -> str:
        """
        Generate human-readable metadata string to include in VLM prompt.
        This enriches VLM accuracy by 15-20%.
        
        Example output:
        "Local CV metadata for frames 1.0s - 3.0s:
         Hands: right hand detected (conf 0.92), left hand detected (conf 0.88)
         Objects: wrench (conf 0.94, center [200,340]), bolt (conf 0.87, center [180,390])
         Interactions: right hand contacting wrench (IoU 0.35, grasp: power, conf 0.78)
         Activity: 0.85 (active manipulation)
         Depth: hand at 0.45m, wrench at 0.43m (contact consistent)"
        """

    def close(self):
        """Release all model resources."""

Fully implemented. This is the most critical integration module. Test with a real image if possible,
or create a synthetic test that validates the data flow between all models.
```

---

## STEP 7: VLM Clients (Stage 4a — Bootstrap)

### Prompt:

```
Build the VLM client layer — 4 files:

1. vlm/base_client.py

class BaseVLMClient:
    """Abstract base for all VLM backends."""
    
    def __init__(self, api_key: str, max_concurrent: int = 3):
        self.client = httpx.AsyncClient(timeout=60.0)
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.total_cost = 0.0
        self.total_calls = 0

    async def predict_batch(self, frames: List[str], context: str) -> dict:
        """Send frame batch to VLM. Must be implemented by subclass."""
        raise NotImplementedError

    async def predict_batch_with_retry(self, frames, context) -> dict:
        """
        Wrap predict_batch with:
        - Semaphore for concurrency control
        - Tenacity retry: 3 attempts, exponential backoff (2s, 4s, 8s)
        - If JSON parse fails, retry with stricter prompt
        - Track cost per call
        - Log call details
        """

    def parse_response(self, raw_text: str) -> dict:
        """
        Parse VLM JSON response.
        Strip markdown code fences if present.
        Validate required fields exist.
        Return parsed dict or raise ValueError.
        """


2. vlm/gemini_client.py

class GeminiClient(BaseVLMClient):
    """Google Gemini Flash API client."""
    
    def __init__(self, api_key: str, max_concurrent: int = 5):
        super().__init__(api_key, max_concurrent)
        self.model = "gemini-2.0-flash"
        self.api_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
        self.cost_per_call = 0.05  # approximate INR per 5-frame batch

    async def predict_batch(self, frame_base64_list: List[str], context: str) -> dict:
        """
        Build Gemini API request with:
        - System instruction: the egocentric action recognition prompt
        - User content: inline_data images (base64) + context text
        - POST to API with API key as query param
        
        Parse response from candidates[0].content.parts[0].text
        """


3. vlm/claude_client.py

class ClaudeClient(BaseVLMClient):
    """Anthropic Claude API client."""
    
    def __init__(self, api_key: str, model="claude-sonnet-4-20250514", max_concurrent: int = 3):
        super().__init__(api_key, max_concurrent)
        self.model = model
        self.api_url = "https://api.anthropic.com/v1/messages"
        self.cost_per_call = 0.35  # INR per batch for Sonnet

    async def predict_batch(self, frame_base64_list: List[str], context: str) -> dict:
        """
        Build Anthropic Messages API request:
        - Headers: x-api-key, anthropic-version 2023-06-01
        - System: egocentric action recognition prompt
        - User message: list of image content blocks (base64, media_type image/jpeg)
          + text block with context
        - max_tokens: 500
        """


4. vlm/openai_client.py

class OpenAIClient(BaseVLMClient):
    """OpenAI GPT-4o-mini client."""
    
    def __init__(self, api_key: str, max_concurrent: int = 3):
        super().__init__(api_key, max_concurrent)
        self.model = "gpt-4o-mini"
        self.api_url = "https://api.openai.com/v1/chat/completions"
        self.cost_per_call = 0.10

    async def predict_batch(self, frame_base64_list: List[str], context: str) -> dict:
        """
        Build OpenAI Chat Completions request:
        - System message: egocentric prompt
        - User message: list of image_url content (base64 data URIs) + text
        - max_tokens: 500
        """

CRITICAL: Store the full egocentric VLM system prompt as a constant in vlm/base_client.py called EGOCENTRIC_SYSTEM_PROMPT. All three clients use the same system prompt. The prompt is:

[Include the full egocentric system prompt from the v3 document — the one that starts with "You are an expert at recognizing human manipulation actions from EGOCENTRIC (first-person) video." and includes the full action taxonomy and manipulation phases]

All clients must use async httpx. All must handle rate limiting, JSON parse errors, and network timeouts. Log every call with timestamp, cost, and response time. Track cumulative cost.
```

---

## STEP 8: Task Labeller (Stage 4 Integration)

### Prompt:

```
Build pipeline/task_labeller.py — dual-mode task labelling.

from vlm.gemini_client import GeminiClient
from vlm.claude_client import ClaudeClient
from vlm.openai_client import OpenAIClient
from utils.image_utils import frame_to_base64, resize_frame

@dataclass
class TaskLabel:
    action: str
    action_description: str
    objects_involved: List[str]
    grasp_type: str
    hand_used: str
    manipulation_phase: str
    task_hierarchy: dict  # high_level, mid_level, low_level
    confidence: float
    is_idle: bool
    is_transition: bool
    labelling_method: str  # "vlm" | "local_model" | "local_model_vlm_fallback"
    was_fallback: bool
    vlm_backend: str | None
    raw_response: dict

class TaskLabeller:
    def __init__(self, mode: str, config: AnnotatorConfig):
        """
        mode: "bootstrap" or "annotate"
        
        bootstrap: use VLM client based on config.vlm_backend
        annotate: use local model, with optional VLM fallback
        """
        self.mode = mode
        if mode == "bootstrap" or config.fallback_vlm:
            self.vlm_client = self._create_vlm_client(config)
        if mode == "annotate":
            self.local_model = self._load_local_model(config)

    def _create_vlm_client(self, config) -> BaseVLMClient:
        """Create appropriate VLM client based on config.vlm_backend"""

    def _load_local_model(self, config):
        """
        Load fine-tuned VideoMAE-v2 model from config.model_path.
        If model_path is empty or file doesn't exist, raise clear error.
        Load class_mapping.json from same directory.
        Load confused_pairs.json for fallback triggers.
        """

    async def label_batch(self, frame_paths: List[Tuple[float, str]], 
                          frame_annotations: List[FrameAnnotation]) -> TaskLabel:
        """
        Label a batch of frames.
        
        If bootstrap mode:
        1. Resize and base64 encode frames
        2. Generate VLM context from frame_annotations (local CV metadata)
        3. Send to VLM client
        4. Parse response into TaskLabel
        
        If annotate mode:
        1. Run local model on frames
        2. If confidence < config.fallback_threshold AND fallback enabled:
           a. Check if action is in confused_pairs → definitely fallback
           b. Send to VLM as fallback
           c. Mark was_fallback = True
        3. Return TaskLabel
        """

    async def label_video(self, good_frames: List[Tuple[float, str]],
                          frame_annotations: List[FrameAnnotation],
                          idle_frames: List[Tuple[float, str]]) -> List[TaskLabel]:
        """
        Process entire video:
        1. Create batches of batch_size consecutive good frames
        2. Run label_batch on each batch (async, respecting max_concurrent)
        3. For idle_frames, create TaskLabel with action="idle" directly (no VLM call)
        4. Show tqdm progress bar
        5. Print running cost estimate every 10 batches
        6. Return all TaskLabels in timestamp order
        """

    def get_cost_summary(self) -> dict:
        """Return total VLM calls made, total cost in INR, cost per minute of video."""

Fully implemented. Handle all error cases. The async batching with semaphore is critical for performance.
```

---

## STEP 9: Trajectory Extraction (Stage 5)

### Prompt:

```
Build pipeline/trajectory_extractor.py — 6-DoF trajectory extraction from pose + depth.

Pure numpy/scipy computation, no ML models needed. This should be fast.

@dataclass
class Trajectory:
    hand: str  # "left" | "right"
    timestamps: List[float]
    positions_2d: np.ndarray          # (T, 2) pixel coords of wrist
    positions_3d: np.ndarray | None   # (T, 3) if depth available
    orientations: np.ndarray | None   # (T, 3) roll, pitch, yaw from hand keypoints
    trajectory_6dof: np.ndarray | None  # (T, 6) combined [x,y,z,r,p,y]
    velocity: np.ndarray              # (T-1, 2 or 3) 
    acceleration: np.ndarray          # (T-2, 2 or 3)
    grasp_signal: np.ndarray          # (T,) float 0=open, 1=closed
    tool_tip_trajectory: np.ndarray | None  # (T, 2 or 3) if tool detected
    path_length: float
    max_velocity: float
    coordinate_frame: str             # "pixel" | "camera_relative"

@dataclass
class GraspEvent:
    timestamp: float
    event_type: str  # "open" | "close"
    duration_to_next: float | None

class TrajectoryExtractor:
    def __init__(self, smoothing_window=5):
        self.smoothing_window = smoothing_window

    def extract(self, frame_annotations: List[FrameAnnotation]) -> dict:
        """
        Extract trajectories for each detected hand.
        
        Returns dict: {
            "left": Trajectory | None,
            "right": Trajectory | None,
            "grasp_events": List[GraspEvent],
            "bimanual": bool  # whether both hands active
        }
        """

    def extract_wrist_trajectory(self, annotations, hand="right") -> np.ndarray:
        """
        Get wrist position (landmark 0) across frames.
        Handle missing detections: interpolate gaps < 5 frames, 
        mark gaps > 5 frames as NaN.
        Return (T, 2) array of pixel positions.
        """

    def extract_3d_trajectory(self, annotations, hand="right") -> np.ndarray | None:
        """
        If depth available, combine wrist (x, y) with depth at wrist → (x, y, z).
        z = depth_map[wrist_y, wrist_x]
        Return (T, 3) array or None if no depth data.
        """

    def extract_orientation(self, annotations, hand="right") -> np.ndarray | None:
        """
        Compute hand orientation from keypoints:
        - Palm normal: cross product of (MCP_index - wrist) × (MCP_pinky - wrist)  
        - Finger direction: wrist → middle_finger_MCP vector
        - Convert to roll, pitch, yaw using atan2
        Return (T, 3) array.
        """

    def compute_grasp_signal(self, annotations, hand="right") -> np.ndarray:
        """
        Track thumb-tip to index-tip distance over time.
        Normalize: max observed distance = 0 (open), min = 1 (closed).
        Apply sigmoid smoothing so transitions are gradual.
        Return (T,) float array.
        """

    def detect_grasp_events(self, grasp_signal: np.ndarray, 
                            timestamps: List[float]) -> List[GraspEvent]:
        """
        Find grasp open/close transitions.
        Close event: grasp_signal crosses 0.5 going up.
        Open event: grasp_signal crosses 0.5 going down.
        """

    def smooth_trajectory(self, trajectory: np.ndarray) -> np.ndarray:
        """
        Savitzky-Golay filter for smoothing.
        Window = self.smoothing_window, polyorder = 2.
        Handle NaN values (missing detections) by interpolating first.
        """

    def compute_derivatives(self, trajectory: np.ndarray, dt: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute velocity and acceleration from position trajectory.
        velocity = diff(position) / dt
        acceleration = diff(velocity) / dt
        Apply smoothing to derivatives to reduce noise.
        Return (velocity, acceleration).
        """

Fully implemented with numpy and scipy. No ML models needed. This should run in milliseconds per video.
Include standalone test with synthetic data.
```

---

## STEP 10: Segment Merger + Quality Scorer (Stage 6 + 7)

### Prompt:

```
Build two files:

1. pipeline/segment_merger.py — merge per-batch labels into coherent segments

@dataclass
class Segment:
    id: int
    start_time: float
    end_time: float
    duration: float
    action: str
    action_description: str
    task_hierarchy: dict
    objects_involved: List[str]
    grasp_type: str
    hand_used: str
    manipulation_phases: List[str]
    trajectory_summary: dict  # path_length, max_velocity, grasp_events
    confidence_avg: float
    confidence_min: float
    frames_analyzed: int
    frames_skipped: int
    labelling_method: str
    needs_review: bool
    review_reason: str | None  # "low_confidence" | "phase_sequence_broken" | "hand_not_visible"

class SegmentMerger:
    def __init__(self, config: AnnotatorConfig):
        self.min_segment_duration = config.min_segment_duration  # 0.5s
        self.idle_threshold = config.idle_segment_threshold       # 2.0s

    def merge(self, task_labels: List[TaskLabel], 
              trajectories: dict,
              transition_ranges: List[Tuple[float, float]]) -> List[Segment]:
        """
        1. Group consecutive labels with same action into segments
        2. Insert transition segments from Stage 2's transition_ranges
        3. Merge segments shorter than min_segment_duration with adjacent
        4. Split idle segments longer than idle_threshold into explicit idle segments
        5. For each segment:
           - Compute confidence_avg and confidence_min
           - Extract manipulation_phases list
           - Check phase sequence validity (approach→contact→manipulation→release→retreat)
           - Compute trajectory_summary from trajectory data
        6. Flag needs_review:
           - confidence_avg < 0.4 → "low_confidence"
           - phase sequence invalid → "phase_sequence_broken"
           - hand not visible in >50% of frames → "hand_not_visible"
        7. Assign sequential IDs
        8. Return sorted by start_time
        """

    def validate_phase_sequence(self, phases: List[str]) -> bool:
        """
        Check if manipulation phases follow valid progression.
        Valid: approach→contact→manipulation→release→retreat (any subset in order)
        Invalid: retreat→approach (went backwards), manipulation without contact, etc.
        Return True if valid.
        """


2. pipeline/quality_scorer.py — composite quality scoring

@dataclass
class QualityScore:
    composite: float  # weighted average 0-1
    pose_completeness: float
    object_visibility: float
    trajectory_smoothness: float
    action_clarity: float
    frame_quality: float
    depth_consistency: float
    may_discard: bool  # composite < 0.2

class QualityScorer:
    def __init__(self):
        self.weights = {
            "pose_completeness": 0.25,
            "object_visibility": 0.20,
            "trajectory_smoothness": 0.20,
            "action_clarity": 0.15,
            "frame_quality": 0.10,
            "depth_consistency": 0.10
        }

    def score_segment(self, segment: Segment, 
                      frame_annotations: List[FrameAnnotation],
                      trajectory: Trajectory | None) -> QualityScore:
        """
        Compute each sub-score:
        
        pose_completeness: % of frames with hand pose detected, 
            weighted by number of keypoints visible (21 = perfect)
        
        object_visibility: average object detection confidence across frames,
            penalize frames where manipulated object not detected
        
        trajectory_smoothness: inverse of mean jerk magnitude (smoothness metric).
            Normalize so smooth trajectory = 1.0, very jerky = 0.0.
            If no trajectory data, default 0.5.
        
        action_clarity: task label confidence averaged across batches in segment.
            Bonus if phase sequence is valid (+0.1).
        
        frame_quality: average Laplacian variance of frames in segment,
            normalized to 0-1. Also factor in brightness consistency.
        
        depth_consistency: check depth values at hand/object positions are 
            physically plausible (no sudden jumps > 0.5m between consecutive frames).
            If no depth data, default 0.5.
        
        composite = weighted sum
        may_discard = composite < 0.2
        """

    def score_video(self, segments, frame_annotations, trajectories) -> dict:
        """Score all segments. Return dict mapping segment_id → QualityScore.
        Also return aggregate stats: mean, median, min, distribution."""

Both fully implemented. All computation is numpy-based, no ML models needed.
```

---

## STEP 11: Output Writer (Stage 8)

### Prompt:

```
Build pipeline/output_writer.py — generates all output formats.

class OutputWriter:
    def __init__(self, config: AnnotatorConfig):
        self.output_format = config.output_format
    
    def write_all(self, output_dir: str, video_info: dict,
                  segments: List[Segment], frame_annotations: List[FrameAnnotation],
                  trajectories: dict, quality_scores: dict,
                  processing_stats: dict) -> List[str]:
        """
        Generate all outputs. Always write JSON + CSV + timeline + review queue.
        Additionally write RLDS or HDF5 based on config.
        Return list of output file paths.
        """

    def write_json(self, output_path: str, video_info, segments, 
                   quality_scores, processing_stats):
        """
        Full structured JSON with:
        - video_file, video_type, camera_mount, duration, fps, vlm_backend
        - processing_stats: frames extracted/skipped by category/sent to VLM/cost
        - segments: full segment data with all fields
        - action_summary: per-action counts, durations, avg confidence
        - manipulation_profile: dominant hand, bimanual%, idle%, most used grasp, etc.
        """

    def write_csv(self, output_path: str, segments):
        """
        Flat CSV. Columns:
        segment_id, start_time, end_time, duration, action, description, 
        objects, grasp_type, hand, manipulation_phases, confidence_avg,
        quality_score, needs_review, review_reason, labelling_method
        """

    def write_timeline(self, output_path: str, segments):
        """
        Human-readable timeline:
        00:00 - 00:05  [0.85] reaching_wrench — Reaching toward wrench (right, approach→contact)
        00:05 - 00:08  [0.92] grasping_wrench — Grasping wrench with power grip (right, contact→manip)
        00:53 - 00:56  [---]  TRANSITION — Head movement / walking
        00:56 - 01:02  [0.45] unknown — ⚠️ NEEDS REVIEW (low confidence)
        """

    def write_review_queue(self, output_path: str, segments, frame_annotations):
        """
        JSON with segments needing review:
        {
          "review_clips": [
            {
              "segment_id": 12,
              "start_time": 56.0, "end_time": 62.0,
              "reason": "low_confidence",
              "suggested_action": "inspecting_part",
              "confidence": 0.35,
              "frame_paths": [...]
            }
          ],
          "total_review_needed": 3,
          "estimated_review_time_minutes": 15
        }
        """

    def write_rlds(self, output_path: str, segments, frame_annotations, trajectories):
        """
        RLDS format — TFRecord files.
        
        Dataset structure matching Open X-Embodiment:
        - Each segment becomes an "episode"
        - Each frame in the episode becomes a "step"
        - step:
          - observation:
            - image: (H, W, 3) uint8 — RGB frame resized to 256x256
            - depth: (H, W) float32 — depth map (if available)
          - action: (7,) float32 — [dx, dy, dz, droll, dpitch, dyaw, grasp]
            (delta from previous frame, computed from trajectory)
          - language_instruction: str — action_description
          - is_first: bool
          - is_last: bool
          - is_terminal: bool
        
        Use tensorflow.data and tf.io.TFRecordWriter.
        Handle tensorflow import gracefully — if not installed, skip with warning.
        """

    def write_hdf5(self, output_path: str, segments, frame_annotations, trajectories):
        """
        HDF5 format matching DROID/robomimic:
        /episode_N/
            /obs/
                /images/cam_head     (T, 256, 256, 3) uint8
                /depth/cam_head      (T, 256, 256) float32
                /hand_pose           (T, 2, 21, 2) float32  [2 hands, 21 kps, xy]
            /action                  (T, 7) float32
            /language_instruction    str attribute
            /quality_scores          (T,) float32
            /action_labels           variable-length string array
        
        Use h5py.
        """

Fully implemented. Handle missing data gracefully (no depth → skip depth fields, no trajectory → skip action deltas). Log output file sizes.
```

---

## STEP 12: Main Orchestrator (Wire Everything Together)

### Prompt:

```
Now wire everything together. Update annotator.py to be the full orchestrator.

The main pipeline flow:

async def run_bootstrap(video_path: str, config: AnnotatorConfig):
    """Bootstrap mode: VLM-based labelling with local CV enrichment."""
    
    logger.info(f"Starting bootstrap annotation: {video_path}")
    
    # Stage 1: Extract frames
    extractor = FrameExtractor(config)
    extraction = extractor.extract(video_path, config.output_dir)
    logger.info(f"Extracted {extraction.total_frames} frames")
    
    # Stage 2: Filter bad frames
    quality_filter = FrameQualityFilter(config)
    filter_result = quality_filter.filter(extraction.frame_paths)
    quality_filter.print_stats(filter_result)
    
    # Cost estimate checkpoint
    estimator = CostEstimator()
    cost = estimator.estimate(
        total_frames=extraction.total_frames,
        frames_after_filter=len(filter_result.good_frames),
        vlm_backend=config.vlm_backend,
        batch_size=config.batch_size
    )
    logger.info(f"Estimated cost: ₹{cost['total_inr']:.2f}")
    
    if config.cost_estimate:
        estimator.print_report(cost)
        return
    
    # Stage 3: Local vision pipeline
    vision = LocalVisionPipeline(config)
    frame_annotations = vision.process_frames(filter_result.good_frames)
    
    # Stage 4: VLM task labelling with enriched context
    labeller = TaskLabeller("bootstrap", config)
    task_labels = await labeller.label_video(
        filter_result.good_frames, frame_annotations, 
        idle_frames=[]  # frames with activity < threshold
    )
    labeller_cost = labeller.get_cost_summary()
    logger.info(f"VLM cost: ₹{labeller_cost['total_inr']:.2f}")
    
    # Stage 5: Trajectory extraction
    traj_extractor = TrajectoryExtractor()
    trajectories = traj_extractor.extract(frame_annotations)
    
    # Stage 6: Segment merging
    merger = SegmentMerger(config)
    segments = merger.merge(task_labels, trajectories, filter_result.skipped_transition)
    
    # Stage 7: Quality scoring
    scorer = QualityScorer()
    quality_scores = scorer.score_video(segments, frame_annotations, trajectories)
    
    # Stage 8: Output
    writer = OutputWriter(config)
    processing_stats = {
        "total_frames_extracted": extraction.total_frames,
        "frames_skipped_blur": len(filter_result.skipped_blur),
        "frames_skipped_duplicate": len(filter_result.skipped_duplicate),
        "frames_skipped_transition": len(filter_result.skipped_transition),
        "frames_skipped_dark": len(filter_result.skipped_dark),
        "frames_sent_to_vlm": labeller_cost["total_calls"] * config.batch_size,
        "vlm_calls_made": labeller_cost["total_calls"],
        "estimated_cost_inr": labeller_cost["total_inr"],
        "processing_time_seconds": time.time() - start_time
    }
    output_files = writer.write_all(
        config.output_dir, extraction.video_metadata,
        segments, frame_annotations, trajectories, quality_scores, processing_stats
    )
    
    # Cleanup
    vision.close()
    
    logger.info(f"Done! {len(segments)} segments, {len([s for s in segments if s.needs_review])} need review")
    logger.info(f"Output files: {output_files}")


async def run_annotate(video_path: str, config: AnnotatorConfig):
    """Production mode: local model + optional VLM fallback."""
    # Same flow as bootstrap but Stage 4 uses local model
    # Same structure, just different TaskLabeller mode


def handle_batch(input_dir: str, config: AnnotatorConfig, mode: str):
    """
    Process all video files in a directory.
    Supported formats: .mp4, .avi, .mov, .mkv, .webm
    Process each video sequentially (GPU memory constraint).
    Print overall summary at end.
    Support --resume: skip videos that already have output files.
    """

# Update the argparse CLI to call these functions.
# Handle Ctrl+C gracefully: save partial results via ProgressTracker.
# At the end, print a nice summary table with:
#   - Total videos processed
#   - Total segments
#   - Total cost
#   - Average quality score
#   - Number of segments needing review

Wire everything together. Import all modules. Handle all edge cases.
This is the final integration step — make sure the full pipeline runs end-to-end.

Also create a file called run_colab.py that provides a simple function interface 
for calling from Colab cells (without argparse):

def bootstrap_video(video_path, output_dir, vlm="gemini", fps=2, verbose=True):
    """Convenience function for Colab usage."""

def annotate_video(video_path, model_path, output_dir, output_format="rlds", verbose=True):
    """Convenience function for Colab usage."""
```

---

## STEP 13: Training Pipeline (Mode 2)

### Prompt:

```
Build the training pipeline — 4 files:

1. training/dataset.py

class ActionVideoDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir: str, min_confidence=0.6, 
                 num_frames=16, frame_size=224, augment=False, use_metadata=False):
        """
        Load bootstrap labels from data_dir.
        
        1. Find all *_labels.json files in data_dir
        2. Load segments from each, filter by confidence > min_confidence
        3. Build class vocabulary from unique action labels
        4. For each segment, store: video_path, start_time, end_time, action_class
        5. If use_metadata: also load corresponding FrameAnnotation data
        
        Save class_mapping as self.class_to_idx and self.idx_to_class
        """

    def __getitem__(self, idx):
        """
        1. Load video file for this segment
        2. Extract num_frames evenly spaced between start_time and end_time
        3. Resize to frame_size × frame_size
        4. Apply augmentations if enabled
        5. Normalize: ImageNet mean/std
        6. Stack into (C, T, H, W) tensor
        7. If use_metadata: also return feature vector from FrameAnnotation
        8. Return (video_tensor, class_label) or (video_tensor, metadata_features, class_label)
        """

    def __len__(self):
        return len(self.samples)

    def get_class_distribution(self) -> dict:
        """Return count per class. Useful for class-weighted loss."""

    def save_class_mapping(self, path: str):
        """Save class_mapping.json"""


2. training/augmentations.py

class EgocentricAugmentation:
    """Augmentations specific to egocentric video."""
    
    def __init__(self):
        self.transforms = []

    def __call__(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """
        Apply random subset of augmentations to ALL frames in clip consistently
        (same transform to all frames to maintain temporal coherence):
        
        - Random rotation ±15° (simulates head tilt)
        - Random brightness ±20% and contrast ±15% (factory lighting)
        - Random crop from center-biased Gaussian distribution
          (hands/action usually center-bottom of egocentric frame)
        - Temporal jitter: randomly shift which frames are selected by ±2
        - Horizontal flip with 50% probability
          IMPORTANT: if flipped, swap left/right hand labels
        - Color jitter: hue ±10, saturation ±20% (different glove colors, lighting)
        - Random Gaussian blur (simulates slight focus issues)
        
        DO NOT apply:
        - Vertical flip (physically impossible in egocentric)
        - Cutout/erasing on center region (would destroy hand/action area)
        - Extreme rotation >20° (unrealistic head motion)
        """


3. training/trainer.py

class ActionModelTrainer:
    def __init__(self, config: dict):
        """
        config keys: model_name, num_classes, num_frames, frame_size,
                     lr, epochs, batch_size, output_dir, use_metadata, gpu
        
        Load pretrained model:
        - "videomae-v2-small" → VideoMAE v2 Small from HuggingFace
          transformers: VideoMAEForVideoClassification
        - "timesformer-small" → TimeSformer from HuggingFace
        - Replace classification head with num_classes output
        
        If use_metadata: add a small MLP that takes metadata features,
        concatenate with video features before final classification head.
        """

    def train(self, train_dataset, val_dataset):
        """
        Standard PyTorch training loop:
        1. Create DataLoaders with appropriate batch_size, num_workers
        2. Use class-weighted CrossEntropyLoss (handle class imbalance)
        3. AdamW optimizer with cosine annealing scheduler
        4. For each epoch:
           - Train on train_dataset, track loss and accuracy
           - Validate on val_dataset
           - Save best model checkpoint (by val accuracy)
           - Print epoch summary
           - Early stopping if val loss doesn't improve for 5 epochs
        5. Save final model, class_mapping, training_config
        """

    def save_checkpoint(self, path, epoch, val_accuracy):
        """Save model state dict + optimizer + epoch + accuracy."""

    def load_checkpoint(self, path):
        """Load and resume training from checkpoint."""


4. training/evaluation.py

class ModelEvaluator:
    def __init__(self, model, class_mapping, device):
        pass

    def evaluate(self, val_dataset) -> EvalResult:
        """
        Run model on validation set.
        Compute:
        - Overall accuracy
        - Per-class accuracy
        - Macro/weighted F1 score
        - Confusion matrix (as numpy array)
        - Confused pairs: class pairs where misclassification rate > 10%
        """

    def plot_confusion_matrix(self, save_path: str):
        """Plot and save confusion matrix using seaborn heatmap."""

    def export_confused_pairs(self, save_path: str):
        """
        Save confused_pairs.json:
        [
          {"class_a": "tightening_bolt", "class_b": "loosening_bolt", "confusion_rate": 0.23},
          ...
        ]
        These pairs trigger VLM fallback in production mode.
        """

    def print_report(self):
        """Print formatted evaluation report."""


Now update train.py CLI to wire these together:
- Parse args: --data-dir, --model, --output-dir, --epochs, --batch-size, --lr, 
  --num-classes (auto), --min-confidence, --augment, --use-metadata, 
  --validation-split, --gpu
- Load dataset, split train/val, create trainer, train, evaluate, save outputs
- Print final summary with accuracy, F1, number of confused pairs

All fully implemented. Use HuggingFace transformers for model loading.
Handle GPU/CPU transparently. Handle cases where dataset is too small (<100 segments).
```

---

## STEP 14: Final Testing + Colab Notebook

### Prompt:

```
Create two final files:

1. test_pipeline.py — end-to-end smoke test

"""
Smoke test that validates the full pipeline works without real video or API keys.

1. Create a synthetic test video: 10 seconds, 640x480, 30fps
   - First 3 seconds: static background with a colored rectangle (simulates workbench)
   - Next 4 seconds: a circle moves across the frame (simulates hand motion)
   - Last 3 seconds: static again
   Use OpenCV VideoWriter to create this.

2. Run Stage 1 (frame extraction) on synthetic video
3. Run Stage 2 (quality filter) — should detect duplicates in static sections
4. Run Stage 3 (local vision) — will detect hands if MediaPipe finds them 
   (may not on synthetic, that's OK — test graceful failure)
5. Skip Stage 4 VLM (no API key needed for smoke test)
   Instead, create mock TaskLabels manually
6. Run Stage 5 (trajectory extraction) on whatever pose data exists
7. Run Stage 6 (segment merging) on mock labels
8. Run Stage 7 (quality scoring) on segments
9. Run Stage 8 (output) — JSON, CSV, timeline

Validate:
- All output files exist and are valid JSON/CSV
- Segment count > 0
- Timeline is human-readable
- No crashes, no unhandled exceptions

Print: PASS/FAIL for each stage.
Run with: python test_pipeline.py
"""


2. colab_notebook.py — Template for the Colab notebook cells

"""
This file contains the code for each Colab cell, clearly separated.
Copy each section into a separate Colab cell.
"""

# ============ CELL 1: SETUP ============
CELL_1_SETUP = '''
# Clone the repo
!git clone https://github.com/YOUR_USERNAME/manudata-annotator.git
%cd manudata-annotator

# Install dependencies
!pip install -r requirements.txt -q

# Verify GPU
import torch
print(f"GPU available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
'''

# ============ CELL 2: MOUNT DRIVE ============
CELL_2_DRIVE = '''
from google.colab import drive
drive.mount('/content/drive')

# Create output directory
!mkdir -p "/content/drive/MyDrive/manudata/annotations"
!mkdir -p "/content/drive/MyDrive/manudata/models"
'''

# ============ CELL 3: API KEYS ============
CELL_3_KEYS = '''
import os
os.environ['GOOGLE_API_KEY'] = 'YOUR_GEMINI_KEY'      # Required for bootstrap
os.environ['ANTHROPIC_API_KEY'] = 'YOUR_CLAUDE_KEY'    # Optional
os.environ['OPENAI_API_KEY'] = 'YOUR_OPENAI_KEY'       # Optional
'''

# ============ CELL 4: SMOKE TEST ============
CELL_4_TEST = '''
!python test_pipeline.py
'''

# ============ CELL 5: BOOTSTRAP SINGLE VIDEO ============
CELL_5_BOOTSTRAP_SINGLE = '''
from run_colab import bootstrap_video

bootstrap_video(
    video_path="/content/drive/MyDrive/factory_videos/test_clip.mp4",
    output_dir="/content/drive/MyDrive/manudata/annotations/",
    vlm="gemini",
    fps=2,
    verbose=True
)
'''

# ============ CELL 6: BOOTSTRAP BATCH ============
CELL_6_BOOTSTRAP_BATCH = '''
# Process all videos in a folder
!python annotator.py bootstrap \
  "/content/drive/MyDrive/factory_videos/" \
  --vlm gemini \
  --batch \
  --fps 2 \
  --output-dir "/content/drive/MyDrive/manudata/annotations/" \
  --verbose
'''

# ============ CELL 7: TRAIN LOCAL MODEL ============
CELL_7_TRAIN = '''
!python train.py \
  --data-dir "/content/drive/MyDrive/manudata/annotations/" \
  --model videomae-v2-small \
  --output-dir "/content/drive/MyDrive/manudata/models/" \
  --epochs 30 \
  --batch-size 8 \
  --lr 1e-4 \
  --augment \
  --validation-split 0.15 \
  --gpu 0
'''

# ============ CELL 8: PRODUCTION ANNOTATION ============
CELL_8_ANNOTATE = '''
from run_colab import annotate_video

annotate_video(
    video_path="/content/drive/MyDrive/factory_videos/",
    model_path="/content/drive/MyDrive/manudata/models/action_model_best.pt",
    output_dir="/content/drive/MyDrive/manudata/annotations/",
    output_format="rlds",
    verbose=True
)
'''

# ============ CELL 9: CHECK RESULTS ============
CELL_9_RESULTS = '''
import json
import os

output_dir = "/content/drive/MyDrive/manudata/annotations/"
for f in os.listdir(output_dir):
    if f.endswith("_labels.json"):
        with open(os.path.join(output_dir, f)) as fh:
            data = json.load(fh)
            print(f"\n{f}:")
            print(f"  Segments: {data['total_segments']}")
            print(f"  Need review: {data['needs_review_count']}")
            print(f"  Cost: ₹{data['processing_stats']['estimated_cost_inr']:.2f}")
            print(f"  Top actions:")
            for action, stats in sorted(data['action_summary'].items(), 
                                         key=lambda x: x[1]['total_duration'], reverse=True)[:5]:
                print(f"    {action}: {stats['count']}x, {stats['total_duration']:.1f}s")
'''

print("Colab notebook template ready!")
print("Copy each CELL_N variable into a separate Colab cell.")

Create both files with complete, runnable code.
```

---

## DONE!

After all 14 steps, you'll have a complete, production-ready annotation engine.

### Quick checklist before pushing to GitHub:

```powershell
# In Claude Code, ask:
"Run test_pipeline.py and fix any import errors or bugs"

# Then:
git init
git add .
git commit -m "ManuData Annotator v3 - complete annotation engine"
# Create repo on GitHub, then:
git remote add origin https://github.com/avinashpandey/manudata-annotator.git
git push -u origin main
```

### Total Claude Code sessions needed: 3-4
- Session 1: Steps 1-4 (skeleton + frames + filtering + hands)
- Session 2: Steps 5-8 (object/depth models + VLM + task labeller)  
- Session 3: Steps 9-12 (trajectory + merger + output + orchestrator)
- Session 4: Steps 13-14 (training + testing + Colab template)
