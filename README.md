# ManuData Annotator

Automated annotation pipeline for manufacturing process videos. Extracts frames, filters for quality, annotates via Vision-Language Models (VLMs), and optionally trains local models for cost-efficient inference.

This repository also contains **`manudata-qc-render`** (the `qc/` package), a separate batch tool that produces customer-facing hand-pose QC videos. See [Hand-pose QC renders](#hand-pose-qc-renders) below.

## Setup

```bash
# Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Linux/macOS

# Install dependencies
pip install -r requirements.txt

# Configure API keys
copy .env.example .env       # Windows
cp .env.example .env         # Linux/macOS
# Edit .env with your API keys
```

Ensure **FFmpeg** is installed and available on your PATH.

## Usage

### Mode 1: Bootstrap (VLM annotation)

Extract frames from a video and annotate using a cloud VLM:

```bash
python annotator.py bootstrap video.mp4 --vlm gemini --fps 2

# Cost estimate only
python annotator.py bootstrap video.mp4 --cost-estimate

# Process an entire directory
python annotator.py bootstrap ./videos/ --batch --vlm claude_haiku

# Resume an interrupted run
python annotator.py bootstrap video.mp4 --resume
```

### Mode 2: Annotate (local model + VLM fallback)

Run a fine-tuned local model with optional VLM fallback for low-confidence frames:

```bash
python annotator.py annotate video.mp4 --model-path weights/best.pt

# Custom output format
python annotator.py annotate video.mp4 --model-path weights/best.pt --output-format hdf5

# Disable VLM fallback
python annotator.py annotate video.mp4 --model-path weights/best.pt --no-fallback
```

### Mode 3: Train (fine-tune local model)

Fine-tune a local model on VLM-annotated data:

```bash
python train.py ./annotated_data/ --model-arch florence2 --epochs 10
```

## Hand-pose QC renders

`manudata-qc-render` turns a directory of source videos into one side-by-side QC
MP4 per video (RGB + 2D skeleton overlay | orbiting 3D skeleton), plus a batch
quality report and a combined customer reel.

```bash
pip install -e ".[wilor]"      # omit [wilor] to render from cached keypoints only

# 1. Label the footage (site pre-filled from folder names; fill in task)
manudata-qc-render init-manifest E:/videos --out E:/manudata_qc/manifest.csv

# 2. Analyse everything and rank it, rendering nothing
manudata-qc-render run E:/videos --out E:/manudata_qc --analyze-only

# 3. Render, resumably
manudata-qc-render run E:/videos --out E:/manudata_qc --resume --max-size-mb 50
```

Useful flags: `--clips-only` (render just the recommended 20–30 s window),
`--top N`, `--limit N`, `--smoke-test 10`, `--with-slam --slam-dir DIR`,
`--pose-backend cached` (re-render from existing `.npz`, no GPU or torch needed).

**Visibility reporting.** The two ways a frame can lack a pose are counted
separately and never merged: `hand_visible=0` means no hand was detected (a fact
about the factory floor), while `hand_visible=1, valid=0` means a hand was there
and the tracker lost it. `pose_recovery_pct` is computed over visible-hand slots
only, and that is what the ranking sorts on — so occlusion neither flatters nor
penalises the tracking figure. Both flags ship in the delivered `.npz`.

See [`docs/README_FOR_CUSTOMER.md`](docs/README_FOR_CUSTOMER.md) for the
conventions and flag semantics to send with a delivery, and
[`docs/VAST_SETUP.md`](docs/VAST_SETUP.md) for the rented-GPU runbook.

```bash
pytest tests/test_qc.py                       # unit tests, no GPU or footage needed
python -m tests.make_fixture /tmp/fixtures    # synthetic video + keypoints to try it on
```

## Project Structure

```
manudata-annotator/
├── qc/                   # manudata-qc-render (hand-pose QC renderer)
│   ├── cli.py            # CLI entry point
│   ├── io/               # ffmpeg decode/encode pipes, size targeting
│   ├── pose/             # keypoint schema, WiLoR backend, cache, gap handling
│   ├── render/           # 2D overlay, 3D viewport, timeline, compositing
│   └── report/           # analysis, ranking, qc_report.csv
├── docs/                 # customer README and Vast.ai runbook
├── annotator.py          # Main CLI (bootstrap + annotate)
├── train.py              # Training CLI
├── config.py             # Configuration and defaults
├── pipeline/             # Pipeline stage modules
├── models/               # Local model definitions
├── training/             # Training loop and data loading
├── vlm/                  # VLM backend clients
├── utils/                # Shared utilities
│   ├── image_utils.py    # Frame resizing, encoding, I/O
│   ├── video_utils.py    # FFprobe metadata extraction
│   ├── cost_estimator.py # API cost estimation
│   ├── progress_tracker.py # Resume support
│   └── logging_config.py # Logging setup
├── requirements.txt
└── .env.example
```
