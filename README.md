# ManuData Annotator

Automated annotation pipeline for manufacturing process videos. Extracts frames, filters for quality, annotates via Vision-Language Models (VLMs), and optionally trains local models for cost-efficient inference.

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

## Project Structure

```
manudata-annotator/
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
