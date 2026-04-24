"""ManuData Annotator — Colab Notebook Template.

This file contains the code for each Colab cell, clearly separated.
Copy each section into a separate Colab cell.

Usage:
    Run this file to print all cells:
        python colab_notebook.py
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
!python annotator.py bootstrap \\
  "/content/drive/MyDrive/factory_videos/" \\
  --vlm gemini \\
  --batch \\
  --fps 2 \\
  --output-dir "/content/drive/MyDrive/manudata/annotations/" \\
  --verbose
'''

# ============ CELL 7: TRAIN LOCAL MODEL ============
CELL_7_TRAIN = '''
!python train.py \\
  "/content/drive/MyDrive/manudata/annotations/" \\
  --model videomae-v2-small \\
  --output-dir "/content/drive/MyDrive/manudata/models/" \\
  --epochs 30 \\
  --batch-size 8 \\
  --lr 1e-4 \\
  --augment \\
  --validation-split 0.15 \\
  --gpu 0
'''

# ============ CELL 8: PRODUCTION ANNOTATION ============
CELL_8_ANNOTATE = '''
from run_colab import annotate_video

annotate_video(
    video_path="/content/drive/MyDrive/factory_videos/new_video.mp4",
    model_path="/content/drive/MyDrive/manudata/models/best.pt",
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
            print(f"\\n{f}:")
            print(f"  Segments: {data['total_segments']}")
            print(f"  Need review: {data['needs_review_count']}")
            cost = data.get('processing_stats', {}).get('estimated_cost_inr', 0)
            print(f"  Cost: \\u20b9{cost:.2f}")
            print(f"  Top actions:")
            action_summary = data.get('action_summary', {})
            for action, stats in sorted(action_summary.items(),
                                         key=lambda x: x[1]['total_duration'],
                                         reverse=True)[:5]:
                print(f"    {action}: {stats['count']}x, "
                      f"{stats['total_duration']:.1f}s")
'''


# ── cell registry ──────────────────────────────────────────────────

CELLS = [
    ("CELL 1: SETUP", CELL_1_SETUP),
    ("CELL 2: MOUNT DRIVE", CELL_2_DRIVE),
    ("CELL 3: API KEYS", CELL_3_KEYS),
    ("CELL 4: SMOKE TEST", CELL_4_TEST),
    ("CELL 5: BOOTSTRAP SINGLE VIDEO", CELL_5_BOOTSTRAP_SINGLE),
    ("CELL 6: BOOTSTRAP BATCH", CELL_6_BOOTSTRAP_BATCH),
    ("CELL 7: TRAIN LOCAL MODEL", CELL_7_TRAIN),
    ("CELL 8: PRODUCTION ANNOTATION", CELL_8_ANNOTATE),
    ("CELL 9: CHECK RESULTS", CELL_9_RESULTS),
]


if __name__ == "__main__":
    print("=" * 60)
    print("  ManuData Annotator — Colab Notebook Template")
    print("=" * 60)
    print()
    print("Copy each cell below into a separate Colab cell.")
    print()

    for title, code in CELLS:
        print(f"# {'=' * 20} {title} {'=' * 20}")
        print(code.strip())
        print()
        print()

    print("Colab notebook template ready!")
    print("Copy each CELL_N variable into a separate Colab cell.")
