# Vast.ai setup for `manudata-qc-render`

Everything here is billed by the hour, so the order matters: get the
instance right, then run the smoke test, then commit to the batch. The
smoke test exists specifically so a broken WiLoR install costs three
minutes instead of three hours.

---

## 1. Instance to rent

| | |
|---|---|
| GPU | 1× RTX 4090 (24 GB) — or 3090 / A5000. WiLoR needs ~6 GB, so VRAM is not the constraint |
| vCPU | **≥ 12 cores** — see the note below; this is the one people under-spec |
| RAM | ≥ 32 GB |
| Disk | ≥ 100 GB (source video + keypoint cache + renders) |
| Image | `pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime` |
| CUDA | 12.1 (match the torch build in step 3) |

**Why the CPU spec matters.** The GPU only runs WiLoR inference. Frame
compositing and H.264 encoding are CPU work, measured at **~11.5 ms per
1080p frame per core** for the two-panel layout. On a 4-core box the CPU
becomes the bottleneck and the GPU idles while you pay for it. Filter
Vast.ai listings on cores, not just GPU model.

Sort listings by `$/hr` among instances meeting the above; a 4090 with 16
cores is usually better value here than an A100 with 8.

## 2. Base system packages

```bash
apt-get update && apt-get install -y ffmpeg git wget
ffmpeg -version | head -1        # confirm it is on PATH
nvidia-smi                       # confirm the GPU and driver
```

`nvidia-smi` reporting a CUDA version **lower** than 12.1 means the
driver predates the image; pick a different instance rather than
downgrading torch.

## 3. Python environment

```bash
git clone <this-repo> manudata && cd manudata
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
```

Install torch **first**, from the index matching the instance's CUDA
runtime. This is the single most common way to lose an hour:

```bash
# CUDA 12.1 (matches the image recommended above)
pip install torch==2.5.1 torchvision==0.20.1 \
    --index-url https://download.pytorch.org/whl/cu121
```

If `nvidia-smi` shows CUDA 11.8, use `--index-url .../whl/cu118` and the
matching torch build instead. Verify before going further:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

`torch.cuda.is_available()` must print `True`. If it prints `False`, stop
and fix it — WiLoR will otherwise silently run on CPU roughly two orders
of magnitude slower, and the tool will warn but still proceed.

Then the tool itself:

```bash
pip install -e ".[wilor]"
```

## 4. WiLoR weights

WiLoR is installed from upstream and its weights are downloaded
separately:

```bash
pip install git+https://github.com/rolpotamias/WiLoR.git

mkdir -p pretrained_models && cd pretrained_models
# From the WiLoR release / HuggingFace mirror named in the upstream README:
wget -O wilor_final.ckpt   <WILOR_CHECKPOINT_URL>
wget -O model_config.yaml  <WILOR_CONFIG_URL>
wget -O detector.pt        <DETECTOR_URL>
cd ..
```

Put `pretrained_models/` on the instance's **persistent volume** if the
rental has one. A restarted instance that has to re-download ~2 GB of
weights before it can resume is pure billed time.

Expected layout:

```
pretrained_models/
├── wilor_final.ckpt
├── model_config.yaml
└── detector.pt
```

> The exact download URLs move between upstream releases, so they are not
> hard-coded here — take them from the WiLoR repository's current README.
> `manudata-qc-render` checks all three files exist before loading
> anything and names the missing one if not.

## 5. Get the footage onto the instance

From the Windows workstation, with the videos on `E:`:

```powershell
scp -P <port> -r "E:\tangerine shoes vl" root@<host>:/workspace/videos/
scp -P <port> -r "E:\alpine Vl"          root@<host>:/workspace/videos/
```

`rsync` is better for anything large or resumed:

```bash
rsync -avP -e "ssh -p <port>" "/e/tangerine shoes vl" root@<host>:/workspace/videos/
```

## 6. Manifest

Site and task labels are burned into every frame, so the tool refuses to
render without them.

```bash
manudata-qc-render init-manifest /workspace/videos --out /workspace/out/manifest.csv
```

That writes one row per video with `site` pre-filled from the folder name
(`tangerine shoes vl` → `Tangerine Shoes VL`). **Fill in the `task`
column**, then continue. Re-running `init-manifest` after adding videos
keeps what you already typed and only appends new rows.

## 7. Smoke test — do this before the full batch

Renders ten seconds of a single video, end to end: WiLoR inference, both
panels, encoding, and the report.

```bash
manudata-qc-render run /workspace/videos \
    --out /workspace/out \
    --manifest /workspace/out/manifest.csv \
    --wilor-weights ./pretrained_models \
    --limit 1 \
    --smoke-test 10 \
    --no-reel
```

Then **look at the output** — `/workspace/out/renders/*_qc.mp4`:

- [ ] The skeleton sits on the hands, not beside them or a beat behind.
- [ ] `L` is on the left hand and `R` on the right.
- [ ] The 3D panel orbits smoothly and the skeleton does not change size.
- [ ] The header shows the right site and task.
- [ ] The log says `Encoding ... with h264_nvenc` (not `libx264`).

If the encoder fell back to `libx264`, NVENC is unavailable on that
instance — the render is still correct, just slower. Force the check with
`--encoder nvenc` to see the reason.

Pull it down to look at it:

```bash
scp -P <port> root@<host>:/workspace/out/renders/*_qc.mp4 .
```

## 8. Analysis pass over the whole batch

Cheap relative to rendering, and it tells you what is worth rendering.
Note that it still runs WiLoR inference on every video — but that work is
cached, so it is never repeated.

```bash
manudata-qc-render run /workspace/videos \
    --out /workspace/out \
    --manifest /workspace/out/manifest.csv \
    --wilor-weights ./pretrained_models \
    --analyze-only
```

Prints the ranked table and writes `/workspace/out/qc_report.csv`.

## 9. Full batch

```bash
manudata-qc-render run /workspace/videos \
    --out /workspace/out \
    --manifest /workspace/out/manifest.csv \
    --wilor-weights ./pretrained_models \
    --resume \
    --max-size-mb 50 \
    2>&1 | tee -a /workspace/out/batch.log
```

Run it under `tmux` so an SSH drop does not kill the batch:

```bash
tmux new -s render
# ... start the command, then Ctrl-B D to detach
tmux attach -t render
```

**`--resume` is safe to use always.** Outputs are written to a temporary
name and renamed only on a clean encoder exit, and a completed render
carries a `.done.json` sidecar recording its frame count and settings.
A video is skipped only when the sidecar matches, the frame count matches,
and ffprobe can still decode the file. Keypoints are cached per video, so
a killed instance re-infers nothing it already finished.

Useful variations:

```bash
--top 8          # render only the 8 best-ranked videos
--clips-only     # render just the recommended 20-30s window of each
--limit 3        # trial run over the first 3 videos
--with-slam --slam-dir /workspace/slam   # add the trajectory panel
```

## 10. Retrieve the results

```
/workspace/out/
├── qc_report.csv
├── manudata_qc_reel.mp4      <- attach this to the customer email
├── renders/                  <- one QC MP4 per video
├── reel_clips/               <- the excerpts the reel was built from
├── keypoints/                <- .npz per video; ship these with the data
└── logs/                     <- per-run log files
```

```bash
scp -P <port> -r root@<host>:/workspace/out/ ./out/
```

Copy `keypoints/` down too, even if you only wanted the videos —
regenerating it costs another GPU rental, and it is what lets you
re-render on a laptop with `--pose-backend cached` (no GPU, no torch).

## 11. Rough time budget

For 25 videos averaging 5 minutes at 1080p60 (~450 000 frames):

| Stage | Rate | Estimate |
|---|---|---|
| WiLoR inference | GPU-bound, batched | benchmark it during the smoke test |
| Compositing | ~11.5 ms/frame/core | ~1.5 h on 12 cores |
| H.264 encode | NVENC, parallel with the above | negligible |

Inference is the term to measure rather than trust a table for — it swings
with how many hands are actually in frame. The smoke test prints frames
per second for one video; multiply that out before committing to the
batch.

To cut cost: run `--analyze-only` first, then render only what you will
actually send with `--top N --clips-only`. Rendering 8 clips of 25 s
instead of 25 full videos is roughly a twentieth of the compositing time.

## Troubleshooting

**`ffmpeg not found`** — `apt-get install -y ffmpeg`, or set
`MANUDATA_FFMPEG=/path/to/ffmpeg`.

**`Missing WiLoR weight files`** — the message names the missing file;
re-check step 4 and `--wilor-weights`.

**`torch.cuda.is_available()` is False** — the torch build does not match
the driver. Reinstall from the correct `--index-url` (step 3).

**`Frame-count mismatch`** — a cached `.npz` was computed from a different
version of that video file. Delete that one `.npz` and re-run; the tool
refuses to render rather than risk offsetting the panels.

**Encoder fell back to `libx264`** — NVENC is missing or busy. The tool
test-encodes one frame before trusting it, so a listed-but-broken NVENC is
detected rather than failing mid-batch. Correct output, just slower.

**Output larger than `--max-size-mb`** — the bitrate cap hit its quality
floor. Shorten the window (`--clips-only`) or raise the ceiling; the log
says which video and by how much.

**Ran out of disk** — clear `reel_clips/` first, it is fully regenerable
from `renders/` and `keypoints/`.
