# Vast.ai setup for `manudata-qc-render`

Everything here is billed by the hour, so the order matters: get the
instance right, then run the smoke test, then commit to the batch. The
smoke test exists specifically so a broken WiLoR install costs three
minutes instead of three hours.

---

## 0. Getting an agent onto the pod

A Claude Code session running in Anthropic's cloud **cannot reach your
Vast.ai pod**. That environment has no SSH client, raw outbound TCP is
blocked, and `vast.ai` is not on its network allowlist — so there is no
tunnel to open and nothing to configure. Verified, not assumed.

The working arrangement is the other way round: **run Claude Code on the
pod**, where the GPU, the weights and the footage all are.

```bash
# On the pod, after the bootstrap in step 2:
curl -fsSL https://deb.nodesource.com/setup_20.x | bash -
apt-get install -y nodejs
npm install -g @anthropic-ai/claude-code

cd /workspace/manudata-annotator
claude          # authenticate once, then work in-repo with local GPU access
```

That session can run inference, inspect renders, and fix things in place.
A cloud session (like the one that wrote this) is still useful for editing
code and pushing commits the pod pulls — it just cannot execute anything
on the pod.

**Getting a video to a cloud Claude session.** If you want a cloud session
to work on real frames, Google Drive links will not work (blocked), but
**GitHub release assets are reachable** and take files up to 2 GB. Attach
a clip to a release on this repo and share the download URL. Note the
cloud session still has no GPU, so it can validate decode, analysis and
rendering on real footage — not WiLoR inference.

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
compositing and H.264 encoding are CPU work. Measured on a 1080p60
source, two-panel layout: **~19 ms/frame to composite**, and **~63
ms/frame for the whole decode → composite → encode loop** (≈16 fps).
On a 4-core box the CPU becomes the bottleneck and the GPU idles while
you pay for it. Filter Vast.ai listings on cores, not just GPU model.

Note the tool renders **one video at a time**, and compositing is
single-threaded (x264 does use multiple threads). Extra cores therefore
help the encoder more than the compositor — see the time budget in
step 11 before assuming a 32-core box finishes proportionally faster.

Sort listings by `$/hr` among instances meeting the above; a 4090 with 16
cores is usually better value here than an A100 with 8.

## 2. One-command setup

Steps 3 and 4 are automated. On a fresh instance:

```bash
bash <(curl -sSL https://raw.githubusercontent.com/alpha1avi/manudata-annotator/claude/manudata-hand-pose-qc-jqn3a8/scripts/vast_bootstrap.sh)
```

It installs ffmpeg and the repo, picks the torch build matching the
instance's driver (cu121 or cu118), **verifies `torch.cuda.is_available()`
and stops if it is False**, installs WiLoR, fetches the weights, runs the
unit tests, and reports whether NVENC is usable. It is idempotent — re-run
it after a failure or an instance restart.

If it reports missing weight files, fetch those three by hand (the
upstream location moves between releases) and re-run. Everything else
will already be in place.

The manual equivalent is steps 3–4 below.

## 2b. Base system packages

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

Install torch **first**, from the index matching the **GPU architecture**,
not just the driver version. This is the single most common way to lose an
hour:

```bash
nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader
```

| Compute capability | GPU | Install |
|---|---|---|
| **12.0** | RTX 5090 / 5080 (Blackwell) | `torch==2.7.0 torchvision==0.22.0` from `.../whl/cu128` |
| 8.9 / 8.6 | RTX 4090 / 3090 (Ada, Ampere) | `torch==2.5.1 torchvision==0.20.1` from `.../whl/cu121` |
| older, CUDA 11.8 driver | — | same versions from `.../whl/cu118` |

```bash
# Blackwell (RTX 50-series)
pip install torch==2.7.0 torchvision==0.22.0 \
    --index-url https://download.pytorch.org/whl/cu128

# Ada / Ampere
pip install torch==2.5.1 torchvision==0.20.1 \
    --index-url https://download.pytorch.org/whl/cu121
```

**Blackwell is the trap.** A cu121 build reports the 5090 as available and
then dies at the first kernel launch with *"no kernel image is available
for execution on the device"* — which reads like a broken driver rather
than a wrong wheel. `torch.cuda.is_available()` returning True proves
nothing here; only running an actual op does.

Verify before going further:

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

## 8b. Full-length renders across several pods

For long source videos (20-30 min) delivered whole, two flags matter.

**`--workers N`** processes N videos concurrently, one process each —
inference, analysis and render for a video all stay inside one worker, so
nothing about frame alignment is split across processes. Verified
byte-identical to a sequential run.

Budget VRAM at roughly **6 GB per worker** for WiLoR:

| GPU | Sensible `--workers` |
|---|---|
| RTX 4090 / 3090 (24 GB) | 3 |
| A6000 / L40S (48 GB) | 4-6 |

Also give it cores: compositing is ~19 ms/frame/worker and single-threaded
per video, so 4 workers wants 16+ vCPU to avoid starving x264.

**`--shard I/N`** splits the video list round-robin across pods, so three
instances cover one batch with no overlap and no manual file-splitting.
Round-robin rather than contiguous blocks, so one pod does not inherit
every long file.

```bash
# Pod 1                      # Pod 2                      # Pod 3
--shard 1/3 --workers 3      --shard 2/3 --workers 3      --shard 3/3 --workers 3
```

Each pod needs the whole video directory and the same manifest; it will
only touch its own shard. Outputs land in each pod's own `--out`, so
collect the three `renders/` directories afterwards.

### Size ceiling on long renders

`--max-size-mb 50` is right for a 25-second clip and **wrong for a
25-minute one** — it works out around 260 kbps, which is unusable for
judging keypoint accuracy. For full-length renders either raise it a lot
or turn it off:

```bash
--max-size-mb 0      # no cap; quality set by CRF, ~600-900 MB per 25 min
```

The tool warns before rendering when a ceiling implies under ~2500 kbps,
naming the video and the size that would actually work. A 25-minute render
is a Drive link, not an email attachment — keep the 50 MB ceiling for the
reel, which is the thing that gets attached.

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

For 25 videos averaging 5 minutes at 1080p60 — about **450 000 frames**.

Measured end to end on a 1080p60 source (decode → composite → libx264,
sequential, one video at a time):

| Stage | Measured | 450 k frames |
|---|---|---|
| Composite only | 18.8 ms/frame (53 fps) | ~2.3 h |
| Full render loop | 63 ms/frame (15.9 fps) | **~7.9 h** |
| WiLoR inference | not measured — no GPU available where this was built | benchmark in the smoke test |

**Read that 7.9 h as a ceiling, not a target.** It is single-video,
sequential, with a software encoder on a shared CPU. NVENC removes most
of the encode term, and a dedicated many-core box will beat it. But do
not assume cores divide it: the tool does not render videos in parallel.

So the honest planning advice is to **not render everything**:

```bash
# 1. Analyse only — cheap, and tells you what is worth rendering
manudata-qc-render run ... --analyze-only

# 2. Render only what you will actually send
manudata-qc-render run ... --top 8 --clips-only
```

Eight 25-second clips is ~12 000 frames against 450 000 — about **13
minutes** instead of eight hours, for the footage that actually goes in
the customer reel. Render full-length videos only for the ones you have
a specific reason to ship whole.

Inference is the term to measure rather than trust a table for — it swings
with how many hands are actually in frame. The smoke test prints frames
per second for one video; multiply that out before committing to the
batch.

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
