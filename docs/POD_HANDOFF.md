# Pod handoff — current state and what to do next

You are Claude Code running **on the Vast.ai GPU instance**, with direct
access to the GPU, this repository, and the source footage. Earlier
sessions ran in a cloud container that could not reach this machine, so
every pod-side fact below was established by the user pasting output
back. You can now verify things yourself — do that rather than trusting
this file where it is cheap to check.

Read this whole file before running anything. Most of the constraints
below were learned by losing an hour to them.

---

## The goal

Hand-pose QC videos for a technical evaluator at Google DeepMind
Robotics. Each output is a side-by-side 1080p MP4: source frame with a
21-keypoint 2D skeleton on the left, the same keypoints as an orbiting
3D skeleton on the right, frame-exact between panels. Plus
`qc_report.csv`, a shippable reel, and `docs/README_FOR_CUSTOMER.md`.

Target: **ten full-length videos**, then the reel and README go out.

## One command does everything

```bash
cd /root/manudata-qc
./scripts/deliver.sh <drive-url-or-id> [more...] [--workers N]
```

That fetches from Drive, fixes the filename, pulls the repo current,
checks the GPU, writes the manifest row, runs inference and both renders,
verifies output byte sizes, and stages everything for download at
`http://localhost:8080/`.

For videos already on disk:

```bash
./scripts/run_video.sh /workspace/videos/*.mp4 --workers 3
```

**`--workers` only helps across videos.** One video is one worker's job
by design — nothing about a video's pipeline splits across processes, so
there is no shared cursor to desynchronise. Three videos and
`--workers 3` runs three at once; one video and `--workers 3` runs one.

## What has already been done

| | |
|---|---|
| `India_Faridabad_AlpineFootwear01_ShoeAssembly_008_016.mp4` | **complete** — 45,030 frames, keypoints + 1.05 GB render, verified |
| `2026_0619_141716_003.MP4` | rendered, but with **placeholder site/task burned into every frame** — needs real labels then a re-render (no re-inference) |
| `India_Faridabad_CosmoReflectors01_FrameFabrication_006_002.mp4` | next up, 3.78 GB, on Drive |

Measured on the completed video, and worth quoting to the customer:

- Hand visible in **86.1%** of hand-slots; both hands posed in **76.2%**
  of frames; longest gap 6.9 s.
- Wrist→middle-MCP **0.095 m on both hands independently**, thumb-tip to
  pinky-tip reaching **0.183 m** at maximum spread. Anatomically correct,
  which is what rules out a joint-ordering bug — the low *median* span
  (0.07 m) is just hands closed around workpieces.
- `valid == hand_visible` everywhere, because WiLoR-mini poses every hand
  it detects. See the README's callout: `pose_recovery_pct` is
  structurally 100% here and must **not** be presented as a measured
  accuracy.

## Constraints that have already cost hours

**Always work inside tmux.** Both scripts relaunch themselves into it, so
just use them. A run started in a bare SSH session dies with the session;
this killed a render mid-encode.

**Pull before running.** One run produced no progress output for 78
minutes because the commit that *added* progress output had never been
pulled. `run_video.sh` now fast-forwards automatically when the tree is
clean.

**The NVIDIA driver breaks under you.** When Vast upgrades the host
driver, `nvidia-smi` starts reporting a driver/library version mismatch.
Already-running CUDA processes survive; new ones fail, and NVENC
disappears — which turns a 2-minute encode into 20. Fix is **Stop then
Start** the instance in the Vast console (never Destroy, that loses the
disk). `run_video.sh` refuses to start when it detects this.

**Site and task are burned into every frame.** They come from the
filename convention `Country_City_Site_Task_NNN_NNN` or from a manifest
row, and are never guessed. Drive's `Copy of ` prefix breaks the parser;
`deliver.sh` strips it on arrival.

**Inference checkpoints every 2000 frames** to `<stem>.npz.partial`, so a
killed pass loses a minute, not an hour. A partial is only reused when
the source, backend and every parameter that changes the numbers match —
including the derived depth scale. Do not loosen that: mixing two depth
scales gives a track that passes every consistency check and is wrong in
half its frames.

**Use `python3`,** not `python` — the latter is not installed.

## Known-good invocation

```bash
python3 -m qc.cli run VIDEO \
    --out /workspace/out \
    --keypoints-dir /workspace/out/keypoints \
    --pose-backend wilor_mini \
    --wilor-weights /root/pretrained_models \
    --resume \
    --max-size-mb 0          # 0 = uncapped, for the archival render
```

The reel is a second pass over the same cached keypoints with
`--clips-only --max-size-mb 50`. `run_video.sh` does both.

## Rough timings (RTX 4090, one video, 25 min of 1080p+ footage)

| Stage | Time |
|---|---|
| Model load / first-run setup | ~25 min once per container |
| Inference | ~78 min (GPU only 7–10% used — it is CPU-bound) |
| Render, NVENC | ~2–5 min |
| Render, libx264 fallback | ~20 min |

Because inference is CPU-bound, running 3 videos concurrently is close to
a true 3× rather than a partial win. That is the single biggest lever on
the ten-video batch.

## What needs doing

1. **Run the remaining videos.** Batch them 3 at a time with
   `--workers 3`.
2. **`2026_0619_141716_003.MP4` needs a real site and task** from the
   user. Its name cannot be parsed, so there is nothing to derive. Once
   you have them, add the manifest row and re-render — the keypoints are
   cached, so this costs a render, not an inference.
3. **Build the reel** across all finished videos and confirm it is under
   50 MB.
4. **Check `docs/README_FOR_CUSTOMER.md`** still matches what shipped —
   particularly the intrinsics section, which rests on an assumed 65°
   horizontal FOV. If the user supplies the SJCAM's true FOV, re-run with
   `--assumed-hfov` and update the doc. Depth is linear in focal length,
   so this rescales `kp3d` depth without moving a single 2D keypoint.

## Things to raise with the user rather than decide

- **The MANO licence restricts commercial use.** WiLoR-mini
  auto-downloads `MANO_RIGHT.pkl`, which does not change those terms.
  This is worth a deliberate decision before the keypoints are embedded
  in a paid dataset delivery.
- **Absolute depth is an assumption, not a measurement** (65° assumed
  FOV). Relative geometry is well conditioned; absolute hand-to-camera
  distance is not. The README says so — keep it that way.

## Never

- Paste an SSH private key, a pod password, or a Drive token into chat or
  into a commit. The `.pub` half is fine.
- Present `pose_recovery_pct` as a measured tracking accuracy.
- Ship a render whose byte size disagrees with its `.done.json` sidecar.
  `+faststart` writes the index at the front of the file, so a truncated
  MP4 still reports its full frame count — byte size is the only check
  that catches it.
