# Pod handoff — current state and what to do next

You are running on a rented Vast.ai GPU instance with direct access to the
GPU, this repository, and the source footage. A previous Claude Code
session built this tool from a cloud container that could not reach this
machine, so everything below was verified either on that container (the
render engine) or by the user pasting output back (the machine setup).

Read this whole file before running anything. The environment has several
non-obvious constraints that have already cost hours.

---

## The goal

Produce hand-pose QC videos for a prospective customer (a technical
evaluator at Google DeepMind Robotics). Each output is a side-by-side
1080p MP4: source frame with a 21-keypoint 2D skeleton on the left, the
same keypoints as an orbiting 3D skeleton on the right, frame-exact
between panels. Plus `qc_report.csv` and a combined reel.

The immediate milestone is a **10-second smoke test on one real video**
that a human looks at. Everything else waits on that.

## Machine state as of handoff

| | |
|---|---|
| GPU | RTX 4090, 24 GB |
| Python | **system `python3` 3.10** — use this, not the venv |
| torch | 2.5.0+cu124 — **`torch.cuda.is_available()` was returning False** |
| wilor_mini | installed and imports OK |
| ffmpeg | 4.4.2, and **NVENC works** (verified by `doctor`) |
| Repo | `~/manudata-qc`, branch `claude/manudata-hand-pose-qc-jqn3a8` |
| WiLoR weights | `~/pretrained_models` — detector.pt, wilor_final.ckpt, model_config.yaml |
| Video | `/workspace/videos/2026_0619_141716_003.MP4` (2.8 GB) |
| Output dir | `/workspace/out` |

**There is a split-brain Python problem.** A virtualenv at
`~/manudata-qc/.venv` (or `~/venv`) has torch 2.6.0 and the QC tool;
system Python has torch 2.5.0 and wilor_mini. Consolidate on **system
Python**. Do not activate the venv.

## Blockers to clear first

### 1. CUDA initialisation failure

```
RuntimeError: CUDA unknown error ... Setting the available devices to be zero.
```

torch 2.6.0 in the venv previously worked on this same GPU, so the GPU and
driver are fine — this is a wedged CUDA context, not a wheel mismatch.
Check `nvidia-smi`, then `echo $CUDA_VISIBLE_DEVICES`. If `nvidia-smi` is
healthy and torch still reports False, the instance needs a stop/start
from the Vast dashboard (`/root` and `/workspace` persist).

Do not proceed past this. WiLoR on CPU is ~100× slower and would silently
burn the rental.

### 2. Editable install rejected

```
build backend is missing the 'build_editable' hook
```

setuptools predates PEP 660:

```bash
pip install -U pip setuptools wheel
cd ~/manudata-qc && pip install -e ".[dev]"
```

Then `python3 -m pytest tests/test_qc.py -q` — **44 tests must pass.**
They cover the render engine, visibility state machine, clip selection and
resume logic, and need no GPU or footage.

## The real work: the pose backend does not match the installed library

`qc/pose/wilor_backend.py` was written against **upstream WiLoR**
(`from wilor.models import load_wilor`, `ViTDetDataset`, etc.). Upstream
has no `setup.py`, is not pip-installable, and additionally requires
`MANO_RIGHT.pkl` from a licence-gated download. It is **not** what is
installed here.

What *is* installed is **WiLoR-mini**, with a completely different API:

```python
from wilor_mini.pipelines.wilor_hand_pose3d_estimation_pipeline import (
    WiLorHandPose3dEstimationPipeline,
)
pipe = WiLorHandPose3dEstimationPipeline(device=torch.device("cuda"),
                                         dtype=torch.float16)
outputs = pipe.predict(rgb_image)      # note: RGB, not BGR
```

**Step one is to probe its real output structure**, not to guess at it:

```bash
cd /workspace/videos
ffmpeg -v error -i "2026_0619_141716_003.MP4" -vf "select=eq(n\,600)" \
    -vsync 0 -frames:v 1 -y /tmp/probe.png

python3 - <<'PY'
import cv2, torch
from wilor_mini.pipelines.wilor_hand_pose3d_estimation_pipeline import WiLorHandPose3dEstimationPipeline
pipe = WiLorHandPose3dEstimationPipeline(device=torch.device("cuda"), dtype=torch.float16)
img = cv2.cvtColor(cv2.imread("/tmp/probe.png"), cv2.COLOR_BGR2RGB)
out = pipe.predict(img)
print(type(out), len(out) if hasattr(out, "__len__") else "")
def show(o, i=0):
    p = "  " * i
    if isinstance(o, dict):
        for k, v in o.items():
            if isinstance(v, (dict, list)): print(f"{p}{k}:"); show(v, i+1)
            elif hasattr(v, "shape"): print(f"{p}{k}: {tuple(v.shape)} {v.dtype}")
            else: print(f"{p}{k}: {type(v).__name__} = {str(v)[:80]}")
    elif isinstance(o, list):
        print(f"{p}list[{len(o)}]")
        if o: show(o[0], i+1)
show(out)
PY
```

Then write `qc/pose/wilor_mini_backend.py` producing a `PoseTrack`
(see `qc/pose/schema.py`) and register it as a `--pose-backend` choice in
`qc/cli.py` and `qc/parallel.py`. Leave `wilor_backend.py` in place for
anyone using upstream.

### Semantics that must be preserved

The entire report and render design rests on **separating two kinds of
missing pose**. Do not collapse them:

| Flag | Meaning |
|---|---|
| `hand_visible[t, h]` | a hand detector found a hand — a fact about the factory floor |
| `valid[t, h]` | a pose was recovered for it — a fact about our tracker |

`pose_recovery_pct` is computed over visible-hand slots only, so occlusion
neither flatters nor penalises the tracking number. This is the figure the
customer README leads with.

If WiLoR-mini does **not** expose detection separately from pose (i.e. it
only returns successful hands), say so explicitly rather than faking the
distinction. In that case `hand_visible` must be set equal to `valid`, and
the operator told that case (b) cannot be measured with this backend —
which materially changes what the customer README may claim. Flag it; do
not paper over it.

Other invariants:

- **Frame-exactness.** One absolute frame index drives decode, pose
  lookup, both panels and the header. Frame windows use ffmpeg's `select`
  filter on frame *number*, never a timestamp seek. The render aborts on
  any length disagreement rather than publishing offset panels.
- Left hand is slot 0, right is slot 1. Handedness must come from the
  model, not from x-position.
- 3D keypoints are camera-space metres, +X right / +Y down / +Z forward.
- Missing keypoints stay `NaN`. Ghosted poses in the render are display
  only and must never reach the saved `.npz`.

## Verification gates

```bash
# 1. preflight — GPU, weights, videos, manifest, disk, workers vs VRAM
manudata-qc-render doctor /workspace/videos --out /workspace/out \
    --wilor-weights ~/pretrained_models --workers 3

# 2. labels (site and task get burned into every frame)
manudata-qc-render init-manifest /workspace/videos --out /workspace/out/manifest.csv
# this file has a raw camera name, so the convention parser will not fire —
# the task column must be filled in by hand before rendering

# 3. smoke test — 10 seconds, then LOOK AT THE MP4
manudata-qc-render run /workspace/videos --out /workspace/out \
    --limit 1 --smoke-test 10 --no-reel --pose-backend <new backend>
```

Check on the smoke test output: skeleton sits on the hands and does not
lag, `L` is on the left hand and `R` on the right, the 3D skeleton does
not change size as it orbits, header shows the right site and task.

Report the inference **fps** — it is the only real throughput number
available and everything downstream is estimated from it.

## Environment lessons already paid for

- **Never scp footage to this pod.** Measured at 70 KB/s from the user's
  connection (~10 h for one 2.6 GB file). Google Drive via `gdown` on the
  pod measured **20.3 MB/s** — roughly 290× faster. Route all data
  through Drive.
- WiLoR-mini pins `torch<=2.5`. Do not upgrade torch past 2.5.
- `python` does not exist; use `python3`.
- Use `tmux` — SSH has dropped repeatedly and killed long jobs.
- 1080p60 measured on the render engine: ~18.8 ms/frame to composite,
  ~63 ms/frame for the whole decode→composite→encode loop with libx264.
  NVENC works here, so expect better.
- `--max-size-mb 50` is right for a 25 s clip and useless for a 25 min one
  (~260 kbps). Use `--max-size-mb 0` for full-length renders.

## Scale, once the smoke test passes

The user wants ~10 full-length renders. Source files are ~2.6 GB each,
disk is 274 GB, so storage is not a constraint. Use `--workers 3` (each
WiLoR worker needs roughly 6 GB of the 24 GB VRAM) and `--resume`.
`--shard I/N` splits the list across several pods without overlap.

Run `--analyze-only` first and read the ranked table: it sorts by
`pose_recovery_pct` and names the best 20–30 s window per video, which is
how the reel clips get chosen.
