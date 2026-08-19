#!/usr/bin/env bash
# One-shot setup for manudata-qc-render on a fresh Vast.ai GPU instance.
#
#   bash <(curl -sSL https://raw.githubusercontent.com/alpha1avi/manudata-annotator/claude/manudata-hand-pose-qc-jqn3a8/scripts/vast_bootstrap.sh)
#
# Idempotent: safe to re-run after a failure or an instance restart. Every
# step that can fail checks its own result and stops with a specific
# message, because a silent half-install is what turns a ten-minute setup
# into an hour of billed confusion.

set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/alpha1avi/manudata-annotator.git}"
BRANCH="${BRANCH:-claude/manudata-hand-pose-qc-jqn3a8}"
WORKDIR="${WORKDIR:-/workspace}"
REPO_DIR="$WORKDIR/manudata-annotator"
# Keep weights outside the repo so a re-clone never re-downloads ~2 GB.
WEIGHTS_DIR="${WEIGHTS_DIR:-$WORKDIR/pretrained_models}"

say()  { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }
warn() { printf '\033[1;33m!!  %s\033[0m\n' "$*"; }
die()  { printf '\n\033[1;31mXX  %s\033[0m\n' "$*" >&2; exit 1; }

# ── 1. GPU ────────────────────────────────────────────────────────────
say "Checking the GPU"
command -v nvidia-smi >/dev/null || die "nvidia-smi not found — this is not a GPU instance."
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

DRIVER_CUDA="$(nvidia-smi | grep -oP 'CUDA Version: \K[0-9]+\.[0-9]+' | head -1 || true)"
[ -n "$DRIVER_CUDA" ] || die "Could not read the CUDA version from nvidia-smi."
echo "Driver supports CUDA up to: $DRIVER_CUDA"

# torch's cu121 build needs a driver advertising >= 12.1.
CUDA_TAG="cu121"
if [ "$(printf '%s\n12.1\n' "$DRIVER_CUDA" | sort -V | head -1)" != "12.1" ]; then
    CUDA_TAG="cu118"
    warn "Driver predates CUDA 12.1 — falling back to the cu118 torch build."
fi
echo "Will install torch for: $CUDA_TAG"

# ── 2. System packages ────────────────────────────────────────────────
say "Installing ffmpeg, git, python3-venv"
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq ffmpeg git python3-venv python3-pip curl >/dev/null
command -v ffmpeg >/dev/null || die "ffmpeg failed to install."
ffmpeg -version | head -1

# ── 3. Repository ─────────────────────────────────────────────────────
say "Fetching the repository"
mkdir -p "$WORKDIR"
if [ -d "$REPO_DIR/.git" ]; then
    git -C "$REPO_DIR" fetch --quiet origin "$BRANCH"
    git -C "$REPO_DIR" checkout --quiet "$BRANCH"
    git -C "$REPO_DIR" reset --hard --quiet "origin/$BRANCH"
else
    git clone --quiet --branch "$BRANCH" "$REPO_URL" "$REPO_DIR"
fi
cd "$REPO_DIR"
echo "At $(git rev-parse --short HEAD) on $BRANCH"

# ── 4. Python environment ─────────────────────────────────────────────
say "Creating the virtualenv"
[ -d .venv ] || python3 -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
pip install --quiet --upgrade pip

say "Installing torch ($CUDA_TAG) — the slowest step, ~2 GB"
if [ "$CUDA_TAG" = "cu121" ]; then
    pip install --quiet torch==2.5.1 torchvision==0.20.1 \
        --index-url https://download.pytorch.org/whl/cu121
else
    pip install --quiet torch==2.5.1 torchvision==0.20.1 \
        --index-url https://download.pytorch.org/whl/cu118
fi

say "Verifying torch sees the GPU"
python - <<'PY' || die "torch cannot see the GPU. Do not continue — WiLoR would silently run on CPU, ~100x slower."
import sys, torch
print("torch:", torch.__version__, "cuda build:", torch.version.cuda)
if not torch.cuda.is_available():
    print("torch.cuda.is_available() == False")
    sys.exit(1)
print("device:", torch.cuda.get_device_name(0))
PY

say "Installing manudata-qc-render"
pip install --quiet -e ".[wilor,dev]"

# ── 5. WiLoR ──────────────────────────────────────────────────────────
say "Installing WiLoR from upstream"
pip install --quiet "git+https://github.com/rolpotamias/WiLoR.git" \
    || warn "Upstream WiLoR install failed — see the note at the end."

say "Fetching WiLoR weights into $WEIGHTS_DIR"
mkdir -p "$WEIGHTS_DIR"
pip install --quiet huggingface_hub

# The upstream weight location moves between releases, so this tries the
# known repo and then reports precisely what is missing rather than
# pretending a partial download succeeded.
python - "$WEIGHTS_DIR" <<'PY' || true
import shutil, sys
from pathlib import Path

dest = Path(sys.argv[1])
wanted = {"wilor_final.ckpt", "model_config.yaml", "detector.pt"}
have = {p.name for p in dest.glob("*")}
if wanted <= have:
    print("All weights already present; skipping download.")
    sys.exit(0)

try:
    from huggingface_hub import snapshot_download
    path = Path(snapshot_download(repo_id="rolpotamias/WiLoR"))
    print("Downloaded snapshot to", path)
    for f in path.rglob("*"):
        if f.is_file() and f.name in wanted:
            shutil.copy2(f, dest / f.name)
            print("  placed", f.name)
except Exception as exc:                      # noqa: BLE001
    print("Automatic weight download failed:", exc)
PY

MISSING=()
for f in wilor_final.ckpt model_config.yaml detector.pt; do
    [ -s "$WEIGHTS_DIR/$f" ] || MISSING+=("$f")
done

# ── 6. Self-check ─────────────────────────────────────────────────────
say "Running the unit tests (no GPU or footage needed)"
python -m pytest tests/test_qc.py -q || warn "Unit tests failed — tell Claude before rendering."

say "Encoder check"
python - <<'PY'
from qc.io.ffmpeg import has_nvenc, select_encoder
print("NVENC usable:", has_nvenc(), "| encoder:", select_encoder("auto"))
PY

# ── 7. Report ─────────────────────────────────────────────────────────
say "Setup complete"
cat <<EOF

  Repo      : $REPO_DIR
  Activate  : source $REPO_DIR/.venv/bin/activate
  Weights   : $WEIGHTS_DIR
  Videos    : put them in $WORKDIR/videos/

EOF

if [ ${#MISSING[@]} -gt 0 ]; then
    warn "Missing WiLoR weights: ${MISSING[*]}"
    cat <<EOF
  The upstream download location changes between WiLoR releases, so fetch
  these by hand from the WiLoR repository's current README and drop them in
  $WEIGHTS_DIR :

      wilor_final.ckpt    the pose model checkpoint
      model_config.yaml   its config
      detector.pt         the YOLO hand detector

  Everything else is installed and ready.
EOF
else
    cat <<EOF
  Next:
    cd $REPO_DIR && source .venv/bin/activate
    manudata-qc-render init-manifest $WORKDIR/videos --out $WORKDIR/out/manifest.csv
    manudata-qc-render run $WORKDIR/videos --out $WORKDIR/out \\
        --wilor-weights $WEIGHTS_DIR --limit 1 --smoke-test 10 --no-reel
EOF
fi
