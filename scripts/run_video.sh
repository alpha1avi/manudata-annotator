#!/usr/bin/env bash
#
# One command, one video, start to downloadable output.
#
#     ./scripts/run_video.sh /workspace/videos/Some_Capture_001.mp4
#
# Everything after the video path is passed through to `qc run`, so
# `--workers 3`, `--assumed-hfov 78`, `--force-inference` and friends all
# still work.
#
# This script exists because the pipeline itself was never the problem.
# Every hour lost so far went to the surrounding process: code that was
# not pulled, a run outside tmux killed by a dropped SSH session, a
# broken GPU driver discovered only after an hour of CPU encoding, and a
# silent pass that could not be told apart from a hang. Each of those is
# checked for here, before any GPU time is spent.
#
# What it does, in order:
#   1. Re-executes itself inside tmux, so a dropped connection is a
#      non-event rather than a lost run.
#   2. Fails if the checkout is behind its remote — running last week's
#      code is how a run ends up with no progress output.
#   3. Fails if the NVIDIA driver is broken (the NVML mismatch that
#      follows a host driver upgrade), because that silently costs NVENC
#      and turns a 2-minute encode into a 20-minute one.
#   4. Fails if the site/task label is missing or a placeholder — those
#      get burned into every one of ~45,000 frames and the re-render is
#      the whole render.
#   5. Runs the archival render, then the shippable reel from the same
#      cached keypoints.
#   6. Verifies the output and stages it for download.

set -euo pipefail

SESSION="manudata-run"
OUT="${OUT:-/workspace/out}"
WEIGHTS="${WEIGHTS:-/root/pretrained_models}"
BACKEND="${BACKEND:-wilor_mini}"
REEL_MAX_MB="${REEL_MAX_MB:-50}"

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

die() { echo; echo "✗ $*" >&2; echo; exit 1; }
say() { echo "── $*"; }

# ── 1. tmux ───────────────────────────────────────────────────────────
# A run started in a bare SSH session dies with the session. This has
# cost real GPU hours, so the script puts itself somewhere safe rather
# than reminding you to.

if [[ -z "${TMUX:-}" ]]; then
    if ! command -v tmux >/dev/null; then
        die "tmux is not installed. apt-get install -y tmux"
    fi
    CMD=$(printf '%q ' "$0" "$@")
    echo "Not inside tmux — relaunching in session '$SESSION'."
    echo "Detach with Ctrl-b then d;  reattach with: tmux attach -t $SESSION"
    sleep 2
    exec tmux new-session -A -s "$SESSION" "$CMD; echo; echo '[finished — press enter to close]'; read"
fi

# ── arguments ─────────────────────────────────────────────────────────

if (( $# < 1 )); then
    die "usage: $0 VIDEO [extra qc run flags...]"
fi
VIDEO="$1"; shift
[[ -f "$VIDEO" ]] || die "No such video: $VIDEO"

STAMP=$(date +%Y%m%d-%H%M%S)
LOG="$OUT/logs/run-$(basename "${VIDEO%.*}")-$STAMP.log"
mkdir -p "$OUT/logs" "$OUT/keypoints"

# ── 2. the checkout is current ────────────────────────────────────────
# The last run produced no progress output for 78 minutes because the
# commit that added progress output had never been pulled onto the pod.

say "Checking the checkout is current"
if git rev-parse --git-dir >/dev/null 2>&1; then
    git fetch --quiet origin || echo "  (fetch failed — continuing offline)"
    BRANCH=$(git rev-parse --abbrev-ref HEAD)
    if git rev-parse --quiet --verify "origin/$BRANCH" >/dev/null; then
        BEHIND=$(git rev-list --count "HEAD..origin/$BRANCH")
        if (( BEHIND > 0 )); then
            if [[ -z "$(git status --porcelain --untracked-files=no)" ]]; then
                say "  $BEHIND commit(s) behind — pulling"
                git merge --ff-only "origin/$BRANCH"
            else
                die "$BEHIND commit(s) behind origin/$BRANCH and the working tree is dirty.
   Commit or stash, then: git pull"
            fi
        fi
    fi
    say "  at $(git rev-parse --short HEAD)"
fi

# ── 3. the GPU actually works ─────────────────────────────────────────
# A host driver upgrade under a running container breaks NVML and NVENC
# while leaving already-running CUDA processes alive. Discovered late,
# it costs an entire render in CPU encoding.

say "Checking the GPU"
if ! command -v nvidia-smi >/dev/null; then
    die "nvidia-smi not found. This needs a GPU instance."
fi
if ! SMI=$(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>&1); then
    die "The NVIDIA driver is not usable:
   $SMI

   This is the driver/library mismatch that follows a host driver
   upgrade. Already-running CUDA processes survive it; new ones do not,
   and NVENC is gone until it is fixed.

   Fix: Stop, then Start the instance in the Vast.ai console (Stop/Start,
   NOT Destroy — destroying loses the disk). Then run this again."
fi
say "  $SMI"

# ── 4. the label is real ──────────────────────────────────────────────
# Site and task are burned into every frame. A placeholder discovered
# after the fact means re-rendering all 45,000 of them.

say "Checking the site/task label"
LABEL=$(python3 - "$VIDEO" "$OUT" <<'PY'
import csv
import sys
from pathlib import Path

from qc.manifest import DEFAULT_MANIFEST_NAME, FIELDNAMES, load, parse_filename

video, out = Path(sys.argv[1]), Path(sys.argv[2])
manifest = out / DEFAULT_MANIFEST_NAME

labels = {}
if manifest.exists():
    try:
        labels = load(manifest)
    except Exception as exc:
        print(f"ERROR|could not read {manifest}: {exc}")
        raise SystemExit(0)

label = labels.get(video.name.lower())
added = False

if label is None:
    # A name following the delivery convention carries its own site and
    # task. Deriving the row from the filename is reading what is already
    # there; it is not the same as inventing a label, which is why an
    # off-convention name still stops the run.
    label = parse_filename(video.stem)
    if label is None or not label.task.strip():
        print(f"ERROR|{video.name} has no manifest row and its name does not "
              f"follow the Country_City_Site_Task_NNN_NNN convention, so the "
              f"site and task cannot be derived from it.")
        raise SystemExit(0)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    exists = manifest.exists()
    with manifest.open("a", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDNAMES)
        if not exists:
            writer.writeheader()
        writer.writerow({"filename": video.name, "site": label.site,
                         "task": label.task, "notes": label.notes})
    added = True

bad = {"factory-floor", "hand-manipulation", "unknown", "site", "task", ""}
if label.site.strip().lower() in bad or label.task.strip().lower() in bad:
    print(f"ERROR|placeholder label in {manifest}: "
          f"site={label.site!r} task={label.task!r}")
else:
    print(f"{'ADDED' if added else 'OK'}|{label.site}|{label.task}")
PY
) || die "label check failed to run"

if [[ "$LABEL" == ADDED\|* ]]; then
    say "  added a manifest row from the filename"
fi

if [[ "$LABEL" == ERROR\|* ]]; then
    die "${LABEL#ERROR|}

   Site and task are burned into every rendered frame, so this is
   checked before the GPU time is spent rather than after.

   Fix either by naming the file to the delivery convention
   (Country_City_Site_Task_NNN_NNN.mp4), or by adding a row to
   $OUT/manifest.csv with a real site and task."
fi
IFS='|' read -r _ SITE TASK <<<"$LABEL"
say "  site=$SITE  task=$TASK"

# ── 5. run ────────────────────────────────────────────────────────────

COMMON=( --out "$OUT"
         --keypoints-dir "$OUT/keypoints"
         --pose-backend "$BACKEND"
         --wilor-weights "$WEIGHTS"
         --resume )

echo
say "Logging to $LOG"
say "Watch from anywhere:  tail -f $LOG"
echo

# Archival render: no size cap. The cap exists to make a clip emailable
# and would starve a 25-minute render of bitrate.
say "Pass 1/2 — analysis and full render"
python3 -m qc.cli run "$VIDEO" "${COMMON[@]}" --max-size-mb 0 "$@" 2>&1 | tee -a "$LOG"

# Shippable reel: the recommended window only, capped. Reuses the cached
# keypoints, so this is seconds of encoding rather than another pass.
say "Pass 2/2 — recommended clip, under ${REEL_MAX_MB} MB"
python3 -m qc.cli run "$VIDEO" "${COMMON[@]}" \
    --out "$OUT/reel" --clips-only --max-size-mb "$REEL_MAX_MB" "$@" 2>&1 | tee -a "$LOG"

# ── 6. verify and stage ───────────────────────────────────────────────

echo
say "Verifying outputs"
python3 - "$OUT" "$VIDEO" <<'PY'
import json, sys
from pathlib import Path

out, video = Path(sys.argv[1]), Path(sys.argv[2])
stem = video.stem
ok = True

npz = out / "keypoints" / f"{stem}.npz"
if npz.exists():
    print(f"  keypoints  {npz.name}  {npz.stat().st_size/1e6:.1f} MB")
else:
    print(f"  MISSING keypoints: {npz}"); ok = False

for label, root in (("render", out / "renders"), ("clip", out / "reel" / "renders")):
    for side in sorted(root.glob(f"{stem}*.done.json")):
        meta = json.loads(side.read_text())
        mp4 = side.with_suffix("")
        actual = mp4.stat().st_size if mp4.exists() else -1
        match = "ok" if actual == meta.get("size_bytes") else "SIZE MISMATCH"
        print(f"  {label:9} {mp4.name}  {meta['frames']} frames  "
              f"{meta['size_mb']:.0f} MB  [{match}]")
        if match != "ok":
            ok = False
    else:
        if not list(root.glob(f"{stem}*.done.json")):
            print(f"  no {label} produced")

raise SystemExit(0 if ok else 1)
PY

echo
say "Staging for download"
"$REPO/scripts/serve_downloads.sh" "$OUT"
