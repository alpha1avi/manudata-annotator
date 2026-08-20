#!/usr/bin/env bash
#
# Google Drive link in, downloadable QC render out. One command.
#
#     ./scripts/deliver.sh https://drive.google.com/file/d/1lapJj.../view
#     ./scripts/deliver.sh <id> <id> <id> --workers 3
#     ./scripts/deliver.sh /workspace/videos/Already_Here.mp4
#
# Sources may be Drive URLs, bare Drive file IDs, or local paths, mixed
# freely. Anything starting with a dash is a flag and passes through to
# `qc run` via run_video.sh.
#
# The download step exists mainly to fix names. Drive hands back files
# called "Copy of India_Faridabad_Site_Task_006_002.mp4", and that prefix
# stops the name matching the delivery convention — which means the site
# and task can no longer be read out of it, and the run stops. Rather
# than have someone hit that after a 4 GB download, the prefix is
# stripped on arrival.

set -euo pipefail

SESSION="manudata-run"
VIDEO_DIR="${VIDEO_DIR:-/workspace/videos}"

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

die() { echo; echo "✗ $*" >&2; echo; exit 1; }
say() { echo "── $*"; }

# Everything long-running belongs in tmux, downloads included: a 4 GB
# transfer killed by a dropped SSH session is the same wasted hour as a
# killed render.
if [[ -z "${TMUX:-}" ]]; then
    command -v tmux >/dev/null || die "tmux is not installed. apt-get install -y tmux"
    CMD=$(printf '%q ' "$0" "$@")
    echo "Not inside tmux — relaunching in session '$SESSION'."
    echo "Detach with Ctrl-b then d;  reattach with: tmux attach -t $SESSION"
    sleep 2
    exec tmux new-session -A -s "$SESSION" "$CMD; echo; echo '[finished — press enter to close]'; read"
fi

SOURCES=()
while (( $# )) && [[ "$1" != -* ]]; do
    SOURCES+=( "$1" ); shift
done
(( ${#SOURCES[@]} )) || die "usage: $0 <drive-url|drive-id|path> [more...] [qc flags...]"

mkdir -p "$VIDEO_DIR"

# ── normalise a downloaded name ───────────────────────────────────────
# "Copy of X.mp4" -> "X.mp4";  "X (1).mp4" -> "X.mp4".  Both are Drive
# artefacts, neither carries meaning, and both break the convention
# parser that reads site and task out of the filename.
normalise() {
    local base="$1"
    while [[ "$base" == "Copy of "* ]]; do base="${base#Copy of }"; done
    base="$(sed -E 's/ \([0-9]+\)(\.[A-Za-z0-9]+)$/\1/' <<<"$base")"
    printf '%s' "$base"
}

VIDEOS=()

for src in "${SOURCES[@]}"; do
    if [[ -f "$src" ]]; then
        say "Local file: $(basename "$src")"
        VIDEOS+=( "$src" )
        continue
    fi

    # Accept a full Drive URL or a bare id.
    id="$src"
    if [[ "$src" == *drive.google.com* ]]; then
        id=$(sed -nE 's#.*/d/([^/?]+).*#\1#p; s#.*[?&]id=([^&]+).*#\1#p' <<<"$src" | head -1)
    fi
    [[ -n "$id" ]] || die "Could not read a Drive file id out of: $src"

    command -v gdown >/dev/null || {
        say "Installing gdown"
        pip install --quiet gdown
    }

    say "Fetching $id from Drive"
    # Download into a scratch dir so gdown's chosen name is visible and
    # can be corrected before it lands next to the real videos.
    tmp=$(mktemp -d "$VIDEO_DIR/.fetch-XXXXXX")
    ( cd "$tmp" && gdown --continue "$id" ) || { rm -rf "$tmp"; die "gdown failed for $id"; }

    got=$(find "$tmp" -maxdepth 1 -type f | head -1)
    [[ -n "$got" ]] || { rm -rf "$tmp"; die "gdown produced no file for $id"; }

    clean="$(normalise "$(basename "$got")")"
    dest="$VIDEO_DIR/$clean"
    if [[ "$clean" != "$(basename "$got")" ]]; then
        say "  renamed: $(basename "$got")  ->  $clean"
    fi
    mv -f "$got" "$dest"
    rm -rf "$tmp"
    say "  $(du -h "$dest" | cut -f1)  $dest"
    VIDEOS+=( "$dest" )
done

# Check the labels are readable before handing off, so a bad name is
# reported next to the download that produced it rather than several
# steps later.
say "Checking names resolve to a site and task"
python3 - "${VIDEOS[@]}" <<'PY'
import sys
from pathlib import Path
from qc.manifest import parse_filename

bad = []
for arg in sys.argv[1:]:
    stem = Path(arg).stem
    label = parse_filename(stem)
    if label and label.task.strip():
        print(f"     {stem}\n        -> {label.site} / {label.task}")
    else:
        print(f"     {stem}\n        -> NOT PARSEABLE")
        bad.append(Path(arg).name)

if bad:
    print()
    print("These names do not follow Country_City_Site_Task_NNN_NNN, so the")
    print("site and task cannot be read from them:")
    for name in bad:
        print(f"  {name}")
    print()
    print("Either rename them to the convention, or add a row with a real")
    print("site and task to the manifest before running. They are burned")
    print("into every frame, so they are not guessed.")
    raise SystemExit(1)
PY

echo
say "Handing off to run_video.sh"
echo
exec "$REPO/scripts/run_video.sh" "${VIDEOS[@]}" "$@"
