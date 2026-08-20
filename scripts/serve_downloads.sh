#!/usr/bin/env bash
#
# Serve a run's outputs over the SSH tunnel so they can be downloaded
# from a browser instead of fought over with scp.
#
# Why this exists: scp through Vast's jump host has been anywhere from
# tolerable to 70 KB/s, and a 1 GB render at 70 KB/s is a four-hour
# transfer that dies on the first dropped connection with nothing to
# resume from. A browser download over the same tunnel shows a real
# progress bar and survives you closing the terminal.
#
# It is also resumable, but only if the server honours Range requests —
# and Python's stdlib http.server does not, so a dropped download would
# silently restart from zero. We install rangehttpserver when we can and
# say so plainly when we cannot, because "resumable" that isn't is worse
# than knowing you have one shot at it.
#
# Usage, on the pod:
#     ./scripts/serve_downloads.sh [OUT_DIR] [PORT]
#
# Then on your own machine, connect with the tunnel (Vast's Connect
# button usually includes it already):
#     ssh -p <PORT> root@<HOST> -L 8080:localhost:8080
# and open http://localhost:8080/ in a browser.
#
# The server binds to 127.0.0.1, never 0.0.0.0. Bound to all interfaces
# it would publish your keypoints and renders to anyone who can reach
# the instance's public IP — these are customer deliverables, so the
# tunnel is the only way in.

set -euo pipefail

OUT_DIR="${1:-/workspace/out}"
PORT="${2:-8080}"
SESSION="manudata-dl"

if [[ ! -d "$OUT_DIR" ]]; then
    echo "No such output directory: $OUT_DIR" >&2
    exit 1
fi

STAGE="$OUT_DIR/_download"
rm -rf "$STAGE"
mkdir -p "$STAGE"

shopt -s nullglob

# The keypoints are the point of the whole exercise: an hour of GPU time
# each, and the only artefact that cannot be cheaply regenerated. Refuse
# to serve a download that silently lacks them — a missing .npz noticed
# after the instance is destroyed is unrecoverable.
KEYPOINTS=( "$OUT_DIR"/keypoints/*.npz )
if (( ${#KEYPOINTS[@]} == 0 )); then
    echo "No .npz keypoint files in $OUT_DIR/keypoints/." >&2
    echo "Nothing worth downloading yet — has the analysis pass finished?" >&2
    exit 1
fi

# Collect what exists rather than assuming a fixed layout; a run stopped
# early legitimately has no renders yet.
BUNDLE_ITEMS=()
for item in qc_report.csv manifest.csv batch.log logs keypoints; do
    [[ -e "$OUT_DIR/$item" ]] && BUNDLE_ITEMS+=( "$item" )
done
for sidecar in "$OUT_DIR"/renders/*.done.json; do
    BUNDLE_ITEMS+=( "renders/$(basename "$sidecar")" )
done

# One tarball for the many small files — many small files over HTTP means
# many round trips.
BUNDLE="$STAGE/manudata_qc_data.tar.gz"
echo "Bundling ${#KEYPOINTS[@]} keypoint file(s), report and logs..."
tar czf "$BUNDLE" -C "$OUT_DIR" "${BUNDLE_ITEMS[@]}"

# ...and the keypoints again, individually, so they are visible in the
# browser listing. Buried inside a tarball they look absent, and the file
# you must not leave behind is the one that should be hardest to miss.
for npz in "${KEYPOINTS[@]}"; do
    ln -sf "$npz" "$STAGE/$(basename "$npz")"
done

# Renders are linked rather than copied: they are gigabytes each and
# duplicating them to stage a download is a good way to fill the disk.
for mp4 in "$OUT_DIR"/renders/*.mp4 "$OUT_DIR"/*.mp4; do
    ln -sf "$mp4" "$STAGE/$(basename "$mp4")"
done

# Small enough to be worth having loose as well as in the bundle.
for extra in "$OUT_DIR"/qc_report.csv "$OUT_DIR"/manifest.csv; do
    [[ -e "$extra" ]] && ln -sf "$extra" "$STAGE/$(basename "$extra")"
done

shopt -u nullglob

echo
echo "Checksums — compare these after downloading:"
( cd "$STAGE" && sha256sum -- * 2>/dev/null | sed 's/^/  /' ) || true
( cd "$STAGE" && sha256sum -- * > SHA256SUMS 2>/dev/null ) || true

echo
echo "Staged in $STAGE:"
ls -lh "$STAGE" | sed 's/^/  /'

# Range support decides whether an interrupted 1 GB download can be
# resumed or has to start over. Worth a two-second install.
SERVER_MODULE="http.server"
RESUMABLE="no — an interrupted download restarts from the beginning"
if python3 -c "import RangeHTTPServer" 2>/dev/null \
   || pip install --quiet rangehttpserver 2>/dev/null; then
    if python3 -c "import RangeHTTPServer" 2>/dev/null; then
        SERVER_MODULE="RangeHTTPServer"
        RESUMABLE="yes — your browser can resume an interrupted download"
    fi
fi

tmux kill-session -t "$SESSION" 2>/dev/null || true
tmux new-session -d -s "$SESSION" \
    "cd '$STAGE' && python3 -m $SERVER_MODULE $PORT --bind 127.0.0.1"

sleep 1
if ! tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "Failed to start the server. Is port $PORT already in use?" >&2
    exit 1
fi

cat <<EOF

────────────────────────────────────────────────────────────────
Serving on 127.0.0.1:$PORT (tunnel-only — not reachable from outside).
Server: $SERVER_MODULE.  Resumable: $RESUMABLE.

On your own machine, connect with the tunnel:

    ssh -p <VAST_PORT> root@<VAST_HOST> -L $PORT:localhost:$PORT

then open:

    http://localhost:$PORT/

The .npz keypoint files are listed individually AND inside
manudata_qc_data.tar.gz — take either, but do not leave without them.
They cost an hour of GPU each and nothing else regenerates them; the
MP4s can be re-rendered from them in twenty minutes.

Verify against SHA256SUMS when they land: a truncated MP4 still plays
and still reports its full duration, so size alone proves nothing.

Stop the server when you are done:

    tmux kill-session -t $SESSION
────────────────────────────────────────────────────────────────
EOF
