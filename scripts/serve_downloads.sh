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
mkdir -p "$STAGE"

# The metadata is many small files, and many small files over HTTP means
# many round trips. One tarball is a single click and a single transfer.
BUNDLE="$STAGE/manudata_qc_data.tar.gz"
echo "Bundling keypoints, report and logs..."
tar czf "$BUNDLE" -C "$OUT_DIR" \
    --exclude="_download" \
    $(cd "$OUT_DIR" && ls -d qc_report.csv manifest.csv batch.log logs keypoints 2>/dev/null) \
    $(cd "$OUT_DIR" && ls renders/*.done.json 2>/dev/null || true)

# Renders are linked rather than copied: they are gigabytes each and
# duplicating them to stage a download is a good way to fill the disk.
shopt -s nullglob
for mp4 in "$OUT_DIR"/renders/*.mp4 "$OUT_DIR"/*.mp4; do
    ln -sf "$mp4" "$STAGE/$(basename "$mp4")"
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

Download manudata_qc_data.tar.gz first — it is small and holds the
keypoints, which are the part that cannot be regenerated cheaply.
Then the .mp4 files. Verify against SHA256SUMS when they land: a
truncated MP4 still plays and still reports its full duration.

Stop the server when you are done:

    tmux kill-session -t $SESSION
────────────────────────────────────────────────────────────────
EOF
