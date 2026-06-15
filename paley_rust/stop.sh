#!/usr/bin/env bash
# stop.sh — gracefully stop the Paley search.  Sends SIGINT so the binary's
# Ctrl-C handler finishes the current prime and saves the bookmark before exit.
# Usage:
#   ./stop.sh           # graceful (SIGINT)
#   ./stop.sh --force   # immediate (SIGKILL)

set -euo pipefail
cd "$(dirname "$0")"

PIDFILE="paley.pid"

if [[ ! -f "$PIDFILE" ]]; then
    echo "[stop] no PID file; not running."
    exit 0
fi

PID="$(cat "$PIDFILE")"
if ! kill -0 "$PID" 2>/dev/null; then
    echo "[stop] PID $PID is not running; cleaning up."
    rm -f "$PIDFILE"
    exit 0
fi

if [[ "${1:-}" == "--force" ]]; then
    echo "[stop] SIGKILL -> $PID"
    kill -9 "$PID" || true
else
    echo "[stop] SIGINT -> $PID (graceful, will finish current prime)"
    kill -INT "$PID" || true
fi

# Wait briefly for it to die
for _ in $(seq 1 30); do
    if ! kill -0 "$PID" 2>/dev/null; then
        echo "[stop] stopped."
        rm -f "$PIDFILE"
        exit 0
    fi
    sleep 1
done

echo "[stop] still alive after 30s; consider ./stop.sh --force"
exit 1
