#!/usr/bin/env bash
# resume.sh — un-suspend a previously ./suspend.sh'd search.

set -euo pipefail
cd "$(dirname "$0")"

PIDFILE="paley.pid"

if [[ ! -f "$PIDFILE" ]]; then
    echo "[resume] no PID file; nothing to resume."
    exit 1
fi

PID="$(cat "$PIDFILE")"
if ! kill -0 "$PID" 2>/dev/null; then
    echo "[resume] PID $PID not running; clean up with ./stop.sh first."
    exit 1
fi

STATE=$(awk '{print $3}' /proc/"$PID"/stat 2>/dev/null || echo "?")
if [[ "$STATE" != "T" ]]; then
    echo "[resume] PID $PID isn't stopped (state=$STATE); nothing to do."
    exit 0
fi

echo "[resume] SIGCONT -> $PID"
kill -CONT "$PID"
