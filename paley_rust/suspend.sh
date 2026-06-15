#!/usr/bin/env bash
# suspend.sh — pause the running search WITHOUT losing in-flight verify progress.
# Sends SIGSTOP, which freezes the process in place (RAM kept).  Resume with
# ./resume.sh.  Use this instead of ./stop.sh when verify is mid-flight on a
# slow prime and you don't want to throw away the work.

set -euo pipefail
cd "$(dirname "$0")"

PIDFILE="paley.pid"

if [[ ! -f "$PIDFILE" ]]; then
    echo "[suspend] no PID file; not running."
    exit 0
fi

PID="$(cat "$PIDFILE")"
if ! kill -0 "$PID" 2>/dev/null; then
    echo "[suspend] PID $PID not running; cleaning up."
    rm -f "$PIDFILE"
    exit 0
fi

STATE=$(awk '{print $3}' /proc/"$PID"/stat 2>/dev/null || echo "?")
if [[ "$STATE" == "T" ]]; then
    echo "[suspend] PID $PID already stopped."
    exit 0
fi

echo "[suspend] SIGSTOP -> $PID  (resume with ./resume.sh)"
kill -STOP "$PID"
