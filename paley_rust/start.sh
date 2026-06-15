#!/usr/bin/env bash
# start.sh — build (if needed) and launch the Paley A(8) search in the background.
# Usage:
#   ./start.sh                      # default: A 8 with default flags
#   ./start.sh A 8 --lo 2683        # custom args (forwarded verbatim to the binary)
#   ./start.sh SK 7 --threads 16
#
# Writes the PID to ./paley.pid and the live log to ./paley.log.

set -euo pipefail
cd "$(dirname "$0")"

PIDFILE="paley.pid"
LOGFILE="paley.log"

if [[ -f "$PIDFILE" ]] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
    echo "[start] already running with PID $(cat "$PIDFILE"); use ./stop.sh first."
    exit 1
fi

echo "[start] building release binary ..."
cargo build --release --quiet

ARGS=("$@")
if [[ ${#ARGS[@]} -eq 0 ]]; then
    ARGS=("A" "8")
fi

# Archive the previous log (if any) before starting a fresh one.
if [[ -s "$LOGFILE" ]]; then
    STAMP=$(date +%Y%m%d-%H%M%S)
    ARCHIVED="${LOGFILE}.${STAMP}"
    mv "$LOGFILE" "$ARCHIVED"
    echo "[start] archived previous log -> $ARCHIVED"
fi

echo "[start] launching: paley ${ARGS[*]}"
nohup nice -n 10 ./target/release/paley "${ARGS[@]}" >>"$LOGFILE" 2>&1 &
echo $! > "$PIDFILE"
echo "[start] PID $(cat "$PIDFILE"), log = $LOGFILE"
echo "[start] tail with: tail -f $LOGFILE"
