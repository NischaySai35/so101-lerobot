#!/usr/bin/env bash
# Start the whole chess pipeline and open its window:
#   arm -> home pose, gripper camera + webcam, YOLO board watcher (once trained), Stockfish, browser UI.
# Usage:  scripts/run_pipeline.sh [--dry-run] [--skill 0-20]       (Ctrl+C here to stop; the arm stays holding at home)
# The game starts PAUSED: press Play (or Space) in the window.
set -e
cd "$(dirname "$0")/.."
source scripts/env.sh
PORT="${PORT:-8770}"
URL="http://localhost:$PORT/"

if pgrep -f "chess_player.ui.server" >/dev/null; then
  echo "The pipeline is already running: opening the window."
else
  python -m chess_player.ui.server --port "$PORT" "$@" &
  SERVER=$!
  trap 'kill $SERVER 2>/dev/null; wait $SERVER 2>/dev/null' EXIT INT TERM
  echo -n "starting"
  for _ in $(seq 1 120); do
    curl -s -o /dev/null "$URL/api/state" && break
    kill -0 $SERVER 2>/dev/null || { echo; echo "server exited, see the messages above"; exit 1; }
    echo -n "."; sleep 0.5
  done
  echo
fi

if command -v google-chrome >/dev/null; then
  google-chrome --app="$URL" --window-size=1680,1000 >/dev/null 2>&1 &
else
  xdg-open "$URL" >/dev/null 2>&1 &
fi
echo "UI: $URL   (Ctrl+C to stop)"
[ -n "${SERVER:-}" ] && wait $SERVER
