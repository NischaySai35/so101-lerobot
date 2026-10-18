#!/usr/bin/env bash
# Fetch what git doesn't store: Stockfish, the Robo voice, and the pretrained chess YOLO model.
#   setup/download_models.sh
# (Whisper and Qwen are downloaded automatically by transformers on the first pipeline start.)
# Your own trained YOLO model (data/yolo/best.pt) isn't downloaded: retrain it from data/dataset with
#   python -m chess_player.vision.train_yolo
set -e
cd "$(dirname "$0")/.."

if [ ! -x bin/stockfish ]; then
  echo "=== Stockfish 19 (Linux x86-64) ==="
  mkdir -p bin /tmp/sf_dl
  curl -L -o /tmp/sf_dl/sf.tar.gz \
    https://github.com/official-stockfish/Stockfish/releases/download/sf_19/stockfish-linux-x86-64-universal.tar.gz
  tar -xzf /tmp/sf_dl/sf.tar.gz -C /tmp/sf_dl
  cp "$(find /tmp/sf_dl -type f -name 'stockfish*' ! -name '*.tar.gz' -perm -u+x | head -1)" bin/stockfish
  chmod +x bin/stockfish
  rm -rf /tmp/sf_dl
fi

VOICE=data/models/piper
if [ ! -f "$VOICE/en_US-ryan-medium.onnx" ]; then
  echo "=== Piper voice (en_US ryan medium) ==="
  mkdir -p "$VOICE"
  base=https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/ryan/medium
  curl -L -o "$VOICE/en_US-ryan-medium.onnx" "$base/en_US-ryan-medium.onnx"
  curl -L -o "$VOICE/en_US-ryan-medium.onnx.json" "$base/en_US-ryan-medium.onnx.json"
fi

YOLO=data/models/pretrained_chess/KanisornPutta_chess-model-yolov8m
if [ ! -f "$YOLO/chess-model-yolov8m.pt" ]; then
  echo "=== Pretrained chess YOLO (KanisornPutta/chess-model-yolov8m) ==="
  mkdir -p "$YOLO"
  curl -L -o "$YOLO/chess-model-yolov8m.pt" \
    https://huggingface.co/KanisornPutta/chess-model-yolov8m/resolve/main/chess-model-yolov8m.pt
fi

echo "done: bin/stockfish, $VOICE, $YOLO"
