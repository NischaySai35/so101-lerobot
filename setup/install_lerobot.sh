#!/usr/bin/env bash
# LeRobot SO-101 install for Ubuntu 24.04 x86_64
# Based on https://wiki.seeedstudio.com/lerobot_so100m_new/ (Ubuntu 22.04 section)
set -e

echo "=== [1/7] System packages (sudo password may be asked) ==="
sudo apt update
sudo apt install -y build-essential git wget
# Ubuntu 24: brltty can grab USB-serial adapters; remove it if present
if dpkg -l brltty 2>/dev/null | grep -q '^ii'; then sudo apt remove -y brltty; fi
# Permanent serial access instead of chmod after every replug
sudo usermod -aG dialout "$USER"
sudo chmod 666 /dev/ttyACM* 2>/dev/null || true

echo "=== [2/7] Miniforge ==="
if [ ! -d "$HOME/miniforge3" ]; then
  cd "$HOME"
  wget -O Miniforge3-Linux-x86_64.sh https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
  chmod +x Miniforge3-Linux-x86_64.sh
  ./Miniforge3-Linux-x86_64.sh -b -p "$HOME/miniforge3"
  "$HOME/miniforge3/bin/conda" init bash
fi
source "$HOME/miniforge3/etc/profile.d/conda.sh"

echo "=== [3/7] conda env 'lerobot' (python 3.12) ==="
conda env list | grep -q '^lerobot ' || conda create -y -n lerobot python=3.12
conda activate lerobot

echo "=== [4/7] ffmpeg ==="
conda install -y ffmpeg -c conda-forge

echo "=== [5/7] Clone Seeed lerobot ==="
[ -d "$HOME/lerobot" ] || git clone https://github.com/Seeed-Projects/lerobot.git "$HOME/lerobot"

echo "=== [6/7] pip install lerobot[feetech] ==="
cd "$HOME/lerobot" && pip install -e ".[feetech]"

echo "=== [7/7] Verify ==="
python -c "import torch, lerobot; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
which lerobot-find-port lerobot-calibrate lerobot-setup-motors
ls -l /dev/ttyACM*
echo "=== DONE ==="
