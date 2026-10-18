# Shared settings for all scripts. Usage (from the repo folder): source scripts/env.sh
source ~/miniforge3/etc/profile.d/conda.sh
conda activate lerobot
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"

FOLLOWER_PORT=/dev/so100_follower
# Leader: stable udev name if the leader rule is installed, else whichever ttyACM isn't the follower
if [ -e /dev/so100_leader ]; then
  LEADER_PORT=/dev/so100_leader
else
  LEADER_PORT=$(ls /dev/ttyACM* 2>/dev/null | grep -v "$(readlink -f "$FOLLOWER_PORT")" | head -1)
fi
GRIPPER_CAM=/dev/v4l/by-id/usb-Sonix_Technology_Co.__Ltd._USB2.0_CAM1_USB2.0_CAM1-video-index0
FOLLOWER_ID=follower_arm
LEADER_ID=leader_arm
