#!/usr/bin/env bash
# Recalibrate the leader. If a calibration exists, type "c" + Enter to redo it (plain Enter keeps the old one).
set -e
source "$(dirname "$0")/env.sh"
echo "Leader on $LEADER_PORT"
lerobot-calibrate --teleop.type=so100_leader --teleop.port="$LEADER_PORT" --teleop.id="$LEADER_ID"
