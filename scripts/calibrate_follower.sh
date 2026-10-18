#!/usr/bin/env bash
# Recalibrate the follower. If a calibration exists, type "c" + Enter to redo it (plain Enter keeps the old one).
set -e
source "$(dirname "$0")/env.sh"
lerobot-calibrate --robot.type=so100_follower --robot.port="$FOLLOWER_PORT" --robot.id="$FOLLOWER_ID"
