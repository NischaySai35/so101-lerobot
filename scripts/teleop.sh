#!/usr/bin/env bash
# Leader -> follower teleop, no camera. Put the leader in the follower's pose first. Ctrl+C to stop (follower goes limp).
set -e
source "$(dirname "$0")/env.sh"
echo "Follower: $FOLLOWER_PORT  Leader: $LEADER_PORT"
lerobot-teleoperate \
  --robot.type=so100_follower --robot.port="$FOLLOWER_PORT" --robot.id="$FOLLOWER_ID" \
  --teleop.type=so100_leader --teleop.port="$LEADER_PORT" --teleop.id="$LEADER_ID" \
  --display_data=false
