#!/usr/bin/env bash
# Leader -> follower teleop with the gripper camera shown live in Rerun. Ctrl+C to stop (follower goes limp).
set -e
source "$(dirname "$0")/env.sh"
echo "Follower: $FOLLOWER_PORT  Leader: $LEADER_PORT  Camera: $GRIPPER_CAM"
lerobot-teleoperate \
  --robot.type=so100_follower --robot.port="$FOLLOWER_PORT" --robot.id="$FOLLOWER_ID" \
  --robot.cameras="{ gripper: {type: opencv, index_or_path: $GRIPPER_CAM, width: 640, height: 480, fps: 30}}" \
  --teleop.type=so100_leader --teleop.port="$LEADER_PORT" --teleop.id="$LEADER_ID" \
  --display_data=true
