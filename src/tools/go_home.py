"""Move the SO100 follower slowly to the saved home pose and hold it there.

Usage:  source scripts/env.sh && python src/tools/go_home.py [--seconds 4] [--release]
  --seconds  how long the move takes (default 4 s)
  --release  turn torque off after arriving (arm goes limp; support it!)
Without --release the servos keep holding the pose after the script exits.
"""
import argparse
import json
import time
from pathlib import Path

from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus

HERE = Path(__file__).resolve().parents[2]  # repo root
PORT = "/dev/so100_follower"
NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]

ap = argparse.ArgumentParser()
ap.add_argument("--pose", default=str(HERE / "data" / "poses" / "home_pose.json"))
ap.add_argument("--seconds", type=float, default=4.0)
ap.add_argument("--release", action="store_true")
args = ap.parse_args()

target = json.loads(Path(args.pose).read_text())["raw"]
motors = {n: Motor(i + 1, "sts3215", MotorNormMode.DEGREES) for i, n in enumerate(NAMES)}
bus = FeetechMotorsBus(PORT, motors)
bus.connect(handshake=True)

start = bus.sync_read("Present_Position", normalize=False)
print("from:", start)
print("to:  ", target)

# Hold the current position first so enabling torque doesn't jerk the arm.
bus.sync_write("Goal_Position", start, normalize=False)
for n in NAMES:
    bus.write("Torque_Enable", n, 1)

steps = max(1, int(args.seconds * 50))
for k in range(1, steps + 1):
    a = k / steps
    a = a * a * (3 - 2 * a)  # smoothstep: gentle start and stop
    goal = {n: round(start[n] + (target[n] - start[n]) * a) for n in NAMES}
    bus.sync_write("Goal_Position", goal, normalize=False)
    time.sleep(1 / 50)

time.sleep(0.5)
now = bus.sync_read("Present_Position", normalize=False)
print("now: ", now)
print("error (counts):", {n: now[n] - target[n] for n in NAMES})

if args.release:
    for n in NAMES:
        bus.write("Torque_Enable", n, 0)
    print("torque released")
else:
    print("holding home pose (torque on)")
bus.port_handler.closePort()
