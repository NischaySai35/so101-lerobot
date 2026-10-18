"""Print the follower's joint positions (raw + calibrated) and torque state. Doesn't move or lock the arm.

Usage: python src/tools/read_positions.py [--save data/poses/NAME.json]
"""
import argparse
import json
from pathlib import Path

from lerobot.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus

PORT = "/dev/so100_follower"
CAL = Path.home() / ".cache/huggingface/lerobot/calibration/robots/so_follower/follower_arm.json"
NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]

ap = argparse.ArgumentParser()
ap.add_argument("--save", help="write the pose to this JSON file (same format as data/poses/home_pose.json)")
args = ap.parse_args()

cal = {k: MotorCalibration(**v) for k, v in json.loads(CAL.read_text()).items()}
motors = {
    n: Motor(i + 1, "sts3215", MotorNormMode.RANGE_0_100 if n == "gripper" else MotorNormMode.DEGREES)
    for i, n in enumerate(NAMES)
}
bus = FeetechMotorsBus(PORT, motors, calibration=cal)
bus.connect(handshake=True)
raw = bus.sync_read("Present_Position", normalize=False)
norm = bus.sync_read("Present_Position")
torque = bus.sync_read("Torque_Enable", normalize=False)
bus.port_handler.closePort()

print(f"{'joint':14} {'raw':>5} {'value':>10}  torque")
for n in NAMES:
    unit = " %" if n == "gripper" else " deg"
    print(f"{n:14} {raw[n]:5d} {norm[n]:8.2f}{unit:4}  {'on' if torque[n] else 'off'}")

if args.save:
    out = {
        "name": Path(args.save).stem,
        "robot_id": "follower_arm",
        "raw": raw,
        "degrees": {n: round(float(norm[n]), 2) for n in NAMES if n != "gripper"},
        "gripper_percent": round(float(norm["gripper"]), 2),
    }
    Path(args.save).parent.mkdir(parents=True, exist_ok=True)
    Path(args.save).write_text(json.dumps(out, indent=2))
    print("saved", args.save)
