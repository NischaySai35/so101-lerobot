"""Turn torque OFF on every follower servo. The arm goes limp, so support it first!

Usage:  source scripts/env.sh && python src/tools/release.py
"""
from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus

PORT = "/dev/so100_follower"
NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]

motors = {n: Motor(i + 1, "sts3215", MotorNormMode.DEGREES) for i, n in enumerate(NAMES)}
bus = FeetechMotorsBus(PORT, motors)
bus.connect(handshake=False)  # still release whatever servos answer, even if one is missing

failed = []
for n in NAMES:
    try:
        bus.write("Torque_Enable", n, 0, num_retry=3)
    except Exception as e:
        failed.append(n)
        print(f"{n}: could not release ({e})")

state = {}
for n in NAMES:
    try:
        state[n] = bus.read("Torque_Enable", n, num_retry=3)
    except Exception:
        state[n] = "?"
bus.port_handler.closePort()

print("torque:", state)
if failed or any(v != 0 for v in state.values()):
    print("WARNING: not every servo confirmed torque off")
else:
    print("all servos released (torque off)")
