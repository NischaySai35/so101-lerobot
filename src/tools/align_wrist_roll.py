"""Make the LEADER's wrist roll read the same as the follower's when both grippers point the same way.

  python src/tools/align_wrist_roll.py

Why: the wrist roll turns a full 360 deg, so calibration can't find its zero from end stops; it is wherever the
wrist sat when you pressed Enter while calibrating. If leader and follower sat differently then, teleop turns the
follower's gripper to a different angle than the leader's. This shifts only the leader's wrist-roll zero
(Homing_Offset), in the motor and in its calibration file (a backup is kept). The follower is not touched, so the
pick & place settings stay valid. Nothing moves: both arms are only read (the leader has no torque anyway).

Close teleop / record_demo / the pipeline first (they hold the ports).
"""
import json
import select
import shutil
import sys
import time
from pathlib import Path

from lerobot.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus

CAL = Path.home() / ".cache/huggingface/lerobot/calibration"
FOLLOWER = ("/dev/so100_follower", CAL / "robots/so_follower/follower_arm.json")
LEADER = ("/dev/so100_leader", CAL / "teleoperators/so_leader/leader_arm.json")
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
J = "wrist_roll"


def open_bus(port, cal_path):
    cal = {k: MotorCalibration(**v) for k, v in json.loads(cal_path.read_text()).items()}
    motors = {j: Motor(i + 1, "sts3215", MotorNormMode.RANGE_0_100 if j == "gripper" else MotorNormMode.DEGREES)
              for i, j in enumerate(JOINTS)}
    bus = FeetechMotorsBus(port, motors, calibration=cal)
    bus.connect(handshake=True)
    return bus


def raw(bus):
    return int(bus.read("Present_Position", J, normalize=False))


def deg(r):
    return (r - 2047.5) * 360 / 4095  # range 0..4095 on both arms: calibrated degrees


def wrap(d):
    return (d + 2048) % 4096 - 2048


def main():
    f_bus, l_bus = open_bus(*FOLLOWER), open_bus(*LEADER)
    try:
        print(__doc__)
        print("Turn the LEADER's gripper (wrist roll only) until it points exactly like the follower's gripper.\n"
              "Live readings below; press Enter when they physically match (Ctrl+C = cancel).\n")
        while True:
            f, l = raw(f_bus), raw(l_bus)
            print(f"\r  follower {deg(f):7.1f} deg   leader {deg(l):7.1f} deg   difference {deg(f) - deg(l):+7.1f} deg   ",
                  end="", flush=True)
            if select.select([sys.stdin], [], [], 0.1)[0]:
                sys.stdin.readline()
                break
        f, l = raw(f_bus), raw(l_bus)
        delta = wrap(f - l)  # the leader must read this much more
        old = int(l_bus.read("Homing_Offset", J, normalize=False))
        new = wrap(old - delta)  # Present = Actual - Homing_Offset
        print(f"\n\nleader wrist_roll: homing offset {old} -> {new} (shift {delta * 360 / 4096:+.1f} deg)")
        l_bus.write("Torque_Enable", J, 0)
        l_bus.write("Homing_Offset", J, new)
        time.sleep(0.2)
        f2, l2 = raw(f_bus), raw(l_bus)
        err = wrap(f2 - l2) * 360 / 4096
        print(f"check: follower {deg(f2):.1f} deg, leader {deg(l2):.1f} deg (difference {err:+.1f} deg)")
        if abs(err) > 3.0:
            l_bus.write("Homing_Offset", J, old)
            raise SystemExit("didn't line up (did an arm move?): leader offset put back, calibration unchanged")
        path = LEADER[1]
        backup = path.with_name(path.stem + time.strftime("_backup_%Y%m%d_%H%M%S") + ".json")
        shutil.copy(path, backup)
        d = json.loads(path.read_text())
        d[J]["homing_offset"] = new
        path.write_text(json.dumps(d, indent=4))
        print(f"saved {path} (old one kept as {backup.name}). Teleop wrist roll now matches.")
    finally:
        for b in (f_bus, l_bus):
            b.port_handler.closePort()


if __name__ == "__main__":
    main()
