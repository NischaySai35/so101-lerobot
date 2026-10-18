"""Sanity test after board_to_robot: hover the jaw tip a few mm above the four corner squares, then go back to
the look pose. Watch it and check by eye. The arm MOVES.

  python -m chess_player.calib.hover_test [--height 15] [--squares a1 h1 h8 a8] [--no-sag-fix] [--dry-run]

At each stop it holds for --hold seconds (default 1.5, below the servos' 2 s overload trip) and logs, every
0.25 s, the tip height from the measured joints, the shoulder/elbow load (% of max) and the servo status, then
goes back up by itself. Feetech overload protection: >80% load for 2 s drops that servo to 20% torque.
"""
import time
import argparse

import chess
import numpy as np

from chess_player import config as C
from chess_player.geometry import BoardToRobot, square_center_mm
from chess_player.motion.arm import Arm, FakeArm, look_pose
from chess_player.motion.ik import IK, solve_or_raise


def tip_height_mm(ik, T, joints):
    p = ik.fk(np.array([joints[j] for j in C.ARM_JOINTS]))[:3, 3]
    return float((T.R.T @ (p - T.t))[2] * 1000)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--height", type=float, default=15.0, help="mm above the board surface")
    ap.add_argument("--squares", nargs="+", default=["a1", "h1", "h8", "a8"])
    ap.add_argument("--no-sag-fix", action="store_true", help="skip the sag correction (to compare)")
    ap.add_argument("--hold", type=float, default=1.5, help="seconds at the low point (logged)")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    T, ik = BoardToRobot.load(), IK()
    arm = FakeArm() if a.dry_run else Arm()
    look = {**look_pose(), "gripper": C.GRIPPER_LOOK}
    try:
        q = np.array([look[j] for j in C.ARM_JOINTS])
        q[4] = C.WRIST_ROLL_DEG
        for name in a.squares:
            xy = square_center_mm(chess.parse_square(name))
            hi = solve_or_raise(ik, T(*xy, C.Z_SAFE), q, what=f"{name} high")
            lo = solve_or_raise(ik, T(*xy, a.height), hi, C.GRASP_MAX_TILT, what=f"{name} low")
            arm.move_to({**dict(zip(C.ARM_JOINTS, hi)), "gripper": 0.0})
            if not a.no_sag_fix:
                arm.settle()
            arm.move_to(dict(zip(C.ARM_JOINTS, lo)), C.DESCEND_S)
            print(f"  {name}: holding {a.hold:.1f} s at {a.height:.0f} mm  (t, tip mm, lift load %, elbow load %, status)")
            t0 = time.time()
            while time.time() - t0 < a.hold:
                now = arm.present()
                load, status = arm.load_status() if hasattr(arm, "load_status") else ({}, {})
                flag = "  <-- OVERLOAD" if any(v >= 80 for v in load.values()) or any(status.values()) else ""
                print(f"    {time.time() - t0:4.2f}s  tip {tip_height_mm(ik, T, now):5.1f}  "
                      f"lift {load.get('shoulder_lift', 0):4.0f}  elbow {load.get('elbow_flex', 0):4.0f}  "
                      f"status {status}{flag}")
                time.sleep(0.25)
            arm.move_to(dict(zip(C.ARM_JOINTS, hi)), C.DESCEND_S)
            q = hi
        arm.move_to(look, 2.0)
    finally:
        arm.close()


if __name__ == "__main__":
    main()
