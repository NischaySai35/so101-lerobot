"""Measure joint sag over the board and fit the gravity sag model (motion/sag.py). The arm MOVES.

  python -m chess_player.calib.sag_calibration [--height 15] [--squares a1 e1 h1 a5 e5 h5 a8 e8 h8]

Visits each square at travel height and at --height above it (nothing is grasped; clear those squares),
measures at each stop how much correction the joints needed (settle), then fits
correction = k * gravity_torque + b per joint. Every stop is kept short (under the servos' 2 s overload
trip). Afterwards every move uses the model, so the arm aims right the first time anywhere on the board.
"""
import argparse

import chess
import numpy as np

from chess_player import config as C
from chess_player.geometry import BoardToRobot, square_center_mm
from chess_player.motion.arm import Arm, look_pose
from chess_player.motion.ik import IK, solve_or_raise
from chess_player.motion.sag import SagModel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--height", type=float, default=15.0)
    ap.add_argument("--squares", nargs="+", default=["a1", "e1", "h1", "a5", "e5", "h5", "a8", "e8", "h8"])
    a = ap.parse_args()
    T, ik = BoardToRobot.load(), IK()
    arm = Arm(use_sag_model=False)  # measure raw sag, not the old model's leftovers
    look = {**look_pose(), "gripper": C.GRIPPER_LOOK}
    samples = []
    try:
        q = np.array([look[j] for j in C.ARM_JOINTS])
        q[4] = C.WRIST_ROLL_DEG
        for name in a.squares:
            xy = square_center_mm(chess.parse_square(name))
            hi = solve_or_raise(ik, T(*xy, C.Z_SAFE), q, what=f"{name} high")
            lo = solve_or_raise(ik, T(*xy, a.height), hi, C.GRASP_MAX_TILT, what=f"{name} low")
            for label, target, dur in (("high", hi, None), ("low", lo, C.DESCEND_S)):
                goal = {**dict(zip(C.ARM_JOINTS, map(float, target))), "gripper": 0.0}
                arm.move_to(goal, dur)
                left = arm.settle(iters=4)
                corr = {j: round(arm.comp.get(j, 0.0), 2) for j in C.ARM_JOINTS[:4]}
                samples.append((goal, corr))
                print(f"  {name} {label:4}: correction deg {corr}   (left over {left:.1f} deg)")
            arm.move_to({**dict(zip(C.ARM_JOINTS, map(float, hi))), "gripper": 0.0}, C.DESCEND_S)
            q = hi
        arm.reset_residual()
        arm.move_to(look, 2.0)
    finally:
        arm.close()

    k, b, rms = SagModel.fit(samples)
    print("\nfit: correction = k * gravity torque (N·m) + b")
    for j in k:
        print(f"  {j:14} k {k[j]:7.2f} deg/N·m   b {b[j]:6.2f} deg   RMS error {rms[j]:.2f} deg")
    SagModel.save(k, b, rms, samples)
    print(f"saved {C.DATA / 'sag_model.json'}: all moves now use it")


if __name__ == "__main__":
    main()
