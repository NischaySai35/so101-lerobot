"""Teach the gripper width for the two piece groups: king/queen, and everything else. Nothing moves.

  python -m chess_player.calib.teach_grip

The gripper motor is switched off so you can close the jaws by hand. For each group: hold a piece between the
jaws at its very BOTTOM (where the robot grabs), squeeze the jaws closed on it, press Enter. Three
readings per group are averaged. Saves data/grip.json; every move then uses these values.
"""
import json

import numpy as np

from chess_player import config as C


def main():
    from lerobot.motors import Motor, MotorCalibration, MotorNormMode
    from lerobot.motors.feetech import FeetechMotorsBus

    cal_path = C.Path.home() / ".cache/huggingface/lerobot/calibration/robots/so_follower" / f"{C.FOLLOWER_ID}.json"
    cal = json.loads(cal_path.read_text())
    bus = FeetechMotorsBus(C.FOLLOWER_PORT, {"gripper": Motor(6, "sts3215", MotorNormMode.RANGE_0_100)},
                           calibration={"gripper": MotorCalibration(**cal["gripper"])})
    bus.connect(handshake=True)
    bus.write("Torque_Enable", "gripper", 0)  # only the gripper goes limp; the arm joints are untouched
    out = {}
    try:
        for key, what in (("small", "a PAWN (or knight / bishop / rook)"), ("big", "the KING or QUEEN")):
            vals = []
            for i in range(3):
                input(f"[{key}] {i + 1}/3: squeeze the jaws closed on the BOTTOM of {what}, then press Enter...")
                v = float(bus.sync_read("Present_Position")["gripper"])
                vals.append(v)
                print(f"   gripper {v:.1f} %")
            out[key] = round(float(np.mean(vals)), 1)
            print(f"   -> {key}: {out[key]} % (spread {max(vals) - min(vals):.1f})\n")
            if max(vals) - min(vals) > 3:
                print("   readings differ a lot; squeeze the same way each time and run this again if needed")
        input("Close the jaws fully with nothing in them (to check the empty reading), then press Enter...")
        empty = float(bus.sync_read("Present_Position")["gripper"])
    finally:
        bus.port_handler.closePort()
    out["empty"] = round(empty, 1)
    print(f"   empty: {empty:.1f} %")
    if min(out["small"], out["big"]) - empty < 2 * C.GRIP_EMPTY_TOL:
        print("WARNING: holding a piece reads almost the same as empty jaws, so misses can't be detected reliably")
    C.GRIP_FILE.parent.mkdir(parents=True, exist_ok=True)
    C.GRIP_FILE.write_text(json.dumps(out, indent=2))
    print(f"saved {C.GRIP_FILE}: {out}")


if __name__ == "__main__":
    main()
