"""Make the follower's jaw open to the same angle as the leader's in teleop.

  python src/tools/align_gripper.py

Why: the gripper value is 0-100 % between each arm's own closed and open readings. The follower's jaw travels
~1470 counts closed->open, the leader's ~1140, so at the same % the follower opens wider. This gives the LEADER the
follower's span, starting at its own fully-closed reading: one count on the leader = one count on the follower
(same angle, both direct-drive STS3215). Only the leader's calibration changes (motor + file, backup kept); the
follower's is untouched, so the pick & place gripper values stay valid. Nothing moves.

Close teleop / record_demo / the pipeline first (they hold the ports).
"""
import json
import shutil
import time

from align_wrist_roll import FOLLOWER, LEADER, open_bus

J = "gripper"


def main():
    f_cal = json.loads(FOLLOWER[1].read_text())[J]
    span = f_cal["range_max"] - f_cal["range_min"]
    l_bus = open_bus(*LEADER)
    try:
        print(__doc__)
        input("Squeeze the LEADER's gripper fully CLOSED and hold it there, then press Enter...")
        closed = int(l_bus.read("Present_Position", J, normalize=False))
        input("Now open it fully and hold it there, then press Enter...")
        opened = int(l_bus.read("Present_Position", J, normalize=False))
        if opened - closed < 300:
            raise SystemExit(f"closed {closed} / open {opened}: hardly moved, nothing changed. Try again.")
        new_min, new_max = closed, min(4095, closed + span)
        path = LEADER[1]
        d = json.loads(path.read_text())
        old = (d[J]["range_min"], d[J]["range_max"])
        print(f"\nleader gripper range {old[0]}..{old[1]} -> {new_min}..{new_max} "
              f"(follower span {span} counts; your full open will read {100 * (opened - closed) / span:.0f} %)")
        l_bus.write("Torque_Enable", J, 0)
        l_bus.write("Min_Position_Limit", J, new_min)
        l_bus.write("Max_Position_Limit", J, new_max)
        backup = path.with_name(path.stem + time.strftime("_backup_%Y%m%d_%H%M%S") + ".json")
        shutil.copy(path, backup)
        d[J]["range_min"], d[J]["range_max"] = new_min, new_max
        path.write_text(json.dumps(d, indent=4))
        print(f"saved {path} (old one kept as {backup.name}). In teleop the jaws now open to the same angle.")
    finally:
        l_bus.port_handler.closePort()


if __name__ == "__main__":
    main()
