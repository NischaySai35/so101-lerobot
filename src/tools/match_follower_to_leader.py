"""Make the FOLLOWER's wrist roll and gripper match the leader (the leader's calibration is not changed).

  python src/tools/match_follower_to_leader.py [--dry-run]

- leader: put back exactly as in its calibration backup (undoes align_wrist_roll.py, if it was run)
- follower wrist roll: zero shifted by the difference align_wrist_roll.py measured with both grippers pointing the
  same way, so the follower turns to the same angle as the leader
- follower gripper: its 0-100 % now spans the leader's travel (from the follower's own closed reading), so the
  jaws open to the same angle as the leader's

Only the wrist roll and gripper motors are touched (torque off -> write -> torque on at the present position, so
nothing jumps; these two don't hold the arm up). Calibration files are backed up first. Afterwards the gripper %
values in the code are converted (the script prints the factor). Close teleop / record_demo / the pipeline first.
"""
import argparse
import glob
import json
import shutil
import time
from pathlib import Path

from align_wrist_roll import FOLLOWER, LEADER, open_bus, wrap


def latest_backup(path: Path):
    files = sorted(glob.glob(str(path.with_name(path.stem + "_backup_*.json"))))
    return Path(files[0]) if files else None  # the oldest = before any alignment


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    f_path, l_path = FOLLOWER[1], LEADER[1]
    f_cal, l_cal = json.loads(f_path.read_text()), json.loads(l_path.read_text())
    l_orig_path = latest_backup(l_path)
    l_orig = json.loads(l_orig_path.read_text()) if l_orig_path else l_cal

    # wrist roll: align_wrist_roll set leader offset new = old - delta (delta = how much more the leader had to read)
    delta = wrap(l_orig["wrist_roll"]["homing_offset"] - l_cal["wrist_roll"]["homing_offset"])
    # with the leader back at its old offset it reads `delta` less; the follower must read `delta` less too:
    # Present = Actual - Homing_Offset  ->  follower offset + delta
    f_roll_old = f_cal["wrist_roll"]["homing_offset"]
    f_roll_new = wrap(f_roll_old + delta)
    # gripper: follower span := leader span, from the follower's closed reading
    l_span = l_orig["gripper"]["range_max"] - l_orig["gripper"]["range_min"]
    g_min, g_max_old = f_cal["gripper"]["range_min"], f_cal["gripper"]["range_max"]
    g_max_new = g_min + l_span
    factor = (g_max_old - g_min) / l_span
    print(f"leader: restore from {l_orig_path.name if l_orig_path else '(no backup: unchanged)'}")
    print(f"follower wrist_roll homing offset {f_roll_old} -> {f_roll_new} ({-delta * 360 / 4096:+.2f} deg reading)")
    print(f"follower gripper range {g_min}..{g_max_old} -> {g_min}..{g_max_new}; gripper % values x {factor:.4f}")
    if a.dry_run:
        return
    stamp = time.strftime("_backup_%Y%m%d_%H%M%S")
    shutil.copy(f_path, f_path.with_name(f_path.stem + stamp + ".json"))

    l_bus = open_bus(*LEADER)
    try:
        l_bus.write("Torque_Enable", "wrist_roll", 0)
        l_bus.write("Homing_Offset", "wrist_roll", l_orig["wrist_roll"]["homing_offset"])
        for k in ("Min_Position_Limit", "Max_Position_Limit"):
            l_bus.write(k, "gripper", l_orig["gripper"]["range_min" if k.startswith("Min") else "range_max"])
    finally:
        l_bus.port_handler.closePort()
    l_path.write_text(json.dumps(l_orig, indent=4))

    f_bus = open_bus(*FOLLOWER)
    try:
        for j, regs in (("wrist_roll", {"Homing_Offset": f_roll_new}), ("gripper", {"Max_Position_Limit": g_max_new})):
            was_on = int(f_bus.read("Torque_Enable", j))
            f_bus.write("Torque_Enable", j, 0)
            for r, v in regs.items():
                f_bus.write(r, j, v)
            time.sleep(0.05)
            present = f_bus.read("Present_Position", j, normalize=False)
            f_bus.write("Goal_Position", j, present, normalize=False)  # hold where it is: no jump
            if was_on:
                f_bus.write("Torque_Enable", j, 1)
        got = (int(f_bus.read("Homing_Offset", "wrist_roll", normalize=False)),
               int(f_bus.read("Max_Position_Limit", "gripper", normalize=False)))
    finally:
        f_bus.port_handler.closePort()
    f_cal["wrist_roll"]["homing_offset"] = f_roll_new
    f_cal["gripper"]["range_max"] = g_max_new
    f_path.write_text(json.dumps(f_cal, indent=4))
    print(f"done: follower now wrist_roll offset {got[0]}, gripper max {got[1]} (files saved, backups kept)")
    print(f"GRIPPER_FACTOR={factor:.4f} ROLL_SHIFT_DEG={-delta * 360 / 4096:.3f} ROLL_SHIFT_COUNTS={-delta}")


if __name__ == "__main__":
    main()
