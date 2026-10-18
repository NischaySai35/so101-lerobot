"""Record teleoperated demonstrations (leader -> follower) so the motion can be studied and turned into code.

  python -m chess_player.teach.record_demo --task "rook e3 to e5"
  python -m chess_player.teach.record_demo --task "pawn e2 e4 past pieces" --episodes 3
  python -m chess_player.teach.record_demo --task "..." --lerobot      # also write a LeRobot (ACT) dataset

The follower copies the leader at 30 Hz the whole time (and eases onto it at the start instead of jumping).
Enter = START recording, Enter = STOP (or after --seconds), then keep / discard.

Each kept episode is a folder data/demos/<date>_<time>_<task>/ with:
  wrist.mp4, webcam.mp4   the two cameras (black frames if a camera had no picture)
  poses.csv               written while recording (a crash can't lose it), one row per frame: time, follower and
                          leader joints (deg), gripper %, jaw-tip pose in the robot frame (x, y, z mm; roll, pitch,
                          yaw deg), the tip in board coordinates (mm), height above the board, tilt, square under
                          the tip, servo load % (shoulder, elbow; every 5th frame)
  info.json               task, fps, frame count, duration, camera status
Units: LeRobot calibrated degrees, gripper 0 closed .. 100 open.
"""
import argparse
import csv
import json
import re
import queue
import shutil
import sys
import threading
import time

import chess
import cv2
import numpy as np

from chess_player import config as C
from chess_player.geometry import BoardToRobot, mm_to_square
from chess_player.motion.arm import Arm
from chess_player.motion.ik import IK

DEMOS = C.DATA / "demos"
WEB_W, WEB_H = 640, 360
JOINT_KEYS = [f"{j}.pos" for j in C.ALL_JOINTS]


def rpy_deg(R):
    """Roll (x), pitch (y), yaw (z) of a rotation matrix, ZYX convention, degrees."""
    pitch = np.arcsin(-np.clip(R[2, 0], -1, 1))
    roll = np.arctan2(R[2, 1], R[2, 2])
    yaw = np.arctan2(R[1, 0], R[0, 0])
    return np.degrees([roll, pitch, yaw])


# All keyboard input goes through ONE reader thread. (Mixing select()+readline with input() let a key press sit in
# a buffer the other one never looks at: Enter then seemed to be ignored while recording.)
_lines = queue.Queue()


def _read_stdin():
    for line in sys.stdin:
        _lines.put(line)


threading.Thread(target=_read_stdin, daemon=True).start()


def key_pressed():
    try:
        _lines.get_nowait()
        return True
    except queue.Empty:
        return False


def ask(prompt):
    """Like input(), from the same reader."""
    print(prompt, end="", flush=True)
    while True:
        try:
            return _lines.get(timeout=0.1).strip()
        except queue.Empty:
            pass


def find_leader():
    if C.Path("/dev/so100_leader").exists():
        return "/dev/so100_leader"
    import glob
    import os

    follower = os.path.realpath(C.FOLLOWER_PORT)
    others = [p for p in sorted(glob.glob("/dev/ttyACM*")) if os.path.realpath(p) != follower]
    if not others:
        raise SystemExit("leader arm not found: plug in its USB cable")
    return others[0]


class LeRobotWriter:
    """Optional ACT dataset (data/demos/lerobot_<name>). Always gets every feature it was created with."""

    def __init__(self, name, fps):
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        root = DEMOS / f"lerobot_{name}"
        cam = {"names": ["height", "width", "channels"], "info": {"is_depth_map": False}, "dtype": "video"}
        if (root / "meta" / "info.json").exists():
            self.ds = LeRobotDataset.resume(f"local/{name}", root=root, image_writer_threads=4)
        else:
            feats = {"observation.state": {"dtype": "float32", "shape": (6,), "names": JOINT_KEYS},
                     "action": {"dtype": "float32", "shape": (6,), "names": JOINT_KEYS},
                     "observation.images.wrist": {**cam, "shape": (C.CAM_H, C.CAM_W, 3)},
                     "observation.images.webcam": {**cam, "shape": (WEB_H, WEB_W, 3)}}
            self.ds = LeRobotDataset.create(f"local/{name}", fps, feats, root=root, robot_type="so101_follower",
                                            use_videos=True, image_writer_threads=8)

    def add(self, obs, act, wrist_rgb, web_rgb, task):
        self.ds.add_frame({"observation.state": np.array(obs, np.float32), "action": np.array(act, np.float32),
                           "observation.images.wrist": wrist_rgb, "observation.images.webcam": web_rgb, "task": task})

    def keep(self, keep):
        if keep:
            self.ds.save_episode()
        else:
            self.ds.clear_episode_buffer()

    def close(self):
        self.ds.finalize()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, help='what you demonstrate, e.g. "rook e3 to e5"')
    ap.add_argument("--episodes", type=int, default=1)
    ap.add_argument("--seconds", type=float, default=90.0, help="max length of one episode")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--no-webcam", action="store_true")
    ap.add_argument("--lerobot", action="store_true", help="also append to a LeRobot dataset (for ACT)")
    a = ap.parse_args()

    from lerobot.teleoperators.so_leader import SOLeader, SOLeaderTeleopConfig

    from chess_player.ui.server import LiveCam
    from chess_player.vision.ext_camera import ExtCamera

    leader_port = find_leader()
    print(f"leader {leader_port}, follower {C.FOLLOWER_PORT}")
    wrist = LiveCam()
    ext = None
    if not a.no_webcam:
        try:
            ext = ExtCamera()
        except RuntimeError as e:
            print(f"\n*** WEBCAM NOT AVAILABLE ({e}).\n*** The phone serves one viewer at a time: close the pipeline "
                  f"window and any browser tab showing {C.EXT_CAMERA_URL.rsplit('/', 1)[0]}.\n*** Recording black "
                  f"frames for the webcam. Ctrl+C now to fix it first.\n")
    leader = SOLeader(SOLeaderTeleopConfig(port=leader_port, id="leader_arm"))
    leader.connect(calibrate=False)
    arm = Arm(use_sag_model=False)
    ik = IK()
    T = BoardToRobot.load() if C.BOARD_TO_ROBOT.exists() else None
    lr = LeRobotWriter(re.sub(r"\W+", "_", "chess_demos"), a.fps) if a.lerobot else None

    def leader_goal():
        act = leader.get_action()
        return {j: float(act[f"{j}.pos"]) for j in C.ALL_JOINTS}

    def teleop_step():
        goal = leader_goal()
        arm._write(goal)
        return goal

    t_wait = time.time()
    while wrist.get() is None and time.time() - t_wait < 5:
        time.sleep(0.1)
    if wrist.get() is None:
        print("WARNING: gripper camera has no picture yet; recording black frames until it does")
    try:
        goal = leader_goal()
        worst = max(abs(goal[j] - arm.joints()[j]) for j in C.ARM_JOINTS)
        print(f"leader and follower differ by up to {worst:.0f} deg: easing the follower onto the leader over 2 s")
        arm.move_to(goal, 2.0)
        period = 1.0 / a.fps
        done = 0
        while done < a.episodes:
            print(f"\n[episode {done + 1}/{a.episodes}] teleop is live. Press Enter to START recording...")
            while not key_pressed():
                t0 = time.perf_counter()
                teleop_step()
                time.sleep(max(0.0, period - (time.perf_counter() - t0)))
            slug = re.sub(r"\W+", "_", a.task.strip().lower()).strip("_")[:40]
            ep_dir = DEMOS / f"{time.strftime('%Y%m%d_%H%M%S')}_{slug}"
            ep_dir.mkdir(parents=True, exist_ok=True)
            vw = cv2.VideoWriter(str(ep_dir / "wrist.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), a.fps, (C.CAM_W, C.CAM_H))
            ve = cv2.VideoWriter(str(ep_dir / "webcam.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), a.fps, (WEB_W, WEB_H))
            fcsv = open(ep_dir / "poses.csv", "w", newline="")
            writer = None
            n, black = 0, {"wrist": 0, "webcam": 0}
            t_start = time.perf_counter()
            print(f"RECORDING into {ep_dir.name}. Press Enter (or Ctrl+C) to STOP.")
            while not _lines.empty():  # forget any extra Enter pressed before recording started
                _lines.get_nowait()
            try:
                while time.perf_counter() - t_start < a.seconds and not key_pressed():
                    t0 = time.perf_counter()
                    act = teleop_step()
                    obs = arm.present()
                    fw = wrist.get()
                    if fw is None:
                        fw, black["wrist"] = np.zeros((C.CAM_H, C.CAM_W, 3), np.uint8), black["wrist"] + 1
                    fw = cv2.resize(fw, (C.CAM_W, C.CAM_H))
                    fe = None
                    if ext is not None:
                        try:
                            fe = ext.grab(max_age_s=1.0)
                        except RuntimeError:
                            fe = None
                    if fe is None:
                        fe, black["webcam"] = np.zeros((WEB_H, WEB_W, 3), np.uint8), black["webcam"] + 1
                    fe = cv2.resize(fe, (WEB_W, WEB_H))
                    vw.write(fw)
                    ve.write(fe)
                    if lr:
                        lr.add([obs[j] for j in C.ALL_JOINTS], [act[j] for j in C.ALL_JOINTS],
                               cv2.cvtColor(fw, cv2.COLOR_BGR2RGB), cv2.cvtColor(fe, cv2.COLOR_BGR2RGB), a.task)
                    # 6-D jaw-tip pose + board coordinates
                    Tt = ik.fk(np.array([obs[j] for j in C.ARM_JOINTS]))
                    p = Tt[:3, 3] * 1000
                    roll, pitch, yaw = rpy_deg(Tt[:3, :3])
                    tilt = float(np.degrees(np.arccos(np.clip(-Tt[2, 2], -1, 1))))
                    if T is not None:
                        bx, by, bz = T.R.T @ (Tt[:3, 3] - T.t) * 1000
                        sq = mm_to_square(bx, by)
                        sqn = chess.square_name(sq) if sq is not None else "off"
                    else:
                        bx = by = bz = float("nan")
                        sqn = "?"
                    row = {"t": round(time.perf_counter() - t_start, 3), "frame": n}
                    row.update({f"f_{j}": round(obs[j], 2) for j in C.ALL_JOINTS})
                    row.update({f"l_{j}": round(act[j], 2) for j in C.ALL_JOINTS})
                    row.update(x_mm=round(p[0], 1), y_mm=round(p[1], 1), z_mm=round(p[2], 1), roll_deg=round(roll, 1),
                               pitch_deg=round(pitch, 1), yaw_deg=round(yaw, 1), board_x_mm=round(bx, 1),
                               board_y_mm=round(by, 1), height_above_board_mm=round(bz, 1), tilt_deg=round(tilt, 1),
                               square=sqn, lift_load_pct="", elbow_load_pct="")
                    if n % 5 == 0:  # loads cost bus time: sample them at 6 Hz
                        load, _ = arm.load_status()
                        row.update(lift_load_pct=load.get("shoulder_lift"), elbow_load_pct=load.get("elbow_flex"))
                    if writer is None:
                        writer = csv.DictWriter(fcsv, fieldnames=list(row))
                        writer.writeheader()
                    writer.writerow(row)
                    if n % 15 == 0:
                        fcsv.flush()
                    n += 1
                    if n % 30 == 0:
                        print(f"  {n / a.fps:4.1f}s  tip over {sqn:>3}  {bz:5.0f} mm above board  tilt {tilt:4.0f}  "
                              f"gripper {obs['gripper']:5.1f}%")
                    time.sleep(max(0.0, period - (time.perf_counter() - t0)))
            except KeyboardInterrupt:  # Ctrl+C = stop this recording (it can still be kept)
                print("\n(stopped with Ctrl+C)")
            finally:
                vw.release()
                ve.release()
                fcsv.close()
            dur = n / a.fps
            ans = ask(f"stopped after {n} frames ({dur:.1f} s). Keep it? [Y/n] ").lower()
            keep = ans != "n" and n >= 10
            if lr:
                try:
                    lr.keep(keep)
                except Exception as e:
                    print(f"LeRobot dataset save failed ({e}); the episode folder is still saved")
            if not keep:
                shutil.rmtree(ep_dir, ignore_errors=True)
                print("discarded")
                continue
            (ep_dir / "info.json").write_text(json.dumps({
                "task": a.task, "fps": a.fps, "frames": n, "seconds": round(dur, 2),
                "black_frames": black, "board_calibrated": T is not None}, indent=2))
            msg = "" if not any(black.values()) else f" (black frames: {black})"
            print(f"saved {ep_dir}{msg}")
            done += 1
    finally:
        if lr:
            try:
                lr.close()
            except Exception as e:
                print("LeRobot finalize:", e)
        leader.disconnect()
        arm.close()  # follower keeps holding its last pose (torque on): src/tools/release.py lets it go limp
        if ext is not None:
            ext.close()
    print("\nanalyse: python -m chess_player.teach.analyze_demo", flush=True)


if __name__ == "__main__":
    import os

    main()
    # Exit without waiting for the camera reader threads: letting Python tear them down while OpenCV is inside a
    # blocking read aborts with "FATAL: exception not rethrown" (harmless, everything is saved, but ugly).
    sys.stdout.flush()
    os._exit(0)
