"""Choose the look pose: live wrist-camera view with ArUco status, arm torque off so you can position it.

  python -m chess_player.calib.look_pose            # position by hand, press 's' in the window to save, 'q' quits
  python -m chess_player.calib.look_pose --corners  # also click a1, h1, h8, a8 outer corners as a fallback
                                                    # for when the markers can't be seen (data/manual_corners.json)

Saves data/look_pose.json in the same format as read_positions.py --save. A good look pose: all 4 markers
detected, whole board in frame, jaws spread (gripper fully open) so they don't block the view, and as high
and as central as the arm allows so tall pieces hide as little as possible.
"""
import argparse
import json

import cv2
import matplotlib.pyplot as plt

from chess_player import config as C
from chess_player.vision.aruco_homography import detect_markers, draw_overlay, locate_board
from chess_player.vision.camera import Camera


def open_bus():
    from lerobot.motors import Motor, MotorCalibration, MotorNormMode
    from lerobot.motors.feetech import FeetechMotorsBus

    cal_path = C.Path.home() / ".cache/huggingface/lerobot/calibration/robots/so_follower" / f"{C.FOLLOWER_ID}.json"
    cal = {k: MotorCalibration(**v) for k, v in json.loads(cal_path.read_text()).items()}
    motors = {
        j: Motor(i + 1, "sts3215", MotorNormMode.RANGE_0_100 if j == "gripper" else MotorNormMode.DEGREES)
        for i, j in enumerate(C.ALL_JOINTS)
    }
    bus = FeetechMotorsBus(C.FOLLOWER_PORT, motors, calibration=cal)
    bus.connect(handshake=True)
    return bus


def save_pose(bus):
    raw = bus.sync_read("Present_Position", normalize=False)
    norm = bus.sync_read("Present_Position")
    C.LOOK_POSE.parent.mkdir(parents=True, exist_ok=True)
    C.LOOK_POSE.write_text(json.dumps({
        "name": "look", "robot_id": C.FOLLOWER_ID, "raw": raw,
        "degrees": {j: round(float(norm[j]), 2) for j in C.ARM_JOINTS},
        "gripper_percent": round(float(norm["gripper"]), 2),
    }, indent=2))
    print("saved", C.LOOK_POSE)


def click_corners(frame):
    plt.figure("click a1, h1, h8, a8 outer corners (in that order)")
    plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    pts = plt.ginput(4, timeout=0)
    plt.close()
    C.MANUAL_CORNERS.parent.mkdir(parents=True, exist_ok=True)
    C.MANUAL_CORNERS.write_text(json.dumps({"px": [list(map(float, p)) for p in pts]}, indent=2))
    print("saved", C.MANUAL_CORNERS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corners", action="store_true")
    ap.add_argument("--no-arm", action="store_true", help="camera only (arm not connected)")
    a = ap.parse_args()
    bus = None if a.no_arm else open_bus()
    if bus:
        input("Hold the arm: torque goes off when you press Enter...")
        bus.disable_torque()
    cam = Camera()
    plt.switch_backend("TkAgg")
    for k in [k for k in plt.rcParams if k.startswith("keymap.")]:
        plt.rcParams[k] = []  # free 's', 'q', 'c' from matplotlib's default shortcuts
    fig, ax = plt.subplots(num="look pose: s = save, c = click corners, q = quit")
    keys = []
    fig.canvas.mpl_connect("key_press_event", lambda e: keys.append(e.key))
    im = None
    try:
        while plt.fignum_exists(fig.number):
            frame = cam.grab(settle_s=0, flush=1)
            found = detect_markers(frame)
            shown = frame
            try:
                shown = draw_overlay(frame, locate_board(frame))
            except RuntimeError:
                pass
            txt = f"markers {sorted(found)} / {sorted(C.MARKER_CENTERS)}"
            cv2.putText(shown, txt, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            rgb = cv2.cvtColor(shown, cv2.COLOR_BGR2RGB)
            if im is None:
                im = ax.imshow(rgb)
            else:
                im.set_data(rgb)
            plt.pause(0.03)
            while keys:
                k = keys.pop(0)
                if k == "s" and bus:
                    save_pose(bus)
                    cv2.imwrite(str(C.DATA / "look_view.jpg"), frame)
                elif k == "c" or (k == "s" and a.corners):
                    click_corners(frame)
                elif k == "q":
                    plt.close(fig)
    finally:
        cam.close()
        if bus:
            bus.port_handler.closePort()


if __name__ == "__main__":
    main()
