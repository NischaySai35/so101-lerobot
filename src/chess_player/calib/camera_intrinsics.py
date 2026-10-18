"""Wrist-camera intrinsics from a printed checkerboard (optional, but it reduces homography error).

  python -m chess_player.calib.camera_intrinsics --cols 9 --rows 6 --square-mm 25

`cols` x `rows` are INNER corners. Hold the printed pattern in front of the camera at different angles and
distances; press Enter for each shot (about 15-20), 'q' + Enter to finish. Saves data/camera_intrinsics.npz.
"""
import argparse

import cv2
import numpy as np

from chess_player import config as C
from chess_player.vision.camera import Camera


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cols", type=int, default=9)
    ap.add_argument("--rows", type=int, default=6)
    ap.add_argument("--square-mm", type=float, default=25.0)
    a = ap.parse_args()
    pattern = (a.cols, a.rows)
    obj = np.zeros((a.cols * a.rows, 3), np.float32)
    obj[:, :2] = np.mgrid[0 : a.cols, 0 : a.rows].T.reshape(-1, 2) * a.square_mm
    cam = Camera()
    objs, imgs = [], []
    out_dir = C.DATA / "intrinsics_shots"
    out_dir.mkdir(parents=True, exist_ok=True)
    while True:
        if input(f"[{len(imgs)} good shots] Enter = capture, q = finish: ").strip().lower() == "q":
            break
        frame = cam.grab()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ok, corners = cv2.findChessboardCornersSB(gray, pattern)
        if not ok:
            print("  pattern not found, try again")
            continue
        cv2.imwrite(str(out_dir / f"shot_{len(imgs):02d}.png"), frame)
        objs.append(obj)
        imgs.append(corners)
    cam.close()
    if len(imgs) < 8:
        print("need at least 8 good shots")
        return
    rms, K, dist, _, _ = cv2.calibrateCamera(objs, imgs, (C.CAM_W, C.CAM_H), None, None)
    print(f"RMS reprojection error {rms:.3f} px (good: < 0.5)\nK =\n{np.round(K, 1)}\ndist = {np.round(dist.ravel(), 4)}")
    np.savez(C.INTRINSICS, K=K, dist=dist.ravel(), rms=rms)
    print("saved", C.INTRINSICS)


if __name__ == "__main__":
    main()
