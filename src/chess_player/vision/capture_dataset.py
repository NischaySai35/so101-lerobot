"""Capture YOLO training images at the look pose and label them automatically from a known position.

  python -m chess_player.vision.capture_dataset --name pos01 --fen start --shots 10
  python -m chess_player.vision.capture_dataset --name pos02 --fen "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3"
  python -m chess_player.vision.capture_dataset --relabel       # recompute all labels (after changing PIECES sizes)
  python -m chess_player.vision.capture_dataset --presets 1-21  # guided: shows each position from data/
                                                                # dataset_positions.json, you set it up, Enter

Set up the position on the board, run the command. With the arm connected it goes to the look pose and adds a
small random joint jitter per shot (lighting/pose variety); --no-arm uses the camera as it is.
Each box is the projection of the piece's 3-D bounding cylinder (base diameter x height from config.PIECES)
through the camera pose found from the ArUco markers. Check data/dataset/previews/ before training.
"""
import argparse
import json
import random
import time

import chess
import cv2
import numpy as np

from chess_player import config as C
from chess_player.geometry import square_center_mm
from chess_player.vision.aruco_homography import BoardView, locate_board
from chess_player.vision.detect_board import CLASSES

DS = C.DATA / "dataset"


def piece_box(view: BoardView, sq, symbol):
    spec = C.PIECES[symbol.lower()]
    x, y = square_center_mm(sq)
    r, h = spec["diameter"] / 2, spec["height"]
    ang = np.linspace(0, 2 * np.pi, 12, endpoint=False)
    ring = np.c_[x + r * np.cos(ang), y + r * np.sin(ang)]
    pts = np.r_[np.c_[ring, np.zeros(12)], np.c_[ring * 0.7 + np.array([x, y]) * 0.3, np.full(12, h)]]
    px = view.project(pts)
    px += view.mm_to_px([(x, y)])[0] - view.project([(x, y, 0.0)])[0]  # anchor the base to the exact homography
    x1, y1 = px.min(0)
    x2, y2 = px.max(0)
    return x1, y1, x2, y2


def labels_for(view, board):
    W, H = view.size
    lines, boxes = [], []
    for sq, p in board.piece_map().items():
        x1, y1, x2, y2 = piece_box(view, sq, p.symbol())
        cx1, cy1, cx2, cy2 = max(0, x1), max(0, y1), min(W, x2), min(H, y2)
        area, vis = (x2 - x1) * (y2 - y1), max(0, cx2 - cx1) * max(0, cy2 - cy1)
        if area <= 0 or vis / area < 0.4:  # mostly out of frame: leave it unlabeled
            continue
        k = CLASSES.index(p.symbol())
        lines.append(f"{k} {(cx1 + cx2) / 2 / W:.6f} {(cy1 + cy2) / 2 / H:.6f} "
                     f"{(cx2 - cx1) / W:.6f} {(cy2 - cy1) / H:.6f}")
        boxes.append((p.symbol(), int(cx1), int(cy1), int(cx2), int(cy2)))
    return lines, boxes


def write_sample(stem, frame, board, camera="ext"):
    view = locate_board(frame, C.EXT_CORNERS if camera == "ext" else None)
    lines, boxes = labels_for(view, board)
    for d in ("images", "labels", "previews", "meta"):
        (DS / d).mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(DS / "images" / f"{stem}.jpg"), frame)
    (DS / "labels" / f"{stem}.txt").write_text("\n".join(lines) + "\n")
    prev = frame.copy()
    for s, x1, y1, x2, y2 in boxes:
        col = (255, 255, 255) if s.isupper() else (40, 40, 40)
        cv2.rectangle(prev, (x1, y1), (x2, y2), col, 1)
        cv2.putText(prev, s, (x1, y1 + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
    cv2.imwrite(str(DS / "previews" / f"{stem}.jpg"), prev)
    return len(lines), view.source


def guided(a):
    """Show each preset position as seen from the human's seat, wait for Enter, capture shots with the webcam."""
    from chess_player.vision.ext_camera import ExtCamera, robot_color_from_corners

    presets = {d["id"]: d for d in json.loads((C.DATA / "dataset_positions.json").read_text())}
    ids = []
    for part in a.presets.split(","):
        lo, _, hi = part.partition("-")
        ids += list(range(int(lo), int(hi or lo) + 1))
    human_side = chess.WHITE if robot_color_from_corners() == "black" else chess.BLACK
    cam = ExtCamera()
    try:
        for i in ids:
            d = presets[i]
            board = chess.Board(d["fen"])
            print(f"\n=== position {i}/{max(presets)}: {d['name']}  ({len(board.piece_map())} pieces; "
                  f"capital letters / white symbols = white)")
            print(board.unicode(borders=True, orientation=human_side))
            ranks = range(8) if human_side == chess.BLACK else range(7, -1, -1)
            files = range(7, -1, -1) if human_side == chess.BLACK else range(8)
            for r in ranks:
                print(f"  {r + 1}  " + " ".join((board.piece_at(chess.square(f, r)) or ".").__str__() for f in files))
            print("     " + " ".join("abcdefgh"[f] for f in files))
            print("(as you see it from your seat: bottom row = nearest you; CAPITALS = white, small = black, "
                  "K king, Q queen, R rook, B bishop, N knight, P pawn)")
            ans = input("Set it up, hands away from the board, then Enter (s = skip, q = quit): ").strip().lower()
            if ans == "q":
                break
            if ans == "s":
                continue
            name = f"pos{i:02d}"
            for k in range(a.shots):
                stem = f"{name}_{k:02d}"
                time.sleep(1.0)
                n, _ = write_sample(stem, cam.grab(), board, "ext")
                (DS / "meta" / f"{stem}.json").write_text(json.dumps({"fen": board.fen(), "position": name, "camera": "ext"}))
                print(f"  {stem}: {n} boxes")
            print(f"  check one: {DS / 'previews' / (name + '_00.jpg')}")
    finally:
        cam.close()
    print(f"\ndone. {len(list((DS / 'meta').glob('*.json')))} images in {DS}. Next: python -m chess_player.vision.train_yolo")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", help="position name, e.g. pos07 (train/val split is by this)")
    ap.add_argument("--fen", default="start")
    ap.add_argument("--shots", type=int, default=10)
    ap.add_argument("--jitter-deg", type=float, default=2.0)
    ap.add_argument("--no-arm", action="store_true")
    ap.add_argument("--relabel", action="store_true")
    ap.add_argument("--presets", help="guided session over data/dataset_positions.json, e.g. 1-21 or 3,5,8")
    ap.add_argument("--camera", choices=["ext", "wrist"], default="ext" if C.EXT_CORNERS.exists() else "wrist",
                    help="ext = fixed third-person camera (default once its corners are clicked; arm not needed)")
    a = ap.parse_args()

    if a.presets:
        return guided(a)
    if a.relabel:
        for meta in sorted((DS / "meta").glob("*.json")):
            m = json.loads(meta.read_text())
            n, _ = write_sample(meta.stem, cv2.imread(str(DS / "images" / f"{meta.stem}.jpg")), chess.Board(m["fen"]),
                                m.get("camera", "wrist"))
            print(meta.stem, n, "boxes")
        return

    board = chess.Board() if a.fen == "start" else chess.Board(a.fen)
    print(board, "\n")
    if a.camera == "ext":
        from chess_player.vision.ext_camera import ExtCamera

        cam = ExtCamera()
        try:  # fixed camera: shots a second apart; move the lamp / your shadow between them for variety
            for i in range(a.shots):
                stem = f"{a.name}_{i:02d}"
                time.sleep(1.0)
                n, src = write_sample(stem, cam.grab(), board, "ext")
                (DS / "meta" / f"{stem}.json").write_text(json.dumps({"fen": board.fen(), "position": a.name, "camera": "ext"}))
                print(f"  {stem}: {n} boxes")
        finally:
            cam.close()
        print(f"previews in {DS / 'previews'}")
        return
    from chess_player.motion.arm import Arm, look_pose
    from chess_player.vision.camera import Camera

    arm = None if a.no_arm else Arm()
    cam = Camera()
    try:
        look = {**look_pose(), "gripper": C.GRIPPER_LOOK}
        if arm:
            arm.move_to(look, 2.0)
        for i in range(a.shots):
            if arm and i:
                jit = {j: look[j] + random.uniform(-a.jitter_deg, a.jitter_deg) for j in C.ARM_JOINTS}
                arm.move_to(jit, 0.8)
            stem = f"{a.name}_{i:02d}"
            try:
                n, src = write_sample(stem, cam.grab(settle_s=0.6), board, "wrist")
            except RuntimeError as e:
                print(f"  {stem}: skipped ({e})")
                continue
            (DS / "meta" / f"{stem}.json").write_text(json.dumps({"fen": board.fen(), "position": a.name, "camera": "wrist"}))
            print(f"  {stem}: {n} boxes ({src})")
        if arm:
            arm.move_to(look, 1.0)
    finally:
        cam.close()
        if arm:
            arm.close()
    print(f"previews in {DS / 'previews'}")


if __name__ == "__main__":
    main()
