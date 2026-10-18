"""Frame -> what's on each square.

Two sensors, both feeding game/state.py's move inference (python-chess stays the source of truth):

- detect_pieces(): YOLOv8 boxes, bottom-center of each box through the homography to a square.
  Returns {square: (symbol, confidence)}, highest-confidence box per square.
- square_change(): per-square image difference between two frames of the warped board (no model needed).
  Works before YOLO is trained: the move that best explains which squares changed wins.

python -m chess_player.vision.detect_board [image.jpg]   prints detections for one frame (camera if no file)
"""
import sys
import threading

import chess
import cv2
import numpy as np

from chess_player import config as C
from chess_player.geometry import mm_to_square
from chess_player.vision.aruco_homography import locate_board

CLASSES = ["P", "N", "B", "R", "Q", "K", "p", "n", "b", "r", "q", "k"]  # YOLO class ids 0..11

_model = None
_symbols = None  # model class id -> python-chess symbol (None = not a piece, e.g. "board", "Hand")
_lock = threading.Lock()  # webcam and gripper-camera watchers share one model


def available():
    """'own' (data/yolo/best.pt), 'pretrained' (public chess model) or None."""
    if C.YOLO_WEIGHTS.exists():
        return "own"
    if C.YOLO_PRETRAINED.exists():
        return "pretrained"
    return None


def _symbol(name):
    n = name.lower().replace("-", "_").replace(" ", "_")
    if len(n) == 2 and n[0] in "wb" and n[1] in "pnbrqk":  # wP, bK ...
        return n[1].upper() if n[0] == "w" else n[1]
    color = "w" if n.startswith(("white", "w_")) else "b" if n.startswith(("black", "b_")) else None
    kind = next((c for w, c in (("pawn", "p"), ("knight", "n"), ("bishop", "b"), ("rook", "r"), ("queen", "q"),
                                ("king", "k")) if w in n), None)
    if color and kind:
        return kind.upper() if color == "w" else kind
    if n in [c.lower() for c in CLASSES] and len(name) == 1:
        return name
    return None


def model():
    global _model, _symbols
    if _model is None:
        from ultralytics import YOLO

        path = C.YOLO_WEIGHTS if C.YOLO_WEIGHTS.exists() else C.YOLO_PRETRAINED
        _model = YOLO(str(path))
        names = _model.names
        _symbols = {i: _symbol(n) for i, n in names.items()}
    return _model


def predict(frame, conf=0.35, imgsz=None):
    m = model()
    with _lock:
        return m.predict(frame, conf=conf, imgsz=imgsz or C.YOLO_IMGSZ, verbose=False)[0]


def piece_height_mm(view, x1, y1, x2, y2):
    """Height (mm) of the piece in this box: the 3-D point above its base that projects onto the box top."""
    base = np.array([(x1 + x2) / 2, y2 - 0.12 * (y2 - y1)])
    bx, by = view.px_to_mm([base])[0]
    lo, hi = 0.0, 150.0
    for _ in range(30):
        m = (lo + hi) / 2
        if view.project([(bx, by, m)])[0][1] > y1:
            lo = m
        else:
            hi = m
    return (lo + hi) / 2


def _kq_by_height(items):
    """items: [dict(sym, conf, h, sq)]. Re-decide uncertain kings/queens from their measured height, scaled by how
    the other pieces in this frame measure against their known heights (the estimate has a global bias)."""
    others = [it["h"] / C.PIECES[it["sym"].lower()]["height"] for it in items if it["sym"].lower() not in "kq"]
    scale = float(np.median(others)) if len(others) >= 3 else 0.93
    k_h, q_h = C.PIECES["k"]["height"] * scale, C.PIECES["q"]["height"] * scale
    pawn_max = (C.PIECES["p"]["height"] + C.PIECES["n"]["height"]) / 2 * scale
    for it in items:
        if it["sym"].lower() == "p" and it["h"] > pawn_max:  # far too tall for a pawn: YOLO's guess is worth little
            it["note"] = f"YOLO said pawn, but it is {it['h']:.0f} mm tall"
            it["conf"] *= 0.3
        if it["sym"].lower() in "kq" and it["conf"] < C.KQ_TRUST_CONF:
            by_h = "k" if abs(it["h"] - k_h) < abs(it["h"] - q_h) else "q"
            new = by_h.upper() if it["sym"].isupper() else by_h
            if new != it["sym"]:
                it["note"] = f"YOLO said {it['sym']} ({it['conf']:.2f}); height {it['h']:.0f} mm says {new}"
                it["sym"] = new


def detect_pieces(frame, view=None, conf=0.35, boxes_out=None, imgsz=None, kq_height=True):
    """{square: (symbol, conf)}; if boxes_out is a list, also appends
    (symbol, conf, x1, y1, x2, y2, square, height_mm, note)."""
    view = view or locate_board(frame)
    res = predict(frame, conf, imgsz)
    out, items = {}, []
    for (x1, y1, x2, y2), c, k in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.conf.cpu().numpy(),
                                      res.boxes.cls.cpu().numpy().astype(int)):
        # bottom-center, nudged up a little: the box bottom is the front edge of the base, not its center
        base = np.array([(x1 + x2) / 2, y2 - 0.12 * (y2 - y1)])
        sym = _symbols.get(int(k))
        if sym is None:  # "board", "Hand", ...
            continue
        sq = mm_to_square(*view.px_to_mm([base])[0])
        items.append(dict(sym=sym, conf=float(c), box=(float(x1), float(y1), float(x2), float(y2)), sq=sq,
                          h=piece_height_mm(view, x1, y1, x2, y2), note=""))
    if C.KQ_HEIGHT_CHECK and kq_height:  # (off for the gripper camera: too close for good height estimates)
        _kq_by_height(items)
    for it in items:
        if boxes_out is not None:
            boxes_out.append((it["sym"], it["conf"], *it["box"], it["sq"], it["h"], it["note"]))
        sq = it["sq"]
        if sq is not None and (sq not in out or it["conf"] > out[sq][1]):
            out[sq] = (it["sym"], it["conf"])
    return out


def square_patches(frame, view, px_per_mm=2.5, crop=0.3):
    """Center crops of each square in the top-down warp: {square: patch}."""
    w = view.warp(frame, px_per_mm)
    s = C.SQUARE_MM * px_per_mm
    m = int(s * crop)
    out = {}
    for sq in chess.SQUARES:
        f, r = chess.square_file(sq), chess.square_rank(sq)
        x0, y0 = int(f * s), int((7 - r) * s)
        out[sq] = w[y0 + m : y0 + int(s) - m, x0 + m : x0 + int(s) - m]
    return out


def square_change(before, after, view_before=None, view_after=None):
    """{square: change score} between two frames, ~0 = unchanged. Each frame is warped with its own view."""
    vb = view_before or locate_board(before)
    va = view_after or locate_board(after)
    pb, pa = square_patches(before, vb), square_patches(after, va)
    out = {}
    for sq in chess.SQUARES:
        a = cv2.GaussianBlur(pb[sq], (5, 5), 0).astype(np.float32)
        b = cv2.GaussianBlur(pa[sq], (5, 5), 0).astype(np.float32)
        a -= a.mean((0, 1)) * 0.5  # partly normalize brightness: lighting drift shouldn't look like a move
        b -= b.mean((0, 1)) * 0.5
        out[sq] = float(np.abs(a - b).mean())
    return out


def render(dets):
    rows = []
    for r in range(7, -1, -1):
        rows.append(f"{r + 1} " + " ".join(dets.get(chess.square(f, r), (".", 0))[0] for f in range(8)))
    return "\n".join(rows + ["  a b c d e f g h"])


if __name__ == "__main__":
    if len(sys.argv) > 1:
        frame = cv2.imread(sys.argv[1])
    else:
        from chess_player.vision.camera import Camera

        cam = Camera()
        frame = cam.grab()
        cam.close()
    print(render(detect_pieces(frame)))
