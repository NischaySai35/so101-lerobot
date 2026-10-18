"""The gripper ("robot eye") camera as a second judge of which piece is on which square.

At the look pose the gripper camera sees most of the board from the robot's side, the webcam from the human's
side. YOLO runs on both; fuse() combines them per square:

- occupancy and colour come from the webcam (it sees every square; the gripper hides the near ranks);
- the TYPE is trusted when both cameras agree (confidence rises), and is weak when they disagree (so a known piece
  keeps its type instead of flipping to a misreading).

Calibration is automatic (no clicking): with the arm at the look pose, the pieces the webcam sees are matched by
colour to the gripper camera's detections (RANSAC homography), the grid is slid until the light/dark squares line
up, then snapped to the square corners. It is checked every startup and redone if the view no longer matches.

  python -m chess_player.vision.wrist_view calibrate   # needs the pipeline running (uses its camera streams)
"""
import collections
import json
import random
import sys
import threading
import time

import chess
import cv2
import numpy as np

from chess_player import config as C
from chess_player.geometry import square_center_mm

S = C.SQUARE_MM


def _inside(p, size):
    """Point usable in the gripper image: in frame and not on the gripper's own fingers."""
    w, h = size
    x0, x1, y0 = C.WRIST_GRIPPER_MASK
    x, y = p
    return 12 < x < w - 12 and 12 < y < h - 12 and not (x0 * w < x < x1 * w and y > y0 * h)


def view_from_px(px, size):
    from chess_player.vision.aruco_homography import BoardView

    B = C.PLAY_MM
    return BoardView(px, [(0, 0), (B, 0), (B, B), (0, B)], "wrist", size, intrinsics_file=None)


def load_view(size=(C.CAM_W, C.CAM_H)):
    if not C.WRIST_CORNERS.exists():
        return None
    return view_from_px(json.loads(C.WRIST_CORNERS.read_text())["px"], size)


def visible_squares(view, size=(C.CAM_W, C.CAM_H)):
    """Squares whose centre the gripper camera sees (not off-frame, not behind its fingers)."""
    pts = view.mm_to_px([square_center_mm(sq) for sq in chess.SQUARES])
    return {sq for sq, p in zip(chess.SQUARES, pts) if _inside(p, size)}


# ---------------------------------------------------------------- calibration
def _bases(frame):
    """[(x, y, is_white)] base points of the pieces YOLO finds (one per piece)."""
    from chess_player.vision import detect_board as D

    res = D.predict(frame, conf=0.3, imgsz=max(frame.shape[:2]))
    out = []
    for (x1, y1, x2, y2), c, k in sorted(zip(res.boxes.xyxy.cpu().numpy(), res.boxes.conf.cpu().numpy(),
                                             res.boxes.cls.cpu().numpy().astype(int)), key=lambda r: -r[1]):
        s = D._symbols.get(int(k))
        if not s:
            continue
        b = ((x1 + x2) / 2, y2 - 0.12 * (y2 - y1))
        if all(np.hypot(b[0] - o[0], b[1] - o[1]) > 15 for o in out):  # same piece, two class guesses
            out.append((float(b[0]), float(b[1]), s.isupper()))
    return out


def _fit_by_pieces(bases, known, iters=30000):
    """RANSAC: board mm -> px from colour-matched pieces. known = {square: is_white}."""
    P = np.float32([b[:2] for b in bases])
    sqs = list(known)
    M = np.float32([square_center_mm(sq) for sq in sqs])
    pairs = [(i, j) for i in range(len(bases)) for j in range(len(sqs)) if bases[i][2] == known[sqs[j]]]
    if len(bases) < 5 or len(pairs) < 4:
        raise RuntimeError(f"the gripper camera sees only {len(bases)} pieces; need 5+")
    rng = random.Random(0)
    best = (0, 1e9, None)
    probe = np.float32([[(S / 2, 4 * S), (7.5 * S, 4 * S), (4 * S, 0.5 * S), (4 * S, 7.5 * S)]])
    robot_white = _robot_white()
    for _ in range(iters):
        sm = rng.sample(pairs, 4)
        if len({a for a, _ in sm}) < 4 or len({b for _, b in sm}) < 4:
            continue
        try:
            H = cv2.getPerspectiveTransform(M[[b for _, b in sm]], P[[a for a, _ in sm]])
            Hi = np.linalg.inv(H)
        except (cv2.error, np.linalg.LinAlgError):
            continue
        q = cv2.perspectiveTransform(probe, H)[0]
        # seen from the robot's side: (robot white) a-file on the left, rank 8 at the top
        left_right = q[0, 0] < q[1, 0] if robot_white else q[0, 0] > q[1, 0]
        far_top = q[3, 1] < q[2, 1] if robot_white else q[3, 1] > q[2, 1]
        if not (left_right and far_top and 150 < np.linalg.norm(q[1] - q[0]) < 2500):
            continue
        mm = cv2.perspectiveTransform(P[None], Hi)[0]
        inl, err = 0, 0.0
        for i, (x, y) in enumerate(mm):
            d = min(np.hypot(x - M[j, 0], y - M[j, 1]) for j in range(len(sqs)) if known[sqs[j]] == bases[i][2])
            if d < 0.4 * S:
                inl, err = inl + 1, err + d
        if inl > best[0] or (inl == best[0] and err < best[1]):
            best = (inl, err, H)
    if best[2] is None or best[0] < max(5, 0.6 * len(bases)):
        raise RuntimeError(f"couldn't match the two cameras' pieces ({best[0]} of {len(bases)})")
    return best[2], best[0]


def _robot_white():
    from chess_player.vision.ext_camera import robot_color_from_corners

    return not C.EXT_CORNERS.exists() or robot_color_from_corners() == "white"


def checker_score(gray, H, occupied, size):
    """Light minus dark brightness over the visible empty squares (high = grid sits on the real squares)."""
    off = np.linspace(-0.25, 0.25, 5) * S
    light, dark = [], []
    for sq in chess.SQUARES:
        if sq in occupied:
            continue
        cx, cy = square_center_mm(sq)
        pts = np.float32([(cx + a, cy + b) for a in off for b in off])
        px = cv2.perspectiveTransform(pts[None], H)[0]
        if not all(_inside(p, size) for p in px[[0, 4, 20, 24]]):
            continue
        v = float(cv2.remap(gray, px[:, 0].reshape(1, -1), px[:, 1].reshape(1, -1), cv2.INTER_LINEAR).mean())
        (dark if (chess.square_file(sq) + chess.square_rank(sq)) % 2 == 0 else light).append(v)
    if len(light) < 4 or len(dark) < 4:
        return -1e9
    return float(np.median(light) - np.median(dark))


def _slide(gray, H0, occupied, size):
    """Small shift/rotation/scale of the board frame that best lines the checker pattern up."""
    c = 4 * S

    def moved(dx, dy, rot, sc):
        r = np.deg2rad(rot)
        A = np.array([[sc * np.cos(r), -sc * np.sin(r), 0], [sc * np.sin(r), sc * np.cos(r), 0], [0, 0, 1]])
        return H0 @ np.array([[1, 0, c + dx], [0, 1, c + dy], [0, 0, 1]]) @ A @ np.array([[1, 0, -c], [0, 1, -c], [0, 0, 1]])

    best = (checker_score(gray, H0, occupied, size), (0, 0, 0, 1.0))
    for step, span, rots, scs in ((4.0, 0.6 * S, (-3, 0, 3), (0.95, 1.0, 1.05)),
                                  (1.0, 4.0, (-1.5, 0, 1.5), (0.98, 1.0, 1.02))):
        dx0, dy0, r0, s0 = best[1]
        for dx in np.arange(dx0 - span, dx0 + span + 1e-6, step):
            for dy in np.arange(dy0 - span, dy0 + span + 1e-6, step):
                for rot in (r0 + r for r in rots):
                    for sc in (s0 * s for s in scs):
                        sc_ = checker_score(gray, moved(dx, dy, rot, sc), occupied, size)
                        if sc_ > best[0]:
                            best = (sc_, (dx, dy, rot, sc))
    return moved(*best[1]), best[0]


def _snap(gray_u8, H, size):
    """Snap the grid lines' crossings to the real square corners (sub-pixel), refit."""
    grid = np.float32([(i * S, j * S) for i in range(9) for j in range(9)])
    used = 0
    for _ in range(3):
        guess = cv2.perspectiveTransform(grid[None], H)[0]
        keep = [k for k, p in enumerate(guess) if _inside(p, size)]
        if len(keep) < 10:
            break
        G = guess[keep]
        ref = cv2.cornerSubPix(gray_u8, G.copy().reshape(-1, 1, 2), (7, 7), (-1, -1),
                               (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.01)).reshape(-1, 2)
        ok = np.linalg.norm(ref - G, axis=1) < 6
        if ok.sum() < 10:
            break
        H2, inl = cv2.findHomography(grid[keep][ok], ref[ok], cv2.RANSAC, 2.0)
        if H2 is None or inl.sum() < 10:
            break
        H, used = H2, int(inl.sum())
    return H, used


def calibrate(frame, known):
    """frame: gripper image at the look pose; known: {square: symbol} the webcam reads now.
    Saves data/wrist_corners.json and returns a report dict."""
    size = (frame.shape[1], frame.shape[0])
    gray_u8 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = gray_u8.astype(np.float32)
    occupied = set(known)
    bases = _bases(frame)
    H, matched = _fit_by_pieces(bases, {sq: s.isupper() for sq, s in known.items()})
    H, score = _slide(gray, H, occupied, size)
    H, corners_used = _snap(gray_u8, H, size)
    score = checker_score(gray, H, occupied, size)
    if score < C.WRIST_MIN_SCORE:
        raise RuntimeError(f"the grid doesn't line up with the squares (contrast {score:.0f})")
    B = C.PLAY_MM
    px = cv2.perspectiveTransform(np.float32([[(0, 0), (B, 0), (B, B), (0, B)]]), H)[0].tolist()
    rep = {"px": px, "pieces_matched": matched, "of": len(bases), "corners_used": corners_used,
           "contrast": round(score, 1), "size": list(size), "saved": time.strftime("%Y-%m-%d %H:%M:%S")}
    C.WRIST_CORNERS.write_text(json.dumps(rep, indent=1))
    from chess_player.vision.aruco_homography import draw_overlay

    cv2.imwrite(str(C.DATA / "wrist_corners_check.jpg"), draw_overlay(frame, view_from_px(px, size)))
    return rep


def still_fits(frame, occupied):
    """Does the saved calibration still sit on the squares in this frame?"""
    v = load_view((frame.shape[1], frame.shape[0]))
    if v is None:
        return False, None
    size = (frame.shape[1], frame.shape[0])
    s = checker_score(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32), v.H, occupied, size)
    return s >= C.WRIST_MIN_SCORE, s


# ---------------------------------------------------------------- live readings + fusion
class WristWatcher:
    """YOLO on the gripper camera while `active()` (arm resting at the look pose). Keeps, per frame, every piece
    it sees with its base point on the board (mm); match() pairs them with the webcam's pieces."""

    def __init__(self, get_frame, active, period=C.WRIST_WATCH_PERIOD_S):
        from chess_player.vision import detect_board as D

        self.get_frame, self.active, self.period, self.D = get_frame, active, period, D
        self.history = collections.deque(maxlen=6)
        self.boxes = (0.0, [])
        self.view = None
        self.visible = set()
        self._lock = threading.Lock()
        self._stop = False
        self.reload()
        threading.Thread(target=self._loop, daemon=True).start()

    def reload(self):
        self.view = load_view()
        self.visible = visible_squares(self.view) if self.view is not None else set()
        self.clear()

    def _loop(self):
        while not self._stop:
            t0 = time.time()
            if self.view is None or not self.active():
                self.clear()
            else:
                frame = self.get_frame()
                if frame is not None:
                    try:
                        boxes = []
                        self.D.detect_pieces(frame, self.view, conf=0.3, boxes_out=boxes,
                                             imgsz=max(frame.shape[:2]), kq_height=False)
                        size = (frame.shape[1], frame.shape[0])
                        pieces = []
                        for sym, c, x1, y1, x2, y2, *_ in sorted(boxes, key=lambda b: -b[1]):
                            base = ((x1 + x2) / 2, y2 - 0.12 * (y2 - y1))
                            if not _inside(base, size):
                                continue
                            mx, my = self.view.px_to_mm([base])[0]
                            if all(np.hypot(mx - p[2], my - p[3]) > 0.4 * S for p in pieces):  # one guess per piece
                                pieces.append((sym, c, float(mx), float(my)))
                        with self._lock:
                            self.history.append((t0, pieces))
                            self.boxes = (t0, boxes)
                    except Exception as e:
                        print(f"[wrist watcher] {type(e).__name__}: {e}")
            time.sleep(max(0.0, self.period - (time.time() - t0)))

    def clear(self):
        with self._lock:
            self.history.clear()
            self.boxes = (0.0, [])

    def match(self, ext, n=2):
        """{square: (symbol, conf)} for the webcam's pieces `ext` that the gripper camera also sees: in each of the
        last n frames, the nearest gripper-camera piece of the same colour within 3/4 of a square (tall pieces lean
        in this close view, so their base point lands off-centre), and the same type in all n frames.
        None = no recent readings (arm moving / not calibrated)."""
        with self._lock:
            if len(self.history) < n:
                return None
            frames = [h[1] for h in list(self.history)[-n:]]
        per_frame = []
        for pieces in frames:
            pairs = sorted((np.hypot(px - square_center_mm(sq)[0], py - square_center_mm(sq)[1]), sq, i)
                           for sq, (sym, _) in ext.items() if sq in self.visible
                           for i, (ps, _, px, py) in enumerate(pieces) if ps.isupper() == sym.isupper())
            got, used = {}, set()
            for d, sq, i in pairs:  # greedy, nearest first, one-to-one
                if d < 0.75 * S and sq not in got and i not in used:
                    got[sq], _ = pieces[i][:2], used.add(i)
            per_frame.append(got)
        out = {}
        for sq in per_frame[0]:
            syms = [f.get(sq, (None,))[0] for f in per_frame]
            if all(s == syms[0] for s in syms):
                out[sq] = (syms[0], float(np.mean([f[sq][1] for f in per_frame])))
        return out

    def extras(self, ext, n=3):
        """Pieces the gripper camera sees where the webcam sees none (e.g. pawns hidden behind the back rank from
        the webcam's side): {square: (symbol, conf)}. Only if the same piece sits well inside that square in the last
        n frames, and it isn't just a webcam piece leaning into the next square in this close view."""
        from chess_player.geometry import mm_to_square

        with self._lock:
            if len(self.history) < n:
                return {}
            frames = [h[1] for h in list(self.history)[-n:]]
        per = []
        for pieces in frames:
            got = {}
            for sym, c, px, py in pieces:
                sq = mm_to_square(px, py)
                if sq is None or sq in ext or sq not in self.visible or c < C.EYE_ADD_MIN_CONF:
                    continue
                cx, cy = square_center_mm(sq)
                if np.hypot(px - cx, py - cy) > 0.45 * S:  # off-centre: probably a leaning neighbour
                    continue
                if any(np.hypot(px - square_center_mm(e)[0], py - square_center_mm(e)[1]) < 0.75 * S
                       and s.isupper() == sym.isupper() for e, (s, _) in ext.items()):
                    continue
                if sq not in got or c > got[sq][1]:
                    got[sq] = (sym, c)
            per.append(got)
        return {sq: (v[0], float(np.mean([f[sq][1] for f in per])) * 0.8) for sq, v in per[0].items()
                if all(f.get(sq, (None,))[0] == v[0] for f in per)}

    def close(self):
        self._stop = True


def fuse(ext, wrist, visible):
    """Combine webcam `ext` and gripper `wrist` readings ({square: (symbol, conf)}).

    Returns (seen, info, votes): seen = {square: (symbol, conf)} for merge_seen, where conf is high only when the
    cameras agree; info = {square name: {"web": [sym, conf], "eye": [sym, conf] | None, "agree": bool | None}};
    votes = {square: [webcam symbol, gripper symbol]} where both see a piece."""
    seen, info, votes = {}, {}, {}
    for sq, (sym, c) in ext.items():
        w = wrist.get(sq) if wrist else None
        if w is None or sq not in visible:
            seen[sq] = (sym, c)  # only the webcam sees it
            agree = None
        elif w[0] == sym:
            seen[sq] = (sym, 1 - (1 - c) * (1 - w[1]))  # both say the same: more sure
            agree = True
        elif w[0].isupper() == sym.isupper():
            win = (sym, c) if c >= w[1] else w
            seen[sq] = (win[0], 0.5 * abs(c - w[1]))  # same colour, different type: weak
            agree = False
        else:
            seen[sq] = (sym, 0.5 * c)  # colours disagree: the webcam decides occupancy, but weakly
            agree = False
        if agree is not None:
            votes[sq] = [sym, w[0]]
        info[chess.square_name(sq)] = {"web": [sym, round(c, 2)], "eye": [w[0], round(w[1], 2)] if w else None,
                                       "agree": agree}
    return seen, info, votes


def _grab_mjpeg(url):
    import urllib.request

    r = urllib.request.urlopen(url, timeout=5)
    buf = b""
    while True:
        buf += r.read(4096)
        a = buf.find(b"\xff\xd8")
        b = buf.find(b"\xff\xd9", a)
        if a >= 0 and b > 0:
            return cv2.imdecode(np.frombuffer(buf[a:b + 2], np.uint8), 1)


def main():
    if sys.argv[1:] != ["calibrate"]:
        print(__doc__)
        return
    from chess_player.vision import detect_board as D
    from chess_player.vision.ext_camera import view

    port = 8770
    wrist = _grab_mjpeg(f"http://localhost:{port}/stream/wrist")
    ext = _grab_mjpeg(f"http://localhost:{port}/stream/ext_raw")
    known = {sq: s for sq, (s, _) in D.detect_pieces(ext, view(ext)).items()}
    rep = calibrate(wrist, known)
    print(f"saved {C.WRIST_CORNERS}: {rep}\ncheck the grid in {C.DATA / 'wrist_corners_check.jpg'}")


if __name__ == "__main__":
    main()
