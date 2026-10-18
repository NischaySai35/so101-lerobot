"""Fixed third-person camera (MJPEG over HTTP), plus a board watcher that runs YOLO every WATCH_PERIOD_S.

  python -m chess_player.vision.ext_camera corners   # once (and whenever the camera or board moves):
                                                     # click the 4 playing-area corners a1, h1, h8, a8
  python -m chess_player.vision.ext_camera auto      # same, automatically: needs the pieces in the START position
  python -m chess_player.vision.ext_camera watch     # live view: grid + YOLO pieces per square, every second

Corners are the outer corners of the corner SQUARES (inside the thin light line), the same points the arm
touches in calib/board_to_robot.py. The corner click checks the colours: a1 and h8 must be dark squares,
h1 and a8 light ("light on the right" for both players).
"""
import collections
import json
import sys
import threading
import time

import chess
import cv2
import numpy as np

from chess_player import config as C
from chess_player.vision.aruco_homography import corners_view, draw_overlay


class WhiteBalance:
    """Make the webcam look like the robot (gripper) camera: per-channel gains so the brightest parts of the scene
    (light squares, white pieces) get the robot camera's white tone (EXT_WB_TINT, measured from it) and brightness
    (EXT_WB_WHITE_LEVEL). Removes the strong yellow/orange cast of warm room light. Gains are re-estimated every
    few frames, smoothed (no flicker), and applied with lookup tables."""

    def __init__(self, every=10, smooth=0.2, max_gain=2.8, top=0.08):
        self.every, self.smooth, self.max_gain, self.top = every, smooth, max_gain, top
        self.tint = np.asarray(C.EXT_WB_TINT, float)
        self.gains = np.ones(3)
        self.n = 0
        self._lut = None

    def __call__(self, frame):
        if self.n % self.every == 0:
            small = frame[::8, ::8].reshape(-1, 3).astype(np.float32)
            lum = small.mean(1)
            ok = lum < 250  # ignore blown-out pixels
            if ok.sum() > 100:
                ref = small[ok & (lum >= np.quantile(lum[ok], 1 - self.top))].mean(0)
                g = self.tint * ref.mean() / np.maximum(ref, 1.0)
                g *= C.EXT_WB_WHITE_LEVEL / max(float((ref * g).max()), 1.0)
                target = np.clip(g, 1 / self.max_gain, self.max_gain)
                self.gains = self.gains * (1 - self.smooth) + target * self.smooth if self.n else target
                x = np.arange(256, dtype=np.float32)
                self._lut = [np.clip(x * gi, 0, 255).astype(np.uint8) for gi in self.gains]
        self.n += 1
        if self._lut is None:
            return frame
        b, g, r = cv2.split(frame)
        return cv2.merge([cv2.LUT(b, self._lut[0]), cv2.LUT(g, self._lut[1]), cv2.LUT(r, self._lut[2])])


def diagnose(url):
    """Why a camera gave no frames, in words: unplugged, busy, phone unreachable."""
    import urllib.request

    if str(url).startswith("/dev"):
        if not C.Path(url).exists():
            return f"the webcam {url} isn't plugged in (not found)"
        return (f"the webcam {url} is busy: another program has it open (only one can), and the pipeline window "
                f"isn't running to share it")

    base = url.rsplit("/", 1)[0] + "/"
    try:
        urllib.request.urlopen(base, timeout=3).read(64)
    except Exception:
        return (f"the webcam phone at {base} doesn't answer: wake the phone, open its camera-stream app, and check "
                f"it's on the same Wi-Fi as this laptop")
    return (f"the webcam at {base} is up but sends no frames to this program: it serves ONE viewer at a time. "
            f"Close the pipeline window (Ctrl+C in its terminal) and any browser tab showing {base}")


class ExtCamera:
    """Reads the stream in a background thread and keeps the latest frame (the stream can't be polled).
    Frames are white-balanced (EXT_WHITE_BALANCE) for everything downstream: display, dataset, YOLO."""

    def __init__(self, url=C.EXT_CAMERA_URL, timeout_s=5.0, white_balance=None):
        self.url = url
        self.source = "?"
        self.wb = WhiteBalance() if (C.EXT_WHITE_BALANCE if white_balance is None else white_balance) else None
        self._frame, self._t = None, 0.0
        self._stop = False
        self._lock = threading.Lock()
        self._th = threading.Thread(target=self._loop, daemon=True)
        self._th.start()
        t0 = time.time()
        while self._frame is None and time.time() - t0 < timeout_s:
            time.sleep(0.05)
        if self._frame is None:
            self.close()
            raise RuntimeError(diagnose(url))

    def _open(self):
        """(capture, already_white_balanced). A USB device is opened as MJPG at EXT_CAMERA_SIZE; if another program
        has it, fall back to the pipeline's shared copy of the stream (already white-balanced)."""
        if str(self.url).startswith("/dev"):
            cap = cv2.VideoCapture(str(self.url), cv2.CAP_V4L2)
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, C.EXT_CAMERA_SIZE[0])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, C.EXT_CAMERA_SIZE[1])
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if cap.isOpened() and cap.read()[0]:
                self.source = "device"
                return cap, False
            cap.release()
            cap = cv2.VideoCapture(C.EXT_SHARED_URL)
            self.source = "shared"
            return cap, True
        self.source = "url"
        return cv2.VideoCapture(self.url), False

    def _loop(self):
        while not self._stop:
            cap, balanced = self._open()
            while not self._stop and cap.isOpened():
                ok, f = cap.read()
                if not ok:
                    break
                if self.wb is not None and not balanced:
                    f = self.wb(f)
                with self._lock:
                    self._frame, self._t = f, time.time()
            cap.release()
            if not self._stop:
                time.sleep(1.0)  # stream dropped: reconnect

    def grab(self, max_age_s=1.0):
        with self._lock:
            f, t = self._frame, self._t
        if f is None or time.time() - t > max_age_s:
            raise RuntimeError("third-person camera stream is stale")
        return f.copy()

    def close(self):
        self._stop = True


def view(frame):
    return corners_view(frame, C.EXT_CORNERS)


def robot_color_from_corners(path=C.EXT_CORNERS):
    """The webcam sits on the human's side, so whichever back rank is nearer the bottom of its image is the
    human's. Clicked order is a1, h1, h8, a8: rank 1 lower in the image -> human white, robot black."""
    px = json.loads(path.read_text())["px"]
    rank1_y = (px[0][1] + px[1][1]) / 2
    rank8_y = (px[2][1] + px[3][1]) / 2
    return "black" if rank1_y > rank8_y else "white"


def colour_check(frame, v):
    """(ok, message). Dark squares are (file + rank) even: a1, c1, ..., h8."""
    patches = {}
    w = v.warp(frame, 2.5)
    s = C.SQUARE_MM * 2.5
    for sq in chess.SQUARES:
        f, r = chess.square_file(sq), chess.square_rank(sq)
        x0, y0 = int(f * s + s * 0.3), int((7 - r) * s + s * 0.3)
        patches[sq] = float(cv2.cvtColor(w[y0 : y0 + int(s * 0.4), x0 : x0 + int(s * 0.4)], cv2.COLOR_BGR2GRAY).mean())
    dark = np.median([v for sq, v in patches.items() if (chess.square_file(sq) + chess.square_rank(sq)) % 2 == 0])
    light = np.median([v for sq, v in patches.items() if (chess.square_file(sq) + chess.square_rank(sq)) % 2 == 1])
    if light - dark > 15:
        return True, f"colours OK (dark squares {dark:.0f}, light {light:.0f})"
    if dark - light > 15:
        return False, ("a1 came out LIGHT: the board is a quarter turn off. Turn the board 90 degrees so each "
                       "player has a light square in the right-hand corner, then click again.")
    return False, f"can't tell dark from light squares ({dark:.0f} vs {light:.0f}): check the clicks"


def auto_corners(frame, robot_color=None):
    """Find the 4 playing-area corners (image px of a1, h1, h8, a8) from YOLO detections, with the pieces in the
    START position: rank 1/2 on the robot's side (far from the webcam) when the robot plays white.

    The bases of the front three rows the camera sees un-occluded (ranks 2, 7, 8 for robot = white) give 24 known
    square centres; a RANSAC homography maps them, and the board corners follow. Returns (px, report) or raises.
    """
    from chess_player.geometry import square_center_mm
    from chess_player.vision import detect_board as D

    robot_color = robot_color or (robot_color_from_corners() if C.EXT_CORNERS.exists() else "white")
    res = D.predict(frame, conf=0.35)
    pts = []
    for (x1, y1, x2, y2), k in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.cls.cpu().numpy().astype(int)):
        if D._symbols.get(int(k)):
            pts.append(((x1 + x2) / 2, y2 - 0.12 * (y2 - y1)))
    if len(pts) < 24:
        raise RuntimeError(f"only {len(pts)} pieces seen; set up the full start position, nothing in front of the board")
    pts = np.array(pts)
    # rows of pieces by image height: far (top) to near (bottom); keep the 4 biggest clusters of 6+ pieces
    order = np.argsort(pts[:, 1])
    rows, cur = [], [order[0]]
    gaps = np.diff(pts[order, 1])
    thr = max(8.0, np.median(gaps) * 4)
    for i, g in zip(order[1:], gaps):
        if g > thr:
            rows.append(cur)
            cur = []
        cur.append(i)
    rows.append(cur)
    rows = [r for r in rows if len(r) >= 6]
    if len(rows) < 4:
        raise RuntimeError(f"couldn't separate the 4 rows of pieces (found {len(rows)}); is it the start position?")
    rows = sorted(rows, key=lambda r: np.mean(pts[r, 1]))[:4]
    # far-to-near ranks and left-to-right files as seen from the webcam (the human's side)
    ranks = [0, 1, 6, 7] if robot_color == "white" else [7, 6, 1, 0]  # 0-based rank index
    img, brd = [], []
    for r_i, rank in zip(range(4), ranks):
        row = sorted(rows[r_i], key=lambda i: pts[i, 0])[:8]
        if len(row) < 7 or r_i == 0:  # the far back rank hides behind its pawns: skip it
            continue
        files = list(range(8))[::-1] if robot_color == "white" else list(range(8))  # human's left = h when robot is white
        for i, f in zip(row, files if len(row) == 8 else files[:len(row)]):
            img.append(pts[i])
            brd.append(square_center_mm(chess.square(f, rank)))
    H, inl = cv2.findHomography(np.float32(brd), np.float32(img), cv2.RANSAC, 12.0)
    if H is None:
        raise RuntimeError("couldn't fit the board")
    B = C.PLAY_MM
    corners = cv2.perspectiveTransform(np.float32([[(0, 0), (B, 0), (B, B), (0, B)]]), H)[0]
    err = np.linalg.norm(cv2.perspectiveTransform(np.float32([brd]), H)[0] - np.float32(img), axis=1)
    return [list(map(float, c)) for c in corners], {"points": len(img), "inliers": int(inl.sum()),
                                                    "median_err_px": float(np.median(err))}


def refine_corners(frame, px, iters=2):
    """Snap a rough board fit to the board itself: the inner square corners of the empty middle (rank lines 2|3 ..
    6|7, file lines 1..7) are found to sub-pixel accuracy and the homography refitted. Returns (px, n_used)."""
    B, S = C.PLAY_MM, C.SQUARE_MM
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    grid = np.float32([(i * S, j * S) for i in range(1, 8) for j in range(2, 7)])
    used = 0
    for _ in range(iters):
        H = cv2.getPerspectiveTransform(np.float32([(0, 0), (B, 0), (B, B), (0, B)]), np.float32(px))
        guess = cv2.perspectiveTransform(grid[None], H)[0]
        win = int(max(4, min(12, np.median(np.linalg.norm(np.diff(guess[::5], axis=0), axis=1)) / 5)))
        ref = cv2.cornerSubPix(gray, guess.copy().reshape(-1, 1, 2), (win, win), (-1, -1),
                               (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.01)).reshape(-1, 2)
        ok = np.linalg.norm(ref - guess, axis=1) < win * 0.9  # moved to a real corner, not wandered off
        if ok.sum() < 12:
            break
        H2, inl = cv2.findHomography(grid[ok], ref[ok], cv2.RANSAC, 3.0)
        if H2 is None:
            break
        used = int(inl.sum())
        px = cv2.perspectiveTransform(np.float32([[(0, 0), (B, 0), (B, B), (0, B)]]), H2)[0].tolist()
    return px, used


def auto_corners_cmd():
    cam = ExtCamera()
    time.sleep(1.5)
    frame = cam.grab()
    cam.close()
    px, rep = auto_corners(frame)
    px, rep["refined_with_corners"] = refine_corners(frame, px)
    C.EXT_CORNERS.parent.mkdir(parents=True, exist_ok=True)
    old = C.EXT_CORNERS.read_text() if C.EXT_CORNERS.exists() else None
    C.EXT_CORNERS.write_text(json.dumps({"px": px, "url": str(C.EXT_CAMERA_URL), "auto": rep}, indent=2))
    v = view(frame)
    ok, msg = colour_check(frame, v)
    cv2.imwrite(str(C.DATA / "ext_corners_check.jpg"), draw_overlay(frame, v))
    print(f"fitted from {rep['points']} pieces ({rep['inliers']} agree), then snapped to {rep['refined_with_corners']} "
          f"square corners; {msg}")
    print(f"grid drawn on {C.DATA / 'ext_corners_check.jpg'}")
    if not ok:
        if old:
            C.EXT_CORNERS.write_text(old)
        raise SystemExit("colour check failed: kept the old corners. Use the manual click: ext_camera corners")
    print(f"saved {C.EXT_CORNERS}")


def click_corners():
    import matplotlib.pyplot as plt

    cam = ExtCamera()
    frame = cam.grab()
    cam.close()
    plt.switch_backend("TkAgg")
    plt.figure("click the playing-area corners: a1, h1, h8, a8 (outer corner of each corner square)")
    plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    plt.title("1) a1 your near-left (dark)  2) h1 near-right  3) h8 far-right  4) a8 far-left")
    pts = plt.ginput(4, timeout=0)
    plt.close()
    if len(pts) != 4:
        raise SystemExit("need 4 clicks")
    C.EXT_CORNERS.parent.mkdir(parents=True, exist_ok=True)
    C.EXT_CORNERS.write_text(json.dumps({"px": [list(map(float, p)) for p in pts], "url": C.EXT_CAMERA_URL}, indent=2))
    v = view(frame)
    ok, msg = colour_check(frame, v)
    print(msg)
    print(f"rank 1 is on the {'human' if robot_color_from_corners() == 'black' else 'robot'}'s side: "
          f"robot plays {robot_color_from_corners()}")
    cv2.imwrite(str(C.DATA / "ext_corners_check.jpg"), draw_overlay(frame, v))
    print(f"saved {C.EXT_CORNERS}; grid drawn on {C.DATA / 'ext_corners_check.jpg'}")
    print(f"robot plays {robot_color_from_corners()} (rank 1 is on the {'robot' if robot_color_from_corners() == 'white' else 'human'}'s side)")
    if not ok:
        sys.exit(1)


class BoardWatcher:
    """Runs YOLO on the third-person camera every `period` seconds in a background thread.

    latest() -> (time, {square: (symbol, conf)}). stable(n) returns a reading only when the last n readings
    agree on every square (no hand over the board, nothing moving), else None.
    """

    def __init__(self, cam: ExtCamera, period=C.WATCH_PERIOD_S, conf=0.35):
        from chess_player.vision import detect_board as D

        self.cam, self.period, self.conf, self.D = cam, period, conf, D
        self.history = collections.deque(maxlen=10)
        self.boxes = (0.0, [])  # latest (time, [(symbol, conf, x1, y1, x2, y2, square)]) for display
        self.paused = False
        self._stop = False
        self._lock = threading.Lock()
        self._th = threading.Thread(target=self._loop, daemon=True)
        self._th.start()

    def _loop(self):
        while not self._stop:
            t0 = time.time()
            if not self.paused:
                try:
                    frame = self.cam.grab()
                    boxes = []
                    dets = self.D.detect_pieces(frame, view(frame), conf=self.conf, boxes_out=boxes)
                    with self._lock:
                        self.history.append((t0, dets, frame))
                        self.boxes = (t0, boxes)
                except Exception as e:  # keep watching; report once per failure kind
                    print(f"[watcher] {type(e).__name__}: {e}")
            time.sleep(max(0.0, self.period - (time.time() - t0)))

    def latest(self):
        with self._lock:
            return self.history[-1][:2] if self.history else (0.0, {})

    def clear(self):
        """Forget old readings (call after the arm moves, so only fresh views count)."""
        with self._lock:
            self.history.clear()

    def stable(self, n=3, vote_n=6):
        """A reading once the last n frames agree on which squares hold a piece and of which colour (no hand over
        the board, nothing moving); each square's piece TYPE is the majority over the last vote_n frames, so one
        flickering square (bishop/pawn at low confidence) doesn't hold up the whole board."""
        with self._lock:
            if len(self.history) < n:
                return None
            hist = list(self.history)
        occ = [{sq: d[0].isupper() for sq, d in h[1].items()} for h in hist[-n:]]
        if not all(o == occ[0] for o in occ):
            return None
        votes = [h[1] for h in hist[-vote_n:] if {sq: d[0].isupper() for sq, d in h[1].items()} == occ[0]]
        out = {}
        for sq in occ[0]:
            c = collections.Counter(v[sq][0] for v in votes if sq in v)
            sym, k = c.most_common(1)[0]
            conf = float(np.mean([v[sq][1] for v in votes if v.get(sq, ("",))[0] == sym]))
            out[sq] = (sym, conf * k / len(votes))  # a split vote lowers the confidence
        return out

    def close(self):
        self._stop = True


def watch():
    import matplotlib.pyplot as plt

    from chess_player.vision import detect_board as D

    cam = ExtCamera()
    use_yolo = D.available() is not None
    if not use_yolo:
        print("no YOLO model yet: showing the grid only")
    plt.switch_backend("TkAgg")
    fig, ax = plt.subplots(num="third-person camera (close the window to stop)")
    im = None
    try:
        while plt.fignum_exists(fig.number):
            t0 = time.time()
            frame = cam.grab()
            v = view(frame)
            shown = draw_overlay(frame, v)
            if use_yolo:
                dets = D.detect_pieces(frame, v)
                for sq, (sym, c) in dets.items():
                    p = v.mm_to_px([((chess.square_file(sq) + 0.5) * C.SQUARE_MM, (chess.square_rank(sq) + 0.5) * C.SQUARE_MM)])[0]
                    cv2.putText(shown, sym, tuple(p.astype(int) + [-6, 6]), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 255), 2)
                print("\n" + D.render(dets))
            rgb = cv2.cvtColor(shown, cv2.COLOR_BGR2RGB)
            if im is None:
                im = ax.imshow(rgb)
            else:
                im.set_data(rgb)
            plt.pause(max(0.05, C.WATCH_PERIOD_S - (time.time() - t0)))
    finally:
        cam.close()


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "watch"
    {"corners": click_corners, "auto": auto_corners_cmd, "watch": watch}[cmd]()
