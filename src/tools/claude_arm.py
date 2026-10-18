"""Live control window for the SO-101 follower, driven by Claude (or anyone) over a local HTTP API.

Window: live gripper camera, joints (current / target), status, log of Claude's decisions,
STOP / RESUME / RELEASE / HOLD buttons. Spacebar = STOP (freeze in place, torque stays on).

API (127.0.0.1:8765, JSON):
  GET  /state                      -> joints (deg / gripper %), targets, raw, load, flags
  GET  /snap?name=x                -> saves the latest camera frame to runs/<session>/x.jpg, returns path
  POST /move  {"joints": {...}, "relative": false, "speed": 15, "wait": true, "timeout": 20}
  POST /say   {"text": "..."}      -> adds a line to the window log
  POST /stop | /resume | /release | /hold

Units: arm joints in degrees (LeRobot calibration convention), gripper in % (0 closed, 100 open).
Usage: source scripts/env.sh && python src/tools/claude_arm.py
"""
import json
import queue
import threading
import time
import tkinter as tk
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import cv2
from PIL import Image, ImageTk

from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus

PORT = "/dev/so100_follower"
CAM = "/dev/v4l/by-id/usb-Sonix_Technology_Co.__Ltd._USB2.0_CAM1_USB2.0_CAM1-video-index0"
CAL = Path.home() / ".cache/huggingface/lerobot/calibration/robots/so_follower/follower_arm.json"
NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
HTTP_PORT = 8765
RATE_HZ = 50
MARGIN = 15  # raw counts kept away from calibrated end stops
MAX_SPEED = 40.0  # deg/s (or %/s for gripper) hard cap
RUN_DIR = Path(__file__).resolve().parents[2] / "data" / "runs" / datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_DIR.mkdir(parents=True, exist_ok=True)

cal = json.loads(CAL.read_text())


# ---------- unit conversion (same formulas as LeRobot) ----------
def to_unit(n, raw):
    c = cal[n]
    if n == "gripper":
        return (raw - c["range_min"]) / (c["range_max"] - c["range_min"]) * 100
    return (raw - (c["range_min"] + c["range_max"]) / 2) * 360 / 4095


def to_raw(n, val):
    c = cal[n]
    if n == "gripper":
        raw = c["range_min"] + val / 100 * (c["range_max"] - c["range_min"])
    else:
        raw = val * 4095 / 360 + (c["range_min"] + c["range_max"]) / 2
    return int(round(min(max(raw, c["range_min"] + MARGIN), c["range_max"] - MARGIN)))


def clamp_unit(n, val):
    return to_unit(n, to_raw(n, val))


# ---------- shared state ----------
class State:
    def __init__(self):
        self.lock = threading.Lock()
        self.present = {}  # unit
        self.raw = {}
        self.load = {}
        self.setpoint = {}  # unit, what we command right now (interpolated)
        self.target = {}  # unit, where we're heading
        self.speed = 15.0
        self.stopped = False
        self.torque = False
        self.status = "starting"
        self.error = None
        self.frame = None
        self.frame_time = 0.0


S = State()
log_q = queue.Queue()


def log(text, who="claude"):
    line = f"{datetime.now().strftime('%H:%M:%S')}  {text}"
    log_q.put((who, line))
    with open(RUN_DIR / "log.txt", "a") as f:
        f.write(f"[{who}] {line}\n")


# ---------- arm control thread ----------
def make_bus():
    motors = {n: Motor(i + 1, "sts3215", MotorNormMode.DEGREES) for i, n in enumerate(NAMES)}
    bus = FeetechMotorsBus(PORT, motors)
    bus.connect(handshake=True)
    # Same servo setup as LeRobot's SOFollower.configure()
    for n in NAMES:
        bus.write("Torque_Enable", n, 0, num_retry=3)
    bus.configure_motors()
    for n in NAMES:
        bus.write("Operating_Mode", n, 0, num_retry=3)
        bus.write("P_Coefficient", n, 16, num_retry=3)
        bus.write("I_Coefficient", n, 0, num_retry=3)
        bus.write("D_Coefficient", n, 32, num_retry=3)
    bus.write("Max_Torque_Limit", "gripper", 500, num_retry=3)
    bus.write("Protection_Current", "gripper", 250, num_retry=3)
    bus.write("Overload_Torque", "gripper", 25, num_retry=3)
    return bus


def set_torque(bus, on):
    raw = bus.sync_read("Present_Position", normalize=False, num_retry=3)
    if on:
        # hold exactly where we are so enabling torque never jerks the arm
        bus.sync_write("Goal_Position", raw, normalize=False)
    for n in NAMES:
        bus.write("Torque_Enable", n, 1 if on else 0, num_retry=3)
    with S.lock:
        for n in NAMES:
            S.setpoint[n] = S.target[n] = to_unit(n, raw[n])
        S.torque = on


cmd_q = queue.Queue()  # "release" / "hold" requests for the control thread


def control_loop():
    bus = None
    k = 0
    while True:
        try:
            if bus is None:
                with S.lock:
                    S.status = "connecting arm"
                bus = make_bus()
                set_torque(bus, True)
                with S.lock:
                    S.status = "ready"
                    S.error = None
                log("arm connected, torque ON, holding current pose", "sys")
            while not cmd_q.empty():
                c = cmd_q.get()
                set_torque(bus, c == "hold")
                log(f"torque {'ON (hold)' if c == 'hold' else 'OFF (released)'}", "sys")

            raw = bus.sync_read("Present_Position", normalize=False, num_retry=3)
            if k % 5 == 0:
                load = bus.sync_read("Present_Load", normalize=False, num_retry=3)
            dt = 1 / RATE_HZ
            with S.lock:
                S.raw = raw
                S.present = {n: to_unit(n, raw[n]) for n in NAMES}
                if k % 5 == 0:
                    # LeRobot already decodes the sign bit; magnitude is in 0.1 % units
                    S.load = {n: abs(v) / 10 for n, v in load.items()}
                moving = False
                if S.torque and not S.stopped:
                    step = min(S.speed, MAX_SPEED) * dt
                    for n in NAMES:
                        d = S.target[n] - S.setpoint[n]
                        if abs(d) > 1e-3:
                            moving = True
                            S.setpoint[n] += max(-step, min(step, d))
                    goal = {n: to_raw(n, S.setpoint[n]) for n in NAMES}
                if S.torque:
                    S.status = "STOPPED" if S.stopped else ("moving" if moving else "holding")
                else:
                    S.status = "released (torque off)"
            if S.torque and not S.stopped and moving:
                bus.sync_write("Goal_Position", goal, normalize=False)
            k += 1
            time.sleep(dt)
        except Exception as e:  # USB dropout etc.: freeze and reconnect
            with S.lock:
                S.status = "ERROR: reconnecting"
                S.error = repr(e)
                S.stopped = True
            log(f"arm error: {e!r} - STOPPED, reconnecting", "sys")
            try:
                bus.port_handler.closePort()
            except Exception:
                pass
            bus = None
            time.sleep(2)


# ---------- camera thread ----------
def camera_loop():
    cap = None
    while True:
        if cap is None:
            cap = cv2.VideoCapture(CAM, cv2.CAP_V4L2)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        ok, f = cap.read()
        if not ok:
            cap.release()
            cap = None
            time.sleep(1)
            continue
        with S.lock:
            S.frame = f
            S.frame_time = time.time()


# ---------- HTTP API ----------
def state_dict():
    with S.lock:
        r = lambda d: {n: round(float(d[n]), 2) for n in d}
        return {
            "status": S.status,
            "stopped": S.stopped,
            "torque": S.torque,
            "present": r(S.present),
            "target": r(S.target),
            "raw": dict(S.raw),
            "load_pct": dict(S.load),
            "speed": S.speed,
            "error": S.error,
        }


class Handler(BaseHTTPRequestHandler):
    def _send(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass

    def do_GET(self):
        u = urlparse(self.path)
        if u.path == "/state":
            return self._send(state_dict())
        if u.path == "/snap":
            name = parse_qs(u.query).get("name", [datetime.now().strftime("%H%M%S")])[0]
            with S.lock:
                f = None if S.frame is None else S.frame.copy()
            if f is None:
                return self._send({"error": "no frame"}, 503)
            p = RUN_DIR / f"{name}.jpg"
            cv2.imwrite(str(p), f)
            return self._send({"path": str(p), **state_dict()})
        self._send({"error": "unknown"}, 404)

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(n) or b"{}")
        p = urlparse(self.path).path
        if p == "/say":
            log(body.get("text", ""))
            return self._send({"ok": True})
        if p == "/stop":
            with S.lock:
                S.stopped = True
                S.target = dict(S.setpoint)
            log("STOP (API)", "sys")
            return self._send(state_dict())
        if p == "/resume":
            with S.lock:
                S.stopped = False
                S.target = dict(S.setpoint)
            log("resumed", "sys")
            return self._send(state_dict())
        if p in ("/release", "/hold"):
            cmd_q.put(p[1:])
            time.sleep(0.3)
            return self._send(state_dict())
        if p == "/move":
            with S.lock:
                if S.stopped or not S.torque:
                    return self._send({"error": "stopped or torque off", **{}}, 409)
                rel = body.get("relative", False)
                for name, v in body.get("joints", {}).items():
                    if name not in NAMES:
                        return self._send({"error": f"bad joint {name}"}, 400)
                    base = S.target[name] if rel else 0
                    S.target[name] = clamp_unit(name, base + float(v))
                S.speed = float(body.get("speed", S.speed))
                tgt = {k: round(v, 1) for k, v in S.target.items()}
            log(f"move -> {body.get('joints')} {'(rel)' if rel else ''} @ {S.speed:.0f}/s", "act")
            if body.get("wait", True):
                t0 = time.time()
                while time.time() - t0 < body.get("timeout", 20):
                    with S.lock:
                        done = all(abs(S.target[k] - S.setpoint[k]) < 1e-3 for k in NAMES)
                        stop = S.stopped
                    if done or stop:
                        break
                    time.sleep(0.05)
                time.sleep(body.get("settle", 0.4))
            return self._send({"commanded_target": tgt, **state_dict()})
        self._send({"error": "unknown"}, 404)


# ---------- window ----------
class App:
    def __init__(self, root):
        self.root = root
        root.title("Claude arm control - SO-101 follower")
        root.configure(bg="#1e1e1e")
        root.bind("<space>", lambda e: self.stop())
        left = tk.Frame(root, bg="#1e1e1e")
        left.pack(side="left", padx=8, pady=8)
        right = tk.Frame(root, bg="#1e1e1e")
        right.pack(side="left", fill="both", expand=True, padx=8, pady=8)

        self.img_label = tk.Label(left, bg="black")
        self.img_label.pack()
        self.status = tk.Label(left, text="", font=("DejaVu Sans Mono", 14, "bold"), fg="white", bg="#1e1e1e")
        self.status.pack(pady=6, anchor="w")
        btns = tk.Frame(left, bg="#1e1e1e")
        btns.pack(anchor="w")
        tk.Button(btns, text="STOP (space)", bg="#c62828", fg="white", font=("DejaVu Sans", 14, "bold"),
                  width=14, command=self.stop).pack(side="left", padx=3)
        tk.Button(btns, text="RESUME", width=9, font=("DejaVu Sans", 12), command=self.resume).pack(side="left", padx=3)
        tk.Button(btns, text="RELEASE", width=9, font=("DejaVu Sans", 12),
                  command=lambda: cmd_q.put("release")).pack(side="left", padx=3)
        tk.Button(btns, text="HOLD", width=7, font=("DejaVu Sans", 12),
                  command=lambda: cmd_q.put("hold")).pack(side="left", padx=3)

        self.joints = tk.Label(right, text="", justify="left", font=("DejaVu Sans Mono", 12), fg="#e0e0e0",
                               bg="#252526", anchor="nw", padx=8, pady=6)
        self.joints.pack(fill="x")
        tk.Label(right, text="Claude's decisions & actions", fg="#9cdcfe", bg="#1e1e1e",
                 font=("DejaVu Sans", 12, "bold")).pack(anchor="w", pady=(10, 2))
        self.log = tk.Text(right, width=62, height=26, bg="#252526", fg="#e0e0e0", font=("DejaVu Sans Mono", 11),
                           wrap="word", borderwidth=0)
        self.log.pack(fill="both", expand=True)
        for tag, col in (("claude", "#ffd479"), ("act", "#9cdcfe"), ("sys", "#a0a0a0")):
            self.log.tag_config(tag, foreground=col)
        self.tick()

    def stop(self):
        with S.lock:
            S.stopped = True
            S.target = dict(S.setpoint)
        log("STOP pressed in window", "sys")

    def resume(self):
        with S.lock:
            S.stopped = False
            S.target = dict(S.setpoint)
        log("RESUME pressed in window", "sys")

    def tick(self):
        with S.lock:
            f = None if S.frame is None else S.frame.copy()
            st, pres, tgt, load = S.status, dict(S.present), dict(S.target), dict(S.load)
        if f is not None:
            h, w = f.shape[:2]
            cv2.drawMarker(f, (w // 2, h // 2), (0, 255, 255), cv2.MARKER_CROSS, 24, 1)
            img = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)))
            self.img_label.configure(image=img)
            self.img_label.image = img
        color = {"STOPPED": "#ff5252", "moving": "#69f0ae"}.get(st, "#ff5252" if "ERROR" in st else "white")
        self.status.configure(text=f"status: {st}", fg=color)
        lines = [f"{'joint':14}{'now':>9}{'target':>9}{'load%':>7}"]
        for n in NAMES:
            if n in pres:
                u = "%" if n == "gripper" else "°"
                lines.append(f"{n:14}{pres[n]:8.1f}{u}{tgt.get(n, 0):8.1f}{u}{load.get(n, 0):7.1f}")
        self.joints.configure(text="\n".join(lines))
        while not log_q.empty():
            who, line = log_q.get()
            self.log.insert("end", line + "\n", who)
            self.log.see("end")
        self.root.after(40, self.tick)


if __name__ == "__main__":
    log(f"session {RUN_DIR.name}", "sys")
    threading.Thread(target=camera_loop, daemon=True).start()
    threading.Thread(target=control_loop, daemon=True).start()
    srv = ThreadingHTTPServer(("127.0.0.1", HTTP_PORT), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    root = tk.Tk()
    App(root)
    root.mainloop()
    # window closed: servos keep holding their last goal (torque stays on)
