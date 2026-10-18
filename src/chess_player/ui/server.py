"""Chess pipeline with a browser UI: arm + cameras + YOLO watcher + Stockfish, all in one process.

  python -m chess_player.ui.server [--port 8770] [--dry-run] [--skill N]
  (scripts/run_pipeline.sh starts this and opens the window)

One worker thread owns the arm and the game (Conductor); the HTTP handlers only queue jobs and read state, so
nothing ever drives the arm from two places. Endpoints:
  GET  /                  the UI          GET /api/state   JSON state      GET /board.svg   current position
  GET  /stream/wrist      gripper camera (MJPEG)          GET /stream/ext  webcam + grid + YOLO boxes (MJPEG)
  POST /api/action        {"action": play|pause|new_game|home|i_moved|done_by_hand|undo}
  POST /api/human_move    {"move": "e4" | "e2e4" | "Nf3" | "O-O"}
  POST /api/manual        {"from": "e4", "to": "a3", "sync": true}   physical pick & place (any squares)
"""
import argparse
import collections
import json
import queue
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import chess
import chess.svg
import cv2
import numpy as np

from chess_player import config as C
from chess_player.game import state as GS
from chess_player.geometry import square_center_mm

HTML = Path(__file__).with_name("index.html")


# ---------------------------------------------------------------- cameras for display
class LiveCam:
    """Gripper camera for the main view: the device itself, or wrist_stream.py's stream if that owns it."""

    def __init__(self):
        self.frame, self.t, self.source = None, 0.0, "none"
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while True:
            cap = cv2.VideoCapture(str(C.CAMERA), cv2.CAP_V4L2)
            if cap.isOpened() and cap.read()[0]:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, C.CAM_W)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, C.CAM_H)
                self.source = "device"
            else:
                cap.release()
                cap = cv2.VideoCapture(C.WRIST_STREAM_URL)
                self.source = "wrist_stream" if cap.isOpened() else "none"
            while cap.isOpened():
                ok, f = cap.read()
                if not ok:
                    break
                self.frame, self.t = f, time.time()
            cap.release()
            self.source = "none"
            time.sleep(1.0)

    def get(self):
        return self.frame if self.frame is not None and time.time() - self.t < 2.0 else None


# ---------------------------------------------------------------- the conductor (owns arm + game)
class Conductor:
    def __init__(self, dry_run=False, skill=None, wrist=None):
        from chess_player.game.session import Session

        self.log_lines = collections.deque(maxlen=80)
        self.jobs = queue.Queue()
        self.mode = "paused"  # paused | playing
        self.phase = "starting"
        self.status = "Starting up..."
        self.help = None  # what the human must do: {"kind": ..., ...}
        self.busy = False
        self.eval = "0.00"
        self.plan = []
        self.on_event = None  # set by the voice assistant
        from chess_player.vision.detect_board import available

        self.yolo_kind = available()  # "own" | "pretrained" | None
        self._read_frame = None  # webcam frame behind the last "Read board" (saved as training data on Apply)
        self.live_sync = C.LIVE_SYNC_DEFAULT  # when on (and not mid-game), the game board follows what the webcam sees
        self.seen = None  # {"fen", "diff": [squares where camera != game], "t"} from the YOLO watcher
        self.san_before_edit = []  # move list from before a manual edit (editing the position clears python-chess's stack)
        self.robot_color = "white"
        if C.EXT_CORNERS.exists():
            from chess_player.vision.ext_camera import robot_color_from_corners

            self.robot_color = robot_color_from_corners()
        self.arm_mode = "simulated" if dry_run else "connected"
        try:
            self.s = Session(dry_run=dry_run, robot_color=self.robot_color, skill=skill, log=self.log)
        except Exception as e:  # arm busy / unplugged: keep the UI usable
            self.log(f"arm not available ({type(e).__name__}: {e}); running SIMULATED")
            self.arm_mode = "simulated"
            self.s = Session(dry_run=True, robot_color=self.robot_color, skill=skill, log=self.log)
        from chess_player import board_state

        self.board_state = board_state
        saved = board_state.load(default_start=False)
        if saved is not None and saved.board_fen() != chess.Board().board_fen():
            self.s.board.set_fen(saved.fen())
            self.log("loaded the saved real-board position (change it with Edit position or Start)")
        self._saved_fen = self.s.board.fen()
        self.ext = self.s.ext
        if self.ext is None and C.EXT_CORNERS.exists():
            from chess_player.vision.ext_camera import ExtCamera

            try:
                self.ext = ExtCamera()
            except RuntimeError as e:
                self.log(f"webcam not available: {e}")
        # gripper camera as the second judge of piece types (only while the arm rests at the look pose)
        self.wrist, self.eye, self.fused = wrist, None, {}
        self.eye_status = "off"
        self._parked_since = None
        if wrist is not None and self.ext is not None and self.yolo_kind:
            from chess_player.vision.wrist_view import WristWatcher

            self.eye = WristWatcher(wrist.get, self._eye_active)
            self.eye_status = "calibrated" if self.eye.view is not None else "not calibrated"
            self.s.refine = lambda dets: self._fused(dets)[0]  # human-move detection uses both cameras too
        threading.Thread(target=self._loop, daemon=True).start()
        self.jobs.put(("startup", {}))

    def _eye_active(self):
        """The gripper camera's view counts only when the arm has rested at the look pose for a moment."""
        still = self.s.parked and not self.busy and getattr(self.s.arm, "torque_on", True)
        if not still:
            self._parked_since = None
            return False
        self._parked_since = self._parked_since or time.time()
        return time.time() - self._parked_since > 1.0

    def _fused(self, ext_dets):
        """Webcam reading + gripper-camera reading -> (seen for merge_seen, per-square info for the UI)."""
        from chess_player.vision.wrist_view import fuse

        eye = self.eye.match(ext_dets) if self.eye is not None and self.eye.view is not None else None
        seen, info, votes = fuse(ext_dets, eye or {}, self.eye.visible if eye is not None else set())
        if eye is not None:  # pieces only the robot eye can see (hidden from the webcam behind other pieces)
            for sq, (sym, c) in self.eye.extras(ext_dets).items():
                seen[sq] = (sym, c)
                info[chess.square_name(sq)] = {"web": None, "eye": [sym, round(c, 2)], "agree": None}
        self.fused = {"t": time.time(), "info": info, "eye_on": eye is not None}
        return seen, votes

    def calibrate_eye(self, force=False):
        """Fit the gripper camera's view of the board (arm at the look pose), unless the saved one still fits."""
        from chess_player.vision import wrist_view as W

        if self.eye is None:
            return
        w = self.s.watcher
        t_end, dets, frame = time.time() + 20.0, None, None
        while time.time() < t_end and (dets is None or frame is None):  # cameras and YOLO may still be warming up
            time.sleep(0.5)
            dets = w.stable(C.WATCH_STABLE_N) if w is not None else None
            frame = self.wrist.get()
        if dets is None or frame is None:
            self.eye_status = "not calibrated (no camera picture)"
            self.log("robot-eye calibration: no steady picture from both cameras; try again with nothing over the board")
            return
        if not force:
            ok, score = W.still_fits(frame, set(dets))
            if ok:
                self.eye_status = "calibrated"
                self.log(f"robot eye: saved calibration still fits (contrast {score:.0f})")
                return
        self.eye_status = "calibrating..."
        try:
            rep = W.calibrate(frame, {sq: s for sq, (s, _) in dets.items()})
            self.eye.reload()
            self.eye_status = "calibrated"
            self.log(f"robot eye calibrated: {rep['pieces_matched']}/{rep['of']} pieces matched, "
                     f"{rep['corners_used']} square corners, contrast {rep['contrast']:.0f}")
        except Exception as e:
            self.eye_status = "not calibrated"
            self.log(f"robot-eye calibration failed: {e}")

    def job_calibrate_eye(self):
        self.say("moving", "Calibrating the robot eye...")
        self.s.park()
        self.calibrate_eye(force=True)
        self.say("idle", f"Robot eye {self.eye_status}.")

    # ------------------------------------------------ helpers
    def log(self, msg, *a):
        line = f"{time.strftime('%H:%M:%S')}  {msg}"
        self.log_lines.append(line)
        print(line, flush=True)

    def say(self, phase, status):
        self.phase, self.status = phase, status

    def emit(self, kind, **data):
        """Tell the voice assistant (if any) about a major moment."""
        if self.on_event:
            try:
                self.on_event(kind, **data)
            except Exception as e:
                self.log(f"event handler: {e}")

    def _human_pushed(self, san, before):
        """After a human move: refresh the eval and report it with how much it helped the robot, and what
        Stockfish would have played instead."""
        from chess_player.voice.assistant import eval_cp

        best_san = None
        try:
            b = self.s.board
            last = b.pop()
            best = self.s.engine.best_move(b, 0.3)
            best_san = b.san(best) if best != last else None
            b.push(last)
        except Exception:
            pass
        self._refresh_eval()
        sign = 1 if self.robot_color == "white" else -1
        self.emit("human_move", san=san, swing_for_robot=sign * (eval_cp(self.eval) - eval_cp(before)),
                  better=best_san, eval=self.eval)

    def human_color(self):
        return chess.BLACK if self.robot_color == "white" else chess.WHITE

    def _refresh_eval(self):
        try:
            self.eval = self.s.engine.evaluate(self.s.board, 0.2) if self.s.board.is_valid() else "—"
        except Exception:
            self.eval = "—"

    # ------------------------------------------------ worker
    def _loop(self):
        while True:
            try:
                job, args = self.jobs.get(timeout=0.15)
            except queue.Empty:
                job, args = None, None
            try:
                if job:
                    self.busy = True
                    getattr(self, f"job_{job}")(**args)
                elif self.mode == "playing" and not self.help:
                    self.tick()
                else:
                    self.observe()
            except Exception as e:
                from chess_player.motion.arm import ArmStopped

                self.mode = "paused"
                if isinstance(e, ArmStopped):
                    self.s.arm.stop_event.clear()
                    self.s.parked = False
                    self.say("stopped", "STOPPED. The arm is holding where it stopped. Check the board, then press "
                             "▶ Play (or ⌂ Home).")
                    self.log("EMERGENCY STOP: arm frozen, game paused")
                else:
                    self.log(f"error in {job or 'play loop'}: {type(e).__name__}: {e}")
                    traceback.print_exc()
                    self.say("need_help", f"Error: {e}. Paused.")
                    self.emit("error", why=e)
            finally:
                self.busy = False
                if self.s.board.fen() != self._saved_fen:  # keep data/board_state.json = the real board
                    self._saved_fen = self.s.board.fen()
                    try:
                        self.board_state.save(self.s.board)
                    except OSError as e:
                        self.log(f"couldn't save the board position: {e}")

    def tick(self):
        b = self.s.board
        if not b.is_valid():
            self.mode = "paused"
            self.say("need_help", f"I can't play from this position: {'; '.join(GS.why_illegal(b))}. Fix it with "
                     "Edit position, or ⟳ Recheck if the camera misread a piece.")
            return
        if b.is_game_over():
            self.mode = "paused"
            self.say("game_over", f"Game over: {b.result()} ({b.outcome().termination.name.lower()})")
            w = b.outcome().winner
            self.emit("game_over", winner=None if w is None else ("robot" if (w == chess.WHITE) == (self.robot_color == "white") else "human"))
            return
        if not self.s.human_to_move():
            self.busy = True
            self.robot_move()
            return
        if self.s.watcher is not None:
            self.say("watching", "Your move. I'm watching the board.")
            inf = self.s.poll_human_move()
            if inf is None and self.s.illegal_seen:
                mv, why = self.s.illegal_seen
                self.say("need_help", f"I see {mv}, but that's not legal: {why}. Put it back and play a legal move.")
                if getattr(self, "_illegal_logged", None) != mv:
                    self._illegal_logged = mv
                    self.log(f"camera sees an illegal move: {mv} ({why})")
            if inf is not None:
                san = b.san(inf.move)
                before = self.eval
                self.s.push_human(inf.move)
                self.log(f"you played {san} (seen by YOLO)")
                self._human_pushed(san, before)
        else:
            self.say("watching", "Your move. Make it on the board, then press 'I moved', or type it.")

    def observe(self):
        """Compare what the webcam sees (stable YOLO reading) with the game board; with live sync, adopt it."""
        w = self.s.watcher
        if w is None:
            self.seen = None
            return
        dets = w.stable(C.WATCH_STABLE_N)
        if dets is None:
            return
        game = self.s.board
        # chess sense: types persist unless the cameras agree on a change, moved pieces keep theirs, one king
        seen, votes = self._fused(dets)
        b, unsure = GS.merge_seen(game, seen, votes=votes)
        if GS.looks_like_start(b):  # the starting setup: exact standard position, whatever types YOLO guessed
            b, unsure = chess.Board(), []
        diff = [chess.square_name(sq) for sq in chess.SQUARES
                if (game.piece_at(sq) and game.piece_at(sq).symbol()) != (b.piece_at(sq) and b.piece_at(sq).symbol())]
        self.seen = {"fen": b.board_fen(), "diff": diff, "t": time.time(), "unsure": unsure}
        # only adopt a change the cameras keep showing (a square flickering in and out must not flip the board)
        pend = getattr(self, "_pending", None)
        if not diff or pend is None or pend[0] != b.board_fen():
            self._pending = (b.board_fen(), time.time()) if diff else None
            return
        if time.time() - pend[1] < C.LIVE_SYNC_HOLD_S:
            return
        if self.live_sync and diff and not self.help and self.mode != "playing":
            self.log(f"live sync: board updated from the cameras ({len(diff)} squares changed: {' '.join(diff[:8])})")
            with w._lock:
                self._read_frame = w.history[-1][2] if w.history else None
            self.job_set_position(b.fen(), from_camera=False, quiet=True)
            self.seen["diff"] = []

    def job_recheck(self):
        """Fresh look: drop old readings, wait for new agreeing ones from more frames, re-sync the board."""
        w = self.s.watcher
        if w is None:
            self.say("need_help", "No camera reading: the webcam or YOLO is off.")
            return
        self.say("thinking", "Rechecking the board...")
        w.clear()
        t_end, dets = time.time() + 4.0, None
        while time.time() < t_end and dets is None:
            time.sleep(0.2)
            dets = w.stable(max(5, C.WATCH_STABLE_N + 2))
        if dets is None:
            self.say("need_help", "The camera view keeps changing (hand in the way?). Try ⟳ Recheck again.")
            return
        seen, votes = self._fused(dets)
        b, unsure = GS.merge_seen(self.s.board, seen, type_conf=C.RECHECK_TYPE_CONF, votes=votes)
        if GS.looks_like_start(b):
            b, unsure = chess.Board(), []
            self.log("recheck: the starting position")
        self.job_set_position(b.fen(), quiet=True)
        self.log(f"recheck: {len(b.piece_map())} pieces" + (f", unsure: {' '.join(unsure)}" if unsure else ""))
        if not GS.why_illegal(b):
            self.say("idle", "Rechecked: board synced. Press Play." + (f" Unsure squares: {' '.join(unsure)}." if unsure else ""))

    def emergency_stop(self):
        """From the HTTP thread: freeze the arm now, drop queued work, pause."""
        self.mode = "paused"
        while not self.jobs.empty():
            try:
                self.jobs.get_nowait()
            except queue.Empty:
                break
        arm = self.s.arm
        arm.stop_event.set()
        t_end = time.time() + 0.5
        while self.busy and time.time() < t_end:  # the worker sees the flag within one control tick
            time.sleep(0.02)
        if not self.busy and arm.stop_event.is_set():  # nothing was moving: freeze and clear here
            try:
                arm.freeze()
            except Exception as e:
                self.log(f"freeze: {e}")
            arm.stop_event.clear()
            self.say("stopped", "STOPPED. Press ▶ Play to continue.")
            self.log("EMERGENCY STOP")
        self.s.parked = False

    def set_torque(self, on):
        arm = self.s.arm
        if on:
            arm.hold()
            self.log("torque on: holding the current pose")
            self.say("idle", "Torque on. Press ⌂ Home or ▶ Play.")
        else:
            self.emergency_stop()
            t_end = time.time() + 1.5
            while self.busy and time.time() < t_end:
                time.sleep(0.02)
            arm.release()
            self.s.parked = False
            self.log("torque OFF: the arm is limp (move it by hand)")
            self.say("stopped", "Torque off: the arm is limp. Press Torque on (or ⌂ Home) before playing.")

    def job_set_turn(self, white):
        """Choose who moves first (keeps the pieces)."""
        b = self.s.board.copy(stack=False)
        b.turn = chess.WHITE if white else chess.BLACK
        b.ep_square = None
        why = GS.why_illegal(b)
        if why:
            self.say("need_help", f"Can't give the move to {'white' if white else 'black'}: {'; '.join(why)}.")
            return
        self.job_set_position(b.fen(), quiet=True)
        who = "Robot" if (b.turn == chess.WHITE) == (self.robot_color == "white") else "You"
        self.say("idle", f"{who} ({'white' if white else 'black'}) to move first. Press Play.")
        self.log(f"side to move: {'white' if white else 'black'}")

    def recover(self, t, err):
        """After a failed grab or a dropped piece: go home, look with the webcam, find the piece.
        Returns "done" (it is on the target), the board-mm xy to grab it from again, or None (ask the human)."""
        from chess_player.geometry import mm_to_square, square_center_mm

        self.log(f"recovering: {err}")
        self.emit("recovering", why=str(err))
        self.say("moving", "Something slipped. Looking for the piece...")
        self.s.pp.go_look()
        self.s.parked = True
        w = self.s.watcher
        if w is None:
            return None
        w.clear()
        w.paused = False
        try:
            t_end, dets = time.time() + 5.0, None
            while time.time() < t_end and dets is None:
                time.sleep(0.2)
                dets = w.stable(C.WATCH_STABLE_N)
        finally:
            w.paused = True
        if dets is None:
            self.log("recovery: no stable view of the board")
            return None
        white = t.piece.isupper()
        same = {sq for sq, (sym, _) in dets.items() if sym.isupper() == white}
        src, dst = mm_to_square(*t.src), mm_to_square(*t.dst)
        if dst is not None and dst in same:
            self.log(f"recovery: the piece is on {chess.square_name(dst)} already")
            return "done"
        if src is not None and src in same:
            self.log(f"recovery: still on {chess.square_name(src)}, grabbing again")
            return tuple(square_center_mm(src))
        expected = {sq for sq, p in self.s.board.piece_map().items() if p.color == white}
        new = sorted(same - expected, key=lambda sq: np.linalg.norm(square_center_mm(sq) - np.asarray(t.dst)))
        if new:
            self.log(f"recovery: found it on {chess.square_name(new[0])}, picking it up from there")
            self.emit("found_piece", square=chess.square_name(new[0]))
            return tuple(square_center_mm(new[0]))
        self.log("recovery: can't see where it went")
        return None

    def robot_move(self):
        self.say("thinking", "Thinking...")
        move = self.s.engine_move()
        san = self.s.board.san(move)
        report = self.s.check(move)
        self.plan = report.get("steps", [])
        if not report.get("physical_ok"):
            why = "; ".join(st.get("details", "") for st in self.plan if not st["ok"])
            self.log(f"robot wants {san} but can't do it: {why}")
            self.help = {"kind": "robot_by_hand", "move": move.uci(), "san": san, "why": why}
            self.say("need_help", f"Please play {san} for me by hand, then press 'Done'.")
            self.emit("need_help", san=san)
            return
        self.say("moving", f"Playing {san}...")
        self.log(f"robot plays {san}")
        victim = None
        if self.s.board.is_capture(move):
            vsq = move.to_square if not self.s.board.is_en_passant(move) else chess.square(
                chess.square_file(move.to_square), chess.square_rank(move.from_square))
            victim = chess.piece_name(self.s.board.piece_at(vsq).piece_type)
        mover = chess.piece_name(self.s.board.piece_at(move.from_square).piece_type)
        self.emit("robot_move", san=san, victim=victim, piece=mover, src=chess.square_name(move.from_square),
                  dst=chess.square_name(move.to_square), castle=self.s.board.is_castling(move), eval=self.eval)
        from chess_player.motion.pick_place import GraspFailed
        from chess_player.motion.ik import Unreachable

        from chess_player.motion.pick_place import PieceDropped

        try:
            self.s.execute(move, recover=self.recover)
        except (Unreachable, GraspFailed, PieceDropped) as e:
            self.log(f"{type(e).__name__}: {e}")
            self.help = {"kind": "robot_by_hand", "move": move.uci(), "san": san, "why": str(e)}
            self.say("need_help", f"I couldn't finish {san}. Please complete it by hand, then press 'Done'.")
            self.emit("need_help", san=san)
            return
        self._refresh_eval()
        self.emit("robot_done", san=san, check=self.s.board.is_check(), eval=self.eval)
        if self.s.has_vision:
            ok, details = self.s.verify_position()
            if not ok:
                self.log(f"board check after {san}: {details}")
                self.help = {"kind": "fix_board", "why": details}
                self.say("need_help", "The board doesn't look right. Fix it, then press 'Done'.")
                self.emit("fix_board")
                return
        self.say("watching", "Your move.")

    # ------------------------------------------------ jobs (queued from the UI)
    def job_startup(self):
        self.say("moving", "Going home...")
        self.s.park()
        if self.s.has_vision:
            try:
                self.s.snapshot()
            except Exception as e:
                self.log(f"first look failed: {e}")
        self._refresh_eval()
        if self.eye is not None:
            threading.Thread(target=self.calibrate_eye, daemon=True).start()
        who = "I play" if self.robot_color == "white" else "You play"
        self.say("idle", f"Ready. {who} white. Press Play.")
        self.emit("ready")
        self.log(f"ready: robot {self.robot_color}, vision {self.s.vision if self.s.has_vision else 'off'}, arm {self.arm_mode}")

    def job_play(self):
        if self.mode != "playing":
            self.emit("game_start" if not self.s.board.move_stack else "play")
        self.mode = "playing"
        self.log("play")

    def job_pause(self):
        self.mode = "paused"
        self.say("idle", "Paused.")
        self.log("pause")
        self.emit("pause")

    def job_new_game(self):
        from chess_player.motion.pick_place import Slots

        self.s.board.reset()
        self.san_before_edit = []
        self.s.slots = Slots(self.s.pp.T)
        self.help, self.plan, self.mode = None, [], "paused"
        self.s.park()
        if self.s.has_vision:
            self.s.snapshot()
        self._refresh_eval()
        self.say("idle", "New game. Set up the pieces, then press Play.")
        self.emit("new_game")
        self.log("new game")

    def job_home(self):
        self.say("moving", "Going home...")
        self.emit("home")
        self.s.parked = False
        self.s.park()
        self.say("idle" if self.mode == "paused" else "watching", "Home.")

    def job_i_moved(self):
        if not self.s.human_to_move():
            self.log("'I moved' ignored: it's the robot's turn")
            return
        self.say("thinking", "Looking at the board...")
        inf = self.s.see_human_move()
        if inf.move is None:
            cands = [self.s.board.san(m) for m, _ in inf.ranked[:3]]
            self.help = {"kind": "which_move", "candidates": cands, "why": inf.reason}
            self.say("need_help", "Not sure what you played. Pick or type it.")
            self.emit("which_move")
            return
        san = self.s.board.san(inf.move)
        before = self.eval
        self.s.push_human(inf.move)
        self.log(f"you played {san}")
        self._human_pushed(san, before)
        self.say("watching", f"You played {san}.")

    def job_human_move(self, move):
        m = GS.parse_human_move(self.s.board, move)
        if not self.s.human_to_move():
            raise ValueError("it's the robot's turn")
        san = self.s.board.san(m)
        before = self.eval
        self.s.push_human(m)
        if self.s.has_vision:
            self.s.snapshot()
        self.help = None
        self.log(f"you played {san} (typed)")
        self._human_pushed(san, before)
        self.say("watching", f"You played {san}.")

    def job_done_by_hand(self):
        h, self.help = self.help, None
        if h and h["kind"] == "robot_by_hand":
            self.s.push_robot_by_hand(chess.Move.from_uci(h["move"]))
            self.log(f"{h['san']} done by hand")
            self._refresh_eval()
        if self.s.has_vision:
            self.s.snapshot()
        self.say("watching", "Thanks.")

    def job_undo(self):
        n = 2 if len(self.s.board.move_stack) >= 2 else len(self.s.board.move_stack)
        for _ in range(n):
            self.s.board.pop()
        self.help = None
        self.mode = "paused"
        self.log(f"took back {n} move(s): put the pieces back by hand")
        self.say("idle", "Took back. Restore the pieces, then press Play.")

    def job_set_position(self, fen, from_camera=False, quiet=False):
        """Make the game position match the real board (typed / clicked in the editor, or read by YOLO).
        If it came from a camera reading (possibly corrected by hand), that frame + position become YOLO training
        data, so our own model learns this board."""
        b = chess.Board(fen)
        if from_camera and self._read_frame is not None:
            try:
                from chess_player.vision.capture_dataset import DS, write_sample

                stem = time.strftime("sync_%Y%m%d_%H%M%S")
                n, _ = write_sample(stem, self._read_frame, b, "ext")
                (DS / "meta" / f"{stem}.json").write_text(json.dumps({"fen": b.fen(), "position": stem, "camera": "ext"}))
                self.log(f"saved this board as YOLO training data ({n} pieces): {stem}")
            except Exception as e:
                self.log(f"couldn't save training data: {e}")
            self._read_frame = None
        self.san_before_edit = self.state()["moves"] + ["(position set)"] if self.s.board.move_stack or self.san_before_edit else []
        self.s.board.set_fen(b.fen())
        self.help, self.plan, self.mode = None, [], "paused"
        if self.s.has_vision:
            self.s.snapshot()
        self._refresh_eval()
        why = GS.why_illegal(self.s.board)
        ok = not why
        self.say("idle", "Position set. Press Play." if ok else f"Not a legal chess position: {'; '.join(why)}. "
                 "Fix it with Edit position (or ⟳ Recheck if the camera misread it).")
        if not quiet:
            self.log(f"position set: {self.s.board.fen()}" + ("" if ok else " (not a legal position)"))
            self.emit("position_set")

    def read_board(self, n=5):
        """Position proposed from the YOLO watcher: per square, the piece seen in most of the last n readings.
        Returns (fen, info) without applying it."""
        w = self.s.watcher
        if w is None:
            raise ValueError("no piece detector: the webcam or its corners aren't set up (vision is off)")
        with w._lock:
            recent = list(w.history)[-n:]
        hist = [h[1] for h in recent]
        if recent:
            self._read_frame = recent[-1][2]
        if len(hist) < 2:
            raise ValueError("not enough webcam readings yet: wait a few seconds")
        votes = collections.defaultdict(collections.Counter)
        for dets in hist:
            for sq in chess.SQUARES:
                votes[sq][dets[sq][0] if sq in dets else "."] += 1
        b = chess.Board(None)
        unsure = []
        for sq, c in votes.items():
            sym, k = c.most_common(1)[0]
            if k < len(hist) * 0.6:
                unsure.append(chess.square_name(sq))
            if sym != ".":
                b.set_piece_at(sq, chess.Piece.from_symbol(sym))
        b.turn = self.s.board.turn
        b.castling_rights = b.clean_castling_rights()
        if GS.looks_like_start(b):  # the starting setup: exact standard position (king e1, queen d1, White to move)
            b, unsure = chess.Board(), []
        return b.fen(), {"readings": len(hist), "pieces": len(b.piece_map()), "unsure": unsure, "valid": b.is_valid()}

    def job_manual(self, src, dst, sync=True):
        """Physical pick & place between any two squares (not a chess move)."""
        from chess_player.motion.pick_place import GraspFailed, Transfer, obstacles_for

        s_sq, d_sq = chess.parse_square(src), chess.parse_square(dst)
        b = self.s.board
        piece = b.piece_at(s_sq) or chess.Piece(chess.PAWN, chess.WHITE)
        if b.piece_at(d_sq) is not None:
            raise ValueError(f"{dst} is occupied")
        t = Transfer(piece.symbol(), tuple(square_center_mm(s_sq)), tuple(square_center_mm(d_sq)), f"manual {src}->{dst}")

        class _S:
            occupied = self.s.slots.occupied

        (t, obst), = obstacles_for(b, _S(), [t])
        self.say("moving", f"Manual: {src} -> {dst}")
        self.emit("manual", src=src, dst=dst)
        self.log(f"manual pick & place {piece.symbol()} {src} -> {dst}")
        self.s.parked = False
        from chess_player.motion.pick_place import PieceDropped

        if self.s.watcher:
            self.s.watcher.paused = True
        try:
            for attempt in range(C.RECOVERY_ATTEMPTS + 1):
                try:
                    self.s.pp.execute(t, obstacles=obst)
                    break
                except (GraspFailed, PieceDropped) as e:
                    where = self.recover(t, e) if attempt < C.RECOVERY_ATTEMPTS else None
                    if where == "done":
                        break
                    if where is None:
                        self.log(f"gave up: {e}")
                        self.emit("need_help_manual", src=src, dst=dst)
                        sync = False
                        break
                    t = Transfer(t.piece, tuple(where), t.dst, t.label + " (retry)")
        finally:
            if self.s.watcher:
                self.s.watcher.clear()
                self.s.watcher.paused = False
            self.s.park()
        if sync and b.piece_at(s_sq) is not None:
            self.san_before_edit = self.state()["moves"] + [f"({src}→{dst})"]
            b.remove_piece_at(s_sq)
            b.set_piece_at(d_sq, piece)
            self.log("game position updated")
        if self.s.has_vision:
            self.s.snapshot()
        self.say("idle" if self.mode == "paused" else "watching", f"Moved {src} -> {dst}.")

    # ------------------------------------------------ state for the UI
    def state(self):
        b = self.s.board
        tmp = b.root()  # position at the start of the current move stack (after any manual edit)
        sans = list(self.san_before_edit)
        for m in b.move_stack:
            sans.append(tmp.san(m))
            tmp.push(m)
        w = self.s.watcher
        return {
            "mode": self.mode, "phase": self.phase, "status": self.status, "help": self.help, "busy": self.busy,
            "fen": b.fen(), "turn": "white" if b.turn else "black", "robot_color": self.robot_color,
            "moves": sans, "last_move": b.peek().uci() if b.move_stack else None, "eval": self.eval,
            "vision": (self.s.vision if self.s.has_vision else "off"), "yolo": self.yolo_kind is not None, "yolo_kind": self.yolo_kind,
            "watcher_age": round(time.time() - w.latest()[0], 1) if w and w.latest()[0] else None,
            "arm": self.arm_mode, "plan": self.plan, "log": list(self.log_lines)[-14:],
            "robo": self.assistant.status() if getattr(self, "assistant", None) else None,
            "live_sync": self.live_sync,
            "hover": self.hover_info(),
            "torque": bool(getattr(self.s.arm, "torque_on", True)),
            "seen_diff": (self.seen or {}).get("diff", []) if self.seen and time.time() - self.seen["t"] < 5 else [],
            "seen_age": round(time.time() - self.seen["t"], 1) if self.seen else None,
        }

    def hover_info(self):
        """For the board hover: each piece's possible moves (legal if it were its side's turn) and what the
        camera saw on each square (symbol, confidence, height, note)."""
        b = self.s.board
        moves = {}
        for color in (chess.WHITE, chess.BLACK):
            t = b.copy(stack=False)
            t.turn = color
            t.ep_square = b.ep_square if color == b.turn else None
            gen = t.legal_moves if t.is_valid() else t.pseudo_legal_moves
            for m in gen:
                moves.setdefault(chess.square_name(m.from_square), []).append(chess.square_name(m.to_square))
        det = {}
        w = self.s.watcher
        if w is not None:
            t0, boxes = w.boxes
            if time.time() - t0 < 3.0:
                for sym, conf, x1, y1, x2, y2, sq, *rest in boxes:
                    if sq is not None:
                        h = rest[0] if rest else None
                        note = rest[1] if len(rest) > 1 else ""
                        name = chess.square_name(sq)
                        if name not in det or conf > det[name][1]:
                            det[name] = [sym, round(conf, 2), round(h, 0) if h else None, note]
        f = self.fused if self.fused and time.time() - self.fused["t"] < 5 else {}
        for name, i in f.get("info", {}).items():  # the fused reading: webcam + what the gripper camera says
            if i["web"] is None:  # seen by the robot eye only
                det[name] = [i["eye"][0], i["eye"][1], None, "seen by the robot eye only (hidden from the webcam)",
                             i["eye"], None]
                continue
            if name in det:
                det[name][:2] = i["web"]
                det[name] += [i["eye"], i["agree"]]
        eye = self.eye_status
        if self.eye is not None and eye == "calibrated":
            eye += " · watching" if f.get("eye_on") else " · waiting (arm not at rest)"
        return {"moves": {k: sorted(set(v)) for k, v in moves.items()}, "det": det, "eye": eye}

    def board_svg(self):
        b = self.s.board
        lm = b.peek() if b.move_stack else None
        arrows = []
        if self.help and self.help.get("kind") == "robot_by_hand":
            m = chess.Move.from_uci(self.help["move"])
            arrows = [chess.svg.Arrow(m.from_square, m.to_square, color="#e5484d")]
        fill = {}
        if self.seen and time.time() - self.seen["t"] < 5:
            for name in self.seen["diff"]:
                fill[chess.parse_square(name)] = "#ef4b5599"  # camera sees something else here
        return chess.svg.board(b, orientation=self.human_color(), lastmove=lm, arrows=arrows, size=420, fill=fill,
                               colors={"square light": "#e9dcc0", "square dark": "#a7794f",
                                       "square light lastmove": "#f3d37a", "square dark lastmove": "#cfa646",
                                       "margin": "#15171c", "coord": "#9aa3b2"})


# ---------------------------------------------------------------- HTTP
def overlay_ext(frame, cond):
    from chess_player.vision.aruco_homography import draw_overlay
    from chess_player.vision.ext_camera import view

    try:
        out = draw_overlay(frame, view(frame))
    except Exception:
        out = frame.copy()
    w = cond.s.watcher
    if w is not None:
        t, boxes = w.boxes
        if time.time() - t < 3.0:
            for sym, conf, x1, y1, x2, y2, sq, *_ in boxes:
                col = (245, 245, 245) if sym.isupper() else (40, 40, 40)
                cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), (80, 220, 255), 2)
                cv2.putText(out, f"{sym} {conf:.2f}", (int(x1), int(y1) - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)
    return out


def overlay_wrist(frame, cond):
    """Gripper image + its board grid and YOLO boxes (while it's judging pieces)."""
    eye = cond.eye
    if frame is None or eye is None or eye.view is None:
        return frame
    from chess_player.vision.aruco_homography import draw_overlay

    t, boxes = eye.boxes
    if time.time() - t > 3.0:
        return frame
    try:
        out = draw_overlay(frame, eye.view)
    except Exception:
        out = frame.copy()
    for sym, conf, x1, y1, x2, y2, sq, *_ in boxes:
        col = (245, 245, 245) if sym.isupper() else (40, 40, 40)
        cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), (80, 220, 255), 2)
        cv2.putText(out, f"{sym} {conf:.2f}", (int(x1), int(y1) - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)
    return out


def make_handler(cond, wrist):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, body, ctype):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, code=200):
            self._send(code, json.dumps(obj).encode(), "application/json")

        def _mjpeg(self, get_frame, fps):
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            blank = np.full((360, 640, 3), 24, np.uint8)
            cv2.putText(blank, "no signal", (250, 185), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (90, 90, 90), 2)
            try:
                while True:
                    f = get_frame()
                    ok, jpg = cv2.imencode(".jpg", f if f is not None else blank, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                     + str(len(jpg)).encode() + b"\r\n\r\n" + jpg.tobytes() + b"\r\n")
                    time.sleep(1.0 / fps)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def do_GET(self):
            path = self.path.split("?")[0]
            if path == "/":
                self._send(200, HTML.read_bytes(), "text/html; charset=utf-8")
            elif path == "/api/state":
                self._json(cond.state())
            elif path == "/board.svg":
                self._send(200, cond.board_svg().encode(), "image/svg+xml")
            elif path == "/stream/wrist_raw":  # the gripper camera without drawings (for tools / tests)
                self._mjpeg(wrist.get, 10)
            elif path == "/stream/wrist":
                self._mjpeg(lambda: overlay_wrist(wrist.get(), cond), 15)
            elif path == "/stream/ext_raw":  # the webcam as is (white-balanced), for other tools while we hold it
                def raw_frame():
                    if cond.ext is None:
                        return None
                    try:
                        return cond.ext.grab(max_age_s=2.0)
                    except RuntimeError:
                        return None

                self._mjpeg(raw_frame, 15)
            elif path == "/stream/ext":
                def ext_frame():
                    if cond.ext is None:
                        return None
                    try:
                        return overlay_ext(cond.ext.grab(max_age_s=2.0), cond)
                    except RuntimeError:
                        return None

                self._mjpeg(ext_frame, 8)
            else:
                self._send(404, b"not found", "text/plain")

        def do_POST(self):
            n = int(self.headers.get("Content-Length") or 0)
            try:
                body = json.loads(self.rfile.read(n) or b"{}")
            except ValueError:
                return self._json({"error": "bad json"}, 400)
            path = self.path.split("?")[0]
            if path == "/api/action":
                a = body.get("action")
                if a == "stop":
                    cond.emergency_stop()
                    return self._json({"ok": "stopped"})
                if a in ("torque_off", "torque_on"):
                    try:
                        cond.set_torque(a == "torque_on")
                    except Exception as e:
                        return self._json({"error": f"{type(e).__name__}: {e}"}, 500)
                    return self._json({"ok": a})
                if a in ("live_sync_on", "live_sync_off"):
                    cond.live_sync = a == "live_sync_on"
                    cond.log(f"live sync {'on' if cond.live_sync else 'off'}")
                    return self._json({"ok": a})
                if a == "calibrate_eye":
                    cond.jobs.put(("calibrate_eye", {}))
                    return self._json({"queued": "calibrating the robot eye"})
                if a in ("turn_white", "turn_black"):
                    cond.jobs.put(("set_turn", {"white": a == "turn_white"}))
                    return self._json({"queued": "white to move" if a == "turn_white" else "black to move"})
                if a not in ("play", "pause", "new_game", "home", "i_moved", "done_by_hand", "undo", "recheck"):
                    return self._json({"error": f"unknown action {a}"}, 400)
                if a == "pause":
                    cond.mode = "paused"  # takes effect at once, even while a job runs
                cond.jobs.put((a, {}))
                return self._json({"queued": a})
            if path == "/api/human_move":
                txt = str(body.get("move", "")).strip()
                b = cond.s.board
                if not cond.s.human_to_move():
                    who = "white" if b.turn else "black"
                    return self._json({"error": f"it's the robot's turn ({who}): press ▶ Play and let it move first, "
                                                f"then send your move"}, 400)
                try:
                    GS.parse_human_move(b, txt)
                except ValueError:
                    legal = ", ".join(sorted(b.san(m) for m in b.legal_moves)[:20])
                    return self._json({"error": f"'{txt}' isn't a legal move for you here. Legal moves: {legal}"}, 400)
                cond.jobs.put(("human_move", {"move": txt}))
                return self._json({"queued": txt})
            if path == "/api/ask":
                txt = str(body.get("text", "")).strip()
                a = getattr(cond, "assistant", None)
                if not txt or a is None:
                    return self._json({"error": "assistant is off" if a is None else "empty"}, 400)
                a.handle(txt, spoken=False)
                return self._json({"queued": "asked"})
            if path == "/api/voice":
                a, act, val = getattr(cond, "assistant", None), body.get("action"), body.get("value")
                if a is None:
                    return self._json({"error": "assistant is off"}, 400)
                sp, li = a.speaker, a.listener
                if act in ("mute", "unmute") and sp:
                    sp.muted = act == "mute"
                    if sp.muted:
                        sp.stop()
                elif act in ("mic_on", "mic_off") and li:
                    li.enabled = act == "mic_on"
                elif act == "volume" and sp:
                    sp.volume = max(0.0, min(1.5, float(val)))
                elif act == "speed" and sp:
                    sp.speed = max(0.6, min(1.6, float(val)))
                elif act == "stop_talking" and sp:
                    sp.stop()
                elif act in ("wake_on", "wake_off"):
                    C.VOICE_REQUIRE_WAKE = act == "wake_on"
                elif act == "sensitivity":  # higher value = needs louder speech
                    C.VOICE_SNR = max(1.5, min(8.0, float(val)))
                elif act == "say" and sp:
                    a.say(str(val or "Testing, one, two. Is this thing on?")[:300])
                else:
                    return self._json({"error": f"can't {act}"}, 400)
                return self._json({"ok": act})
            if path == "/api/position":
                fen = str(body.get("fen", "")).strip()
                if body.get("preset") == "start":
                    fen = chess.STARTING_FEN
                try:
                    chess.Board(fen)
                except ValueError as e:
                    return self._json({"error": f"bad position: {e}"}, 400)
                cond.jobs.put(("set_position", {"fen": fen, "from_camera": bool(body.get("from_camera"))}))
                return self._json({"queued": "position"})
            if path == "/api/read_board":
                try:
                    fen, info = cond.read_board()
                except ValueError as e:
                    return self._json({"error": str(e)}, 400)
                return self._json({"fen": fen, **info})
            if path == "/api/manual":
                src, dst = str(body.get("from", "")).strip().lower(), str(body.get("to", "")).strip().lower()
                try:
                    chess.parse_square(src), chess.parse_square(dst)
                except ValueError:
                    return self._json({"error": "squares like e4 and a3"}, 400)
                cond.jobs.put(("manual", {"src": src, "dst": dst, "sync": bool(body.get("sync", True))}))
                return self._json({"queued": f"{src}->{dst}"})
            self._json({"error": "not found"}, 404)

    return H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8770)
    ap.add_argument("--dry-run", action="store_true", help="simulated arm")
    ap.add_argument("--skill", type=int, help="Stockfish skill level 0-20")
    ap.add_argument("--no-voice", action="store_true", help="no Robo (microphone, speech, language model)")
    ap.add_argument("--no-mic", action="store_true", help="Robo talks and answers typed questions, but doesn't listen")
    ap.add_argument("--mute", action="store_true", help="start with Robo's voice muted")
    ap.add_argument("--no-webcam", action="store_true", help="don't connect to the third-person webcam")
    a = ap.parse_args()
    if a.no_webcam:
        C.EXT_CORNERS = C.DATA / "__no_webcam__.json"  # makes Session and the UI skip the webcam
    wrist = LiveCam()
    cond = Conductor(dry_run=a.dry_run, skill=a.skill, wrist=wrist)
    if not a.no_voice:
        from chess_player.voice.assistant import Assistant

        cond.assistant = Assistant(cond, wrist, use_mic=not a.no_mic)
        if a.mute and cond.assistant.speaker:
            cond.assistant.speaker.muted = True
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), make_handler(cond, wrist))
    srv.daemon_threads = True
    print(f"UI at http://localhost:{a.port}/", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        cond.mode = "paused"
        try:
            cond.s.close()  # arm home, holding; engine + cameras closed
        except Exception as e:
            print("close:", e)


if __name__ == "__main__":
    main()
