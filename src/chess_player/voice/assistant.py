"""Robo, the talking side of the pipeline: hears you, follows spoken/typed commands, answers questions (with the
camera images), and comments on the game at major moments.

Commands are matched by rules (reliable, instant), never by the language model, so a misheard sentence can't
move the arm in a surprising way. Spoken commands need the wake word ("robot" / "robo") unless VOICE_REQUIRE_WAKE
is off; typed ones don't. Anything that isn't a command is a question for the vision-language model.
"""
import collections
import random
import re
import threading
import time

import chess

from chess_player import config as C
from chess_player.game import state as GS

PIECE_WORDS = {"knight": "N", "night": "N", "bishop": "B", "rook": "R", "castle": "R", "queen": "Q", "king": "K",
               "pawn": ""}
NUM_WORDS = {"one": "1", "two": "2", "to": None, "three": "3", "four": "4", "for": "4", "five": "5", "six": "6",
             "seven": "7", "eight": "8", "ate": "8"}
SAY_PIECE = {"N": "knight", "B": "bishop", "R": "rook", "Q": "queen", "K": "king"}


def spoken_san(san):
    if san.startswith("O-O-O"):
        s = "castles queenside"
    elif san.startswith("O-O"):
        s = "castles kingside"
    else:
        core = san.rstrip("+#")
        piece = SAY_PIECE.get(core[0], "pawn")
        promo = ""
        if "=" in core:
            core, p = core.split("=")
            promo = f", promoting to a {SAY_PIECE.get(p[0], p)}"
        sq = re.findall(r"[a-h][1-8]", core)[-1]
        s = f"{piece} {'takes on' if 'x' in core else 'to'} {sq[0]} {sq[1]}{promo}"
    if san.endswith("#"):
        s += ", checkmate"
    elif san.endswith("+"):
        s += ", check"
    return s


def eval_cp(e):
    """'+0.35' -> 35, 'mate in 3' -> 10000, 'mate in -2' -> -10000 (White's view)."""
    m = re.match(r"mate in (-?\d+)", str(e))
    if m:
        return 10000 if int(m.group(1)) > 0 else -10000
    try:
        return int(float(e) * 100)
    except ValueError:
        return 0


def normalise(text):
    t = text.lower()
    t = re.sub(r"[^\w\s'-]", " ", t)
    words = []
    for w in t.split():
        words.append(NUM_WORDS.get(w, w) or "to")
    t = " ".join(words)
    t = re.sub(r"\b([a-h])\s*-?\s*([1-8])\b", r"\1\2", t)  # "e 4" -> "e4"
    return t


LINES = {  # short on purpose: Robo speaks only when it matters
    "ready": ["Ready."],
    "play": ["Game on."],
    "pause": ["Paused."],
    "new_game": ["New game."],
    "home": ["Going home."],
    "need_help": ["Please play {move} for me."],
    "fix_board": ["Please check the board."],
    "which_move": ["Which move did you play?"],
    "game_over_win": ["Checkmate. Good game!"],
    "game_over_lose": ["You win. Well played!"],
    "game_over_draw": ["Draw."],
    "error": ["Problem: {why}"],
    "manual": ["{src} to {dst}."],
    "position_set": ["Board updated."],
    "game_start": ["Game on."],
    "robot_plan": ["{piece} {src} to {dst}."],
    "robot_plan_capture": ["{piece} takes {dst}."],
    "robot_plan_castle": ["Castling."],
    "robot_done": ["Your move."],
    "robot_done_check": ["Check. Your move."],
    "human_great": ["Strong move."],
    "human_blunder": ["That's a blunder."],
    "recovering": ["Dropped it. Retrying."],
    "found_piece": ["Found it on {square}."],
    "need_help_manual": ["Please put the piece on {dst}."],
}


def line(kind, **kw):
    return random.choice(LINES[kind]).format(**kw)


class Assistant:
    def __init__(self, cond, wrist_cam, use_mic=True):
        self.cond, self.wrist = cond, wrist_cam
        self.chat = collections.deque(maxlen=40)
        self.thinking = False
        self._last_said = ("", 0.0)
        self.speaker = self.listener = self.brain = None
        try:
            from chess_player.voice.audio import Speaker

            self.speaker = Speaker()
        except Exception as e:
            self.note(f"voice output off ({type(e).__name__}: {e})")
        if C.VLM_ENABLED:
            from chess_player.voice.brain import Brain

            self.brain = Brain()
        self._mic_starting = use_mic
        if use_mic:
            threading.Thread(target=self._start_mic, daemon=True).start()
        cond.on_event = self.on_event

    def _start_mic(self):
        try:
            from chess_player.voice.audio import Listener

            self.listener = Listener(self.on_heard, self.speaker)
            self.note("microphone ready (switched " + ("on" if C.VOICE_MIC_ON_AT_START else "off: turn it on with the "
                      "Microphone switch") + "); say \"robot, ...\"")
        except Exception as e:
            self.note(f"microphone off ({type(e).__name__}: {e})")
        finally:
            self._mic_starting = False

    # ------------------------------------------------ output
    def note(self, text):
        self.chat.append({"who": "sys", "text": text, "t": time.strftime("%H:%M:%S")})

    def say(self, text):
        last, t = self._last_said
        if text == last and time.time() - t < 5:
            return
        self._last_said = (text, time.time())
        self.chat.append({"who": "robo", "text": text, "t": time.strftime("%H:%M:%S")})
        if self.speaker:
            self.speaker.say(text)

    # ------------------------------------------------ game events from the conductor
    @staticmethod
    def _sq(name):
        return f"{name[0]} {name[1]}" if name else ""

    def _quip(self, facts, ask, max_age_s=8.0):
        """One short spoken comment from the language model, in the background; dropped if it comes too late."""
        if self.brain is None or self.brain.status != "ready":
            return
        t0 = time.time()

        def run():
            try:
                txt = self.brain.ask(f"{facts}\n\n{ask} One short sentence, playful, no move notation, don't invent "
                                     "tactics you weren't told about.", self.context(), images=(), max_new_tokens=40)
                if txt and time.time() - t0 < max_age_s:
                    self.say(txt)
            except Exception as e:
                print(f"[robo] quip failed: {e}", flush=True)

        threading.Thread(target=run, daemon=True).start()

    def on_event(self, kind, **d):
        if not C.VOICE_COMMENTARY:  # game commentary off: Robo only answers commands and questions
            return
        try:
            if kind == "robot_move":
                if d.get("castle"):
                    self.say(line("robot_plan_castle"))
                elif d.get("victim"):
                    self.say(line("robot_plan_capture", piece=d["piece"], victim=d["victim"], dst=self._sq(d["dst"])))
                else:
                    self.say(line("robot_plan", piece=d["piece"], src=self._sq(d["src"]), dst=self._sq(d["dst"])))
                if C.VOICE_QUIPS:
                    self._quip(f"You (Robo) are about to play {spoken_san(d['san'])}; Stockfish evaluation is {d.get('eval')}.",
                               "Say why you like this move, cheekily.")
            elif kind == "robot_done":
                self.say(line("robot_done_check" if d.get("check") else "robot_done"))
            elif kind == "human_move":
                swing = d.get("swing_for_robot", 0)
                # Only remark on big swings; ordinary moves get no comment (the board shows them).
                if swing > 200:
                    better = f" Better was {spoken_san(d['better'])}." if d.get("better") else ""
                    self.say(line("human_blunder") + better)
                elif swing < -100:
                    self.say(line("human_great"))
            elif kind == "found_piece":
                self.say(line("found_piece", square=self._sq(d["square"])))
            elif kind == "need_help_manual":
                self.say(line("need_help_manual", dst=self._sq(d["dst"])))
            elif kind == "need_help":
                self.say(line("need_help", move=spoken_san(d["san"])))
            elif kind == "game_over":
                self.say(line({"robot": "game_over_win", "human": "game_over_lose"}.get(d.get("winner"), "game_over_draw")))
            elif kind == "error":
                self.say(line("error", why=str(d.get("why", ""))[:80]))
            elif kind == "manual":
                self.say(line("manual", src=f"{d['src'][0]} {d['src'][1]}", dst=f"{d['dst'][0]} {d['dst'][1]}"))
            elif kind in LINES:
                self.say(line(kind))
        except Exception as e:
            print(f"[robo] event {kind}: {e}", flush=True)

    # ------------------------------------------------ input
    def on_heard(self, text):
        t = normalise(text)
        wake = re.search(r"\b(" + "|".join(C.VOICE_WAKE_WORDS) + r")\b", t)
        if C.VOICE_REQUIRE_WAKE and not wake:
            self.chat.append({"who": "heard", "text": text, "t": time.strftime("%H:%M:%S")})
            return
        self.handle(text, spoken=True)

    def handle(self, text, spoken=False):
        self.chat.append({"who": "you", "text": text, "t": time.strftime("%H:%M:%S")})
        t = normalise(text)
        t = re.sub(r"\b(hey|ok|okay)?\s*(" + "|".join(C.VOICE_WAKE_WORDS) + r")\b[\s,]*", " ", t).strip()
        reply = self.command(t)
        if reply is not None:
            if reply:
                self.say(reply)
            return
        threading.Thread(target=self._answer, args=(text,), daemon=True).start()

    def command(self, t):
        """Rule-based commands. Returns a reply ('' = act silently, the event will speak) or None if not a command."""
        c = self.cond
        if re.search(r"\b(be quiet|shut up|mute|stop talking|silence)\b", t):
            if self.speaker:
                self.speaker.say("Fine. Going quiet.")
                self.speaker.muted = True
            return ""
        if re.search(r"\b(unmute|talk again|speak again|you can talk)\b", t):
            if self.speaker:
                self.speaker.muted = False
            return "I'm back. Did you miss me?"
        if c.help and re.search(r"\b(done|did it|finished|fixed|there you go)\b", t):
            c.jobs.put(("done_by_hand", {}))
            return "Thanks!"
        if re.search(r"\b(hint|suggest|what should i (play|do)|best move|help me)\b", t):
            b = c.s.board
            if b.turn == (c.robot_color == "white") or b.is_game_over() or not b.is_valid():
                return "No hints right now, it's not your move."
            try:
                best = c.s.engine.best_move(b, 0.5)
                return f"If I were you, I'd play {spoken_san(b.san(best))}. But don't tell anyone I helped."
            except Exception:
                return "My engine is thinking about something else. Try again in a second."
        if re.search(r"\b(go home|home position|go to home)\b", t):
            c.jobs.put(("home", {}))
            return ""
        if re.search(r"\b(new game|restart|reset the board|start over)\b", t):
            c.jobs.put(("new_game", {}))
            return ""
        if re.search(r"\b(undo|take back|takeback)\b", t):
            c.jobs.put(("undo", {}))
            return "Taking that back. Put the pieces back for me?"
        if re.search(r"\b(pause|stop|wait|hold on|freeze)\b", t):
            c.mode = "paused"
            c.jobs.put(("pause", {}))
            return ""
        human = re.match(r"^(i (played|moved|move|go|play)|my move( is)?)\b\s*(.*)$", t)
        sq = re.findall(r"\b[a-h][1-8]\b", t)
        if not human and re.search(r"\b(move|pick|put|place|shift|carry)\b", t) and len(sq) >= 2:
            c.jobs.put(("manual", {"src": sq[0], "dst": sq[1], "sync": True}))
            return ""
        if human:
            rest = human.group(4).strip()
            if not rest:
                c.jobs.put(("i_moved", {}))
                return "Let me look."
            move = self.parse_spoken_move(rest)
            if move is None:
                return "I didn't get that move. Try something like knight f 3, or e 2 to e 4."
            c.jobs.put(("human_move", {"move": move}))
            return ""
        if re.search(r"\b(play|start|resume|continue|let's go|lets go|let's play|lets play|your turn)\b", t) \
                and not sq:
            c.jobs.put(("play", {}))
            return ""
        # a bare move ("knight f6", "castle", "e5") on the human's turn counts as the human's move
        if len(t.split()) <= 5 and c.s.board.turn != (c.robot_color == "white"):
            move = self.parse_spoken_move(t)
            if move is not None:
                c.jobs.put(("human_move", {"move": move}))
                return ""
        return None

    def parse_spoken_move(self, rest):
        b = self.cond.s.board
        if re.search(r"\b(castle|castles|castling)\b", rest):
            san = "O-O-O" if re.search(r"\b(long|queen ?side)\b", rest) else "O-O"
            return san if self._legal(san) else None
        sq = re.findall(r"\b[a-h][1-8]\b", rest)
        if len(sq) >= 2 and self._legal(sq[0] + sq[1]):
            return sq[0] + sq[1]
        if sq:
            piece = next((PIECE_WORDS[w] for w in rest.split() if w in PIECE_WORDS), "")
            cap = "x" if re.search(r"\b(takes|take|captures|x)\b", rest) else ""
            for cand in (f"{piece}{cap}{sq[-1]}", f"{piece}{sq[-1]}", f"{piece}x{sq[-1]}"):
                if self._legal(cand):
                    return cand
            # pawn captures need the file: let python-chess find the unique legal move to that square
            target = chess.parse_square(sq[-1])
            ms = [m for m in b.legal_moves if m.to_square == target and
                  (not piece or b.piece_at(m.from_square).symbol().upper() == piece)]
            if len(ms) == 1:
                return ms[0].uci()
        return None

    def _legal(self, txt):
        try:
            GS.parse_human_move(self.cond.s.board, txt)
            return True
        except ValueError:
            return False

    # ------------------------------------------------ questions -> vision-language model
    def context(self):
        c, b = self.cond, self.cond.s.board
        st = c.state()
        pieces = {}
        for color in (chess.WHITE, chess.BLACK):
            items = [f"{chess.piece_name(p.piece_type)} {chess.square_name(sq)}"
                     for sq, p in sorted(b.piece_map().items()) if p.color == color]
            pieces[color] = f"{len(items)} pieces: " + ", ".join(items)
        return (f"you (Robo) play {c.robot_color}; it is {'your' if b.turn == (c.robot_color == 'white') else 'the human'}'s "
                f"turn; moves so far: {' '.join(st['moves'][-12:]) or 'none'}; Stockfish evaluation (White's view): "
                f"{c.eval}; status: {c.status}. Position (from the game record, reliable): White has {pieces[chess.WHITE]}. "
                f"Black has {pieces[chess.BLACK]}.")

    def _answer(self, question):
        if self.brain is None:
            self.say("I only understand commands right now, my language brain is switched off.")
            return
        if self.brain.status == "loading":
            self.say("Give me a second, my brain is still booting up.")
        self.thinking = True
        try:
            ext = None
            if self.cond.ext is not None:
                try:
                    ext = self.cond.ext.grab(max_age_s=2.0)
                except RuntimeError:
                    ext = None
            wrist = self.wrist.get()
            q = question.lower()
            wants_web = re.search(r"\b(webcam|board|table|pieces?|see|look|camera|watching)\b", q)
            wants_grip = re.search(r"\b(gripper|hand|claw|your camera|robot camera|holding)\b", q)
            if wants_grip and wrist is None:
                self.say("My gripper camera isn't sending pictures right now, so I'm blind on that side.")
                return
            if wants_web and not wants_grip and ext is None and wrist is None:
                self.say("My cameras aren't sending pictures right now. I can still tell you about the game from memory.")
                return
            reply = self.brain.ask(question, self.context(), images=[wrist, ext])
            self.say(reply or "I have no idea, honestly.")
        except Exception as e:
            self.say(f"My brain hiccuped: {type(e).__name__}.")
            print(f"[robo] answer failed: {e}", flush=True)
        finally:
            self.thinking = False

    # ------------------------------------------------ UI
    def status(self):
        l = self.listener
        return {
            "mic": (l.state if l and l.enabled else "off") if l else ("starting" if self._mic_starting else "off"),
            "level": round(l.level) if l else 0,
            "threshold": round(l.threshold) if l else 0,
            "wake_required": C.VOICE_REQUIRE_WAKE,
            "wake_words": C.VOICE_WAKE_WORDS[:2],
            "sensitivity": C.VOICE_SNR,
            "speaking": bool(self.speaker and self.speaker.speaking),
            "saying": self.speaker.current if self.speaker else "",
            "volume": self.speaker.volume if self.speaker else 0,
            "speed": self.speaker.speed if self.speaker else 1,
            "muted": bool(self.speaker and self.speaker.muted),
            "voice_out": self.speaker is not None,
            "brain": self.brain.status if self.brain else "off",
            "thinking": self.thinking,
            "chat": list(self.chat)[-24:],
        }
