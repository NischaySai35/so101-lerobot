"""Tools the Claude orchestrator can call. Each wraps a Session method and returns a JSON-able dict.

Claude never picks chess moves: execute_robot_move only accepts the move get_engine_move returned.
"""
import json

import chess

from chess_player.game import state
from chess_player.motion.ik import Unreachable
from chess_player.motion.pick_place import GraspFailed

TOOLS = [
    {
        "name": "get_game_state",
        "description": "Current position (FEN), whose turn it is, which color the robot plays, the move list, "
                     "Stockfish's evaluation, and whether the game is over.",
        "input_schema": {"type": "object", "properties": {}, "required": [], "additionalProperties": False},
    },
    {
        "name": "wait_for_human_move",
        "description": "Ask the human to make their move on the physical board, wait until they confirm, then "
                       "look at the board and infer the move. Returns the move if vision is confident, otherwise "
                       "the top candidates; then use confirm_human_move with the move the human states.",
        "input_schema": {"type": "object", "properties": {}, "required": [], "additionalProperties": False},
    },
    {
        "name": "confirm_human_move",
        "description": "Record the human's move (SAN like 'Nf3' or UCI like 'g1f3') after vision was unsure or "
                       "the human typed it. Fails if illegal.",
        "input_schema": {"type": "object", "properties": {"move": {"type": "string"}}, "required": ["move"],
                         "additionalProperties": False},
        "strict": True,
    },
    {
        "name": "get_engine_move",
        "description": "Ask Stockfish for the robot's move in the current position.",
        "input_schema": {"type": "object", "properties": {}, "required": [], "additionalProperties": False},
    },
    {
        "name": "check_robot_move",
        "description": "Feasibility report for a move without moving the arm: per piece transfer, travel height, "
                       "grasp tilt, estimated time, or why it can't be done (unreachable, would hit a piece).",
        "input_schema": {"type": "object", "properties": {"move": {"type": "string", "description": "UCI"}},
                         "required": ["move"], "additionalProperties": False},
        "strict": True,
    },
    {
        "name": "execute_robot_move",
        "description": "Physically play the engine's move with the arm (handles captures, castling, en passant, "
                       "promotion), then check the board with the camera. Only the move from get_engine_move "
                       "is accepted. On failure the result says what went wrong; ask the human for help.",
        "input_schema": {"type": "object", "properties": {"move": {"type": "string", "description": "UCI"}},
                         "required": ["move"], "additionalProperties": False},
        "strict": True,
    },
    {
        "name": "ask_human",
        "description": "Say something to the human and wait for their typed reply (e.g. 'please finish the move by "
                       "hand', 'fix the piece on e4'). Returns what they typed.",
        "input_schema": {"type": "object", "properties": {"message": {"type": "string"}}, "required": ["message"],
                         "additionalProperties": False},
        "strict": True,
    },
    {
        "name": "robot_move_done_by_hand",
        "description": "Record that the human physically completed the robot's engine move after the arm failed.",
        "input_schema": {"type": "object", "properties": {"move": {"type": "string", "description": "UCI"}},
                         "required": ["move"], "additionalProperties": False},
        "strict": True,
    },
]


class ToolBox:
    def __init__(self, session, say=print, ask=input):
        self.s = session
        self.say, self.ask = say, ask
        self.engine_suggestion = None

    def run(self, name, args):
        try:
            return getattr(self, name)(**args)
        except Exception as e:  # report to the model as a tool error
            return {"error": f"{type(e).__name__}: {e}"}

    def get_game_state(self):
        return self.s.describe()

    def wait_for_human_move(self):
        if not self.s.human_to_move():
            return {"error": "it's not the human's turn"}
        typed = self.ask("\n[robot] Your move. Press Enter when done (or type it): ").strip()
        if typed:
            return self.confirm_human_move(typed)
        if not self.s.has_vision:
            return {"error": "no camera: the human must type the move"}
        inf = self.s.see_human_move()
        if inf.move is None:
            return {"recognized": False, "reason": inf.reason,
                    "candidates": [self.s.board.san(m) for m, _ in inf.ranked[:3]]}
        san = self.s.board.san(inf.move)
        self.s.push_human(inf.move)
        return {"recognized": True, "move": san, "fen": self.s.board.fen()}

    def confirm_human_move(self, move):
        m = state.parse_human_move(self.s.board, move)
        san = self.s.board.san(m)
        self.s.push_human(m)
        if self.s.has_vision:
            self.s.snapshot()
        return {"recorded": san, "fen": self.s.board.fen()}

    def get_engine_move(self):
        if self.s.human_to_move() or self.s.board.is_game_over():
            return {"error": "not the robot's turn"}
        m = self.s.engine_move()
        self.engine_suggestion = m
        return {"uci": m.uci(), "san": self.s.board.san(m)}

    def check_robot_move(self, move):
        return self.s.check(chess.Move.from_uci(move))

    def execute_robot_move(self, move):
        m = chess.Move.from_uci(move)
        if m != self.engine_suggestion:
            return {"error": f"refused: only the engine's move {self.engine_suggestion} may be executed"}
        try:
            done = self.s.execute(m)
        except Unreachable as e:
            return {"ok": False, "problem": "unreachable", "details": str(e), "board_unchanged": True}
        except GraspFailed as e:
            return {"ok": False, "problem": "grasp_failed", "details": str(e),
                    "note": "the move may be half done; ask the human to finish it, then robot_move_done_by_hand"}
        self.engine_suggestion = None
        ok, details = self.s.verify_position()
        return {"ok": True, "transfers": done, "board_check": {"matches": ok, "details": details}}

    def robot_move_done_by_hand(self, move):
        m = chess.Move.from_uci(move)
        if m != self.engine_suggestion:
            return {"error": f"expected the engine's move {self.engine_suggestion}"}
        self.s.push_robot_by_hand(m)
        self.engine_suggestion = None
        if self.s.has_vision:
            self.s.snapshot()
        return {"recorded": move, "fen": self.s.board.fen()}

    def ask_human(self, message):
        return {"reply": self.ask(f"\n[robot] {message}\n> ")}


def to_content(result) -> str:
    return json.dumps(result)
